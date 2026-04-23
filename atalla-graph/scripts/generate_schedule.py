#!/usr/bin/env python3
"""
Emit graph_schedule.c by consuming an FX GraphModule that already has DRAM
addresses (node.meta["dram_addr"]) and tensor metadata attached.
"""

from __future__ import annotations

import argparse
import builtins
import operator
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.fx import GraphModule, Node

from graph.memoryallocator import (
    TILE_HEIGHT,
    TILE_WIDTH,
    TILE_BYTES,
    VIEW_FUNCTIONS,
    VIEW_METHODS,
)
from scripts.policies import (
    build_add_plan,
    build_conv_plan,
    build_matmul_plan,
    build_maxpool_plan,
    build_relu_plan,
    build_softmax_plan,
)
from scripts.tile_manager import plan_tile_moves
from scripts.tile_structures import OpPlan, StepKind

MAX_RANK = 8

MATMUL_TARGETS = {
    torch.matmul,
    torch.ops.aten.matmul.default,
}

ADD_TARGETS = {
    operator.add,
    torch.add,
    torch.ops.aten.add.Tensor,
    torch.ops.aten.add_.Tensor,
}

RELU_TARGETS = {
    torch.relu,
    torch.nn.functional.relu,
    torch.ops.aten.relu.default,
}

SOFTMAX_TARGETS = {
    torch.softmax,
    torch.nn.functional.softmax,
}
if hasattr(torch.ops.aten, "_softmax"):
    SOFTMAX_TARGETS.add(torch.ops.aten._softmax.default)

MAXPOOL_TARGETS = {
    torch.nn.functional.max_pool2d,
}

MUL_TARGETS = {
    operator.mul,
    torch.mul,
    torch.ops.aten.mul.Tensor,
}

CONV_TARGETS = {
    torch.nn.functional.conv2d,
}

# Functions like builtins.getattr don't need kernels
IGNORED_FUNCTIONS = {builtins.getattr}

# Regex that matches any non-alphanumeric character
_IDENT_RE = re.compile(r"\W+")



def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


#Remove non-alphanumeric chars and ensure it doesn't start with a digit
def _sanitize(name: str) -> str: 
    cleaned = _IDENT_RE.sub("_", name)
    if not cleaned:
        cleaned = "tensor"
    if cleaned[0].isdigit():
        cleaned = f"n_{cleaned}"
    return cleaned


# Detect whether a node is just a view into another tensor.
def _view_source(node: Node) -> Optional[Node]:
    if node.op == "call_method":
        method = node.target.strip("'")
        if method in VIEW_METHODS:
            arg = node.args[0]
            if isinstance(arg, Node):
                return arg
    if node.op == "call_function" and node.target in VIEW_FUNCTIONS:
        arg = node.args[0]
        if isinstance(arg, Node):
            return arg
    return None


# Pad shape/tiles arrays out to MAX_RANK with zeros.
def _pad(values: Sequence[int]) -> List[int]:
    out = list(values)
    if len(out) > MAX_RANK:
        raise ValueError(f"Rank {len(out)} exceeds MAX_RANK={MAX_RANK}")
    out.extend([0] * (MAX_RANK - len(out)))
    return out


# Grab tensor shape from FX metadata
def _shape_from_meta(node: Node) -> List[int]:
    tensor_meta = node.meta.get("tensor_meta")
    if tensor_meta is None or tensor_meta.shape is None:
        raise ValueError(f"Missing tensor_meta.shape for node {node.name}")
    shape = [int(dim) for dim in tensor_meta.shape]
    if not shape:
        raise ValueError(f"Scalar tensor encountered at node {node.name}")
    return shape


# Compute tile counts per dimension
def _tiles_for_shape(shape: Sequence[int]) -> List[int]:
    dims = list(shape)
    if len(dims) == 1:
        return [_ceil_div(max(1, dims[0]), TILE_WIDTH)]
    if len(dims) == 2:
        return [
            _ceil_div(max(1, dims[0]), TILE_HEIGHT),
            _ceil_div(max(1, dims[1]), TILE_WIDTH),
        ]
    tiles = [max(1, dim) for dim in dims[:-2]]
    tiles.append(_ceil_div(max(1, dims[-2]), TILE_HEIGHT))
    tiles.append(_ceil_div(max(1, dims[-1]), TILE_WIDTH))
    return tiles


# Multiply tile counts to get total tile slots in a tensor
def _tile_count(values: Iterable[int]) -> int:
    product = 1
    for value in values:
        product *= max(1, value)
    return product


# Normalize stride/pad args into (height, width) tuples
def _as_pair(value: Sequence[int] | int) -> Tuple[int, int]:
    if isinstance(value, Sequence):
        if len(value) == 2:
            return int(value[0]), int(value[1])
        if len(value) == 1:
            return int(value[0]), int(value[0])
    return int(value), int(value)


@dataclass
class TensorSpec:
    identifier: str
    shape: List[int]
    tiles_per_dim: List[int]
    base_addr: int
    count: int = field(init=False)
    tiles: List[int] = field(init=False)

    def __post_init__(self) -> None:
        self.count = max(1, _tile_count(self.tiles_per_dim))
        self.tiles = [self.base_addr + i * TILE_BYTES for i in range(self.count)]


# Build a TensorSpec (tiles + metadata) for a graph node
def _tensor_spec_for(node: Node) -> TensorSpec:
    dram_addr = node.meta.get("dram_addr")
    if dram_addr is None:
        raise ValueError(f"Node {node.name} is missing dram_addr metadata")
    base_addr = int(str(dram_addr), 16)

    tensor_meta = node.meta.get("tensor_meta")
    if tensor_meta is None:
        raise ValueError(f"Missing tensor_meta for node {node.name}")
    if getattr(tensor_meta, "dtype", torch.bfloat16) != torch.bfloat16:
        raise ValueError(f"Node {node.name} has unsupported dtype {tensor_meta.dtype}")

    shape = _shape_from_meta(node)
    tiles = _tiles_for_shape(shape)

    expected_bytes = int(node.meta.get("bytes", 0))
    actual_bytes = _tile_count(tiles) * TILE_BYTES
    if expected_bytes and expected_bytes != actual_bytes:
        raise ValueError(
            f"Node {node.name} reserve mismatch: meta bytes={expected_bytes}, "
            f"derived={actual_bytes}"
        )

    return TensorSpec(
        identifier=_sanitize(node.name),
        shape=shape,
        tiles_per_dim=tiles,
        base_addr=base_addr,
    )


# Write C structs for a tensor's tiles and global descriptor.
def _render_tensor(spec: TensorSpec) -> str:
    tiles_lines = ",\n    ".join(f"{{ .base_addr = 0x{addr:08X}u }}" for addr in spec.tiles)
    tile_block = (
        f"static TileDesc32 tensor_{spec.identifier}_tiles[{spec.count}] = {{\n"
        f"    {tiles_lines}\n"
        f"}};\n"
    )
    shape_init = ", ".join(str(x) for x in _pad(spec.shape))
    tiles_init = ", ".join(str(x) for x in _pad(spec.tiles_per_dim))
    gt_block = (
        f"static GlobalTile tensor_{spec.identifier}_gt = {{\n"
        f"    .rank = {len(spec.shape)},\n"
        f"    .shape = {{{shape_init}}},\n"
        f"    .tiles_per_dim = {{{tiles_init}}},\n"
        f"    .count = {spec.count},\n"
        f"    .tiles = tensor_{spec.identifier}_tiles,\n"
        f"}};\n"
    )
    return tile_block + "\n" + gt_block


# return the pointer string of a node's tensor
def _tensor_ref(node: Node, specs: Dict[Node, TensorSpec]) -> str:
    try:
        spec = specs[node]
    except KeyError as exc:
        raise KeyError(f"Missing tensor spec for node {node.name}") from exc
    return f"&tensor_{spec.identifier}_gt"


#  Get a  positional argument from an FX node
def _get_node_arg(node: Node, position: int, *, fallback: Optional[str] = None) -> Node:
    if len(node.args) > position and isinstance(node.args[position], Node):
        return node.args[position]
    if fallback:
        candidate = node.kwargs.get(fallback)
        if isinstance(candidate, Node):
            return candidate
    raise ValueError(f"Node {node.name} lacks argument {position}")


def _kernel_missing(op_name: str, node: Node) -> NotImplementedError:
    return NotImplementedError(f"Kernel does not exist for {op_name} (node {node.name})")


# Prepare lhs/rhs for add kernels.
def _add_operands(
    node: Node,
    specs: Dict[Node, TensorSpec],
) -> Tuple[Node, Node]:
    lhs_obj: object
    rhs_obj: object

    if len(node.args) > 0:
        lhs_obj = node.args[0]
    else:
        lhs_obj = node.kwargs.get("input")
    if len(node.args) > 1:
        rhs_obj = node.args[1]
    else:
        rhs_obj = node.kwargs.get("other")

    if not isinstance(lhs_obj, Node):
        raise _kernel_missing("add", node)
    if not isinstance(rhs_obj, Node):
        if isinstance(rhs_obj, (int, float)):
            raise _kernel_missing("add_scalar", node)
        raise _kernel_missing("add", node)

    lhs = lhs_obj
    rhs = rhs_obj
    alpha = float(node.kwargs.get("alpha", 1.0))
    if alpha != 1.0:
        raise _kernel_missing("add_alpha", node)
    if lhs not in specs or rhs not in specs:
        raise ValueError(f"Add node {node.name} operands missing tensor specs")
    return lhs, rhs


# Collect TensorSpec entries and lookup map for every bf16 tensor node.
def _collect_tensor_specs(
    gm: GraphModule,
) -> Tuple[List[TensorSpec], Dict[Node, TensorSpec]]:
    tensor_specs: List[TensorSpec] = []
    specs_by_node: Dict[Node, TensorSpec] = {}
    for node in gm.graph.nodes:
        if node.op == "output":
            continue
        tensor_meta = node.meta.get("tensor_meta")
        if tensor_meta is None:
            continue
        if getattr(tensor_meta, "dtype", torch.bfloat16) != torch.bfloat16:
            continue
        spec = _tensor_spec_for(node)
        tensor_specs.append(spec)
        specs_by_node[node] = spec
    return tensor_specs, specs_by_node


def _mat_dims_from_spec(spec: TensorSpec) -> Tuple[int, int, int, int]:
    rank = len(spec.shape)
    if rank == 1:
        rows = 1
        cols = int(spec.shape[-1])
        row_tiles = 1
        col_tiles = int(spec.tiles_per_dim[-1])
    else:
        rows = int(spec.shape[-2])
        cols = int(spec.shape[-1])
        row_tiles = int(spec.tiles_per_dim[-2])
        col_tiles = int(spec.tiles_per_dim[-1])
    return rows, cols, row_tiles, col_tiles


# Build compile-time op plans.
def _build_op_plans(
    gm: GraphModule,
    specs_by_node: Dict[Node, TensorSpec],
    attr_nodes: Dict[str, Node],
) -> List[OpPlan]:
    plans: List[OpPlan] = []
    for node in gm.graph.nodes:
        if node.op == "call_module":
            module = gm.get_submodule(node.target)
            if isinstance(module, torch.nn.Linear):
                input_node = _get_node_arg(node, 0)
                weight_attr = f"{node.target}.weight"
                weight_node = attr_nodes.get(weight_attr)
                if weight_node is None:
                    raise ValueError(f"Missing get_attr node for {weight_attr}")
                input_spec = specs_by_node[input_node]
                weight_spec = specs_by_node[weight_node]
                dst_spec = specs_by_node[node]
                _, _, m_tiles, k_tiles = _mat_dims_from_spec(input_spec)
                _, _, _, n_tiles = _mat_dims_from_spec(weight_spec)
                plans.append(
                    build_matmul_plan(
                        node_name=node.name,
                        lhs_shape=input_spec.shape,
                        rhs_shape=weight_spec.shape,
                        lhs_tiles=input_spec.tiles,
                        rhs_tiles=weight_spec.tiles,
                        dst_tiles=dst_spec.tiles,
                        m_tiles=m_tiles,
                        k_tiles=k_tiles,
                        n_tiles=n_tiles,
                        tile_size=TILE_HEIGHT,
                    )
                )

                if module.bias is not None:
                    bias_attr = f"{node.target}.bias"
                    bias_node = attr_nodes.get(bias_attr)
                    if bias_node is None:
                        raise ValueError(f"Missing get_attr node for {bias_attr}")
                    plans.append(
                        build_add_plan(
                            node_name=f"{node.name}_bias_add",
                            lhs_tiles=specs_by_node[node].tiles,
                            lhs_shape=specs_by_node[node].shape,
                            lhs_tiles_per_dim=specs_by_node[node].tiles_per_dim,
                            rhs_tiles=specs_by_node[bias_node].tiles,
                            rhs_shape=specs_by_node[bias_node].shape,
                            rhs_tiles_per_dim=specs_by_node[bias_node].tiles_per_dim,
                            dst_tiles=specs_by_node[node].tiles,
                            dst_shape=specs_by_node[node].shape,
                            dst_tiles_per_dim=specs_by_node[node].tiles_per_dim,
                            tile_size=TILE_HEIGHT,
                        )
                    )
                continue
            if isinstance(module, torch.nn.MaxPool2d):
                input_node = _get_node_arg(node, 0)
                kernel = _as_pair(module.kernel_size)
                stride = module.stride if module.stride is not None else module.kernel_size
                stride = _as_pair(stride)
                padding = _as_pair(module.padding)
                dilation = _as_pair(module.dilation)
                ceil_flag = 1 if module.ceil_mode else 0
                plans.append(
                    build_maxpool_plan(
                        node_name=node.name,
                        src_tiles=specs_by_node[input_node].tiles,
                        src_shape=specs_by_node[input_node].shape,
                        src_tiles_per_dim=specs_by_node[input_node].tiles_per_dim,
                        dst_tiles=specs_by_node[node].tiles,
                        dst_shape=specs_by_node[node].shape,
                        dst_tiles_per_dim=specs_by_node[node].tiles_per_dim,
                        tile_size=TILE_HEIGHT,
                        kernel=kernel,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        ceil_mode=ceil_flag,
                    )
                )
                continue
            if isinstance(module, torch.nn.AdaptiveAvgPool2d):
                raise _kernel_missing("adaptive_avg_pool2d", node)
            continue

        if node.op == "call_function" and node.target in MATMUL_TARGETS:
            lhs_node = _get_node_arg(node, 0)
            rhs_node = _get_node_arg(node, 1)
            lhs_spec = specs_by_node[lhs_node]
            rhs_spec = specs_by_node[rhs_node]
            dst_spec = specs_by_node[node]
            _, _, m_tiles, k_tiles = _mat_dims_from_spec(lhs_spec)
            _, _, _, n_tiles = _mat_dims_from_spec(rhs_spec)
            plans.append(
                build_matmul_plan(
                    node_name=node.name,
                    lhs_shape=lhs_spec.shape,
                    rhs_shape=rhs_spec.shape,
                    lhs_tiles=lhs_spec.tiles,
                    rhs_tiles=rhs_spec.tiles,
                    dst_tiles=dst_spec.tiles,
                    m_tiles=m_tiles,
                    k_tiles=k_tiles,
                    n_tiles=n_tiles,
                    tile_size=TILE_HEIGHT,
                )
            )
            continue
        if node.op == "call_function" and node.target in RELU_TARGETS:
            input_node = _get_node_arg(node, 0)
            plans.append(
                build_relu_plan(
                    node_name=node.name,
                    src_tiles=specs_by_node[input_node].tiles,
                    src_shape=specs_by_node[input_node].shape,
                    src_tiles_per_dim=specs_by_node[input_node].tiles_per_dim,
                    dst_tiles=specs_by_node[node].tiles,
                    dst_shape=specs_by_node[node].shape,
                    dst_tiles_per_dim=specs_by_node[node].tiles_per_dim,
                    tile_size=TILE_HEIGHT,
                )
            )
            continue
        if node.op == "call_method":
            method = node.target.strip("'")
            if method in VIEW_METHODS:
                continue
            if method == "mean":
                raise _kernel_missing("mean", node)
            continue
        if node.op != "call_function":
            continue
        if node.target in IGNORED_FUNCTIONS:
            continue
        if node.op == "call_function" and node.target in SOFTMAX_TARGETS:
            input_node = _get_node_arg(node, 0)
            plans.append(
                build_softmax_plan(
                    node_name=node.name,
                    src_tiles=specs_by_node[input_node].tiles,
                    src_shape=specs_by_node[input_node].shape,
                    src_tiles_per_dim=specs_by_node[input_node].tiles_per_dim,
                    dst_tiles=specs_by_node[node].tiles,
                    dst_shape=specs_by_node[node].shape,
                    dst_tiles_per_dim=specs_by_node[node].tiles_per_dim,
                    tile_size=TILE_HEIGHT,
                )
            )
            continue
        if node.op == "call_function" and node.target in MAXPOOL_TARGETS:
            input_node = _get_node_arg(node, 0)
            kernel = _as_pair(node.kwargs.get("kernel_size", 1))
            stride = _as_pair(node.kwargs.get("stride", kernel))
            padding = _as_pair(node.kwargs.get("padding", 0))
            dilation = _as_pair(node.kwargs.get("dilation", 1))
            ceil_flag = 1 if bool(node.kwargs.get("ceil_mode", False)) else 0
            plans.append(
                build_maxpool_plan(
                    node_name=node.name,
                    src_tiles=specs_by_node[input_node].tiles,
                    src_shape=specs_by_node[input_node].shape,
                    src_tiles_per_dim=specs_by_node[input_node].tiles_per_dim,
                    dst_tiles=specs_by_node[node].tiles,
                    dst_shape=specs_by_node[node].shape,
                    dst_tiles_per_dim=specs_by_node[node].tiles_per_dim,
                    tile_size=TILE_HEIGHT,
                    kernel=kernel,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    ceil_mode=ceil_flag,
                )
            )
            continue
        if node.target in MUL_TARGETS:
            raise _kernel_missing("mul", node)
            continue
        if node.target in ADD_TARGETS:
            lhs_node, rhs_node = _add_operands(node, specs_by_node)
            plans.append(
                build_add_plan(
                    node_name=node.name,
                    lhs_tiles=specs_by_node[lhs_node].tiles,
                    lhs_shape=specs_by_node[lhs_node].shape,
                    lhs_tiles_per_dim=specs_by_node[lhs_node].tiles_per_dim,
                    rhs_tiles=specs_by_node[rhs_node].tiles,
                    rhs_shape=specs_by_node[rhs_node].shape,
                    rhs_tiles_per_dim=specs_by_node[rhs_node].tiles_per_dim,
                    dst_tiles=specs_by_node[node].tiles,
                    dst_shape=specs_by_node[node].shape,
                    dst_tiles_per_dim=specs_by_node[node].tiles_per_dim,
                    tile_size=TILE_HEIGHT,
                )
            )
            continue
        if node.target in CONV_TARGETS:
            input_node = _get_node_arg(node, 0)
            weight_node = _get_node_arg(node, 1)
            weight_spec = specs_by_node[weight_node]
            weight_shape = weight_spec.shape
            kernel = (1, 1)
            if len(weight_shape) >= 2:
                kernel = _as_pair(weight_shape[-2:])
            stride = _as_pair(node.kwargs.get("stride", 1))
            padding = _as_pair(node.kwargs.get("padding", 0))
            dilation = _as_pair(node.kwargs.get("dilation", 1))
            groups = int(node.kwargs.get("groups", 1))
            plans.append(
                build_conv_plan(
                    node_name=node.name,
                    src_tiles=specs_by_node[input_node].tiles,
                    src_shape=specs_by_node[input_node].shape,
                    src_tiles_per_dim=specs_by_node[input_node].tiles_per_dim,
                    dst_tiles=specs_by_node[node].tiles,
                    dst_shape=specs_by_node[node].shape,
                    dst_tiles_per_dim=specs_by_node[node].tiles_per_dim,
                    tile_size=TILE_HEIGHT,
                    kernel=kernel,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
            )
            continue
    return plans


def _render_plan(plan: OpPlan) -> str:
    lines: List[str] = ["    {\n"]
    tile_info = plan.attrs.get("tile_info", {})
    if not isinstance(tile_info, dict):
        tile_info = {}

    for step in plan.steps:
        if step.kind is StepKind.RELEASE:
            continue

        if step.kind is StepKind.LOAD:
            tile_id = step.attrs.get("tile_id")
            if not isinstance(tile_id, str):
                continue
            info = tile_info.get(tile_id, {})
            slot = int(step.attrs.get("slot", 0))
            rows = int(info.get("rows", 1))
            cols = int(info.get("cols", 1))
            full_cols = int(info.get("full_cols", 1))
            addr = int(info.get("addr", 0))
            lines.append(
                f"        scpad_load({slot} * ATALLA_TILE, 0x{addr:08X}, "
                f"sdma_control({rows}, {cols}, {full_cols}));\n"
            )
            continue

        if step.kind is StepKind.STORE:
            tile_id = step.attrs.get("tile_id")
            if not isinstance(tile_id, str):
                continue
            info = tile_info.get(tile_id, {})
            slot = int(step.attrs.get("slot", 0))
            rows = int(info.get("rows", 1))
            cols = int(info.get("cols", 1))
            full_cols = int(info.get("full_cols", 1))
            addr = int(info.get("addr", 0))
            lines.append(
                f"        scpad_store({slot} * ATALLA_TILE, 0x{addr:08X}, "
                f"sdma_control({rows}, {cols}, {full_cols}));\n"
            )
            continue

        if step.kind is StepKind.COMPUTE:
            if step.op == "matmul":
                slot_a = int(step.attrs.get("slot_a", 0))
                slot_b = int(step.attrs.get("slot_b", 1))
                slot_c = int(step.attrs.get("slot_c", 2))
                m_rows = int(step.attrs.get("m_rows", 1))
                n_cols = int(step.attrs.get("n_cols", 1))
                k_cols = int(step.attrs.get("k_cols", 1))
                lines.append(
                    f"        matmul_kernel({slot_a} * ATALLA_TILE, {slot_b} * ATALLA_TILE, "
                    f"{slot_c} * ATALLA_TILE, {m_rows}, {n_cols}, {k_cols});\n"
                )
            elif step.op == "relu":
                lines.append(
                    f"        relu_kernel({int(step.attrs.get('slot_in', 0))} * ATALLA_TILE, "
                    f"{int(step.attrs.get('slot_out', 0))} * ATALLA_TILE, "
                    f"{int(step.attrs.get('rows', 1))}, {int(step.attrs.get('cols', 1))});\n"
                )
            elif step.op == "softmax":
                lines.append(
                    f"        softmax_kernel({int(step.attrs.get('slot_in', 0))} * ATALLA_TILE, "
                    f"{int(step.attrs.get('slot_out', 0))} * ATALLA_TILE, "
                    f"{int(step.attrs.get('rows', 1))}, {int(step.attrs.get('cols', 1))});\n"
                )
            elif step.op == "add":
                lines.append(
                    f"        add_kernel({int(step.attrs.get('slot_lhs', 0))} * ATALLA_TILE, "
                    f"{int(step.attrs.get('slot_rhs', 0))} * ATALLA_TILE, "
                    f"{int(step.attrs.get('slot_out', 0))} * ATALLA_TILE, "
                    f"{int(step.attrs.get('rows', 1))}, {int(step.attrs.get('cols', 1))});\n"
                )
            elif step.op == "maxpool":
                lines.append(
                    f"        maxpool_kernel({int(step.attrs.get('slot_in', 0))} * ATALLA_TILE, "
                    f"{int(step.attrs.get('slot_out', 0))} * ATALLA_TILE, "
                    f"{int(step.attrs.get('rows', 1))}, {int(step.attrs.get('cols', 1))}, "
                    f"{int(step.attrs.get('kernel_h', 1))}, {int(step.attrs.get('kernel_w', 1))}, "
                    f"{int(step.attrs.get('stride_h', 1))}, {int(step.attrs.get('stride_w', 1))}, "
                    f"{int(step.attrs.get('pad_h', 0))}, {int(step.attrs.get('pad_w', 0))}, "
                    f"{int(step.attrs.get('dilation_h', 1))}, {int(step.attrs.get('dilation_w', 1))}, "
                    f"{int(step.attrs.get('ceil_mode', 0))});\n"
                )
            elif step.op == "conv":
                lines.append(
                    f"        conv_kernel({int(step.attrs.get('slot_in', 0))} * ATALLA_TILE, "
                    f"{int(step.attrs.get('slot_out', 0))} * ATALLA_TILE, "
                    f"{int(step.attrs.get('rows', 1))}, {int(step.attrs.get('cols', 1))}, "
                    f"{int(step.attrs.get('kernel_h', 1))}, {int(step.attrs.get('kernel_w', 1))}, "
                    f"{int(step.attrs.get('stride_h', 1))}, {int(step.attrs.get('stride_w', 1))}, "
                    f"{int(step.attrs.get('pad_h', 0))}, {int(step.attrs.get('pad_w', 0))}, "
                    f"{int(step.attrs.get('dilation_h', 1))}, {int(step.attrs.get('dilation_w', 1))}, "
                    f"{int(step.attrs.get('groups', 1))});\n"
                )

    lines.append("    }\n")
    return "".join(lines)


def _render_kernel_calls(op_plans: List[OpPlan]) -> List[str]:
    calls: List[str] = []
    for plan in op_plans:
        if plan.op_type in {
            "matmul",
            "relu",
            "softmax",
            "maxpool",
            "add",
            "conv",
        }:
            calls.append(_render_plan(plan))
            continue
        for step in plan.steps:
            if step.kind is not StepKind.COMPUTE:
                continue
            rendered = step.attrs.get("rendered_call")
            if isinstance(rendered, str):
                calls.append(rendered)
    return calls

#main entry point 
def emit(gm: GraphModule) -> str:
    attr_nodes = {
        node.target: node
        for node in gm.graph.nodes
        if node.op == "get_attr"
    }

    tensor_specs, specs_by_node = _collect_tensor_specs(gm)
    raw_op_plans = _build_op_plans(gm, specs_by_node, attr_nodes)
    op_plans: List[OpPlan] = []
    for plan in raw_op_plans:
        op_plans.append(plan_tile_moves(plan))
    calls = _render_kernel_calls(op_plans)

    pieces: List[str] = [
        "#include \"kernels/kernels.h\"\n\n",
        "#include \"kernels/sdma.h\"\n\n",
    ]
    for spec in tensor_specs:
        pieces.append(_render_tensor(spec))
        pieces.append("\n")
    pieces.append("int run_graph(void) {\n")
    pieces.extend(calls)
    pieces.append("    return 0;\n}\n")
    return "".join(pieces)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate graph_schedule.c from a serialized FX GraphModule."
    )
    parser.add_argument(
        "graph",
        nargs="?",
        help="Path to GraphModule with memory metadata",
    )
    parser.add_argument(
        "--output",
        default="graph_schedule.c",
        help="Destination C file",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.graph:
        raise SystemExit("Provide a path to a GraphModule with metadata.")
    gm: GraphModule = torch.load(args.graph)
    Path(args.output).write_text(emit(gm))


if __name__ == "__main__":
    main()
