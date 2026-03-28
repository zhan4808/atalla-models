"""Assembly emitter: lower FX graph nodes to .in assembly + data.

For each compute node in topological order, generates assembly using the
proven build_*.py generators from functional_sim.
"""
from __future__ import annotations

import os
import sys
import math
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.fx import GraphModule, Node

# Add functional_sim to path for importing build helpers
_FUNC_SIM = Path(__file__).resolve().parent.parent.parent / "functional_sim"
if str(_FUNC_SIM) not in sys.path:
    sys.path.insert(0, str(_FUNC_SIM))

from build import assemble_file, emit_test_format, DRAMWriter, render_testfile
from build_alexnet_layer import (
    make_relu_asm, make_softmax_asm, make_tiled_gemm_asm,
    make_maxpool_asm, im2col, TILE,
)

from graph.tile_planner import TileConfig
from graph.fx_capture import get_node_shape


ADDR_TABLE = 60


def _bf16_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def _to_bf16_array(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to fp32 numpy (bf16 values preserved)."""
    return tensor.detach().float().cpu().numpy()


def _get_module(gm: GraphModule, target: str):
    parts = target.split(".")
    mod = gm
    for p in parts:
        mod = getattr(mod, p)
    return mod


class LayerEmission:
    """Result of emitting one layer."""
    __slots__ = ("instr_text", "dram", "output_addr", "output_shape",
                 "output_elements", "skip_emulator", "numpy_result")

    def __init__(self):
        self.instr_text: str = ""
        self.dram: DRAMWriter = DRAMWriter()
        self.output_addr: int = 0
        self.output_shape: Tuple[int, ...] = ()
        self.output_elements: int = 0
        self.skip_emulator: bool = False
        self.numpy_result: Optional[np.ndarray] = None


def emit_conv(node: Node, gm: GraphModule,
              input_data: np.ndarray, tc: TileConfig) -> LayerEmission:
    p = tc.params
    M, N, K = p["M"], p["N"], p["K"]

    mod = _get_module(gm, node.target) if node.op == "call_module" else None
    if mod is not None:
        weight_np = _to_bf16_array(mod.weight)  # (C_out, C_in, R, S)
        weight_flat = weight_np.reshape(N, K).T  # (K, N)
    else:
        weight_node = node.args[1]
        weight_tensor = None
        if hasattr(weight_node, 'meta') and 'val' in weight_node.meta:
            weight_tensor = weight_node.meta['val']
        if weight_tensor is None:
            attr = gm
            for part in weight_node.target.split("."):
                attr = getattr(attr, part)
            weight_tensor = attr
        weight_np = _to_bf16_array(weight_tensor)
        weight_flat = weight_np.reshape(N, K).T

    # im2col transform: need NHWC layout
    H, W, C_in = p["H"], p["W"], p["C_in"]
    total_expected = H * W * C_in
    flat = input_data.flatten()
    if len(flat) < total_expected:
        padded = np.zeros(total_expected, dtype=np.float32)
        padded[:len(flat)] = flat
        flat = padded
    else:
        flat = flat[:total_expected]
    # Try to figure out if input is NCHW or NHWC
    if input_data.ndim == 4 and input_data.shape[1] == C_in:
        input_nhwc = input_data.transpose(0, 2, 3, 1)
    elif input_data.ndim == 4 and input_data.shape[3] == C_in:
        input_nhwc = input_data
    else:
        # Assume NCHW from FX graph shapes
        input_nhwc = flat.reshape(1, C_in, H, W).transpose(0, 2, 3, 1)
    A_mat = im2col(input_nhwc, 1, H, W, C_in, p["R"], p["S"], p["stride"], p["pad"])
    W_flat = weight_flat

    A_GMEM = 0x1000
    W_GMEM = A_GMEM + _align_data(M * K * 2)
    C_GMEM = W_GMEM + _align_data(K * N * 2)

    asm = make_tiled_gemm_asm(M, N, K)
    instrs = assemble_file(asm)
    instr_text = emit_test_format(instrs)

    img = DRAMWriter()
    _write_gemm_params(img, A_GMEM, W_GMEM, C_GMEM, M, N, K)
    _write_matrix(img, A_GMEM, A_mat, M, K)
    _write_matrix(img, W_GMEM, W_flat, K, N)
    _write_zeros(img, C_GMEM, M * N)

    # Output from GEMM is (Ho*Wo, C_out); store as NCHW for FX consistency
    out_shape = get_node_shape(node)
    em = LayerEmission()
    em.instr_text = instr_text
    em.dram = img
    em.output_addr = C_GMEM
    em.output_shape = out_shape if out_shape else (1, N, p["Ho"], p["Wo"])
    em.output_elements = M * N
    return em


def emit_linear(node: Node, gm: GraphModule,
                input_data: np.ndarray, tc: TileConfig) -> LayerEmission:
    p = tc.params
    M, N, K = p["M"], p["N"], p["K"]

    mod = _get_module(gm, node.target) if node.op == "call_module" else None
    if mod is not None:
        weight_np = _to_bf16_array(mod.weight)  # (N, K)
        W_mat = weight_np.T  # (K, N)
    else:
        weight_node = node.args[1]
        attr = gm
        for part in weight_node.target.split("."):
            attr = getattr(attr, part)
        weight_np = _to_bf16_array(attr)
        W_mat = weight_np.T if weight_np.shape[0] == N else weight_np

    A = input_data.flatten().reshape(1, K)[:, :K]

    A_GMEM = 0x1000
    W_GMEM = A_GMEM + _align_data(M * K * 2)
    C_GMEM = W_GMEM + _align_data(K * N * 2)

    asm = make_tiled_gemm_asm(M, N, K)
    instrs = assemble_file(asm)
    instr_text = emit_test_format(instrs)

    img = DRAMWriter()
    _write_gemm_params(img, A_GMEM, W_GMEM, C_GMEM, M, N, K)
    _write_matrix(img, A_GMEM, A, M, K)
    _write_matrix(img, W_GMEM, W_mat, K, N)
    _write_zeros(img, C_GMEM, M * N)

    out_shape = get_node_shape(node)
    em = LayerEmission()
    em.instr_text = instr_text
    em.dram = img
    em.output_addr = C_GMEM
    em.output_shape = out_shape if out_shape else (M, N)
    em.output_elements = M * N
    return em


def emit_relu(node: Node, input_data: np.ndarray, tc: TileConfig) -> LayerEmission:
    p = tc.params
    total = p["total_elements"]
    width = min(p["width"], 32)
    rows = math.ceil(total / width)

    flat = input_data.flatten()[:total]
    IN_GMEM = 0x1000
    OUT_GMEM = IN_GMEM + _align_data(rows * width * 2)

    asm = make_relu_asm(total, width)
    instrs = assemble_file(asm)
    instr_text = emit_test_format(instrs)

    img = DRAMWriter()
    img.u32(ADDR_TABLE + 0, IN_GMEM)
    img.u32(ADDR_TABLE + 4, OUT_GMEM)

    padded = np.zeros(rows * width, dtype=np.float32)
    padded[:len(flat)] = flat
    for i in range(rows * width):
        img.bf16(IN_GMEM + i * 2, float(padded[i]))

    # Preserve the input's shape for downstream ops (relu is elementwise)
    input_shape = get_node_shape(node)
    out_shape = input_shape if input_shape else (total,)

    em = LayerEmission()
    em.instr_text = instr_text
    em.dram = img
    em.output_addr = OUT_GMEM
    em.output_shape = out_shape
    em.output_elements = total
    return em


def emit_softmax(input_data: np.ndarray, tc: TileConfig) -> LayerEmission:
    p = tc.params
    length = p["length"]
    width = min(length, 32)
    rows = math.ceil(length / 32)

    flat = input_data.flatten()[:length]
    IN_GMEM = 0x1000

    asm = make_softmax_asm(length)
    instrs = assemble_file(asm)
    instr_text = emit_test_format(instrs)

    img = DRAMWriter()
    img.u32(ADDR_TABLE + 0, IN_GMEM)
    img.u32(ADDR_TABLE + 4, 0)

    padded = np.zeros(rows * width, dtype=np.float32)
    padded[:len(flat)] = flat
    for i in range(rows * width):
        img.bf16(IN_GMEM + i * 2, float(padded[i]))

    em = LayerEmission()
    em.instr_text = instr_text
    em.dram = img
    em.output_addr = IN_GMEM  # softmax is in-place
    em.output_shape = (length,)
    em.output_elements = length
    return em


def emit_maxpool(node: Node, input_data: np.ndarray, tc: TileConfig) -> LayerEmission:
    p = tc.params
    H, W, C = p["H"], p["W"], p["channels"]
    pool, stride = p["pool"], p["stride"]
    H_out, W_out = p["H_out"], p["W_out"]

    em = LayerEmission()
    em.skip_emulator = True

    # Ensure NCHW layout
    total = H * W * C
    flat = input_data.flatten()
    if len(flat) < total:
        padded = np.zeros(total, dtype=np.float32)
        padded[:len(flat)] = flat
        flat = padded
    data_nchw = flat[:total].reshape(1, C, H, W)

    out = np.full((1, C, H_out, W_out), -np.inf, dtype=np.float32)
    for c in range(C):
        for oh in range(H_out):
            for ow in range(W_out):
                for pr in range(pool):
                    for pc_i in range(pool):
                        ih = oh * stride + pr
                        iw = ow * stride + pc_i
                        if ih < H and iw < W:
                            out[0, c, oh, ow] = max(out[0, c, oh, ow],
                                                     float(data_nchw[0, c, ih, iw]))

    # Return in NCHW to match FX graph shapes
    out_shape = get_node_shape(node)
    em.numpy_result = out.reshape(out_shape) if out_shape else out
    em.output_shape = out_shape if out_shape else (1, C, H_out, W_out)
    em.output_elements = C * H_out * W_out
    return em


def emit_node(node: Node, gm: GraphModule,
              activation_cache: Dict[str, np.ndarray]) -> Optional[LayerEmission]:
    """Emit assembly for a single node. Returns None for passthrough ops."""
    tc = node.meta.get("tile_config")
    if tc is None:
        return None

    atalla_op = node.meta.get("atalla_op")

    # Get input activation
    input_data = None
    if node.args:
        first_arg = node.args[0]
        if isinstance(first_arg, Node) and first_arg.name in activation_cache:
            input_data = activation_cache[first_arg.name]

    if input_data is None and atalla_op not in ("placeholder", "get_attr"):
        input_data = np.zeros(1, dtype=np.float32)

    if atalla_op == "conv":
        return emit_conv(node, gm, input_data, tc)
    elif atalla_op == "linear":
        return emit_linear(node, gm, input_data, tc)
    elif atalla_op == "relu":
        return emit_relu(node, input_data, tc)
    elif atalla_op == "softmax":
        return emit_softmax(input_data, tc)
    elif atalla_op == "maxpool":
        return emit_maxpool(node, input_data, tc)
    elif atalla_op in ("flatten", "dropout"):
        return None  # passthrough: just reshape in activation_cache
    elif atalla_op == "adaptive_avg_pool":
        em = LayerEmission()
        em.skip_emulator = True
        shape = get_node_shape(node.args[0])
        if shape and len(shape) == 4:
            C = shape[1]
            t = torch.tensor(input_data.reshape(shape), dtype=torch.float32)
            result = torch.nn.functional.adaptive_avg_pool2d(t, 1)
            em.numpy_result = result.numpy().flatten()
        else:
            em.numpy_result = input_data.flatten()
        em.output_shape = em.numpy_result.shape
        em.output_elements = em.numpy_result.size
        return em
    elif atalla_op == "matmul":
        return emit_matmul(node, gm, input_data, activation_cache, tc)
    elif atalla_op == "add":
        return emit_add(node, activation_cache, tc)
    elif atalla_op == "mul":
        return emit_mul(node, activation_cache)

    return None


def emit_matmul(node: Node, gm: GraphModule,
                input_data: np.ndarray,
                activation_cache: Dict[str, np.ndarray],
                tc: TileConfig) -> LayerEmission:
    p = tc.params
    M, N, K = p["M"], p["N"], p["K"]

    rhs_node = node.args[1]
    if isinstance(rhs_node, Node) and rhs_node.name in activation_cache:
        W_mat = activation_cache[rhs_node.name]
    else:
        attr = gm
        for part in rhs_node.target.split("."):
            attr = getattr(attr, part)
        W_mat = _to_bf16_array(attr)

    A = input_data.reshape(-1, K)[:M, :]
    if W_mat.shape != (K, N):
        W_mat = W_mat.reshape(K, N) if W_mat.size == K * N else W_mat.T

    A_GMEM = 0x1000
    W_GMEM = A_GMEM + _align_data(M * K * 2)
    C_GMEM = W_GMEM + _align_data(K * N * 2)

    asm = make_tiled_gemm_asm(M, N, K)
    instrs = assemble_file(asm)
    instr_text = emit_test_format(instrs)

    img = DRAMWriter()
    _write_gemm_params(img, A_GMEM, W_GMEM, C_GMEM, M, N, K)
    _write_matrix(img, A_GMEM, A, M, K)
    _write_matrix(img, W_GMEM, W_mat, K, N)
    _write_zeros(img, C_GMEM, M * N)

    em = LayerEmission()
    em.instr_text = instr_text
    em.dram = img
    em.output_addr = C_GMEM
    em.output_shape = (M, N) if M > 1 else (N,)
    em.output_elements = M * N
    return em


def emit_mul(node: Node, activation_cache: Dict[str, np.ndarray]) -> LayerEmission:
    """Element-wise or scalar multiply computed in NumPy."""
    lhs_arg = node.args[0]
    rhs_arg = node.args[1]

    if isinstance(lhs_arg, Node) and lhs_arg.name in activation_cache:
        lhs = activation_cache[lhs_arg.name]
    elif isinstance(lhs_arg, (int, float)):
        lhs = np.array([lhs_arg], dtype=np.float32)
    else:
        lhs = np.ones(1, dtype=np.float32)

    if isinstance(rhs_arg, Node) and rhs_arg.name in activation_cache:
        rhs = activation_cache[rhs_arg.name]
    elif isinstance(rhs_arg, (int, float)):
        rhs = np.array([rhs_arg], dtype=np.float32)
    else:
        rhs = np.ones(1, dtype=np.float32)

    em = LayerEmission()
    em.skip_emulator = True
    result = (lhs.flatten() * rhs.flatten()).astype(np.float32)
    out_shape = get_node_shape(node)
    em.numpy_result = result.reshape(out_shape) if out_shape else result
    em.output_shape = out_shape if out_shape else result.shape
    em.output_elements = result.size
    return em


def emit_add(node: Node, activation_cache: Dict[str, np.ndarray],
             tc: TileConfig) -> LayerEmission:
    """Element-wise add computed in NumPy (no dedicated hardware add kernel)."""
    lhs = activation_cache.get(node.args[0].name) if isinstance(node.args[0], Node) else None
    rhs_arg = node.args[1]
    if isinstance(rhs_arg, Node) and rhs_arg.name in activation_cache:
        rhs = activation_cache[rhs_arg.name]
    elif isinstance(rhs_arg, (int, float)):
        rhs = np.full_like(lhs, rhs_arg) if lhs is not None else np.array([rhs_arg])
    else:
        rhs = np.zeros_like(lhs) if lhs is not None else np.zeros(1)

    if lhs is None:
        lhs = np.zeros_like(rhs)

    em = LayerEmission()
    em.skip_emulator = True
    em.numpy_result = (lhs.flatten() + rhs.flatten()).astype(np.float32)
    em.output_shape = em.numpy_result.shape
    em.output_elements = em.numpy_result.size
    return em


# ---- Helpers ----

def _align_data(nbytes: int) -> int:
    return int(math.ceil(nbytes / 0x1000) * 0x1000) + 0x1000


def _write_gemm_params(img: DRAMWriter, A_GMEM: int, W_GMEM: int, C_GMEM: int,
                       M: int, N: int, K: int):
    M_tiles = math.ceil(M / TILE)
    N_tiles = math.ceil(N / TILE)
    K_tiles = math.ceil(K / TILE)
    img.u32(ADDR_TABLE + 0, A_GMEM)
    img.u32(ADDR_TABLE + 4, W_GMEM)
    img.u32(ADDR_TABLE + 8, C_GMEM)
    img.u32(ADDR_TABLE + 12, M)
    img.u32(ADDR_TABLE + 16, N)
    img.u32(ADDR_TABLE + 20, K)
    img.u32(ADDR_TABLE + 24, M_tiles)
    img.u32(ADDR_TABLE + 28, N_tiles)
    img.u32(ADDR_TABLE + 32, K_tiles)
    img.u32(ADDR_TABLE + 36, TILE)


def _write_matrix(img: DRAMWriter, base_addr: int, mat: np.ndarray, rows: int, cols: int):
    flat = mat.flatten()
    for i in range(min(rows * cols, len(flat))):
        img.bf16(base_addr + i * 2, float(flat[i]))


def _write_zeros(img: DRAMWriter, base_addr: int, count: int):
    for i in range(count):
        img.bf16(base_addr + i * 2, 0.0)


def render_in_file(emission: LayerEmission) -> str:
    """Render a LayerEmission to a complete .in file string."""
    data_text = emission.dram.render_data_mem(include_zeros=True)
    return render_testfile(emission.instr_text, data_text)
