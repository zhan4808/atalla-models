"""C emitter: generate AtallaC source for each FX graph op.

Generates parameterized AtallaC that uses the same ADDR_TABLE / DRAM layout
conventions as asm_emitter.py, but outputs .c files for the ppci compiler
instead of raw assembly.

Pipeline: .c -> ppci atalla_cc -> .s -> build_compiler.compile_asm -> .in
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.fx import GraphModule, Node

_FUNC_SIM = Path(__file__).resolve().parent.parent.parent / "functional_sim"
if str(_FUNC_SIM) not in sys.path:
    sys.path.insert(0, str(_FUNC_SIM))

_default_compiler = Path(__file__).resolve().parent.parent.parent / "aihw-ppci-compiler"
_COMPILER = Path(os.environ.get("ATALLA_COMPILER_PATH", str(_default_compiler)))

from build import DRAMWriter, render_testfile
from build_alexnet_layer import im2col, TILE
import build_compiler as _bc

from graph.tile_planner import TileConfig
from graph.fx_capture import get_node_shape
from kernels import ADDR_TABLE, TILE, sdma_ctl_val, sdma_ctl_expr
from kernels import gemm_c as _gemm_c
from kernels import relu_c as _relu_c
from kernels import softmax_c as _softmax_c
from kernels import maxpool_c as _maxpool_c


def _to_bf16_array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _get_module(gm: GraphModule, target: str):
    parts = target.split(".")
    mod = gm
    for p in parts:
        mod = getattr(mod, p)
    return mod


class LayerEmission:
    __slots__ = ("c_source", "instr_text", "dram", "output_addr", "output_shape",
                 "output_elements", "skip_emulator", "numpy_result", "maxpool_post")

    def __init__(self):
        self.c_source: str = ""
        self.instr_text: str = ""
        self.dram: DRAMWriter = DRAMWriter()
        self.output_addr: int = 0
        self.output_shape: Tuple[int, ...] = ()
        self.output_elements: int = 0
        self.skip_emulator: bool = False
        self.numpy_result: Optional[np.ndarray] = None
        self.maxpool_post: Optional[dict] = None


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


# ---------------------------------------------------------------------------
# Top-level emitters
# ---------------------------------------------------------------------------

def emit_conv(node: Node, gm: GraphModule,
              input_data: np.ndarray, tc: TileConfig) -> LayerEmission:
    p = tc.params
    M, N, K = p["M"], p["N"], p["K"]

    mod = _get_module(gm, node.target) if node.op == "call_module" else None
    if mod is not None:
        weight_np = _to_bf16_array(mod.weight)
        weight_flat = weight_np.reshape(N, K).T
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

    H, W, C_in = p["H"], p["W"], p["C_in"]
    total_expected = H * W * C_in
    flat = input_data.flatten()
    if len(flat) < total_expected:
        padded = np.zeros(total_expected, dtype=np.float32)
        padded[:len(flat)] = flat
        flat = padded
    else:
        flat = flat[:total_expected]
    if input_data.ndim == 4 and input_data.shape[1] == C_in:
        input_nhwc = input_data.transpose(0, 2, 3, 1)
    elif input_data.ndim == 4 and input_data.shape[3] == C_in:
        input_nhwc = input_data
    else:
        input_nhwc = flat.reshape(1, C_in, H, W).transpose(0, 2, 3, 1)
    A_mat = im2col(input_nhwc, 1, H, W, C_in, p["R"], p["S"], p["stride"], p["pad"])

    A_GMEM = 0x1000
    W_GMEM = A_GMEM + _align_data(M * K * 2)
    C_GMEM = W_GMEM + _align_data(K * N * 2)

    img = DRAMWriter()
    _write_gemm_params(img, A_GMEM, W_GMEM, C_GMEM, M, N, K)
    _write_matrix(img, A_GMEM, A_mat, M, K)
    _write_matrix(img, W_GMEM, weight_flat, K, N)
    _write_zeros(img, C_GMEM, M * N)

    out_shape = get_node_shape(node)
    em = LayerEmission()
    em.c_source = _gemm_c(M, N, K)
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
        W_mat = _to_bf16_array(mod.weight).T
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

    img = DRAMWriter()
    _write_gemm_params(img, A_GMEM, W_GMEM, C_GMEM, M, N, K)
    _write_matrix(img, A_GMEM, A, M, K)
    _write_matrix(img, W_GMEM, W_mat, K, N)
    _write_zeros(img, C_GMEM, M * N)

    out_shape = get_node_shape(node)
    em = LayerEmission()
    em.c_source = _gemm_c(M, N, K)
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

    img = DRAMWriter()
    img.u32(ADDR_TABLE + 0, IN_GMEM)
    img.u32(ADDR_TABLE + 4, OUT_GMEM)

    padded = np.zeros(rows * width, dtype=np.float32)
    padded[:len(flat)] = flat
    for i in range(rows * width):
        img.bf16(IN_GMEM + i * 2, float(padded[i]))

    input_shape = get_node_shape(node)
    out_shape = input_shape if input_shape else (total,)

    em = LayerEmission()
    em.c_source = _relu_c(total, width)
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

    img = DRAMWriter()
    img.u32(ADDR_TABLE + 0, IN_GMEM)
    img.u32(ADDR_TABLE + 4, 0)

    padded = np.zeros(rows * width, dtype=np.float32)
    padded[:len(flat)] = flat
    for i in range(rows * width):
        img.bf16(IN_GMEM + i * 2, float(padded[i]))

    em = LayerEmission()
    em.c_source = _softmax_c(length)
    em.dram = img
    em.output_addr = IN_GMEM
    em.output_shape = (length,)
    em.output_elements = length
    return em


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

    img = DRAMWriter()
    _write_gemm_params(img, A_GMEM, W_GMEM, C_GMEM, M, N, K)
    _write_matrix(img, A_GMEM, A, M, K)
    _write_matrix(img, W_GMEM, W_mat, K, N)
    _write_zeros(img, C_GMEM, M * N)

    em = LayerEmission()
    em.c_source = _gemm_c(M, N, K)
    em.dram = img
    em.output_addr = C_GMEM
    em.output_shape = (M, N) if M > 1 else (N,)
    em.output_elements = M * N
    return em


def emit_maxpool(node: Node, input_data: np.ndarray, tc: TileConfig) -> LayerEmission:
    p = tc.params
    H, W, C = p["H"], p["W"], p["channels"]
    pool, stride = p["pool"], p["stride"]
    H_out, W_out = p["H_out"], p["W_out"]

    total = H * W * C
    flat = input_data.flatten()
    if len(flat) < total:
        padded = np.zeros(total, dtype=np.float32)
        padded[:len(flat)] = flat
        flat = padded

    out_shape = get_node_shape(node) or (1, C, H_out, W_out)
    out_elems = C * H_out * W_out

    if W > 32:
        em = LayerEmission()
        em.skip_emulator = True
        data_nchw = flat[:total].reshape(1, C, H, W)
        out = np.full((1, C, H_out, W_out), -np.inf, dtype=np.float32)
        for c in range(C):
            for oh in range(H_out):
                for ow in range(W_out):
                    for pr in range(pool):
                        for pc_i in range(pool):
                            ih, iw = oh * stride + pr, ow * stride + pc_i
                            if ih < H and iw < W:
                                out[0, c, oh, ow] = max(out[0, c, oh, ow],
                                                         float(data_nchw[0, c, ih, iw]))
        em.numpy_result = out.reshape(out_shape)
        em.output_shape = out_shape
        em.output_elements = out_elems
        return em

    # Kernel outputs full-width rows (vertical max only); horizontal
    # stride-select + pool-width max done in post_process_maxpool.
    IN_GMEM = 0x1000
    channel_in_bytes = H * W * 2
    raw_out_ch_bytes = H_out * W * 2
    total_in_bytes = C * channel_in_bytes
    total_raw_out = C * H_out * W
    OUT_GMEM = IN_GMEM + _align_data(total_in_bytes)

    img = DRAMWriter()
    img.u32(ADDR_TABLE + 0, IN_GMEM)
    img.u32(ADDR_TABLE + 4, OUT_GMEM)

    data_nchw = flat[:total].reshape(C, H, W) if C > 0 else flat[:total].reshape(1, H, W)
    for c in range(C):
        base = IN_GMEM + c * channel_in_bytes
        for r in range(H):
            for col in range(W):
                img.bf16(base + (r * W + col) * 2, float(data_nchw[c, r, col]))

    _write_zeros(img, OUT_GMEM, total_raw_out)

    em = LayerEmission()
    em.c_source = _maxpool_c(H, W, C, pool, stride)
    em.dram = img
    em.output_addr = OUT_GMEM
    em.output_elements = total_raw_out
    em.output_shape = (C, H_out, W)
    em.maxpool_post = dict(C=C, H_out=H_out, W=W, W_out=W_out,
                           pool=pool, stride=stride, final_shape=out_shape)
    return em


def emit_mul(node: Node, activation_cache: Dict[str, np.ndarray]) -> LayerEmission:
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


def emit_node(node: Node, gm: GraphModule,
              activation_cache: Dict[str, np.ndarray]) -> Optional[LayerEmission]:
    tc = node.meta.get("tile_config")
    if tc is None:
        return None

    atalla_op = node.meta.get("atalla_op")

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
        return None
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


# ---------------------------------------------------------------------------
# Compilation pipeline: .c -> ppci -> .s -> build_compiler.compile_asm -> .in
# ---------------------------------------------------------------------------

def compile_c(c_source: str, work_dir: str, tag: str) -> str:
    """Compile AtallaC source to raw ppci .s assembly text."""
    os.makedirs(work_dir, exist_ok=True)
    c_path = os.path.join(work_dir, f"{tag}.c")
    s_path = os.path.join(work_dir, f"{tag}.s")

    Path(c_path).write_text(c_source)

    env = {**os.environ, "PYTHONPATH": str(_COMPILER)}
    result = subprocess.run(
        [sys.executable, "-m", "ppci", "atalla_cc", "-m", "atalla",
         "-S", c_path, "-o", s_path],
        capture_output=True, text=True, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ppci compile failed for {tag}:\n{result.stderr[-2000:]}"
        )

    return Path(s_path).read_text()


def compile_and_assemble(emission: LayerEmission, work_dir: str,
                         tag: str) -> str:
    """Compile C source (if present) or use pre-built instr_text.

    Uses build_compiler.compile_asm() which handles notation conversion,
    scheduling, and encoding - compatible with the updated ISA instruction formats.
    """
    if emission.c_source:
        raw_s = compile_c(emission.c_source, work_dir, tag)
        in_content, _, _ = _bc.compile_asm(raw_s)
        # compile_asm returns a complete .in file; strip the .data section
        # since render_in_file will add the populated DRAMWriter data
        emission.instr_text = in_content.split("\n.data")[0].strip()
    return emission.instr_text


def render_in_file(emission: LayerEmission) -> str:
    """Render a LayerEmission to a complete .in file string."""
    data_text = emission.dram.render_data_mem(include_zeros=True)
    return render_testfile(emission.instr_text, data_text)
