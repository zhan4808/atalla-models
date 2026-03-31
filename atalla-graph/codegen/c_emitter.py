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

# Default: vendored copy inside atalla-models (master + isa-fixes branch).
# Override with ATALLA_COMPILER_PATH if you want to point at the sibling repo directly.
_default_compiler = Path(__file__).resolve().parent.parent.parent / "aihw-ppci-compiler"
_COMPILER = Path(os.environ.get("ATALLA_COMPILER_PATH", str(_default_compiler)))
_COMPILER_EMU = _COMPILER / "emulator"
if str(_COMPILER_EMU) not in sys.path:
    sys.path.insert(0, str(_COMPILER_EMU))

from build import DRAMWriter, render_testfile
from build_alexnet_layer import im2col, TILE
import build_compiler as _bc

from graph.tile_planner import TileConfig
from graph.fx_capture import get_node_shape


ADDR_TABLE = 60


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
                 "output_elements", "skip_emulator", "numpy_result")

    def __init__(self):
        self.c_source: str = ""
        self.instr_text: str = ""
        self.dram: DRAMWriter = DRAMWriter()
        self.output_addr: int = 0
        self.output_shape: Tuple[int, ...] = ()
        self.output_elements: int = 0
        self.skip_emulator: bool = False
        self.numpy_result: Optional[np.ndarray] = None


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
# AtallaC source generators
# ---------------------------------------------------------------------------

def _sdma_ctl_val(sid: int, num_rows: int, num_cols: int, full_cols: int) -> int:
    """Pack scratchpad DMA control register value."""
    return (sid << 30) | ((num_rows - 1) << 25) | ((num_cols - 1) << 20) | (full_cols - 1)


def _sdma_ctl_expr(name: str, sid: int, num_rows: int, num_cols: int, full_cols: int) -> str:
    """Emit C statement that loads a pre-computed sdma_ctl via inline asm li_s.

    The compiler's instruction selector can't handle large constant stores,
    so we use inline asm to let build_compiler expand li_s -> lui_s+addi_s.
    """
    val = _sdma_ctl_val(sid, num_rows, num_cols, full_cols)
    return f'    int {name};\n    asm("li_s %0, {val}" : "=r"({name}));\n'


def _gemm_c(M: int, N: int, K: int) -> str:
    """Generate AtallaC for tiled GEMM reading config from ADDR_TABLE."""
    tm1 = min(TILE, M) - 1
    tn1 = min(TILE, N) - 1
    tk1 = min(TILE, K) - 1
    tile_m = min(TILE, M)
    tile_n = min(TILE, N)
    tile_k = min(TILE, K)
    # Input: SP0 (sid=0). Weights: SP1 (sid=1). Output: SP1 (sid=1).
    # Weights are loaded and preloaded into the systolic array first,
    # then output is loaded into SP1 (overwriting weight data, but the
    # systolic array already has the weights).
    sdma_a_s = _sdma_ctl_expr("sdma_ctl_a", 0, tile_m, tile_k, K)
    sdma_w_s = _sdma_ctl_expr("sdma_ctl_w", 1, tile_k, tile_n, N)
    sdma_c_s = _sdma_ctl_expr("sdma_ctl_c", 1, tile_m, tile_n, N)

    return (
        "int main() {\n"
        f"    int cfg = {ADDR_TABLE};\n"
        "    int A_GMEM; int W_GMEM; int C_GMEM;\n"
        "    int gM; int gN; int gK;\n"
        "    int M_tiles; int N_tiles; int K_tiles; int tile_sz;\n"
        "\n"
        '    asm("lw_s %0, 0(%1)"  : "=r"(A_GMEM)  : "r"(cfg));\n'
        '    asm("lw_s %0, 4(%1)"  : "=r"(W_GMEM)  : "r"(cfg));\n'
        '    asm("lw_s %0, 8(%1)"  : "=r"(C_GMEM)  : "r"(cfg));\n'
        '    asm("lw_s %0, 12(%1)" : "=r"(gM)      : "r"(cfg));\n'
        '    asm("lw_s %0, 16(%1)" : "=r"(gN)      : "r"(cfg));\n'
        '    asm("lw_s %0, 20(%1)" : "=r"(gK)      : "r"(cfg));\n'
        '    asm("lw_s %0, 24(%1)" : "=r"(M_tiles) : "r"(cfg));\n'
        '    asm("lw_s %0, 28(%1)" : "=r"(N_tiles) : "r"(cfg));\n'
        '    asm("lw_s %0, 32(%1)" : "=r"(K_tiles) : "r"(cfg));\n'
        '    asm("lw_s %0, 36(%1)" : "=r"(tile_sz) : "r"(cfg));\n'
        "\n"
        "    int all_mask = -1;\n"
        "    int ncols = 1;\n"
        "    int sp_a = 0;\n"
        "    int sp_w = 0;\n"
        "    int sp_c = 0;\n"
        f"{sdma_a_s}"
        f"{sdma_w_s}"
        f"{sdma_c_s}"
        "\n"
        "    int mi = 0;\n"
        "    while (mi < M_tiles) {\n"
        "        int ni = 0;\n"
        "        while (ni < N_tiles) {\n"
        "            int ki = 0;\n"
        "            while (ki < K_tiles) {\n"
        "                int a_off = mi * tile_sz * gK + ki * tile_sz;\n"
        "                int a_byte = a_off * 2;\n"
        "                int a_addr = A_GMEM + a_byte;\n"
        "\n"
        "                int w_off = ki * tile_sz * gN + ni * tile_sz;\n"
        "                int w_byte = w_off * 2;\n"
        "                int w_addr = W_GMEM + w_byte;\n"
        "\n"
        # Load weights into SP1, preload to systolic array
        '                asm("scpad_ld %0, %1, %2" : : "r"(sp_w), "r"(w_addr), "r"(sdma_ctl_w));\n'
        "\n"
        "                int wi = 0;\n"
        f"                while (wi < {tile_k}) {{\n"
        "                    vec wvec;\n"
        "                    int w_row = sp_w + wi;\n"
        f'                    asm("vreg_ld %0, %1, %2, {tn1}, 1"\n'
        '                        : "=v"(wvec) : "r"(w_row), "r"(ncols));\n'
        '                    asm("lw_vi %0, %1, 0, m0" : "=v"(wvec) : "v"(wvec));\n'
        "                    wi = wi + 1;\n"
        "                }\n"
        "\n"
        # Now load input into SP0, output into SP1 (overwrites weight data)
        '                asm("scpad_ld %0, %1, %2" : : "r"(sp_a), "r"(a_addr), "r"(sdma_ctl_a));\n'
        "\n"
        "                int c_off = mi * tile_sz * gN + ni * tile_sz;\n"
        "                int c_byte = c_off * 2;\n"
        "                int c_addr = C_GMEM + c_byte;\n"
        '                asm("scpad_ld %0, %1, %2" : : "r"(sp_c), "r"(c_addr), "r"(sdma_ctl_c));\n'
        "\n"
        "                int ri = 0;\n"
        f"                while (ri < {tile_m}) {{\n"
        "                    vec a_row;\n"
        "                    vec c_row;\n"
        f'                    asm("vreg_ld %0, %1, %2, {tk1}, 0"\n'
        '                        : "=v"(a_row) : "r"(ri), "r"(ncols));\n'
        f'                    asm("vreg_ld %0, %1, %2, {tn1}, 1"\n'
        '                        : "=v"(c_row) : "r"(ri), "r"(ncols));\n'
        "\n"
        "                    vec result = gemm(a_row, c_row, all_mask);\n"
        "\n"
        f'                    asm("vreg_st %0, %1, %2, {tn1}, 1"\n'
        '                        : : "v"(result), "r"(ri), "r"(ncols));\n'
        "                    ri = ri + 1;\n"
        "                }\n"
        "\n"
        '                asm("scpad_st %0, %1, %2" : : "r"(sp_c), "r"(c_addr), "r"(sdma_ctl_c));\n'
        "                ki = ki + 1;\n"
        "            }\n"
        "\n"
        "            ni = ni + 1;\n"
        "        }\n"
        "        mi = mi + 1;\n"
        "    }\n"
        "\n"
        '    asm("halt");\n'
        "    return 0;\n"
        "}\n"
    )


def _relu_c(total: int, width: int) -> str:
    """Generate AtallaC for ReLU."""
    rows = math.ceil(total / width)
    w_m1 = width - 1
    sp_rows = min(rows, TILE)
    tile_count = math.ceil(rows / sp_rows)
    tile_bytes = sp_rows * width * 2
    sdma_s = _sdma_ctl_expr("sdma_ctl", 0, sp_rows, width, width)

    return (
        "int main() {\n"
        f"    int cfg = {ADDR_TABLE};\n"
        "    int IN_GMEM;\n"
        "    int OUT_GMEM;\n"
        '    asm("lw_s %0, 0(%1)" : "=r"(IN_GMEM)  : "r"(cfg));\n'
        '    asm("lw_s %0, 4(%1)" : "=r"(OUT_GMEM) : "r"(cfg));\n'
        "\n"
        "    int sp = 0;\n"
        "    int all_mask = -1;\n"
        "    int ncols = 1;\n"
        f"{sdma_s}"
        "\n"
        "    vec zero_vec;\n"
        '    asm("vreg_ld %0, %1, %2, ' + str(w_m1) + ', 0" : "=v"(zero_vec) : "r"(0), "r"(ncols));\n'
        '    zero_vec = vec_op_masked("*", zero_vec, 0.0, all_mask);\n'
        "\n"
        "    int tile = 0;\n"
        f"    while (tile < {tile_count}) {{\n"
        '        asm("scpad_ld %0, %1, %2" : : "r"(sp), "r"(IN_GMEM), "r"(sdma_ctl));\n'
        "\n"
        "        int row = 0;\n"
        f"        while (row < {sp_rows}) {{\n"
        "            vec v;\n"
        f'            asm("vreg_ld %0, %1, %2, {w_m1}, 0"\n'
        '                : "=v"(v) : "r"(row), "r"(ncols));\n'
        "\n"
        '            int m_neg = make_mask("<", v, zero_vec, all_mask);\n'
        '            vec result = vec_op_masked("*", v, 0.0, m_neg);\n'
        "\n"
        f'            asm("vreg_st %0, %1, %2, {w_m1}, 0"\n'
        '                : : "v"(result), "r"(row), "r"(ncols));\n'
        "            row = row + 1;\n"
        "        }\n"
        "\n"
        '        asm("scpad_st %0, %1, %2" : : "r"(sp), "r"(OUT_GMEM), "r"(sdma_ctl));\n'
        f"        IN_GMEM = IN_GMEM + {tile_bytes};\n"
        f"        OUT_GMEM = OUT_GMEM + {tile_bytes};\n"
        "        tile = tile + 1;\n"
        "    }\n"
        "\n"
        '    asm("halt");\n'
        "    return 0;\n"
        "}\n"
    )


def _softmax_c(length: int) -> str:
    """Generate AtallaC for softmax using rcp.bf for reciprocal."""
    width = min(length, 32)
    rows = math.ceil(length / 32)
    w_m1 = width - 1
    mask_val = -1 if width == 32 else (1 << width) - 1
    sdma_s = _sdma_ctl_expr("sdma_ctl", 0, rows, width, width)

    return (
        "int main() {\n"
        f"    int cfg = {ADDR_TABLE};\n"
        "    int IN_GMEM;\n"
        "    int dummy;\n"
        '    asm("lw_s %0, 0(%1)" : "=r"(IN_GMEM) : "r"(cfg));\n'
        '    asm("lw_s %0, 4(%1)" : "=r"(dummy)   : "r"(cfg));\n'
        "\n"
        "    int sp = 0;\n"
        f"    int mask_val = {mask_val};\n"
        "    int ncols = 1;\n"
        f"{sdma_s}"
        "\n"
        '    asm("scpad_ld %0, %1, %2" : : "r"(sp), "r"(IN_GMEM), "r"(sdma_ctl));\n'
        "\n"
        "    int row = 0;\n"
        f"    while (row < {rows}) {{\n"
        "        vec v;\n"
        f'        asm("vreg_ld %0, %1, %2, {w_m1}, 0" : "=v"(v) : "r"(row), "r"(ncols));\n'
        "\n"
        '        vec vmax = vec_op_masked("RMAX", v, 0.0, mask_val);\n'
        '        vec shifted = vec_op_masked("-", v, vmax, mask_val);\n'
        '        vec exp_v = vec_op_masked("EXP", shifted, 0.0, mask_val);\n'
        '        vec sum_v = vec_op_masked("RSUM", exp_v, 0.0, mask_val);\n'
        "\n"
        "        float sum_f = sum_v[0];\n"
        "        int sum_bits;\n"
        "        int inv_bits;\n"
        '        asm("stbf_s %0, %1, %2" : "=r"(sum_bits) : "r"(sum_f), "r"(0));\n'
        '        asm("rcp_bf %0, %1, %2" : "=r"(inv_bits) : "r"(sum_bits), "r"(0));\n'
        "        float inv_sum;\n"
        '        asm("bfts_s %0, %1, %2" : "=r"(inv_sum) : "r"(inv_bits), "r"(0));\n'
        '        vec result = vec_op_masked("*", exp_v, inv_sum, mask_val);\n'
        "\n"
        f'        asm("vreg_st %0, %1, %2, {w_m1}, 0" : : "v"(result), "r"(row), "r"(ncols));\n'
        "        row = row + 1;\n"
        "    }\n"
        "\n"
        '    asm("scpad_st %0, %1, %2" : : "r"(sp), "r"(IN_GMEM), "r"(sdma_ctl));\n'
        "\n"
        '    asm("halt");\n'
        "    return 0;\n"
        "}\n"
    )


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

    em = LayerEmission()
    em.skip_emulator = True

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

    out_shape = get_node_shape(node)
    em.numpy_result = out.reshape(out_shape) if out_shape else out
    em.output_shape = out_shape if out_shape else (1, C, H_out, W_out)
    em.output_elements = C * H_out * W_out
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
