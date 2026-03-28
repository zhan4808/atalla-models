"""C emitter: generate AtallaC source for each FX graph op.

Generates parameterized AtallaC that uses the same ADDR_TABLE / DRAM layout
conventions as asm_emitter.py, but outputs .c files for the ppci compiler
instead of raw assembly.

Pipeline: .c -> ppci atalla_cc -> .s -> asm_converter -> assemble -> .in
"""
from __future__ import annotations

import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.fx import GraphModule, Node

_FUNC_SIM = Path(__file__).resolve().parent.parent.parent / "functional_sim"
if str(_FUNC_SIM) not in sys.path:
    sys.path.insert(0, str(_FUNC_SIM))

_COMPILER = Path(__file__).resolve().parent.parent.parent / "aihw-ppci-compiler"

from build import DRAMWriter, assemble_file, emit_test_format, render_testfile
from build_alexnet_layer import (
    im2col, make_tiled_gemm_asm, make_relu_asm, make_softmax_asm, TILE,
)

from graph.tile_planner import TileConfig
from graph.fx_capture import get_node_shape
from codegen.asm_converter import convert as asm_convert


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

def _gemm_c(M: int, N: int, K: int) -> str:
    """Generate AtallaC for tiled GEMM reading config from ADDR_TABLE."""
    tm1 = min(TILE, M) - 1
    tn1 = min(TILE, N) - 1
    tk1 = min(TILE, K) - 1
    tile_m = min(TILE, M)
    tile_n = min(TILE, N)
    tile_k = min(TILE, K)

    return f"""\
int main() {{
    int cfg = {ADDR_TABLE};
    int A_GMEM; int W_GMEM; int C_GMEM;
    int gM; int gN; int gK;
    int M_tiles; int N_tiles; int K_tiles; int tile_sz;

    asm("lw_s %0, 0(%1)"  : "=r"(A_GMEM)  : "r"(cfg));
    asm("lw_s %0, 4(%1)"  : "=r"(W_GMEM)  : "r"(cfg));
    asm("lw_s %0, 8(%1)"  : "=r"(C_GMEM)  : "r"(cfg));
    asm("lw_s %0, 12(%1)" : "=r"(gM)      : "r"(cfg));
    asm("lw_s %0, 16(%1)" : "=r"(gN)      : "r"(cfg));
    asm("lw_s %0, 20(%1)" : "=r"(gK)      : "r"(cfg));
    asm("lw_s %0, 24(%1)" : "=r"(M_tiles) : "r"(cfg));
    asm("lw_s %0, 28(%1)" : "=r"(N_tiles) : "r"(cfg));
    asm("lw_s %0, 32(%1)" : "=r"(K_tiles) : "r"(cfg));
    asm("lw_s %0, 36(%1)" : "=r"(tile_sz) : "r"(cfg));

    int all_mask = -1;
    int sp_a = 0;
    int sp_w = 512;
    int sp_c = 0;

    int mi = 0;
    while (mi < M_tiles) {{
        int ni = 0;
        while (ni < N_tiles) {{
            int c_off = mi * tile_sz * gN + ni * tile_sz;
            int c_byte = c_off * 2;
            int c_addr = C_GMEM + c_byte;

            asm("scpad_ld %0, %1, {tn1}, {tm1}, 1" : : "r"(sp_c), "r"(c_addr));

            int ki = 0;
            while (ki < K_tiles) {{
                int a_off = mi * tile_sz * gK + ki * tile_sz;
                int a_byte = a_off * 2;
                int a_addr = A_GMEM + a_byte;

                int w_off = ki * tile_sz * gN + ni * tile_sz;
                int w_byte = w_off * 2;
                int w_addr = W_GMEM + w_byte;

                asm("scpad_ld %0, %1, {tk1}, {tm1}, 0" : : "r"(sp_a), "r"(a_addr));
                asm("scpad_ld %0, %1, {tn1}, {tk1}, 0" : : "r"(sp_w), "r"(w_addr));

                int wi = 0;
                while (wi < {tile_k}) {{
                    vec wvec;
                    asm("vreg_ld %0, %1, {tn1}, {tk1}, 0, 1, 0"
                        : "=v"(wvec) : "r"(wi));
                    wi = wi + 1;
                }}

                int ri = 0;
                while (ri < {tile_m}) {{
                    vec a_row;
                    vec c_row;
                    asm("vreg_ld %0, %1, {tk1}, {tm1}, 0, 1, 0"
                        : "=v"(a_row) : "r"(ri));
                    asm("vreg_ld %0, %1, {tn1}, {tm1}, 1, 1, 0"
                        : "=v"(c_row) : "r"(ri));

                    vec result = gemm(a_row, c_row, all_mask);

                    asm("vreg_st %0, %1, {tn1}, {tm1}, 1, 1, 0"
                        : : "v"(result), "r"(ri));
                    ri = ri + 1;
                }}

                ki = ki + 1;
            }}

            asm("scpad_st %0, %1, {tn1}, {tm1}, 1" : : "r"(sp_c), "r"(c_addr));
            ni = ni + 1;
        }}
        mi = mi + 1;
    }}

    asm("halt");
    return 0;
}}
"""


def _relu_c(total: int, width: int) -> str:
    """Generate AtallaC for ReLU."""
    rows = math.ceil(total / width)
    w_m1 = width - 1
    sp_rows = min(rows, TILE)
    sp_r_m1 = sp_rows - 1
    tile_count = math.ceil(rows / sp_rows)
    tile_bytes = sp_rows * width * 2

    return f"""\
int main() {{
    int cfg = {ADDR_TABLE};
    int IN_GMEM;
    int OUT_GMEM;
    asm("lw_s %0, 0(%1)" : "=r"(IN_GMEM)  : "r"(cfg));
    asm("lw_s %0, 4(%1)" : "=r"(OUT_GMEM) : "r"(cfg));

    int sp = 0;
    int all_mask = -1;

    vec zero_vec = vec_op_masked("*", zero_vec, 0.0, all_mask);

    int tile = 0;
    while (tile < {tile_count}) {{
        asm("scpad_ld %0, %1, {w_m1}, {sp_r_m1}, 0" : : "r"(sp), "r"(IN_GMEM));

        int row = 0;
        while (row < {sp_rows}) {{
            vec v;
            asm("vreg_ld %0, %1, {w_m1}, {sp_r_m1}, 0, 1, 0"
                : "=v"(v) : "r"(row));

            mask m_neg = make_mask("<", v, zero_vec, all_mask);
            vec result = vec_op_masked("*", v, 0.0, m_neg);

            asm("vreg_st %0, %1, {w_m1}, {sp_r_m1}, 0, 1, 0"
                : : "v"(result), "r"(row));
            row = row + 1;
        }}

        asm("scpad_st %0, %1, {w_m1}, {sp_r_m1}, 0" : : "r"(sp), "r"(OUT_GMEM));
        IN_GMEM = IN_GMEM + {tile_bytes};
        OUT_GMEM = OUT_GMEM + {tile_bytes};
        tile = tile + 1;
    }}

    asm("halt");
    return 0;
}}
"""


def _softmax_c(length: int) -> str:
    """Generate AtallaC for softmax.

    Uses inline asm for div.vs since vec_op_masked("/") tree isn't
    covered by the instruction selector.
    """
    w_m1 = min(length, 32) - 1
    rows = math.ceil(length / 32)
    r_m1 = rows - 1
    mask_val = (1 << min(length, 32)) - 1

    return f"""\
int main() {{
    int cfg = {ADDR_TABLE};
    int IN_GMEM;
    int dummy;
    asm("lw_s %0, 0(%1)" : "=r"(IN_GMEM) : "r"(cfg));
    asm("lw_s %0, 4(%1)" : "=r"(dummy)   : "r"(cfg));

    int sp = 0;
    int mask_val = {mask_val};

    asm("scpad_ld %0, %1, {w_m1}, {r_m1}, 0" : : "r"(sp), "r"(IN_GMEM));

    int row = 0;
    while (row < {rows}) {{
        vec v;
        asm("vreg_ld %0, %1, {w_m1}, {r_m1}, 0, 1, 0"
            : "=v"(v) : "r"(row));

        vec vmax = vec_op_masked("RMAX", v, 0.0, mask_val);
        vec shifted = vec_op_masked("-", v, vmax, mask_val);
        vec exp_v = vec_op_masked("EXP", shifted, 0.0, mask_val);
        vec sum_v = vec_op_masked("RSUM", exp_v, 0.0, mask_val);

        float sum_f = sum_v[0];
        int sum_bits;
        int one_bits;
        asm("stbf_s %0, %1, %2" : "=r"(sum_bits) : "r"(sum_f), "r"(0));
        asm("rcp_bf %0, %1, %2" : "=r"(one_bits) : "r"(sum_bits), "r"(0));
        float inv_sum;
        asm("bfts_s %0, %1, %2" : "=r"(inv_sum) : "r"(one_bits), "r"(0));
        vec result = vec_op_masked("*", exp_v, inv_sum, mask_val);

        asm("vreg_st %0, %1, {w_m1}, {r_m1}, 0, 1, 0"
            : : "v"(result), "r"(row));
        row = row + 1;
    }}

    asm("scpad_st %0, %1, {w_m1}, {r_m1}, 0" : : "r"(sp), "r"(IN_GMEM));

    asm("halt");
    return 0;
}}
"""


# ---------------------------------------------------------------------------
# Top-level emitters (same interface as asm_emitter)
# ---------------------------------------------------------------------------

def _gemm_direct_asm(M: int, N: int, K: int) -> str:
    """Use direct assembly for GEMM (requires lw.vi for systolic weight preload)."""
    asm = make_tiled_gemm_asm(M, N, K)
    instrs = assemble_file(asm)
    return emit_test_format(instrs)


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
    em.instr_text = _gemm_direct_asm(M, N, K)
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
    em.instr_text = _gemm_direct_asm(M, N, K)
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

    asm = make_relu_asm(total, width)
    instrs = assemble_file(asm)

    em = LayerEmission()
    em.instr_text = emit_test_format(instrs)
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

    asm = make_softmax_asm(length)
    instrs = assemble_file(asm)

    em = LayerEmission()
    em.instr_text = emit_test_format(instrs)
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
    em.instr_text = _gemm_direct_asm(M, N, K)
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
# Compilation pipeline: .c -> ppci -> .s -> converter -> assemble -> .in
# ---------------------------------------------------------------------------

def compile_c(c_source: str, work_dir: str, tag: str) -> str:
    """Compile AtallaC source to emulator-compatible assembly text.

    Returns the converted assembly string ready for assemble_file().
    """
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

    compiler_asm = Path(s_path).read_text()
    return asm_convert(compiler_asm)


def compile_and_assemble(emission: LayerEmission, work_dir: str,
                         tag: str) -> str:
    """Compile C source (if present) or use pre-built instr_text."""
    if emission.c_source:
        converted_asm = compile_c(emission.c_source, work_dir, tag)
        instrs = assemble_file(converted_asm)
        emission.instr_text = emit_test_format(instrs)
    return emission.instr_text


def render_in_file(emission: LayerEmission) -> str:
    """Render a LayerEmission to a complete .in file string."""
    data_text = emission.dram.render_data_mem(include_zeros=True)
    return render_testfile(emission.instr_text, data_text)
