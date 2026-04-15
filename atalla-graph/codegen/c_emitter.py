"""C emitter: generate AtallaC source for each FX graph op.

Generates parameterized AtallaC that uses the same ADDR_TABLE / DRAM layout
conventions as asm_emitter.py, but outputs .c files for the ppci compiler
instead of raw assembly.

Pipeline: .c -> ppci atalla_cc -> .s -> build_compiler.compile_asm -> .in
"""
from __future__ import annotations

import importlib.util
import math
import os
from collections import Counter
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.fx import GraphModule, Node

_FUNC_SIM = Path(__file__).resolve().parent.parent.parent / "functional_sim"
if str(_FUNC_SIM) not in sys.path:
    sys.path.insert(0, str(_FUNC_SIM))


def _load_emit_kernels():
    """Graph AtallaC kernels; must not use top-level name `kernels` (functional_sim has its own)."""
    name = "_atalla_emit_kernels"
    if name in sys.modules:
        return sys.modules[name]
    root = Path(__file__).resolve().parent.parent / "kernels"
    spec = importlib.util.spec_from_file_location(
        name, root / "__init__.py", submodule_search_locations=[str(root)]
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_ek = _load_emit_kernels()
ADDR_TABLE = _ek.ADDR_TABLE
TILE = _ek.TILE
sdma_ctl_val = _ek.sdma_ctl_val
sdma_ctl_expr = _ek.sdma_ctl_expr
_gemm_c = _ek.gemm_c
_relu_c = _ek.relu_c
_softmax_c = _ek.softmax_c
_softmax_c_batched = _ek.softmax_c_batched
_maxpool_c = _ek.maxpool_c
_add_c = _ek.add_c
_layernorm_c = _ek.layernorm_c

_default_compiler = Path(__file__).resolve().parent.parent.parent / "aihw-ppci-compiler"
_COMPILER = Path(os.environ.get("ATALLA_COMPILER_PATH", str(_default_compiler)))

from build import DRAMWriter, render_testfile
from build_conv_tiled import im2col
import build_compiler as _bc

from graph.tile_planner import TileConfig
from graph.fx_capture import get_node_shape


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
                 "output_elements", "skip_emulator", "numpy_result", "maxpool_post",
                 "conv_post", "sched_packets", "sched_slots_filled", "sched_slot_efficiency",
                 "sched_slot_histogram", "layer_metrics")

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
        self.conv_post: Optional[dict] = None
        # Filled by compile_and_assemble() from build_compiler VLIW schedule (static program).
        self.sched_packets: int = 0
        self.sched_slots_filled: int = 0
        self.sched_slot_efficiency: float = 0.0
        self.sched_slot_histogram: Dict[str, int] = {}
        # Logical dims / byte estimates for CSV export (run_graph); not used by compiler.
        self.layer_metrics: Dict[str, Any] = {}


def _align_data(nbytes: int) -> int:
    return int(math.ceil(nbytes / 0x1000) * 0x1000) + 0x1000


def _gemm_k_stride(K: int) -> int:
    """Row stride for A and W' in DRAM (pad inner dim to a TILE multiple)."""
    k = max(int(K), 1)
    return int(math.ceil(k / TILE) * TILE)


def _padded_gemm_a(A: np.ndarray, K: int) -> np.ndarray:
    """Left-hand GEMM operand as ``(M, k_stride)`` with tail columns zero."""
    A = np.asarray(A, dtype=np.float32)
    M = int(A.shape[0])
    ks = _gemm_k_stride(K)
    buf = np.zeros((M, ks), dtype=np.float32)
    buf[:, :K] = A[:, :K]
    return buf


def _write_gemm_params(img: DRAMWriter, A_GMEM: int, W_GMEM: int, C_GMEM: int,
                       M: int, N: int, K: int, Z_GMEM: int):
    M_tiles = math.ceil(M / TILE)
    N_tiles = math.ceil(N / TILE)
    K_tiles = math.ceil(K / TILE)
    k_stride = _gemm_k_stride(K)
    img.u32(ADDR_TABLE + 0, A_GMEM)
    img.u32(ADDR_TABLE + 4, W_GMEM)
    img.u32(ADDR_TABLE + 8, C_GMEM)
    img.u32(ADDR_TABLE + 12, M)
    img.u32(ADDR_TABLE + 16, N)
    img.u32(ADDR_TABLE + 20, k_stride)
    img.u32(ADDR_TABLE + 24, M_tiles)
    img.u32(ADDR_TABLE + 28, N_tiles)
    img.u32(ADDR_TABLE + 32, K_tiles)
    img.u32(ADDR_TABLE + 36, TILE)
    img.u32(ADDR_TABLE + 40, Z_GMEM)


def _write_matrix(img: DRAMWriter, base_addr: int, mat: np.ndarray, rows: int, cols: int):
    flat = mat.flatten()
    for i in range(min(rows * cols, len(flat))):
        img.bf16(base_addr + i * 2, float(flat[i]))


def _write_gemm_rhs_weight(img: DRAMWriter, base_addr: int, w_kn: np.ndarray) -> None:
    """Pack RHS for tiled GEMM: logical ``C = A @ W`` with ``W`` shape ``(K, N)``.

    Stored as ``W.T`` row-major (``N`` × ``K``), each row zero-padded to
    ``_gemm_k_stride(K)`` so SDMA can fetch ``tile_n`` lanes when ``K < TILE``.
    """
    w = np.asarray(w_kn, dtype=np.float32)
    k, n = w.shape
    ks = _gemm_k_stride(k)
    buf = np.zeros((n, ks), dtype=np.float32)
    buf[:, :k] = w.T
    _write_matrix(img, base_addr, buf, n, ks)


def _write_zeros(img: DRAMWriter, base_addr: int, count: int):
    for i in range(count):
        img.bf16(base_addr + i * 2, 0.0)


def _gemm_map_metrics(M: int, N: int, K: int, *, kind: str = "gemm") -> Dict[str, Any]:
    """Tiled GEMM / conv-as-GEMM: logical M,N,K, tile counts, DRAM BF16 byte estimates."""
    ks = _gemm_k_stride(K)
    return {
        "map_kind": kind,
        "map_M": int(M),
        "map_N": int(N),
        "map_K": int(K),
        "map_TILE": int(TILE),
        "map_M_tiles": int(math.ceil(M / TILE)),
        "map_N_tiles": int(math.ceil(N / TILE)),
        "map_K_tiles": int(math.ceil(K / TILE)),
        "map_k_stride": int(ks),
        "bytes_est_activation": int(M * ks * 2),
        "bytes_est_weight": int(N * ks * 2),
        "bytes_est_output": int(M * N * 2),
        "bytes_est_Z_tile": int(TILE * 2),
        "map_reuse_note": (
            "A-tile sweeps N; W-tile sweeps M; K outer with accum in C (see tiled GEMM kernel)"
        ),
    }


# ---------------------------------------------------------------------------
# Top-level emitters
# ---------------------------------------------------------------------------

def emit_conv(node: Node, gm: GraphModule,
              input_data: np.ndarray, tc: TileConfig) -> LayerEmission:
    p = tc.params
    M, N, K = p["M"], p["N"], p["K"]
    R, S, C_in = p["R"], p["S"], p["C_in"]

    mod = _get_module(gm, node.target) if node.op == "call_module" else None
    bias_np: Optional[np.ndarray] = None
    if mod is not None:
        weight_np = _to_bf16_array(mod.weight)
        if getattr(mod, "bias", None) is not None:
            bias_np = _to_bf16_array(mod.bias).reshape(-1)
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
        if len(node.args) >= 3 and isinstance(node.args[2], Node):
            bias_node = node.args[2]
            bias_tensor = None
            if hasattr(bias_node, "meta") and bias_node.meta.get("val") is not None:
                bias_tensor = bias_node.meta["val"]
            if bias_tensor is None and bias_node.op == "get_attr":
                attr = gm
                for part in bias_node.target.split("."):
                    attr = getattr(attr, part)
                bias_tensor = attr
            if bias_tensor is not None:
                bias_np = _to_bf16_array(bias_tensor).reshape(-1)
    # PyTorch conv weights are OIHW; im2col emits each patch in (r, s, c) order,
    # so flatten weights in (r, s, c, out) to keep GEMM K aligned.
    weight_flat = weight_np.reshape(N, C_in, R, S).transpose(2, 3, 1, 0).reshape(K, N)

    H, W = p["H"], p["W"]
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
    ks = _gemm_k_stride(K)
    A_dram = _padded_gemm_a(A_mat, K)

    A_GMEM = 0x1000
    W_GMEM = A_GMEM + _align_data(M * ks * 2)
    C_GMEM = W_GMEM + _align_data(N * ks * 2)
    Z_GMEM = C_GMEM + M * N * 2

    img = DRAMWriter()
    _write_gemm_params(img, A_GMEM, W_GMEM, C_GMEM, M, N, K, Z_GMEM)
    _write_matrix(img, A_GMEM, A_dram, M, ks)
    _write_gemm_rhs_weight(img, W_GMEM, weight_flat)
    if bias_np is not None and bias_np.size == N:
        c_init = np.tile(bias_np.reshape(1, N), (M, 1))
        _write_matrix(img, C_GMEM, c_init, M, N)
    else:
        _write_zeros(img, C_GMEM, M * N)
    _write_zeros(img, Z_GMEM, TILE)

    out_shape = get_node_shape(node)
    em = LayerEmission()
    em.c_source = _gemm_c(M, N, K)
    em.dram = img
    em.output_addr = C_GMEM
    em.output_shape = out_shape if out_shape else (1, N, p["Ho"], p["Wo"])
    em.output_elements = M * N
    em.conv_post = {
        "Ho": p["Ho"],
        "Wo": p["Wo"],
        "C": N,
        "final_shape": em.output_shape,
    }
    em.layer_metrics = {
        **_gemm_map_metrics(M, N, K, kind="conv_as_gemm"),
        "map_R": int(R),
        "map_S": int(S),
        "map_C_in": int(C_in),
        "map_H": int(H),
        "map_W": int(W),
        "map_Ho": int(p["Ho"]),
        "map_Wo": int(p["Wo"]),
    }
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

    A = np.asarray(input_data, dtype=np.float32).reshape(M, K)
    ks = _gemm_k_stride(K)
    A_dram = _padded_gemm_a(A, K)

    A_GMEM = 0x1000
    W_GMEM = A_GMEM + _align_data(M * ks * 2)
    C_GMEM = W_GMEM + _align_data(N * ks * 2)
    Z_GMEM = C_GMEM + M * N * 2

    img = DRAMWriter()
    _write_gemm_params(img, A_GMEM, W_GMEM, C_GMEM, M, N, K, Z_GMEM)
    _write_matrix(img, A_GMEM, A_dram, M, ks)
    _write_gemm_rhs_weight(img, W_GMEM, W_mat)
    _write_zeros(img, C_GMEM, M * N)
    _write_zeros(img, Z_GMEM, TILE)

    out_shape = get_node_shape(node)
    em = LayerEmission()
    em.c_source = _gemm_c(M, N, K)
    em.dram = img
    em.output_addr = C_GMEM
    em.output_shape = out_shape if out_shape else (M, N)
    em.output_elements = M * N
    em.layer_metrics = _gemm_map_metrics(M, N, K, kind="gemm")
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
    em.layer_metrics = {
        "map_kind": "relu",
        "map_rows": int(rows),
        "map_width": int(width),
        "bytes_est_in": int(rows * width * 2),
        "bytes_est_out": int(rows * width * 2),
    }
    return em


def emit_softmax(node: Node, input_data: np.ndarray, tc: TileConfig) -> LayerEmission:
    p = tc.params
    if "num_rows" in p:
        num_rows = int(p["num_rows"])
        row_len = int(p["row_len"])
        dim = int(p.get("softmax_dim", -1))
    else:
        num_rows, row_len, dim = 1, int(p.get("length", 1)), -1

    out_shape = get_node_shape(node)
    total = num_rows * row_len

    x = np.asarray(input_data, dtype=np.float32)
    if out_shape:
        x = x.reshape(out_shape)
        nd = x.ndim
        if dim < 0:
            dim += nd
        if 0 <= dim < nd - 1:
            perm = list(range(nd))
            perm[dim], perm[-1] = perm[-1], perm[dim]
            x = np.transpose(x, perm)
        mat = x.reshape(num_rows, row_len)
    else:
        mat = x.reshape(num_rows, row_len)

    flat_rows = mat.reshape(-1)
    if flat_rows.size != total:
        raise ValueError(
            f"softmax layout mismatch: got {flat_rows.size} elems, expect {total}"
        )

    if row_len > 32:
        em = LayerEmission()
        em.skip_emulator = True
        os = tuple(out_shape) if out_shape else (num_rows, row_len)
        xt = torch.from_numpy(np.asarray(input_data, dtype=np.float32).reshape(os))
        d = int(p.get("softmax_dim", -1))
        y = torch.nn.functional.softmax(xt, dim=d).numpy()
        em.numpy_result = y
        em.output_shape = os
        em.output_elements = int(y.size)
        em.dram = DRAMWriter()
        return em

    IN_GMEM = 0x1000
    img = DRAMWriter()
    img.u32(ADDR_TABLE + 0, IN_GMEM)
    img.u32(ADDR_TABLE + 4, 0)

    for i in range(total):
        img.bf16(IN_GMEM + i * 2, float(flat_rows[i]))

    em = LayerEmission()
    if num_rows == 1 and row_len > 32:
        em.c_source = _softmax_c(row_len)
    else:
        em.c_source = _softmax_c_batched(num_rows, row_len)
    em.dram = img
    em.output_addr = IN_GMEM
    em.output_shape = tuple(out_shape) if out_shape else (num_rows, row_len)
    em.output_elements = total
    em.layer_metrics = {
        "map_kind": "softmax",
        "map_num_rows": int(num_rows),
        "map_row_len": int(row_len),
        "bytes_est_io_inplace": int(total * 2),
    }
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
        # Direct matmul(x, W): W is (K, N) in memory (e.g. ParameterList).  Lowered
        # Linear uses matmul(x, W.T) so rhs is a transpose node — use activation_cache.
        attr = gm
        for part in rhs_node.target.split("."):
            attr = getattr(attr, part)
        W_mat = _to_bf16_array(attr)

    A = input_data.reshape(-1, K)[:M, :]
    W_mat = np.asarray(W_mat, dtype=np.float32)
    # RHS must be logical (K, N) for C = A @ W. nn.Linear stores (N, K); lower_linear_modules
    # uses matmul(x, weight.transpose(-1,-2)), so the transpose node's value is already (K, N).
    # For square (K,K), (N,K) and (K,N) shapes collide — do not double-transpose that case.
    rhs_tr = (
        isinstance(rhs_node, Node)
        and rhs_node.op == "call_method"
        and rhs_node.target == "transpose"
    )
    if W_mat.shape == (K, N):
        pass
    elif W_mat.shape == (N, K) and N != K:
        W_mat = W_mat.T
    elif W_mat.shape == (N, K) and N == K:
        if not rhs_tr:
            W_mat = W_mat.T
    elif W_mat.size == K * N:
        W_mat = W_mat.reshape(K, N)
    else:
        W_mat = W_mat.T
    ks = _gemm_k_stride(K)
    A_dram = _padded_gemm_a(A, K)

    A_GMEM = 0x1000
    W_GMEM = A_GMEM + _align_data(M * ks * 2)
    C_GMEM = W_GMEM + _align_data(N * ks * 2)
    Z_GMEM = C_GMEM + M * N * 2

    img = DRAMWriter()
    _write_gemm_params(img, A_GMEM, W_GMEM, C_GMEM, M, N, K, Z_GMEM)
    _write_matrix(img, A_GMEM, A_dram, M, ks)
    _write_gemm_rhs_weight(img, W_GMEM, W_mat)
    _write_zeros(img, C_GMEM, M * N)
    _write_zeros(img, Z_GMEM, TILE)

    out_shape = get_node_shape(node)
    em = LayerEmission()
    em.c_source = _gemm_c(M, N, K)
    em.dram = img
    em.output_addr = C_GMEM
    em.output_shape = tuple(out_shape) if out_shape else ((M, N) if M > 1 else (N,))
    em.output_elements = M * N
    em.layer_metrics = _gemm_map_metrics(M, N, K, kind="gemm")
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

    # Kernel writes per-output-row vectors of width W (vertical max per column);
    # horizontal max over the pool window is applied in run_graph (maxpool_post).
    IN_GMEM = 0x1000
    channel_in_bytes = H * W * 2
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
    em.layer_metrics = {
        "map_kind": "maxpool",
        "map_C": int(C),
        "map_H": int(H),
        "map_W": int(W),
        "map_pool": int(pool),
        "map_stride": int(stride),
        "bytes_est_in": int(C * H * W * 2),
        "bytes_est_out_raw": int(C * H_out * W * 2),
    }
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

    total = max(lhs.size, rhs.size)
    width = min(total, 32)
    rows = math.ceil(total / width)

    a_flat = lhs.flatten()
    b_flat = rhs.flatten()
    if len(b_flat) < len(a_flat):
        b_flat = np.resize(b_flat, a_flat.shape)
    elif len(a_flat) < len(b_flat):
        a_flat = np.resize(a_flat, b_flat.shape)

    A_GMEM = 0x1000
    B_GMEM = A_GMEM + _align_data(rows * width * 2)
    C_GMEM = B_GMEM + _align_data(rows * width * 2)

    img = DRAMWriter()
    img.u32(ADDR_TABLE + 0, A_GMEM)
    img.u32(ADDR_TABLE + 4, B_GMEM)
    img.u32(ADDR_TABLE + 8, C_GMEM)

    padded_a = np.zeros(rows * width, dtype=np.float32)
    padded_b = np.zeros(rows * width, dtype=np.float32)
    padded_a[:len(a_flat)] = a_flat[:total]
    padded_b[:len(b_flat)] = b_flat[:total]
    for i in range(rows * width):
        img.bf16(A_GMEM + i * 2, float(padded_a[i]))
        img.bf16(B_GMEM + i * 2, float(padded_b[i]))

    input_shape = get_node_shape(node)
    out_shape = input_shape if input_shape else (total,)

    em = LayerEmission()
    em.c_source = _add_c(total, width)
    em.dram = img
    em.output_addr = C_GMEM
    em.output_shape = out_shape
    em.output_elements = total
    em.layer_metrics = {
        "map_kind": "add",
        "map_rows": int(rows),
        "map_width": int(width),
        "bytes_est_a": int(rows * width * 2),
        "bytes_est_b": int(rows * width * 2),
        "bytes_est_out": int(rows * width * 2),
    }
    return em


def emit_layernorm(
    node: Node,
    gm: GraphModule,
    input_data: np.ndarray,
    activation_cache: Dict[str, np.ndarray],
    tc: TileConfig,
) -> LayerEmission:
    """LayerNorm: AtallaC when ``D % 32 == 0``, else PyTorch BF16 reference."""
    p = tc.params
    M, D = int(p["M"]), int(p["D"])
    eps = float(p["eps"])
    need = M * D
    flat = np.asarray(input_data, dtype=np.float32).flatten()
    if flat.size < need:
        pad = np.zeros(need, dtype=np.float32)
        pad[: flat.size] = flat
        flat = pad
    else:
        flat = flat[:need].copy()

    w_arr = np.ones(D, dtype=np.float32)
    b_arr = np.zeros(D, dtype=np.float32)
    if node.op == "call_module":
        mod = _get_module(gm, node.target)
        if getattr(mod, "elementwise_affine", True):
            w_arr = mod.weight.detach().float().cpu().numpy().astype(np.float32).flatten()[:D]
            if mod.bias is not None:
                b_arr = mod.bias.detach().float().cpu().numpy().astype(np.float32).flatten()[:D]
    else:
        if len(node.args) > 2 and node.args[2] is not None:
            wn = node.args[2]
            if isinstance(wn, Node) and wn.name in activation_cache:
                w_arr = np.asarray(activation_cache[wn.name], dtype=np.float32).flatten()[:D]
        if len(node.args) > 3 and node.args[3] is not None:
            bn = node.args[3]
            if isinstance(bn, Node) and bn.name in activation_cache:
                b_arr = np.asarray(activation_cache[bn.name], dtype=np.float32).flatten()[:D]

    out_shape = get_node_shape(node)
    os = tuple(out_shape) if out_shape else (M, D)

    use_hw = (D % 32 == 0) and D >= 32
    if use_hw:
        IN0 = 0x1000
        OUT0 = IN0 + _align_data(M * D * 2)
        GM = OUT0 + _align_data(M * D * 2)
        BM = GM + _align_data(D * 2)

        img = DRAMWriter()
        img.u32(ADDR_TABLE + 0, IN0)
        img.u32(ADDR_TABLE + 4, OUT0)
        img.u32(ADDR_TABLE + 8, GM)
        img.u32(ADDR_TABLE + 12, BM)
        img.u32(ADDR_TABLE + 16, M & 0xFFFFFFFF)

        for i in range(M * D):
            img.bf16(IN0 + i * 2, float(flat[i]))
        for i in range(D):
            img.bf16(GM + i * 2, float(w_arr[i]))
            img.bf16(BM + i * 2, float(b_arr[i]))

        em = LayerEmission()
        em.c_source = _layernorm_c(M, D, eps)
        em.dram = img
        em.output_addr = OUT0
        em.output_shape = os
        em.output_elements = need
        em.skip_emulator = False
        em.layer_metrics = {
            "map_kind": "layernorm",
            "map_M_rows": int(M),
            "map_D": int(D),
            "bytes_est_input": int(M * D * 2),
            "bytes_est_output": int(M * D * 2),
            "bytes_est_gamma": int(D * 2),
            "bytes_est_beta": int(D * 2),
        }
        return em

    x = torch.from_numpy(flat.reshape(M, D)).to(torch.bfloat16)
    wt = torch.from_numpy(w_arr.copy()).to(torch.bfloat16)
    bi = torch.from_numpy(b_arr.copy()).to(torch.bfloat16)
    y = torch.nn.functional.layer_norm(x, (D,), wt, bi, eps).float().numpy()

    em = LayerEmission()
    em.skip_emulator = True
    em.numpy_result = y.reshape(os)
    em.output_shape = os
    em.output_elements = int(y.size)
    em.dram = DRAMWriter()
    return em


def emit_gelu(node: Node, input_data: np.ndarray, tc: TileConfig) -> LayerEmission:
    """GELU via PyTorch BF16 (ISA kernel TBD)."""
    p = tc.params
    total = int(p["total_elements"])
    flat = np.asarray(input_data, dtype=np.float32).flatten()[:total]
    if flat.size < total:
        pad = np.zeros(total, dtype=np.float32)
        pad[: flat.size] = flat
        flat = pad
    x = torch.from_numpy(flat).to(torch.bfloat16)
    try:
        y = torch.nn.functional.gelu(x, approximate="tanh")
    except TypeError:
        y = torch.nn.functional.gelu(x)
    y = y.float().numpy()
    out_shape = get_node_shape(node)
    os = tuple(out_shape) if out_shape else (total,)
    em = LayerEmission()
    em.skip_emulator = True
    em.numpy_result = y.reshape(os)
    em.output_shape = os
    em.output_elements = total
    em.dram = DRAMWriter()
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
        return emit_softmax(node, input_data, tc)
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
    elif atalla_op == "layernorm":
        return emit_layernorm(node, gm, input_data, activation_cache, tc)
    elif atalla_op == "gelu":
        return emit_gelu(node, input_data, tc)

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
        in_content, _ready, packets = _bc.compile_asm(raw_s)
        n_pkt = len(packets)
        n_slot = sum(len(p) for p in packets)
        emission.sched_packets = n_pkt
        emission.sched_slots_filled = n_slot
        emission.sched_slot_efficiency = (n_slot / (n_pkt * 4.0)) if n_pkt else 0.0
        emission.sched_slot_histogram = {
            str(k): v for k, v in sorted(Counter(len(p) for p in packets).items())
        }
        # compile_asm returns a complete .in file; strip the .data section
        # since render_in_file will add the populated DRAMWriter data
        emission.instr_text = in_content.split("\n.data")[0].strip()
    return emission.instr_text


def render_in_file(emission: LayerEmission) -> str:
    """Render a LayerEmission to a complete .in file string."""
    data_text = emission.dram.render_data_mem(include_zeros=True)
    return render_testfile(emission.instr_text, data_text)
