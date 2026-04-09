#!/usr/bin/env python3
"""Standalone GEMM golden checks (same DRAM layout as emit_matmul).

Run from ``atalla-graph`` root:
  python scripts/debug_matmul_golden.py --case all
  python scripts/debug_matmul_golden.py --case matmul2 --seed 42

Compares emulator output to PyTorch BF16 matmul (same as a reasonable golden).
Also runs identity / one-hot cases to expose transpose or partial-store bugs.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
_FUNC_SIM = _ROOT.parent / "functional_sim"
for p in (_ROOT, _FUNC_SIM):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from build import DRAMWriter  # noqa: E402
from codegen.c_emitter import (  # noqa: E402
    LayerEmission,
    _align_data,
    _gemm_k_stride,
    _padded_gemm_a,
    _write_gemm_params,
    _write_gemm_rhs_weight,
    _write_matrix,
    _write_zeros,
    compile_and_assemble,
    render_in_file,
)
from kernels.common import TILE  # noqa: E402
from kernels.gemm import gemm_c as _gemm_c  # noqa: E402
from run_graph import _layer_compare_metrics, _read_bf16, _nz_count, _run_emulator  # noqa: E402


def _ref_bf16(A: np.ndarray, W: np.ndarray) -> np.ndarray:
    """C = A @ W with BF16 round-trip on operands (M,K) @ (K,N)."""
    At = torch.from_numpy(np.asarray(A, np.float32)).to(torch.bfloat16)
    Wt = torch.from_numpy(np.asarray(W, np.float32)).to(torch.bfloat16)
    return (At @ Wt).float().numpy()


def _run_gemm(
    A: np.ndarray,
    W: np.ndarray,
    work_dir: str,
    tag: str,
    *,
    weight_layout: str = "emitter",
) -> np.ndarray:
    M, K = A.shape
    K2, N = W.shape
    if K != K2:
        raise ValueError(f"A.shape={A.shape} W.shape={W.shape} need K match")
    ks = _gemm_k_stride(K)
    A_dram = _padded_gemm_a(A, K)
    A_GMEM = 0x1000
    W_GMEM = A_GMEM + _align_data(M * ks * 2)
    C_GMEM = W_GMEM + _align_data(N * ks * 2)
    Z_GMEM = C_GMEM + M * N * 2

    img = DRAMWriter()
    _write_gemm_params(img, A_GMEM, W_GMEM, C_GMEM, M, N, K, Z_GMEM)
    _write_matrix(img, A_GMEM, A_dram, M, ks)
    if weight_layout == "emitter":
        _write_gemm_rhs_weight(img, W_GMEM, W)
    elif weight_layout == "legacy":
        _write_matrix(img, W_GMEM, W, K, N)
    else:
        raise ValueError(weight_layout)
    _write_zeros(img, C_GMEM, M * N)
    _write_zeros(img, Z_GMEM, TILE)

    em = LayerEmission()
    em.c_source = _gemm_c(M, N, K)
    em.dram = img
    em.output_addr = C_GMEM
    em.output_elements = M * N

    os.makedirs(work_dir, exist_ok=True)
    compile_and_assemble(em, work_dir, tag)
    in_path = Path(work_dir) / f"{tag}.in"
    in_path.write_text(render_in_file(em))
    mem, _eu = _run_emulator(str(in_path), work_dir, tag)
    raw = _read_bf16(mem, C_GMEM, M * N)
    return raw.reshape(M, N)


def _print_slice(name: str, mat: np.ndarray, nrows: int = 2, ncols: int = 8) -> None:
    r = min(nrows, mat.shape[0])
    c = min(ncols, mat.shape[1])
    print(f"  {name} shape={mat.shape} first [{r}x{c}]:")
    for i in range(r):
        row = mat[i, :c]
        print(f"    [{i}] " + " ".join(f"{float(x):8.4f}" for x in row))


def _report(
    case: str,
    A: np.ndarray,
    W: np.ndarray,
    verbose: bool,
    *,
    weight_layout: str = "emitter",
) -> dict:
    work = str(_ROOT / "out" / "debug_matmul")
    emu = _run_gemm(A, W, work, case.replace(" ", "_"), weight_layout=weight_layout)
    ref = _ref_bf16(A, W)
    alt = _ref_bf16(A, W.T) if W.shape[0] == W.shape[1] else None

    m = _layer_compare_metrics(ref, emu)
    print(f"\n=== {case} (W DRAM layout={weight_layout}) ===")
    print(
        f"  M,N,K = {A.shape[0]},{W.shape[1]},{A.shape[1]}  "
        f"cos={m['cos_sim']:.6f}  rel_l2={m['rel_l2_error']:.6f}  "
        f"relmax={m['rel_max_abs_error']:.6f}  max_abs={m['max_abs_error']:.6f}"
    )
    print(f"  emu_nz={_nz_count(emu)}  ref_nz={_nz_count(ref)}  "
          f"emu_norm={float(np.linalg.norm(emu)):.6f}  ref_norm={float(np.linalg.norm(ref)):.6f}")

    if alt is not None:
        m_alt = _layer_compare_metrics(alt, emu)
        print(
            f"  If ref used W.T instead: cos={m_alt['cos_sim']:.6f}  "
            f"rel_l2={m_alt['rel_l2_error']:.6f}  (transpose-ref sanity check)"
        )

    if verbose:
        _print_slice("A", A)
        _print_slice("W", W)
        _print_slice("ref", ref)
        _print_slice("emu", emu)
    return m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case",
        choices=("identity", "one_hot", "matmul2", "matmul1", "matmul7", "all"),
        default="all",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--weight-layout",
        choices=("emitter", "legacy"),
        default="emitter",
        help="emitter=match c_emitter._write_gemm_rhs_weight; legacy=old (K,N) row-major (wrong).",
    )
    p.add_argument(
        "--gate",
        action="store_true",
        help="Exit 1 if any run case fails BF16-style thresholds (cos>=0.995, rel_l2<0.02).",
    )
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    wl = args.weight_layout

    def gate_ok(m: dict) -> bool:
        return m["cos_sim"] >= 0.995 and m["rel_l2_error"] < 0.02

    bad: list[str] = []

    if args.case in ("identity", "all"):
        m = _report(
            "identity I_4 @ W (4x4 @ 4x32)",
            np.eye(4, dtype=np.float32),
            np.fromfunction(lambda i, j: 0.01 * (i * 32 + j), (4, 32), dtype=np.float32),
            verbose=True,
            weight_layout=wl,
        )
        if args.gate and not gate_ok(m):
            bad.append("identity")
    if args.case in ("one_hot", "all"):
        A = np.zeros((4, 32), dtype=np.float32)
        A[2, 5] = 1.0
        W = np.arange(32 * 32, dtype=np.float32).reshape(32, 32) * 0.001
        m = _report("one_hot row2 col5 @ W (4x32 @ 32x32)", A, W, verbose=True, weight_layout=wl)
        if args.gate and not gate_ok(m):
            bad.append("one_hot")
    if args.case in ("matmul2", "all"):
        A = rng.standard_normal((4, 32)).astype(np.float32) * 0.02
        W = rng.standard_normal((32, 32)).astype(np.float32) * 0.02
        m = _report("matmul2-style (4x32) @ (32x32)", A, W, verbose=False, weight_layout=wl)
        if args.gate and not gate_ok(m):
            bad.append("matmul2")
    if args.case in ("matmul1", "all"):
        A = rng.standard_normal((4, 4)).astype(np.float32) * 0.05
        W = rng.standard_normal((4, 32)).astype(np.float32) * 0.05
        m = _report("matmul1-style (4x4) @ (4x32)", A, W, verbose=True, weight_layout=wl)
        if args.gate and not gate_ok(m):
            bad.append("matmul1")
    if args.case in ("matmul7", "all"):
        A = rng.standard_normal((4, 32)).astype(np.float32) * 0.02
        W = rng.standard_normal((32, 64)).astype(np.float32) * 0.02
        m = _report("matmul7-style (4x32) @ (32x64)", A, W, verbose=False, weight_layout=wl)
        if args.gate and not gate_ok(m):
            bad.append("matmul7")

    print("\nDone. Artifacts under out/debug_matmul/")
    print("Interpret: cos~1 & low rel_l2 => emulator matches PyTorch BF16 matmul.")
    print("If 'legacy' layout matches ref but 'emitter' does not, RHS packing in c_emitter is wrong.")
    print("If K < 32 cases still show emu_nz << ref_nz, suspect GEMM tile bounds (separate from transpose).")

    if args.gate and bad:
        print(f"\nGATE FAIL: {bad}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
