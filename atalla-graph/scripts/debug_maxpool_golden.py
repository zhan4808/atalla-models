#!/usr/bin/env python3
"""Isolate maxpool errors: vertical raw vs horizontal post vs NumPy / PyTorch gold.

Run from ``atalla-graph`` root::

  python scripts/debug_maxpool_golden.py --case all
  python scripts/debug_maxpool_golden.py --case tiny_tie --verbose

Stages compared (same DRAM layout as ``emit_maxpool``):

1. **vertical_raw** — BF16 read from emulator after the AtallaC kernel (shape ``(C, H_out, W)``).
2. **post** — same ``maxpool_post`` loop as ``run_graph.py`` (shape ``(C, H_out, W_out)``).
3. **numpy_vertical** — NumPy replica of the kernel’s vertical merge (strict ``>``).
4. **numpy_full** — full 2D max over the pool window (gold for post + final).
5. **torch_f32** / **torch_bf16** — ``torch.nn.functional.max_pool2d`` on float32 or BF16 input.

If **vertical_raw** already disagrees with **numpy_vertical**, the bug is in the emulator /
masked vertical path. If vertical matches NumPy but **post** disagrees with **numpy_full**,
the bug is in ``maxpool_post`` indexing. If both match NumPy but not Torch, check dtype /
rounding.

**Regression suite** (default BF16-aware gold + thresholds): ``tiny_tie``, ``all_losers``,
``two_channel`` under ``--case all`` catch masked-merge / vd seeding bugs; ``all_losers``
should match exactly. Pass/fail uses BF16-rounded NumPy gold by default and
``rel_l2 < 1e-2``, ``max_abs < 0.02``, ``cos >= 0.999``. Use ``--f32-ref`` for float32
gold and strict ``1e-5`` checks (legacy).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
_FUNC_SIM = _ROOT.parent / "functional_sim"
for p in (_ROOT, _FUNC_SIM):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from build import DRAMWriter  # noqa: E402
from codegen.c_emitter import (  # noqa: E402
    LayerEmission,
    _align_data,
    _write_zeros,
    compile_and_assemble,
    render_in_file,
)
from kernels import ADDR_TABLE  # noqa: E402
from kernels.maxpool import maxpool_c as _maxpool_c  # noqa: E402
from run_graph import _layer_compare_metrics, _nz_count, _read_bf16, _run_emulator  # noqa: E402


def _apply_maxpool_post(raw_c_hw: np.ndarray, pp: Dict) -> np.ndarray:
    """Same horizontal pass as ``run_graph.run_validate``."""
    C, H_out, W = pp["C"], pp["H_out"], pp["W"]
    W_out, pool, stride = pp["W_out"], pp["pool"], pp["stride"]
    raw = raw_c_hw.reshape(C, H_out, W)
    out = np.empty((C, H_out, W_out), dtype=np.float32)
    for c in range(C):
        for oh in range(H_out):
            for ow in range(W_out):
                base = ow * stride
                out[c, oh, ow] = max(
                    float(raw[c, oh, base + p])
                    for p in range(pool)
                    if base + p < W
                )
    return out


def _numpy_vertical_strict_gt(x_chw: np.ndarray, pool: int, stride: int) -> np.ndarray:
    """Replica of AtallaC vertical merge: ``make_mask('>', v, best)`` + masked pick."""
    x = np.asarray(x_chw, dtype=np.float32)
    C, H, W = x.shape
    H_out = (H - pool) // stride + 1
    raw = np.zeros((C, H_out, W), dtype=np.float32)
    for c in range(C):
        for oh in range(H_out):
            ir = oh * stride
            best = x[c, ir].copy()
            for p in range(1, pool):
                v = x[c, ir + p]
                gt = v > best
                best = np.where(gt, v, best)
            raw[c, oh] = best
    return raw


def _numpy_full_maxpool(x_chw: np.ndarray, pool: int, stride: int) -> np.ndarray:
    """Full 2D max pool (float32 reference)."""
    x = np.asarray(x_chw, dtype=np.float32)
    C, H, W = x.shape
    H_out = (H - pool) // stride + 1
    W_out = (W - pool) // stride + 1
    out = np.full((C, H_out, W_out), -np.inf, dtype=np.float32)
    for c in range(C):
        for oh in range(H_out):
            for ow in range(W_out):
                for pr in range(pool):
                    for pc in range(pool):
                        ih = oh * stride + pr
                        iw = ow * stride + pc
                        if ih < H and iw < W:
                            v = x[c, ih, iw]
                            if v > out[c, oh, ow]:
                                out[c, oh, ow] = v
    return out


def _torch_maxpool(x_chw: np.ndarray, pool: int, stride: int, bf16: bool) -> np.ndarray:
    C, H, W = x_chw.shape
    t = torch.from_numpy(np.asarray(x_chw, np.float32)).view(1, C, H, W)
    if bf16:
        t = t.to(torch.bfloat16)
        y = F.max_pool2d(t, kernel_size=pool, stride=stride)
        return y.float().numpy().reshape(C, y.shape[-2], y.shape[-1])
    y = F.max_pool2d(t, kernel_size=pool, stride=stride)
    return y.numpy().reshape(C, y.shape[-2], y.shape[-1])


def _bf16_round_numpy(x: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(np.asarray(x, np.float32)).to(torch.bfloat16)
    return t.float().numpy()


# Default pass criteria: emulator output is BF16; float32 NumPy without rounding is too strict.
_BF16_PASS_REL_L2 = 1e-2
_BF16_PASS_MAX_ABS = 0.02
_BF16_PASS_COS = 0.999
_STRICT_REL_L2 = 1e-5


def _metrics_pass_bf16(m: Dict[str, float]) -> bool:
    return (
        m["cos_sim"] >= _BF16_PASS_COS
        and m["rel_l2_error"] < _BF16_PASS_REL_L2
        and m["max_abs_error"] < _BF16_PASS_MAX_ABS
    )


def _metrics_pass_strict(m: Dict[str, float]) -> bool:
    return m["rel_l2_error"] < _STRICT_REL_L2


def _build_dram_and_emit(
    x_chw: np.ndarray, pool: int, stride: int, final_shape: Tuple[int, ...]
) -> Tuple[LayerEmission, int]:
    C, H, W = x_chw.shape
    H_out = (H - pool) // stride + 1
    W_out = (W - pool) // stride + 1
    total = C * H * W
    flat = np.asarray(x_chw, dtype=np.float32).reshape(-1)
    if flat.size < total:
        pad = np.zeros(total, dtype=np.float32)
        pad[: flat.size] = flat
        flat = pad
    data_nchw = flat[:total].reshape(C, H, W)

    IN_GMEM = 0x1000
    channel_in_bytes = H * W * 2
    total_in_bytes = C * channel_in_bytes
    total_raw_out = C * H_out * W
    OUT_GMEM = IN_GMEM + _align_data(total_in_bytes)

    img = DRAMWriter()
    img.u32(ADDR_TABLE + 0, IN_GMEM)
    img.u32(ADDR_TABLE + 4, OUT_GMEM)
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
    em.maxpool_post = dict(
        C=C,
        H_out=H_out,
        W=W,
        W_out=W_out,
        pool=pool,
        stride=stride,
        final_shape=final_shape,
    )
    return em, OUT_GMEM


def _run_emu(em: LayerEmission, work_dir: str, tag: str) -> np.ndarray:
    os.makedirs(work_dir, exist_ok=True)
    compile_and_assemble(em, work_dir, tag)
    in_path = Path(work_dir) / f"{tag}.in"
    in_path.write_text(render_in_file(em))
    mem, _eu = _run_emulator(str(in_path), work_dir, tag)
    raw = _read_bf16(mem, em.output_addr, em.output_elements)
    return raw.reshape(em.output_shape)


def _print_block(title: str, a: np.ndarray, r: int = 4, c: int = 8) -> None:
    a = np.asarray(a, dtype=np.float32)
    print(f"  {title} shape={a.shape}")
    rr = min(r, a.shape[0] if a.ndim >= 1 else 0)
    if a.ndim == 1:
        cc = min(c, a.shape[0])
        print("    " + " ".join(f"{float(x):8.4f}" for x in a[:cc]))
        return
    if a.ndim == 2:
        cc = min(c, a.shape[1])
        for i in range(rr):
            print(f"    [{i}] " + " ".join(f"{float(x):8.4f}" for x in a[i, :cc]))
        return
    if a.ndim == 3:
        cc = min(c, a.shape[2])
        for k in range(min(2, a.shape[0])):
            print(f"    [c={k}]")
            for i in range(min(r, a.shape[1])):
                print(
                    f"      [{i}] "
                    + " ".join(f"{float(x):8.4f}" for x in a[k, i, :cc])
                )


def _case_tensor(name: str, seed: int) -> Tuple[np.ndarray, int, int, str]:
    rng = np.random.default_rng(seed)
    pool, stride = 2, 2

    if name == "tiny_pos":
        x = rng.uniform(0.1, 2.0, size=(1, 4, 4)).astype(np.float32)
    elif name == "tiny_neg":
        x = rng.uniform(-2.0, -0.1, size=(1, 4, 4)).astype(np.float32)
    elif name == "tiny_tie":
        # All four rows identical -> vertical strict > never updates; still valid maxpool.
        row = rng.uniform(0.5, 1.5, size=(4,)).astype(np.float32)
        x = np.tile(row, (4, 1))[np.newaxis, :, :].astype(np.float32)
    elif name == "all_losers":
        # Second row in each vertical window strictly smaller -> best stays first row.
        x = np.array(
            [
                [
                    [5.0, 5.0, 5.0, 5.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [5.0, 5.0, 5.0, 5.0],
                    [1.0, 1.0, 1.0, 1.0],
                ]
            ],
            dtype=np.float32,
        )
    elif name == "two_channel":
        a = rng.uniform(0.0, 1.0, size=(1, 6, 6)).astype(np.float32)
        b = rng.uniform(0.5, 1.5, size=(1, 6, 6)).astype(np.float32)
        x = np.concatenate([a, b], axis=0)
    elif name == "alex_tile":
        x = rng.standard_normal(size=(1, 16, 16)).astype(np.float32) * 0.05
    else:
        raise ValueError(f"unknown case {name}")

    return x, pool, stride, name


def _run_case(
    case: str,
    seed: int,
    work_dir: str,
    verbose: bool,
    bf16_input: bool,
    bf16_ref: bool,
    f32_ref: bool,
) -> Tuple[bool, bool]:
    """Returns (vertical_ok, post_ok) under chosen pass criteria."""
    x, pool, stride, tag = _case_tensor(case, seed)
    if bf16_input:
        x = _bf16_round_numpy(x)
    x_gold = x if f32_ref else _bf16_round_numpy(x)
    C, H, W = x_gold.shape
    H_out = (H - pool) // stride + 1
    W_out = (W - pool) // stride + 1
    final_shape = (1, C, H_out, W_out)

    print(f"\n=== case={case} seed={seed}  x.shape={x_gold.shape}  pool={pool} stride={stride} ===")
    if verbose:
        _print_block("input x (gold / DRAM)", x_gold)

    nv = _numpy_vertical_strict_gt(x_gold, pool, stride)
    nf = _numpy_full_maxpool(x_gold, pool, stride)
    tt_f32 = _torch_maxpool(x_gold, pool, stride, bf16=False)
    tt_bf = _torch_maxpool(x_gold, pool, stride, bf16=True)

    em, _out_gmem = _build_dram_and_emit(x_gold, pool, stride, final_shape)
    emu_v = _run_emu(em, work_dir, f"mp_{tag}")
    pp = em.maxpool_post
    assert pp is not None
    emu_post = _apply_maxpool_post(emu_v, pp).reshape(final_shape)
    emu_post_flat = emu_post.reshape(C, H_out, W_out)

    if verbose:
        _print_block("numpy_vertical (gold for kernel raw)", nv)
        _print_block("emu vertical_raw", emu_v)
        _print_block("numpy_full (gold final)", nf)
        _print_block("emu after maxpool_post", emu_post_flat)

    def report(stage: str, ref: np.ndarray, got: np.ndarray) -> None:
        m = _layer_compare_metrics(ref, got)
        print(
            f"  {stage:28s}  cos={m['cos_sim']:.6f}  rel_l2={m['rel_l2_error']:.6f}  "
            f"max_abs={m['max_abs_error']:.6f}  "
            f"nz(ref)={_nz_count(ref)} nz(got)={_nz_count(got)}"
        )

    report("emu_vertical vs numpy_vertical", nv.reshape(-1), emu_v.reshape(-1))
    report("emu_post vs numpy_full", nf.reshape(-1), emu_post_flat.reshape(-1))
    report("emu_post vs torch_f32", tt_f32.reshape(-1), emu_post_flat.reshape(-1))
    if bf16_ref:
        report("emu_post vs torch_bf16", tt_bf.reshape(-1), emu_post_flat.reshape(-1))

    mv = _layer_compare_metrics(nv.reshape(-1), emu_v.reshape(-1))
    mp = _layer_compare_metrics(nf.reshape(-1), emu_post_flat.reshape(-1))
    pass_fn = _metrics_pass_strict if f32_ref else _metrics_pass_bf16
    v_ok = pass_fn(mv)
    p_ok = pass_fn(mp)
    crit = "strict f32" if f32_ref else "BF16-aware"
    v_label = "PASS" if v_ok else "FAIL"
    p_label = "PASS" if p_ok else "FAIL"
    print(f"  -> vertical {v_label} ({crit})   post {p_label} ({crit})")

    if f32_ref:
        if mv["rel_l2_error"] < _STRICT_REL_L2 and mp["rel_l2_error"] >= _STRICT_REL_L2:
            print("  -> (diagnostic) Under strict f32: failure first at **horizontal post**.")
        elif mv["rel_l2_error"] >= _STRICT_REL_L2:
            print("  -> (diagnostic) Under strict f32: failure first at **vertical raw**.")
    elif not v_ok and p_ok:
        print("  -> Note: vertical failed BF16 criteria but post passed — unusual; inspect vertical metrics.")
    return v_ok, p_ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Maxpool golden: vertical vs post vs references")
    ap.add_argument(
        "--case",
        default="all",
        help="tiny_pos | tiny_neg | tiny_tie | all_losers | two_channel | alex_tile | all",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work-dir", default=str(_ROOT / "out" / "debug_maxpool"))
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument(
        "--bf16-input",
        action="store_true",
        help="Round input to BF16 before DRAM write (matches stricter hw path).",
    )
    ap.add_argument(
        "--bf16-ref",
        action="store_true",
        help="Compare emulator to torch max_pool2d with BF16 input (diagnostic).",
    )
    ap.add_argument(
        "--f32-ref",
        action="store_true",
        help="Use float32 NumPy gold (no BF16 input rounding) and strict rel_l2 < 1e-5 pass/fail.",
    )
    args = ap.parse_args()

    cases = (
        ["tiny_pos", "tiny_neg", "tiny_tie", "all_losers", "two_channel", "alex_tile"]
        if args.case == "all"
        else [args.case]
    )
    if not args.f32_ref:
        print(
            "BF16 regression mode: NumPy gold uses BF16-rounded input (same as DRAM); "
            f"pass if cos>={_BF16_PASS_COS}, rel_l2<{_BF16_PASS_REL_L2}, "
            f"max_abs<{_BF16_PASS_MAX_ABS} for vertical and post vs NumPy.\n"
        )
    failures: List[str] = []
    for c in cases:
        v_ok, p_ok = _run_case(
            c,
            args.seed,
            args.work_dir,
            args.verbose,
            args.bf16_input,
            args.bf16_ref,
            args.f32_ref,
        )
        if not (v_ok and p_ok):
            failures.append(f"{c} seed={args.seed}")
    print()
    if failures:
        print(f"FAILED ({len(failures)}): {', '.join(failures)}")
        raise SystemExit(1)
    print(f"OK: all {len(cases)} case(s) passed vertical + post.")


if __name__ == "__main__":
    main()
