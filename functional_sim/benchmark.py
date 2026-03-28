"""Comprehensive Atalla benchmark suite.

Runs all kernel types at multiple sizes, collects metrics, and generates
publication-quality graphs.

Usage:
    python benchmark.py                    # run everything
    python benchmark.py --only-graphs      # re-plot from saved CSV
"""
from __future__ import annotations

import os, sys, time, math, struct, json, csv
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from build import assemble_file, emit_test_format, DRAMWriter, render_testfile
from build_attention import make_attention_asm, f32_to_bf16
from build_gemm_tiled import make_tiled_gemm_asm, TILE
from build_conv_tiled import make_tiled_conv_asm, im2col
from build_maxpool import make_maxpool_asm
from build_alexnet_layer import (
    alexnet_layers, build_layer, make_relu_asm, make_softmax_asm,
    make_tiled_gemm_asm as layer_gemm_asm,
)
from src.functional_sim import run as run_emulator
from src.misc.memory import Memory
from src.components.scalar_register_file import ScalarRegisterFile
from src.components.vector_register_file import VectorRegisterFile
from src.components.execute import ExecuteUnit
from src.components.scpad import Scratchpad


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class BenchResult:
    kernel: str
    params: str
    theoretical_flops: int
    wall_time_ms: float
    cycles: int = 0
    packets: int = 0
    instructions: int = 0
    branches: int = 0
    mem_ops: int = 0
    gemm_ops: int = 0
    sdma_ops: int = 0
    flops_total: int = 0
    flops_scalar: int = 0
    flops_vector: int = 0
    flops_matmul: int = 0
    bytes_loaded: int = 0
    arithmetic_intensity: float = 0.0


# ---------------------------------------------------------------------------
# Emulator runner
# ---------------------------------------------------------------------------
OUT_DIR = "out/bench"


def run_on_emulator(in_text: str, tag: str) -> Dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    in_path = f"{OUT_DIR}/{tag}.in"
    Path(in_path).write_text(in_text)

    mem = Memory(in_path)
    sregs = ScalarRegisterFile()
    mregs = ScalarRegisterFile(num_regs=16)
    vregs = VectorRegisterFile()
    SP0 = Scratchpad(slots_per_bank=32)
    SP1 = Scratchpad(slots_per_bank=32)
    EU = ExecuteUnit()

    prefix = f"{OUT_DIR}/{tag}"
    t0 = time.time()
    run_emulator(
        mem, sregs, mregs, vregs, SP0, SP1, EU, 0, 4,
        f"{prefix}_mem.out", f"{prefix}_sregs.out", f"{prefix}_vregs.out",
        f"{prefix}_mregs.out", f"{prefix}_sp0.out", f"{prefix}_sp1.out",
        f"{prefix}_perf.out", debug=False,
    )
    wall_ms = (time.time() - t0) * 1000

    metrics = {}
    with open(f"{prefix}_perf.out") as f:
        for line in f:
            if ":" in line:
                k, v = line.strip().split(":", 1)
                try:
                    metrics[k.strip()] = float(v.strip())
                except ValueError:
                    pass
    metrics["wall_time_ms"] = wall_ms
    return metrics


# ---------------------------------------------------------------------------
# Kernel builders (return .in text + theoretical flops)
# ---------------------------------------------------------------------------
def build_gemm(M, N, K):
    M_tiles = math.ceil(M / TILE)
    N_tiles = math.ceil(N / TILE)
    K_tiles = math.ceil(K / TILE)

    asm = make_tiled_gemm_asm(M, N, K)
    instrs = assemble_file(asm)
    instr_text = emit_test_format(instrs)

    img = DRAMWriter()
    AT = 60
    A_GMEM = 0x1000
    B_GMEM = A_GMEM + M * K * 2 + 0x1000
    C_GMEM = B_GMEM + K * N * 2 + 0x1000

    img.u32(AT + 0, A_GMEM); img.u32(AT + 4, B_GMEM); img.u32(AT + 8, C_GMEM)
    img.u32(AT + 12, M); img.u32(AT + 16, N); img.u32(AT + 20, K)
    img.u32(AT + 24, M_tiles); img.u32(AT + 28, N_tiles)
    img.u32(AT + 32, K_tiles); img.u32(AT + 36, TILE)

    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K)).astype(np.float32) * 0.5
    B = rng.standard_normal((K, N)).astype(np.float32) * 0.5

    for r in range(M):
        for c in range(K):
            img.bf16(A_GMEM + (r * K + c) * 2, float(A[r, c]))
    for r in range(K):
        for c in range(N):
            img.bf16(B_GMEM + (r * N + c) * 2, float(B[r, c]))
    for i in range(M * N):
        img.bf16(C_GMEM + i * 2, 0.0)

    data_text = img.render_data_mem(include_zeros=True)
    final = render_testfile(instr_text, data_text)
    flops = 2 * M * N * K
    return final, flops


def build_conv(H, W, C, K_out, R, S, stride, pad):
    Ho = (H + 2 * pad - R) // stride + 1
    Wo = (W + 2 * pad - S) // stride + 1
    K_flat = R * S * C
    M = Ho * Wo
    M_tiles = math.ceil(M / TILE)
    N_tiles = math.ceil(K_out / TILE)
    K_tiles = math.ceil(K_flat / TILE)

    asm = make_tiled_conv_asm(M, K_flat, K_out)
    instrs = assemble_file(asm)
    instr_text = emit_test_format(instrs)

    img = DRAMWriter()
    AT = 60
    A_GMEM = 0x1000
    W_GMEM = A_GMEM + M * K_flat * 2 + 0x1000
    C_GMEM = W_GMEM + K_flat * K_out * 2 + 0x1000

    img.u32(AT + 0, A_GMEM); img.u32(AT + 4, W_GMEM); img.u32(AT + 8, C_GMEM)
    img.u32(AT + 12, M); img.u32(AT + 16, K_out); img.u32(AT + 20, K_flat)
    img.u32(AT + 24, M_tiles); img.u32(AT + 28, N_tiles)
    img.u32(AT + 32, K_tiles); img.u32(AT + 36, TILE)

    rng = np.random.default_rng(0)
    ifmap = rng.standard_normal((1, H, W, C)).astype(np.float32) * 0.5
    weights = rng.standard_normal((R, S, C, K_out)).astype(np.float32) * 0.5
    A_mat = im2col(ifmap, 1, H, W, C, R, S, stride, pad)
    W_flat = weights.reshape(K_flat, K_out)

    for r in range(M):
        for c in range(K_flat):
            img.bf16(A_GMEM + (r * K_flat + c) * 2, float(A_mat[r, c]))
    for r in range(K_flat):
        for c in range(K_out):
            img.bf16(W_GMEM + (r * K_out + c) * 2, float(W_flat[r, c]))
    for i in range(M * K_out):
        img.bf16(C_GMEM + i * 2, 0.0)

    data_text = img.render_data_mem(include_zeros=True)
    final = render_testfile(instr_text, data_text)
    flops = 2 * M * K_flat * K_out
    return final, flops


def build_attention(S, d):
    from build_attention import make_attention_asm
    asm = make_attention_asm(S, d)
    instrs = assemble_file(asm)
    instr_text = emit_test_format(instrs)

    AT = 60
    Q_GMEM = 0x1000; Q_SCPAD = 0
    KT_GMEM = 0x2000; KT_SCPAD = 512
    V_GMEM = 0x3000; V_SCPAD = 1024
    OUT_GMEM = 0x4000; OUT_SCPAD = 0
    SCORES_SCPAD = 0

    inv_sqrt_d = 1.0 / math.sqrt(d)
    rng = np.random.default_rng(42)
    Q = rng.standard_normal((S, d)).astype(np.float32) * 0.5
    K = rng.standard_normal((S, d)).astype(np.float32) * 0.5
    V = rng.standard_normal((S, d)).astype(np.float32) * 0.5
    KT = K.T

    img = DRAMWriter()
    img.u32(AT + 0, Q_GMEM); img.u32(AT + 4, Q_SCPAD)
    img.u32(AT + 8, KT_GMEM); img.u32(AT + 12, KT_SCPAD)
    img.u32(AT + 16, V_GMEM); img.u32(AT + 20, V_SCPAD)
    img.u32(AT + 24, OUT_GMEM); img.u32(AT + 28, OUT_SCPAD)
    img.u32(AT + 32, SCORES_SCPAD)
    scale_bits = struct.unpack("<I", struct.pack("<f", inv_sqrt_d))[0]
    img.u32(AT + 36, scale_bits)

    for r in range(S):
        for c in range(d):
            img.bf16(Q_GMEM + (r * d + c) * 2, float(Q[r, c]))
    for r in range(d):
        for c in range(S):
            img.bf16(KT_GMEM + (r * S + c) * 2, float(KT[r, c]))
    for r in range(S):
        for c in range(d):
            img.bf16(V_GMEM + (r * d + c) * 2, float(V[r, c]))
    for i in range(S * d):
        img.bf16(OUT_GMEM + i * 2, 0.0)

    data_text = img.render_data_mem(include_zeros=True)
    final = render_testfile(instr_text, data_text)
    # Q*K^T: 2*S*S*d, scale: S*S, softmax: ~5*S*S, attn*V: 2*S*d*S
    flops = 2 * S * S * d + S * S + 5 * S * S + 2 * S * d * S
    return final, flops


def build_relu_bench(total, width):
    width = min(width, 32)
    rows = math.ceil(total / width)
    sp_rows = min(rows, TILE)

    asm = make_relu_asm(total, width)
    instrs = assemble_file(asm)
    instr_text = emit_test_format(instrs)

    img = DRAMWriter()
    AT = 60
    IN_GMEM = 0x1000
    OUT_GMEM = IN_GMEM + rows * width * 2 + 0x1000
    img.u32(AT + 0, IN_GMEM); img.u32(AT + 4, OUT_GMEM)

    rng = np.random.default_rng(0)
    data = rng.standard_normal(rows * width).astype(np.float32) * 0.5
    for i in range(rows * width):
        img.bf16(IN_GMEM + i * 2, float(data[i]))

    data_text = img.render_data_mem(include_zeros=True)
    final = render_testfile(instr_text, data_text)
    flops = total  # 1 comparison per element
    return final, flops


def build_softmax_bench(length):
    asm = make_softmax_asm(length)
    instrs = assemble_file(asm)
    instr_text = emit_test_format(instrs)

    img = DRAMWriter()
    AT = 60
    IN_GMEM = 0x1000
    width = min(length, 32)
    rows = math.ceil(length / 32)
    img.u32(AT + 0, IN_GMEM); img.u32(AT + 4, 0)

    rng = np.random.default_rng(0)
    data = rng.standard_normal(rows * width).astype(np.float32)
    for i in range(rows * width):
        img.bf16(IN_GMEM + i * 2, float(data[i]))

    data_text = img.render_data_mem(include_zeros=True)
    final = render_testfile(instr_text, data_text)
    flops = 5 * length  # rmax + sub + exp + rsum + div
    return final, flops


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark(tag: str, kernel: str, params: str,
                  in_text: str, theo_flops: int) -> BenchResult:
    metrics = run_on_emulator(in_text, tag)
    return BenchResult(
        kernel=kernel, params=params, theoretical_flops=theo_flops,
        wall_time_ms=metrics.get("wall_time_ms", 0),
        cycles=int(metrics.get("cycles", 0)),
        packets=int(metrics.get("packets", 0)),
        instructions=int(metrics.get("instructions", 0)),
        branches=int(metrics.get("branches", 0)),
        mem_ops=int(metrics.get("mem_ops", 0)),
        gemm_ops=int(metrics.get("gemm_ops", 0)),
        sdma_ops=int(metrics.get("sdma_ops", 0)),
        flops_total=int(metrics.get("flops_total", 0)),
        flops_scalar=int(metrics.get("flops_scalar", 0)),
        flops_vector=int(metrics.get("flops_vector", 0)),
        flops_matmul=int(metrics.get("flops_matmul", 0)),
        bytes_loaded=int(metrics.get("bytes_loaded", 0)),
        arithmetic_intensity=metrics.get("arithmetic_intensity", 0),
    )


def _bench_alexnet_at_scale(scale: float, results: List[BenchResult]):
    """Run all AlexNet layers at a given channel scale."""
    tag_prefix = f"alexnet_s{scale}"
    layers = alexnet_layers(scale=scale)
    for lnum, lname, lspec in layers:
        if lname == "maxpool":
            continue
        print(f"  AlexNet s={scale} L{lnum} ({lname})...", end="", flush=True)
        layer_rng = np.random.default_rng(42 + lnum)
        w = None
        if lname == "conv":
            w = layer_rng.standard_normal(
                (lspec.R, lspec.S, lspec.C_in, lspec.C_out)
            ).astype(np.float32) * 0.1
        elif lname == "fc":
            w = layer_rng.standard_normal(
                (lspec.in_features, lspec.out_features)
            ).astype(np.float32) * 0.1

        try:
            instr_text, img, expected = build_layer(
                lnum, lname, lspec, layer_rng, weights=w)
        except Exception as e:
            print(f" SKIP ({e})")
            continue

        data_text = img.render_data_mem(include_zeros=True)
        final = render_testfile(instr_text, data_text)

        if lname == "conv":
            s = lspec
            Ho = (s.H + 2*s.pad - s.R) // s.stride + 1
            Wo = (s.W + 2*s.pad - s.S) // s.stride + 1
            theo = 2 * Ho * Wo * s.R * s.S * s.C_in * s.C_out
        elif lname == "fc":
            theo = 2 * lspec.in_features * lspec.out_features
        elif lname == "relu":
            theo = lspec.total_elements
        elif lname == "softmax":
            theo = 5 * lspec.length
        else:
            theo = 0

        r = run_benchmark(f"{tag_prefix}_L{lnum}", f"AlexNet-{lname}",
                         f"s{scale}-L{lnum}", final, theo)
        results.append(r)
        print(f" {r.cycles} cyc, {r.instructions} instr")


def run_all_benchmarks() -> List[BenchResult]:
    results = []

    # --- GEMM square scaling ---
    print("\n[1/7] GEMM square scaling")
    for N in [4, 8, 16, 32, 64]:
        print(f"  GEMM {N}x{N}x{N}...", end="", flush=True)
        text, flops = build_gemm(N, N, N)
        r = run_benchmark(f"gemm_{N}", "GEMM", f"{N}x{N}x{N}", text, flops)
        results.append(r)
        print(f" {r.cycles} cyc, {r.instructions} instr")

    # --- GEMM rectangular ---
    print("\n[2/7] GEMM rectangular")
    rect_configs = [
        (1, 32, 32, "1x32x32"),
        (4, 32, 64, "4x32x64"),
        (32, 4, 32, "32x4x32"),
        (64, 32, 8, "64x32x8"),
        (32, 64, 32, "32x64x32"),
        (16, 16, 64, "16x16x64"),
        (64, 64, 4, "64x64x4"),
    ]
    for M, N, K, label in rect_configs:
        print(f"  GEMM {label}...", end="", flush=True)
        text, flops = build_gemm(M, N, K)
        r = run_benchmark(f"gemm_rect_{label}", "GEMM-rect", label, text, flops)
        results.append(r)
        print(f" {r.cycles} cyc, {r.instructions} instr")

    # --- Conv scaling ---
    print("\n[3/7] Conv scaling")
    conv_configs = [
        (4, 4, 3, 4, 3, 3, 1, 0, "4x4x3->4"),
        (8, 8, 3, 4, 3, 3, 1, 0, "8x8x3->4"),
        (8, 8, 3, 8, 3, 3, 1, 1, "8x8x3->8"),
        (13, 13, 4, 8, 3, 3, 1, 1, "13x13x4->8"),
        (16, 16, 4, 4, 3, 3, 1, 1, "16x16x4->4"),
        (16, 16, 8, 8, 3, 3, 1, 1, "16x16x8->8"),
        (16, 16, 8, 16, 3, 3, 1, 1, "16x16x8->16"),
        (32, 32, 3, 4, 3, 3, 1, 1, "32x32x3->4"),
    ]
    for H, W, C, K, R, S, stride, pad, label in conv_configs:
        print(f"  Conv {label}...", end="", flush=True)
        text, flops = build_conv(H, W, C, K, R, S, stride, pad)
        r = run_benchmark(f"conv_{label}", "Conv", label, text, flops)
        results.append(r)
        print(f" {r.cycles} cyc, {r.instructions} instr")

    # --- Attention scaling ---
    print("\n[4/7] Attention scaling")
    for S in [4, 8, 16, 32]:
        d = S
        print(f"  Attention S={S},d={d}...", end="", flush=True)
        text, flops = build_attention(S, d)
        r = run_benchmark(f"attn_{S}", "Attention", f"S={S},d={d}", text, flops)
        results.append(r)
        print(f" {r.cycles} cyc, {r.instructions} instr")

    # --- ReLU scaling ---
    print("\n[5/7] Activation scaling")
    for total in [32, 64, 128, 256, 512, 1024]:
        print(f"  ReLU {total}...", end="", flush=True)
        text, flops = build_relu_bench(total, 32)
        r = run_benchmark(f"relu_{total}", "ReLU", f"N={total}", text, flops)
        results.append(r)
        print(f" {r.cycles} cyc, {r.instructions} instr")

    # --- Softmax scaling ---
    for length in [4, 8, 16, 32]:
        print(f"  Softmax {length}...", end="", flush=True)
        text, flops = build_softmax_bench(length)
        r = run_benchmark(f"softmax_{length}", "Softmax", f"N={length}", text, flops)
        results.append(r)
        print(f" {r.cycles} cyc, {r.instructions} instr")

    # --- AlexNet at multiple scales ---
    print("\n[6/7] AlexNet layer-by-layer (multi-scale)")
    for scale in [0.01, 0.02, 0.05, 0.1]:
        _bench_alexnet_at_scale(scale, results)

    # --- PyTorch pipeline end-to-end ---
    print("\n[7/7] PyTorch pipeline end-to-end")
    try:
        import torch
        _ag = os.path.join(os.path.dirname(__file__), "..", "atalla-graph")
        if _ag not in sys.path:
            sys.path.insert(0, _ag)
        from run_model import run_pipeline
        from model.basic import BasicModule
        from model.alexnet import AlexNetSmall

        torch.manual_seed(42); np.random.seed(42)
        print("  BasicModule...", end="", flush=True)
        model = BasicModule(dim=32, depth=2)
        res = run_pipeline(model, torch.randn(1, 32),
                           out_dir=f"{OUT_DIR}/pipeline_basic", verbose=False)
        results.append(BenchResult(
            kernel="Pipeline-Basic", params="dim=32,depth=2",
            theoretical_flops=2*32*32*2 + 32*2,
            wall_time_ms=res["elapsed_s"] * 1000,
            cycles=0, instructions=0,
        ))
        cos = res.get("cosine_sim", 0)
        print(f" {res['elapsed_s']:.2f}s, cos={cos:.4f}")

        for sc in [0.01, 0.02, 0.05]:
            torch.manual_seed(42); np.random.seed(42)
            print(f"  AlexNet s={sc}...", end="", flush=True)
            model = AlexNetSmall(scale=sc, num_classes=10)
            res = run_pipeline(model, torch.randn(1, 3, 32, 32),
                               out_dir=f"{OUT_DIR}/pipeline_alexnet_s{sc}",
                               verbose=False)
            results.append(BenchResult(
                kernel="Pipeline-AlexNet", params=f"scale={sc}",
                theoretical_flops=0,
                wall_time_ms=res["elapsed_s"] * 1000,
                cycles=0, instructions=0,
            ))
            cos = res.get("cosine_sim", 0)
            print(f" {res['elapsed_s']:.2f}s, cos={cos:.4f}")
    except Exception as e:
        print(f"  Pipeline benchmarks skipped: {e}")

    return results


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------
CSV_PATH = f"{OUT_DIR}/benchmark_results.csv"


def save_csv(results: List[BenchResult]):
    os.makedirs(OUT_DIR, exist_ok=True)
    fields = list(asdict(results[0]).keys())
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    print(f"\nSaved {len(results)} results to {CSV_PATH}")


def load_csv() -> List[BenchResult]:
    results = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in row:
                if k in ("kernel", "params"):
                    continue
                try:
                    row[k] = float(row[k])
                except ValueError:
                    pass
            results.append(BenchResult(**row))
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def generate_graphs(results: List[BenchResult]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import matplotlib.gridspec as gridspec

    GD = f"{OUT_DIR}/graphs"
    os.makedirs(GD, exist_ok=True)

    COLORS = {
        "GEMM": "#2196F3", "GEMM-rect": "#42A5F5", "Conv": "#4CAF50",
        "Attention": "#FF9800", "ReLU": "#E91E63", "Softmax": "#9C27B0",
        "AlexNet-conv": "#1565C0", "AlexNet-relu": "#C62828",
        "AlexNet-fc": "#2E7D32", "AlexNet-softmax": "#6A1B9A",
        "Pipeline-Basic": "#795548", "Pipeline-AlexNet": "#FF5722",
    }

    def color_for(kernel):
        return COLORS.get(kernel, "#607D8B")

    # Helper: only kernel-level results (skip pipeline entries with 0 cycles)
    kern = [r for r in results if int(r.cycles) > 0]

    # ======================================================================
    # 01. GEMM Square Scaling — cycles, instructions, IPC
    # ======================================================================
    gemm = [r for r in results if r.kernel == "GEMM"]
    if gemm:
        sizes = [int(r.params.split("x")[0]) for r in gemm]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, metric, label, alpha in [
            (axes[0], [int(r.cycles) for r in gemm], "Cycles", 0.85),
            (axes[1], [int(r.instructions) for r in gemm], "Instructions", 0.7),
            (axes[2], [int(r.instructions)/max(int(r.cycles),1) for r in gemm], "IPC", 0.85),
        ]:
            ax.bar(range(len(sizes)), metric, color=COLORS["GEMM"], alpha=alpha)
            ax.set_xticks(range(len(sizes)))
            ax.set_xticklabels([f"{s}x{s}" for s in sizes])
            ax.set_xlabel("Matrix Size (NxN)")
            ax.set_ylabel(label)
            ax.set_title(f"GEMM: {label} vs Size")
            if label != "IPC":
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.suptitle("Square GEMM Scaling", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{GD}/01_gemm_scaling.png", dpi=150)
        plt.close()

    # ======================================================================
    # 02. GEMM Throughput — FLOPS/cycle for square GEMMs
    # ======================================================================
    if gemm:
        fig, ax = plt.subplots(figsize=(8, 5))
        gflops = [int(r.theoretical_flops) / max(int(r.cycles), 1) for r in gemm]
        ax.plot(sizes, gflops, "o-", color=COLORS["GEMM"], linewidth=2.5, markersize=9)
        for i, (s, g) in enumerate(zip(sizes, gflops)):
            ax.annotate(f"{g:.2f}", (s, g), textcoords="offset points",
                       xytext=(0, 10), ha="center", fontsize=9)
        ax.set_xlabel("Matrix Size N")
        ax.set_ylabel("FLOPS / Cycle")
        ax.set_title("GEMM: Compute Throughput")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{GD}/02_gemm_throughput.png", dpi=150)
        plt.close()

    # ======================================================================
    # 03. Rectangular GEMM — cycles heatmap-style bar
    # ======================================================================
    rect = [r for r in results if r.kernel == "GEMM-rect"]
    if rect:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        labels_r = [r.params for r in rect]
        cyc_r = [int(r.cycles) for r in rect]
        eff_r = [int(r.theoretical_flops)/max(int(r.cycles),1) for r in rect]
        ax1.barh(range(len(rect)), cyc_r, color=COLORS["GEMM-rect"], alpha=0.8)
        ax1.set_yticks(range(len(rect)))
        ax1.set_yticklabels(labels_r, fontsize=9)
        ax1.set_xlabel("Cycles")
        ax1.set_title("Rectangular GEMM: Cycles")
        ax2.barh(range(len(rect)), eff_r, color="#1976D2", alpha=0.8)
        ax2.set_yticks(range(len(rect)))
        ax2.set_yticklabels(labels_r, fontsize=9)
        ax2.set_xlabel("FLOPS / Cycle")
        ax2.set_title("Rectangular GEMM: Efficiency")
        plt.suptitle("Rectangular GEMM (MxKxN)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{GD}/03_rect_gemm.png", dpi=150)
        plt.close()

    # ======================================================================
    # 04. Conv Scaling — cycles + flops/cycle dual axis
    # ======================================================================
    conv = [r for r in results if r.kernel == "Conv"]
    if conv:
        fig, ax1 = plt.subplots(figsize=(12, 5))
        labels_c = [r.params for r in conv]
        cyc_c = [int(r.cycles) for r in conv]
        eff_c = [int(r.theoretical_flops)/max(int(r.cycles),1) for r in conv]
        x = np.arange(len(conv))
        ax1.bar(x, cyc_c, 0.4, color=COLORS["Conv"], alpha=0.8, label="Cycles")
        ax1.set_ylabel("Cycles", color=COLORS["Conv"])
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels_c, rotation=25, ha="right", fontsize=8)
        ax2 = ax1.twinx()
        ax2.plot(x, eff_c, "D-", color="#E65100", linewidth=2, markersize=7, label="FLOPS/cyc")
        ax2.set_ylabel("FLOPS / Cycle", color="#E65100")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        ax1.set_title("Conv (Tiled im2col GEMM): Scaling")
        plt.tight_layout()
        plt.savefig(f"{GD}/04_conv_scaling.png", dpi=150)
        plt.close()

    # ======================================================================
    # 05. Attention Scaling — cycles, instructions, FLOPS/cyc
    # ======================================================================
    attn = [r for r in results if r.kernel == "Attention"]
    if attn:
        sizes_a = [int(r.params.split(",")[0].split("=")[1]) for r in attn]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].plot(sizes_a, [int(r.cycles) for r in attn], "o-",
                    color=COLORS["Attention"], linewidth=2, markersize=8)
        axes[0].set_title("Cycles"); axes[0].set_xlabel("Seq Length S"); axes[0].grid(True, alpha=0.3)
        axes[1].plot(sizes_a, [int(r.instructions) for r in attn], "s-",
                    color=COLORS["Attention"], linewidth=2, markersize=8)
        axes[1].set_title("Instructions"); axes[1].set_xlabel("Seq Length S"); axes[1].grid(True, alpha=0.3)
        axes[2].plot(sizes_a, [int(r.theoretical_flops)/max(int(r.cycles),1) for r in attn],
                    "^-", color=COLORS["Attention"], linewidth=2, markersize=8)
        axes[2].set_title("FLOPS/Cycle"); axes[2].set_xlabel("Seq Length S"); axes[2].grid(True, alpha=0.3)
        plt.suptitle("Self-Attention Scaling (S=d)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{GD}/05_attention_scaling.png", dpi=150)
        plt.close()

    # ======================================================================
    # 06. ReLU + Softmax Activation Scaling
    # ======================================================================
    relu = [r for r in results if r.kernel == "ReLU"]
    softmax = [r for r in results if r.kernel == "Softmax"]
    if relu or softmax:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        if relu:
            ns = [int(r.params.split("=")[1]) for r in relu]
            ax1.plot(ns, [int(r.cycles) for r in relu], "o-", color=COLORS["ReLU"],
                    linewidth=2, markersize=8, label="Cycles")
            ax1t = ax1.twinx()
            ax1t.plot(ns, [int(r.cycles)/max(int(r.theoretical_flops),1) for r in relu],
                     "s--", color="#880E4F", linewidth=1.5, markersize=6, label="Cyc/elem")
            ax1t.set_ylabel("Cycles / Element", color="#880E4F")
            ax1.set_xlabel("Elements"); ax1.set_ylabel("Cycles"); ax1.set_title("ReLU Scaling")
            ax1.grid(True, alpha=0.3)
        if softmax:
            ns_s = [int(r.params.split("=")[1]) for r in softmax]
            ax2.plot(ns_s, [int(r.cycles) for r in softmax], "o-", color=COLORS["Softmax"],
                    linewidth=2, markersize=8, label="Cycles")
            ax2t = ax2.twinx()
            ax2t.plot(ns_s, [int(r.cycles)/max(int(r.theoretical_flops),1) for r in softmax],
                     "s--", color="#4A148C", linewidth=1.5, markersize=6, label="Cyc/FLOP")
            ax2t.set_ylabel("Cycles / FLOP", color="#4A148C")
            ax2.set_xlabel("Elements"); ax2.set_ylabel("Cycles"); ax2.set_title("Softmax Scaling")
            ax2.grid(True, alpha=0.3)
        plt.suptitle("Activation Kernel Scaling", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{GD}/06_activation_scaling.png", dpi=150)
        plt.close()

    # ======================================================================
    # 07. Cross-Kernel Efficiency Comparison
    # ======================================================================
    knames = ["GEMM", "GEMM-rect", "Conv", "Attention", "ReLU", "Softmax"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for kn in knames:
        sub = [r for r in kern if r.kernel == kn]
        if not sub: continue
        cpf = [max(int(r.cycles),1) / max(int(r.theoretical_flops),1) for r in sub]
        ax1.plot(range(len(sub)), cpf, "o-", label=kn, color=color_for(kn),
                linewidth=2, markersize=5)
    ax1.set_ylabel("Cycles / FLOP"); ax1.set_xlabel("Config index")
    ax1.set_title("Cycles per Theoretical FLOP")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    for kn in knames:
        sub = [r for r in kern if r.kernel == kn]
        if not sub: continue
        ipc = [int(r.instructions)/max(int(r.cycles),1) for r in sub]
        ax2.plot(range(len(sub)), ipc, "o-", label=kn, color=color_for(kn),
                linewidth=2, markersize=5)
    ax2.set_ylabel("IPC"); ax2.set_xlabel("Config index")
    ax2.set_title("Instructions per Cycle")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    plt.suptitle("Cross-Kernel Efficiency", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{GD}/07_efficiency_comparison.png", dpi=150)
    plt.close()

    # ======================================================================
    # 08. Instruction Mix — stacked bar (subset of kernels)
    # ======================================================================
    mix_results = [r for r in kern if r.kernel in ("GEMM", "Conv", "Attention", "ReLU", "Softmax")]
    if mix_results:
        step = max(1, len(mix_results) // 25)
        mix_sub = mix_results[::step]
        fig, ax = plt.subplots(figsize=(14, 6))
        labels_m = [f"{r.kernel}\n{r.params}" for r in mix_sub]
        gemm_f, mem_f, branch_f, scalar_f = [], [], [], []
        for r in mix_sub:
            ti = max(int(r.instructions), 1)
            g = int(r.gemm_ops); m = int(r.sdma_ops) + int(r.mem_ops)
            b = int(r.branches); s = ti - g - m - b
            gemm_f.append(g/ti*100); mem_f.append(m/ti*100)
            branch_f.append(b/ti*100); scalar_f.append(max(0, s/ti*100))
        x = np.arange(len(mix_sub))
        bottom = np.zeros(len(x))
        for data, lbl, col in [(gemm_f,"GEMM ops","#1565C0"), (mem_f,"Memory","#4CAF50"),
                                (branch_f,"Branch","#FF9800"), (scalar_f,"Scalar/Other","#9E9E9E")]:
            ax.bar(x, data, 0.65, bottom=bottom, label=lbl, color=col)
            bottom += np.array(data)
        ax.set_xticks(x); ax.set_xticklabels(labels_m, rotation=50, ha="right", fontsize=7)
        ax.set_ylabel("% of Instructions"); ax.set_ylim(0, 115)
        ax.set_title("Instruction Mix Breakdown"); ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{GD}/08_instruction_mix.png", dpi=150)
        plt.close()

    # ======================================================================
    # 09. Instruction Mix — normalized per kernel type (grouped)
    # ======================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    for kn in ["GEMM", "Conv", "Attention", "ReLU", "Softmax"]:
        sub = [r for r in kern if r.kernel == kn]
        if not sub: continue
        ti = sum(int(r.instructions) for r in sub)
        if ti == 0: continue
        g = sum(int(r.gemm_ops) for r in sub) / ti * 100
        m = sum(int(r.sdma_ops)+int(r.mem_ops) for r in sub) / ti * 100
        b = sum(int(r.branches) for r in sub) / ti * 100
        s = 100 - g - m - b
        ax.barh(kn, g, color="#1565C0", label="GEMM" if kn == "GEMM" else "")
        ax.barh(kn, m, left=g, color="#4CAF50", label="Memory" if kn == "GEMM" else "")
        ax.barh(kn, b, left=g+m, color="#FF9800", label="Branch" if kn == "GEMM" else "")
        ax.barh(kn, max(0,s), left=g+m+b, color="#9E9E9E", label="Scalar" if kn == "GEMM" else "")
    ax.set_xlabel("% of Total Instructions"); ax.set_xlim(0, 105)
    ax.set_title("Average Instruction Mix by Kernel Type")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{GD}/09_instruction_mix_by_type.png", dpi=150)
    plt.close()

    # ======================================================================
    # 10. Arithmetic Intensity — all kernels, horizontal bar
    # ======================================================================
    ai_data = [(f"{r.kernel} {r.params}", float(r.arithmetic_intensity), r.kernel)
               for r in kern if float(r.arithmetic_intensity) > 0]
    if ai_data:
        if len(ai_data) > 30:
            step = max(1, len(ai_data) // 30)
            ai_data = ai_data[::step]
        fig, ax = plt.subplots(figsize=(10, max(5, len(ai_data)*0.3)))
        labels_ai, vals_ai, ktypes = zip(*ai_data)
        colors_ai = [color_for(k) for k in ktypes]
        ax.barh(range(len(vals_ai)), vals_ai, color=colors_ai, alpha=0.8)
        ax.set_yticks(range(len(labels_ai)))
        ax.set_yticklabels(labels_ai, fontsize=7)
        ax.set_xlabel("Arithmetic Intensity (FLOPS / Byte)")
        ax.set_title("Arithmetic Intensity")
        ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{GD}/10_arithmetic_intensity.png", dpi=150)
        plt.close()

    # ======================================================================
    # 11. Roofline — all kernel types
    # ======================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    for kn in knames:
        sub = [r for r in kern if r.kernel == kn and float(r.arithmetic_intensity) > 0]
        if not sub: continue
        ai = [float(r.arithmetic_intensity) for r in sub]
        perf = [int(r.theoretical_flops)/max(int(r.cycles),1) for r in sub]
        ax.scatter(ai, perf, c=color_for(kn), s=60, label=kn, zorder=5, alpha=0.8)
    ax.set_xlabel("Arithmetic Intensity (FLOPS/Byte)")
    ax.set_ylabel("Performance (FLOPS/Cycle)")
    ax.set_title("Roofline Plot — All Kernels")
    ax.set_xscale("log"); ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{GD}/11_roofline.png", dpi=150)
    plt.close()

    # ======================================================================
    # 12. AlexNet per-scale breakdown — one subplot per scale
    # ======================================================================
    alexnet_all = [r for r in kern if r.kernel.startswith("AlexNet")]
    scales_found = sorted(set(r.params.split("-")[0] for r in alexnet_all if "-" in r.params))
    if scales_found:
        n_scales = len(scales_found)
        fig, axes = plt.subplots(1, n_scales, figsize=(6*n_scales, 5), squeeze=False)
        for idx, sc_tag in enumerate(scales_found):
            ax = axes[0][idx]
            sub = [r for r in alexnet_all if r.params.startswith(sc_tag)]
            layer_labels = [r.params.split("-")[1] for r in sub]
            layer_cyc = [int(r.cycles) for r in sub]
            layer_col = [color_for(r.kernel) for r in sub]
            ax.bar(range(len(sub)), layer_cyc, color=layer_col, alpha=0.85)
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels(layer_labels, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Cycles"); ax.set_title(f"AlexNet ({sc_tag})")
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.suptitle("AlexNet Layer-by-Layer Breakdown", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{GD}/12_alexnet_breakdown.png", dpi=150)
        plt.close()

    # ======================================================================
    # 13. AlexNet cycle distribution pie — per scale
    # ======================================================================
    if scales_found:
        fig, axes = plt.subplots(1, len(scales_found), figsize=(5*len(scales_found), 5), squeeze=False)
        for idx, sc_tag in enumerate(scales_found):
            ax = axes[0][idx]
            sub = [r for r in alexnet_all if r.params.startswith(sc_tag)]
            type_cyc = {}
            for r in sub:
                t = r.kernel.split("-")[1]
                type_cyc[t] = type_cyc.get(t, 0) + int(r.cycles)
            if type_cyc:
                ax.pie(list(type_cyc.values()), labels=list(type_cyc.keys()),
                      colors=[color_for(f"AlexNet-{t}") for t in type_cyc],
                      autopct="%1.1f%%", startangle=90)
            ax.set_title(f"{sc_tag}")
        plt.suptitle("AlexNet: Cycle Distribution by Op Type", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{GD}/13_alexnet_pie.png", dpi=150)
        plt.close()

    # ======================================================================
    # 14. AlexNet scale comparison — total cycles across scales
    # ======================================================================
    if len(scales_found) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        total_cyc = []
        total_instr = []
        for sc_tag in scales_found:
            sub = [r for r in alexnet_all if r.params.startswith(sc_tag)]
            total_cyc.append(sum(int(r.cycles) for r in sub))
            total_instr.append(sum(int(r.instructions) for r in sub))
        ax1.bar(range(len(scales_found)), total_cyc, color="#1565C0", alpha=0.85)
        ax1.set_xticks(range(len(scales_found)))
        ax1.set_xticklabels(scales_found)
        ax1.set_ylabel("Total Cycles"); ax1.set_title("Total Cycles vs Scale")
        for i, v in enumerate(total_cyc):
            ax1.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
        ax2.bar(range(len(scales_found)), total_instr, color="#2E7D32", alpha=0.85)
        ax2.set_xticks(range(len(scales_found)))
        ax2.set_xticklabels(scales_found)
        ax2.set_ylabel("Total Instructions"); ax2.set_title("Total Instructions vs Scale")
        for i, v in enumerate(total_instr):
            ax2.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
        plt.suptitle("AlexNet: Scaling Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{GD}/14_alexnet_scale_comparison.png", dpi=150)
        plt.close()

    # ======================================================================
    # 15. Wall-Clock Time — all kernel types
    # ======================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    for kn in ["GEMM", "Conv", "Attention", "ReLU", "Softmax"]:
        sub = [r for r in kern if r.kernel == kn]
        if not sub: continue
        ax.plot(range(len(sub)), [float(r.wall_time_ms) for r in sub],
                "o-", label=kn, color=color_for(kn), linewidth=2, markersize=6)
    ax.set_ylabel("Wall-Clock Time (ms)"); ax.set_xlabel("Config (ascending size)")
    ax.set_title("Emulator Wall-Clock Time"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{GD}/15_wall_time.png", dpi=150)
    plt.close()

    # ======================================================================
    # 16. Cycle Heatmap — all benchmarks in a 2D grid
    # ======================================================================
    if kern:
        fig, ax = plt.subplots(figsize=(14, max(5, len(kern)*0.22)))
        sorted_k = sorted(kern, key=lambda r: int(r.cycles), reverse=True)
        if len(sorted_k) > 40:
            sorted_k = sorted_k[:40]
        labels_h = [f"{r.kernel} {r.params}" for r in sorted_k]
        vals_h = [int(r.cycles) for r in sorted_k]
        colors_h = [color_for(r.kernel) for r in sorted_k]
        ax.barh(range(len(sorted_k)), vals_h, color=colors_h, alpha=0.85)
        ax.set_yticks(range(len(sorted_k)))
        ax.set_yticklabels(labels_h, fontsize=7)
        ax.set_xlabel("Cycles")
        ax.set_title("All Benchmarks Ranked by Cycle Count (top 40)")
        plt.tight_layout()
        plt.savefig(f"{GD}/16_cycle_ranking.png", dpi=150)
        plt.close()

    # ======================================================================
    # 17. Packet Utilization — instructions per packet
    # ======================================================================
    pkt_data = [(f"{r.kernel} {r.params}", int(r.instructions)/max(int(r.packets),1), r.kernel)
                for r in kern if int(r.packets) > 0]
    if pkt_data:
        if len(pkt_data) > 25:
            step = max(1, len(pkt_data) // 25)
            pkt_data = pkt_data[::step]
        fig, ax = plt.subplots(figsize=(12, 5))
        labels_p, vals_p, ktypes_p = zip(*pkt_data)
        colors_p = [color_for(k) for k in ktypes_p]
        ax.bar(range(len(pkt_data)), vals_p, color=colors_p, alpha=0.85)
        ax.set_xticks(range(len(pkt_data)))
        ax.set_xticklabels(labels_p, rotation=50, ha="right", fontsize=7)
        ax.set_ylabel("Instructions / Packet")
        ax.set_title("VLIW Packet Utilization (higher = better slot usage)")
        ax.axhline(y=4.0, color="red", linestyle="--", alpha=0.4, label="Max (4 slots)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{GD}/17_packet_utilization.png", dpi=150)
        plt.close()

    # ======================================================================
    # 18. Memory Bandwidth — bytes loaded vs cycles
    # ======================================================================
    bw_data = [r for r in kern if int(r.bytes_loaded) > 0]
    if bw_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        for kn in knames:
            sub = [r for r in bw_data if r.kernel == kn]
            if not sub: continue
            cyc = [int(r.cycles) for r in sub]
            byt = [int(r.bytes_loaded) for r in sub]
            ax.scatter(cyc, byt, c=color_for(kn), s=50, label=kn, alpha=0.8)
        ax.set_xlabel("Cycles"); ax.set_ylabel("Bytes Loaded")
        ax.set_title("Memory Traffic vs Execution Time")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{GD}/18_memory_traffic.png", dpi=150)
        plt.close()

    # ======================================================================
    # 19. Pipeline E2E — wall time bar chart
    # ======================================================================
    pipe = [r for r in results if r.kernel.startswith("Pipeline")]
    if pipe:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels_pipe = [f"{r.kernel}\n{r.params}" for r in pipe]
        wt = [float(r.wall_time_ms) for r in pipe]
        colors_pipe = [color_for(r.kernel) for r in pipe]
        ax.bar(range(len(pipe)), wt, color=colors_pipe, alpha=0.85)
        ax.set_xticks(range(len(pipe)))
        ax.set_xticklabels(labels_pipe, fontsize=8)
        ax.set_ylabel("Wall-Clock Time (ms)")
        ax.set_title("End-to-End PyTorch Pipeline Runtime")
        for i, v in enumerate(wt):
            ax.text(i, v, f"{v:.0f}ms", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(f"{GD}/19_pipeline_e2e.png", dpi=150)
        plt.close()

    # ======================================================================
    # 20. Summary table (text)
    # ======================================================================
    with open(f"{GD}/summary.txt", "w") as f:
        hdr = (f"{'Kernel':<16} {'Params':<20} {'Cycles':>8} {'Instr':>8} {'Packets':>8} "
               f"{'GEMM':>6} {'SDMA':>6} {'MemOps':>6} {'Branch':>6} "
               f"{'AI':>8} {'TheoFLOP':>10} {'FLOP/cyc':>10} {'IPC':>6} {'Wall(ms)':>10}")
        f.write(hdr + "\n")
        f.write("-" * len(hdr) + "\n")
        for r in results:
            cyc = max(int(r.cycles), 1)
            flop_cyc = int(r.theoretical_flops) / cyc
            ipc = int(r.instructions) / cyc
            f.write(f"{r.kernel:<16} {r.params:<20} {int(r.cycles):>8} "
                    f"{int(r.instructions):>8} {int(r.packets):>8} "
                    f"{int(r.gemm_ops):>6} {int(r.sdma_ops):>6} "
                    f"{int(r.mem_ops):>6} {int(r.branches):>6} "
                    f"{float(r.arithmetic_intensity):>8.2f} "
                    f"{int(r.theoretical_flops):>10} {flop_cyc:>10.2f} "
                    f"{ipc:>6.2f} {float(r.wall_time_ms):>10.1f}\n")

    n_graphs = len([f for f in os.listdir(GD) if f.endswith(".png")])
    print(f"\nGenerated {n_graphs} graphs + summary in {GD}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-graphs", action="store_true")
    args = ap.parse_args()

    if args.only_graphs:
        results = load_csv()
    else:
        print("=" * 60)
        print("Atalla Benchmark Suite")
        print("=" * 60)
        t0 = time.time()
        results = run_all_benchmarks()
        save_csv(results)
        print(f"\nBenchmarks complete in {time.time() - t0:.1f}s")

    generate_graphs(results)


if __name__ == "__main__":
    main()
