#!/usr/bin/env python3
"""Comprehensive Atalla pipeline benchmark suite.

Runs full workloads through the PyTorch->Atalla pipeline at multiple scales,
collects per-layer and aggregate metrics from the emulator, and generates
dense, publication-quality graphs.

Usage:
    python benchmark_full.py                    # default
    python benchmark_full.py --out out/bench    # custom output dir
"""
from __future__ import annotations

import os, sys, time, math, struct, json, argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# Paths
ROOT = Path(__file__).resolve().parent
FUNC_SIM = ROOT / "functional_sim"
GRAPH = ROOT / "atalla-graph"
sys.path.insert(0, str(FUNC_SIM))
sys.path.insert(0, str(GRAPH))

import torch
import torch.nn as nn

from graph.fx_capture import capture, get_node_shape
from graph.remove_ops import remove_ops
from graph.tile_planner import plan_tiles
from codegen.asm_emitter import emit_node, render_in_file, LayerEmission
from codegen.dram_builder import extract_input_data

from build import assemble_file, emit_test_format, DRAMWriter, render_testfile
from build_alexnet_layer import (
    make_relu_asm, make_softmax_asm, make_tiled_gemm_asm, im2col, TILE,
)
from build_attention import make_attention_asm as _make_attention_asm_raw
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
    name: str
    category: str
    label: str
    # emulator metrics
    cycles: int = 0
    packets: int = 0
    instructions: int = 0
    branches: int = 0
    mem_ops: int = 0
    gemm_ops: int = 0
    sdma_ops: int = 0
    # derived
    flops_theoretical: float = 0.0
    wall_time_s: float = 0.0
    elements: int = 0
    # pipeline metrics (for FX pipeline benchmarks)
    nodes_emulated: int = 0
    nodes_numpy: int = 0
    cosine_sim: float = 0.0


# ---------------------------------------------------------------------------
# Emulator runner
# ---------------------------------------------------------------------------
def run_emu(in_text: str, out_dir: str, tag: str) -> Tuple[Memory, Dict[str, float]]:
    """Write .in file, run emulator, parse perf metrics."""
    os.makedirs(out_dir, exist_ok=True)
    in_path = f"{out_dir}/{tag}.in"
    Path(in_path).write_text(in_text)

    mem = Memory(in_path)
    sregs = ScalarRegisterFile()
    mregs = ScalarRegisterFile(num_regs=16)
    vregs = VectorRegisterFile()
    SP0 = Scratchpad(slots_per_bank=32)
    SP1 = Scratchpad(slots_per_bank=32)
    EU = ExecuteUnit()

    prefix = f"{out_dir}/{tag}"
    run_emulator(
        mem, sregs, mregs, vregs, SP0, SP1, EU, 0, 4,
        f"{prefix}_mem.out", f"{prefix}_sregs.out",
        f"{prefix}_vregs.out", f"{prefix}_mregs.out",
        f"{prefix}_sp0.out", f"{prefix}_sp1.out",
        f"{prefix}_perf.out",
    )

    metrics = {}
    perf_path = f"{prefix}_perf.out"
    if os.path.exists(perf_path):
        for line in open(perf_path):
            if ":" in line:
                k, v = line.strip().split(":", 1)
                try:
                    metrics[k.strip()] = float(v.strip())
                except ValueError:
                    pass
    return mem, metrics


def build_gemm(N: int, M: int = None, K: int = None) -> Tuple[str, float]:
    M = M or N
    K = K or N
    A_GMEM, B_GMEM = 0x1000, 0x1000 + M * K * 2 + 0x1000
    C_GMEM = B_GMEM + K * N * 2 + 0x1000
    Mt, Nt, Kt = math.ceil(M / TILE), math.ceil(N / TILE), math.ceil(K / TILE)

    rng = np.random.default_rng(42)
    A = rng.standard_normal((M, K)).astype(np.float32) * 0.5
    B = rng.standard_normal((K, N)).astype(np.float32) * 0.5

    asm = make_tiled_gemm_asm(M, N, K)
    instrs = assemble_file(asm)
    instr_text = emit_test_format(instrs)
    img = DRAMWriter()
    img.u32(60, A_GMEM); img.u32(64, B_GMEM); img.u32(68, C_GMEM)
    img.u32(72, M); img.u32(76, N); img.u32(80, K)
    img.u32(84, Mt); img.u32(88, Nt); img.u32(92, Kt); img.u32(96, TILE)
    for r in range(M):
        for c in range(K):
            img.bf16(A_GMEM + (r * K + c) * 2, float(A[r, c]))
    for r in range(K):
        for c in range(N):
            img.bf16(B_GMEM + (r * N + c) * 2, float(B[r, c]))
    for i in range(M * N):
        img.bf16(C_GMEM + i * 2, 0.0)

    data = img.render_data_mem(include_zeros=True)
    text = render_testfile(instr_text, data)
    flops = 2.0 * M * N * K
    return text, flops


def build_relu(total: int) -> Tuple[str, float]:
    width = min(total, 32)
    rows = math.ceil(total / width)
    IN = 0x1000
    OUT = IN + rows * width * 2 + 0x1000

    asm = make_relu_asm(total, width)
    instrs = assemble_file(asm)
    instr_text = emit_test_format(instrs)
    img = DRAMWriter()
    img.u32(60, IN); img.u32(64, OUT)
    rng = np.random.default_rng(42)
    data = rng.standard_normal(rows * width).astype(np.float32) * 0.5
    for i in range(rows * width):
        img.bf16(IN + i * 2, float(data[i]))
    text = render_testfile(instr_text, img.render_data_mem(include_zeros=True))
    return text, float(total)


def build_softmax(length: int) -> Tuple[str, float]:
    width = min(length, 32)
    rows = math.ceil(length / 32)
    IN = 0x1000

    asm = make_softmax_asm(length)
    instrs = assemble_file(asm)
    instr_text = emit_test_format(instrs)
    img = DRAMWriter()
    img.u32(60, IN); img.u32(64, 0)
    rng = np.random.default_rng(42)
    data = rng.standard_normal(rows * width).astype(np.float32)
    for i in range(rows * width):
        img.bf16(IN + i * 2, float(data[i]))
    text = render_testfile(instr_text, img.render_data_mem(include_zeros=True))
    return text, float(length) * 5  # exp + sub + div + rmax + rsum


def build_attention(S: int, d: int = 32) -> Tuple[str, float]:
    """Build a complete attention .in file for sequence length S, head dim d."""
    asm = _make_attention_asm_raw(S, d)
    instrs = assemble_file(asm)
    instr_text = emit_test_format(instrs)

    ADDR_TABLE = 60
    Q_GMEM, KT_GMEM, V_GMEM, OUT_GMEM = 0x1000, 0x2000, 0x3000, 0x4000
    inv_sqrt_d = 1.0 / math.sqrt(d)

    rng = np.random.default_rng(42)
    Q = rng.standard_normal((S, d)).astype(np.float32) * 0.5
    K = rng.standard_normal((S, d)).astype(np.float32) * 0.5
    V = rng.standard_normal((S, d)).astype(np.float32) * 0.5
    KT = K.T

    img = DRAMWriter()
    img.u32(ADDR_TABLE + 0, Q_GMEM); img.u32(ADDR_TABLE + 4, 0)
    img.u32(ADDR_TABLE + 8, KT_GMEM); img.u32(ADDR_TABLE + 12, 512)
    img.u32(ADDR_TABLE + 16, V_GMEM); img.u32(ADDR_TABLE + 20, 1024)
    img.u32(ADDR_TABLE + 24, OUT_GMEM); img.u32(ADDR_TABLE + 28, 0)
    img.u32(ADDR_TABLE + 32, 0)
    scale_bits = struct.unpack("<I", struct.pack("<f", inv_sqrt_d))[0]
    img.u32(ADDR_TABLE + 36, scale_bits)

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

    text = render_testfile(instr_text, img.render_data_mem(include_zeros=True))
    flops = 2.0 * S * d * S + S * S + 5.0 * S * S + 2.0 * S * S * d
    return text, flops


# ---------------------------------------------------------------------------
# FX pipeline benchmark (AlexNet at various scales)
# ---------------------------------------------------------------------------
def run_alexnet_pipeline(scale: float, out_dir: str) -> Tuple[List[BenchResult], Dict]:
    """Run AlexNet through the full FX pipeline, return per-layer results."""
    from model.alexnet import AlexNetSmall

    torch.manual_seed(42)
    np.random.seed(42)
    model = AlexNetSmall(scale=scale, num_classes=10)
    example_input = torch.randn(1, 3, 32, 32)

    gm = capture(model, example_input)
    gm = remove_ops(gm)
    gm = plan_tiles(gm)

    ref_activations = extract_input_data(gm, example_input.bfloat16())
    activation_cache: Dict[str, np.ndarray] = {}
    results: List[BenchResult] = []
    total_cycles = 0
    total_instrs = 0
    nodes_emulated = 0
    nodes_numpy = 0

    for node in gm.graph.nodes:
        atalla_op = node.meta.get("atalla_op")
        if node.op == "output":
            continue
        if node.op == "placeholder":
            activation_cache[node.name] = example_input.detach().float().cpu().numpy()
            continue
        if node.op == "get_attr":
            attr = gm
            for part in node.target.split("."):
                attr = getattr(attr, part)
            activation_cache[node.name] = attr.detach().float().cpu().numpy() if isinstance(attr, torch.Tensor) else np.array(attr)
            continue

        if atalla_op in ("flatten", "dropout", None):
            if node.args and isinstance(node.args[0], torch.fx.Node):
                prev = node.args[0]
                if prev.name in activation_cache:
                    data = activation_cache[prev.name]
                    out_shape = get_node_shape(node)
                    if out_shape:
                        try:
                            data = data.reshape(out_shape)
                        except ValueError:
                            data = data.flatten()
                    activation_cache[node.name] = data
                    continue
            if node.name in ref_activations:
                activation_cache[node.name] = ref_activations[node.name]
            continue

        emission = emit_node(node, gm, activation_cache)
        if emission is None:
            if node.name in ref_activations:
                activation_cache[node.name] = ref_activations[node.name]
            elif node.args and isinstance(node.args[0], torch.fx.Node) and node.args[0].name in activation_cache:
                activation_cache[node.name] = activation_cache[node.args[0].name]
            continue

        shape = get_node_shape(node)
        elems = 1
        for d in (shape or [1]):
            elems *= d

        if emission.skip_emulator:
            activation_cache[node.name] = emission.numpy_result
            nodes_numpy += 1
            r = BenchResult(
                name=f"alexnet_s{scale}_{node.name}",
                category="AlexNet", label=f"{atalla_op}({node.name})",
                elements=elems,
            )
            results.append(r)
            continue

        in_text = render_in_file(emission)
        t0 = time.time()
        mem, metrics = run_emu(in_text, out_dir, f"alexnet_s{scale}_{node.name}")
        wall = time.time() - t0
        nodes_emulated += 1

        # Read output back
        def bf16_to_f32(bits):
            return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]

        out_data = np.zeros(emission.output_elements, dtype=np.float32)
        for i in range(emission.output_elements):
            w = mem.read_data(emission.output_addr + i * 2)
            out_data[i] = bf16_to_f32(w & 0xFFFF)
        if emission.output_shape:
            try:
                out_data = out_data.reshape(emission.output_shape)
            except ValueError:
                pass
        activation_cache[node.name] = out_data

        cyc = int(metrics.get("cycles", 0))
        instr = int(metrics.get("instructions", 0))
        total_cycles += cyc
        total_instrs += instr

        tc = node.meta.get("tile_config")
        flops = 0
        if atalla_op == "conv" and tc:
            p = tc.params
            flops = 2.0 * p["M"] * p["N"] * p["K"]
        elif atalla_op == "linear" and tc:
            p = tc.params
            flops = 2.0 * p["M"] * p["N"] * p["K"]
        elif atalla_op == "relu":
            flops = float(elems)
        elif atalla_op == "softmax":
            flops = float(elems) * 5

        r = BenchResult(
            name=f"alexnet_s{scale}_{node.name}",
            category="AlexNet", label=f"{atalla_op}({node.name})",
            cycles=cyc,
            packets=int(metrics.get("packets", 0)),
            instructions=instr,
            branches=int(metrics.get("branches", 0)),
            mem_ops=int(metrics.get("mem_ops", 0)),
            gemm_ops=int(metrics.get("gemm_ops", 0)),
            sdma_ops=int(metrics.get("sdma_ops", 0)),
            flops_theoretical=flops,
            wall_time_s=wall,
            elements=elems,
            nodes_emulated=1,
        )
        results.append(r)

    # Validation
    output_node = [n for n in gm.graph.nodes if n.op == "output"][0]
    out_arg = output_node.args[0]
    emu_out = activation_cache.get(out_arg.name) if isinstance(out_arg, torch.fx.Node) else None
    ref_out = ref_activations.get("output")
    cos_sim = 0.0
    if emu_out is not None and ref_out is not None:
        ef, rf = emu_out.flatten(), ref_out.flatten()
        ml = min(len(ef), len(rf))
        ef, rf = ef[:ml], rf[:ml]
        dot = np.dot(ef, rf)
        ne, nr = np.linalg.norm(ef), np.linalg.norm(rf)
        cos_sim = float(dot / (ne * nr + 1e-12))

    summary = {
        "scale": scale,
        "total_cycles": total_cycles,
        "total_instructions": total_instrs,
        "nodes_emulated": nodes_emulated,
        "nodes_numpy": nodes_numpy,
        "cosine_sim": cos_sim,
    }
    return results, summary


# ---------------------------------------------------------------------------
# Standalone kernel benchmarks
# ---------------------------------------------------------------------------
def run_kernel_benchmarks(out_dir: str) -> List[BenchResult]:
    results = []
    os.makedirs(out_dir, exist_ok=True)

    # GEMM scaling
    for N in [4, 8, 16, 32]:
        text, flops = build_gemm(N)
        t0 = time.time()
        _, m = run_emu(text, out_dir, f"gemm_{N}")
        wall = time.time() - t0
        results.append(BenchResult(
            name=f"gemm_{N}", category="GEMM", label=f"{N}x{N}x{N}",
            cycles=int(m.get("cycles", 0)), packets=int(m.get("packets", 0)),
            instructions=int(m.get("instructions", 0)),
            branches=int(m.get("branches", 0)), mem_ops=int(m.get("mem_ops", 0)),
            gemm_ops=int(m.get("gemm_ops", 0)), sdma_ops=int(m.get("sdma_ops", 0)),
            flops_theoretical=flops, wall_time_s=wall, elements=N * N,
        ))

    # Rectangular GEMM
    for M, K, N in [(1, 32, 32), (1, 64, 32), (1, 128, 64), (32, 32, 64), (64, 32, 32)]:
        text, flops = build_gemm(N, M, K)
        t0 = time.time()
        _, m = run_emu(text, out_dir, f"gemm_{M}x{K}x{N}")
        wall = time.time() - t0
        results.append(BenchResult(
            name=f"gemm_{M}x{K}x{N}", category="GEMM-Rect", label=f"{M}x{K}x{N}",
            cycles=int(m.get("cycles", 0)), packets=int(m.get("packets", 0)),
            instructions=int(m.get("instructions", 0)),
            branches=int(m.get("branches", 0)), mem_ops=int(m.get("mem_ops", 0)),
            gemm_ops=int(m.get("gemm_ops", 0)), sdma_ops=int(m.get("sdma_ops", 0)),
            flops_theoretical=flops, wall_time_s=wall, elements=M * N,
        ))

    # ReLU scaling
    for total in [32, 128, 512, 1024, 2048]:
        text, flops = build_relu(total)
        t0 = time.time()
        _, m = run_emu(text, out_dir, f"relu_{total}")
        wall = time.time() - t0
        results.append(BenchResult(
            name=f"relu_{total}", category="ReLU", label=f"{total}",
            cycles=int(m.get("cycles", 0)), packets=int(m.get("packets", 0)),
            instructions=int(m.get("instructions", 0)),
            branches=int(m.get("branches", 0)), mem_ops=int(m.get("mem_ops", 0)),
            gemm_ops=int(m.get("gemm_ops", 0)), sdma_ops=int(m.get("sdma_ops", 0)),
            flops_theoretical=flops, wall_time_s=wall, elements=total,
        ))

    # Softmax scaling
    for length in [10, 16, 32]:
        text, flops = build_softmax(length)
        t0 = time.time()
        _, m = run_emu(text, out_dir, f"softmax_{length}")
        wall = time.time() - t0
        results.append(BenchResult(
            name=f"softmax_{length}", category="Softmax", label=f"{length}",
            cycles=int(m.get("cycles", 0)), packets=int(m.get("packets", 0)),
            instructions=int(m.get("instructions", 0)),
            branches=int(m.get("branches", 0)), mem_ops=int(m.get("mem_ops", 0)),
            gemm_ops=int(m.get("gemm_ops", 0)), sdma_ops=int(m.get("sdma_ops", 0)),
            flops_theoretical=flops, wall_time_s=wall, elements=length,
        ))

    # Attention scaling
    for N in [4, 8, 16, 32]:
        try:
            text, flops = build_attention(N, d=32)
            t0 = time.time()
            _, m = run_emu(text, out_dir, f"attention_{N}")
            wall = time.time() - t0
            results.append(BenchResult(
                name=f"attention_{N}", category="Attention", label=f"N={N},d=32",
                cycles=int(m.get("cycles", 0)), packets=int(m.get("packets", 0)),
                instructions=int(m.get("instructions", 0)),
                branches=int(m.get("branches", 0)), mem_ops=int(m.get("mem_ops", 0)),
                gemm_ops=int(m.get("gemm_ops", 0)), sdma_ops=int(m.get("sdma_ops", 0)),
                flops_theoretical=flops, wall_time_s=wall, elements=N * 32,
            ))
        except Exception as e:
            print(f"  [WARN] attention_{N} failed: {e}")

    return results


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------
def generate_graphs(all_results: List[BenchResult],
                    alexnet_summaries: List[Dict],
                    out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.ticker as ticker

    gdir = f"{out_dir}/graphs"
    os.makedirs(gdir, exist_ok=True)

    # Color palette
    CAT_COLORS = {
        "GEMM": "#2196F3", "GEMM-Rect": "#1565C0",
        "ReLU": "#4CAF50", "Softmax": "#FF9800",
        "Attention": "#9C27B0", "AlexNet": "#F44336",
    }
    def ccolor(cat):
        return CAT_COLORS.get(cat, "#607D8B")

    # --- 1. GEMM Scaling: Cycles & Instructions ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("GEMM Scaling Analysis", fontsize=14, fontweight="bold")
    gemm = [r for r in all_results if r.category == "GEMM"]
    if gemm:
        sizes = [r.label for r in gemm]
        ax1.bar(sizes, [r.cycles for r in gemm], color="#2196F3", alpha=0.85, edgecolor="white")
        ax1.set_xlabel("Matrix Size"); ax1.set_ylabel("Cycles"); ax1.set_title("Cycles vs Size")
        for i, r in enumerate(gemm):
            ax1.text(i, r.cycles, str(r.cycles), ha="center", va="bottom", fontsize=8)

        ax2.bar(sizes, [r.instructions for r in gemm], color="#1976D2", alpha=0.85, edgecolor="white")
        ax2_t = ax2.twinx()
        ax2_t.plot(sizes, [r.gemm_ops for r in gemm], "ro-", label="GEMM ops", markersize=6)
        ax2.set_xlabel("Matrix Size"); ax2.set_ylabel("Instructions"); ax2_t.set_ylabel("GEMM Ops")
        ax2.set_title("Instructions & GEMM Ops vs Size")
        ax2_t.legend(loc="upper left")
    plt.tight_layout(); plt.savefig(f"{gdir}/01_gemm_scaling.png", dpi=150); plt.close()

    # --- 2. GEMM Throughput & Efficiency ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("GEMM Throughput & Efficiency", fontsize=14, fontweight="bold")
    if gemm:
        flops_per_cycle = [r.flops_theoretical / max(r.cycles, 1) for r in gemm]
        sizes = [r.label for r in gemm]
        ax1.bar(sizes, flops_per_cycle, color="#00BCD4", alpha=0.85, edgecolor="white")
        ax1.set_xlabel("Matrix Size"); ax1.set_ylabel("FLOP/Cycle"); ax1.set_title("Throughput (FLOP/Cycle)")
        for i, v in enumerate(flops_per_cycle):
            ax1.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

        # Cycles breakdown: compute vs overhead
        compute_cyc = [r.gemm_ops * 1 for r in gemm]  # 1 cycle per gemm op (minimum)
        overhead_cyc = [r.cycles - c for r, c in zip(gemm, compute_cyc)]
        x = range(len(sizes))
        ax2.bar(x, compute_cyc, label="Compute (GEMM)", color="#4CAF50", alpha=0.85)
        ax2.bar(x, overhead_cyc, bottom=compute_cyc, label="Overhead (load/branch/etc)", color="#FF5722", alpha=0.85)
        ax2.set_xticks(x); ax2.set_xticklabels(sizes)
        ax2.set_xlabel("Matrix Size"); ax2.set_ylabel("Cycles"); ax2.set_title("Compute vs Overhead")
        ax2.legend()
    plt.tight_layout(); plt.savefig(f"{gdir}/02_gemm_throughput.png", dpi=150); plt.close()

    # --- 3. Rectangular GEMM Analysis ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Rectangular GEMM Analysis", fontsize=14, fontweight="bold")
    rect = [r for r in all_results if r.category == "GEMM-Rect"]
    if rect:
        labels = [r.label for r in rect]
        x = range(len(labels))
        ax1.bar(x, [r.cycles for r in rect], color="#1565C0", alpha=0.85, edgecolor="white")
        ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=30, ha="right")
        ax1.set_ylabel("Cycles"); ax1.set_title("Cycles per Shape")

        fpc = [r.flops_theoretical / max(r.cycles, 1) for r in rect]
        ax2.bar(x, fpc, color="#0097A7", alpha=0.85, edgecolor="white")
        ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=30, ha="right")
        ax2.set_ylabel("FLOP/Cycle"); ax2.set_title("Throughput per Shape")
        for i, v in enumerate(fpc):
            ax2.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout(); plt.savefig(f"{gdir}/03_rect_gemm.png", dpi=150); plt.close()

    # --- 4. ReLU Scaling ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ReLU Scaling Analysis", fontsize=14, fontweight="bold")
    relu = [r for r in all_results if r.category == "ReLU"]
    if relu:
        sizes = [int(r.label) for r in relu]
        ax1.plot(sizes, [r.cycles for r in relu], "o-", color="#4CAF50", linewidth=2, markersize=6)
        ax1.set_xlabel("Elements"); ax1.set_ylabel("Cycles"); ax1.set_title("Cycles vs Elements")
        ax1.set_xscale("log", base=2)

        cycles_per_elem = [r.cycles / max(r.elements, 1) for r in relu]
        ax2.plot(sizes, cycles_per_elem, "s-", color="#388E3C", linewidth=2, markersize=6)
        ax2.set_xlabel("Elements"); ax2.set_ylabel("Cycles/Element"); ax2.set_title("Amortization")
        ax2.set_xscale("log", base=2)
    plt.tight_layout(); plt.savefig(f"{gdir}/04_relu_scaling.png", dpi=150); plt.close()

    # --- 5. Softmax & Attention Scaling ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Softmax & Attention Scaling", fontsize=14, fontweight="bold")
    sm = [r for r in all_results if r.category == "Softmax"]
    attn = [r for r in all_results if r.category == "Attention"]
    if sm:
        ax1.bar([r.label for r in sm], [r.cycles for r in sm], color="#FF9800", alpha=0.85, edgecolor="white")
        ax1.set_xlabel("Length"); ax1.set_ylabel("Cycles"); ax1.set_title("Softmax Cycles")
        for i, r in enumerate(sm):
            ax1.text(i, r.cycles, str(r.cycles), ha="center", va="bottom", fontsize=8)
    if attn:
        sizes = [r.label for r in attn]
        ax2.bar(sizes, [r.cycles for r in attn], color="#9C27B0", alpha=0.85, edgecolor="white")
        ax2.set_xlabel("Config"); ax2.set_ylabel("Cycles"); ax2.set_title("Attention Cycles")
        for i, r in enumerate(attn):
            ax2.text(i, r.cycles, str(r.cycles), ha="center", va="bottom", fontsize=7)
    plt.tight_layout(); plt.savefig(f"{gdir}/05_softmax_attention.png", dpi=150); plt.close()

    # --- 6. Instruction Mix Comparison ---
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Instruction Mix Across Kernels", fontsize=14, fontweight="bold")
    kernel_results = [r for r in all_results if r.category != "AlexNet" and r.cycles > 0]
    if kernel_results:
        names = [r.name for r in kernel_results]
        x = np.arange(len(names))
        w = 0.2
        gemm_v = [r.gemm_ops for r in kernel_results]
        sdma_v = [r.sdma_ops for r in kernel_results]
        branch_v = [r.branches for r in kernel_results]
        other_v = [r.instructions - r.gemm_ops - r.sdma_ops - r.branches for r in kernel_results]
        ax.bar(x - 1.5*w, gemm_v, w, label="GEMM", color="#4CAF50", alpha=0.85)
        ax.bar(x - 0.5*w, sdma_v, w, label="SDMA", color="#2196F3", alpha=0.85)
        ax.bar(x + 0.5*w, branch_v, w, label="Branch", color="#FF9800", alpha=0.85)
        ax.bar(x + 1.5*w, other_v, w, label="Other", color="#9E9E9E", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Count"); ax.legend()
    plt.tight_layout(); plt.savefig(f"{gdir}/06_instruction_mix.png", dpi=150); plt.close()

    # --- 7. Stacked Instruction Mix (normalized) ---
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Instruction Mix (Normalized %)", fontsize=14, fontweight="bold")
    if kernel_results:
        names = [r.name for r in kernel_results]
        x = range(len(names))
        totals = [max(r.instructions, 1) for r in kernel_results]
        gemm_pct = [r.gemm_ops / t * 100 for r, t in zip(kernel_results, totals)]
        sdma_pct = [r.sdma_ops / t * 100 for r, t in zip(kernel_results, totals)]
        branch_pct = [r.branches / t * 100 for r, t in zip(kernel_results, totals)]
        other_pct = [100 - g - s - b for g, s, b in zip(gemm_pct, sdma_pct, branch_pct)]
        ax.bar(x, gemm_pct, label="GEMM", color="#4CAF50", alpha=0.85)
        ax.bar(x, sdma_pct, bottom=gemm_pct, label="SDMA", color="#2196F3", alpha=0.85)
        bot2 = [g + s for g, s in zip(gemm_pct, sdma_pct)]
        ax.bar(x, branch_pct, bottom=bot2, label="Branch", color="#FF9800", alpha=0.85)
        bot3 = [b2 + b for b2, b in zip(bot2, branch_pct)]
        ax.bar(x, other_pct, bottom=bot3, label="Other", color="#9E9E9E", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("%"); ax.set_ylim(0, 105); ax.legend(loc="upper right")
    plt.tight_layout(); plt.savefig(f"{gdir}/07_instruction_mix_normalized.png", dpi=150); plt.close()

    # --- 8. Arithmetic Intensity ---
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Arithmetic Intensity (FLOP/Byte)", fontsize=14, fontweight="bold")
    compute_results = [r for r in all_results if r.flops_theoretical > 0 and r.cycles > 0]
    if compute_results:
        names = [r.name for r in compute_results]
        # FLOP / (SDMA ops * 32 * 32 * 2 bytes)
        ai = []
        for r in compute_results:
            bytes_moved = max(r.sdma_ops, 1) * 32 * 32 * 2
            ai.append(r.flops_theoretical / bytes_moved)
        colors = [ccolor(r.category) for r in compute_results]
        x = range(len(names))
        ax.bar(x, ai, color=colors, alpha=0.85, edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("FLOP/Byte"); ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="AI=1.0")
        ax.legend()
    plt.tight_layout(); plt.savefig(f"{gdir}/08_arithmetic_intensity.png", dpi=150); plt.close()

    # --- 9. Cycles/FLOP (compute efficiency) ---
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Compute Efficiency (Cycles per FLOP)", fontsize=14, fontweight="bold")
    if compute_results:
        names = [r.name for r in compute_results]
        cpf = [r.cycles / max(r.flops_theoretical, 1) for r in compute_results]
        colors = [ccolor(r.category) for r in compute_results]
        x = range(len(names))
        ax.bar(x, cpf, color=colors, alpha=0.85, edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Cycles/FLOP (lower=better)")
    plt.tight_layout(); plt.savefig(f"{gdir}/09_cycles_per_flop.png", dpi=150); plt.close()

    # --- 10. AlexNet per-layer breakdown (multi-scale) ---
    for sinfo in alexnet_summaries:
        scale = sinfo["scale"]
        alex = [r for r in all_results if r.name.startswith(f"alexnet_s{scale}_") and r.cycles > 0]
        if not alex:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"AlexNet (scale={scale}) Layer Breakdown", fontsize=14, fontweight="bold")

        # Cycles
        ax = axes[0, 0]
        labels = [r.label for r in alex]
        ax.barh(labels, [r.cycles for r in alex], color="#F44336", alpha=0.85)
        ax.set_xlabel("Cycles"); ax.set_title("Cycles per Layer")

        # Instructions
        ax = axes[0, 1]
        ax.barh(labels, [r.instructions for r in alex], color="#E91E63", alpha=0.85)
        ax.set_xlabel("Instructions"); ax.set_title("Instructions per Layer")

        # GEMM vs SDMA
        ax = axes[1, 0]
        x = range(len(alex))
        ax.bar(x, [r.gemm_ops for r in alex], label="GEMM", color="#4CAF50", alpha=0.85)
        ax.bar(x, [r.sdma_ops for r in alex], bottom=[r.gemm_ops for r in alex],
               label="SDMA", color="#2196F3", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Ops"); ax.set_title("GEMM & SDMA Ops"); ax.legend()

        # Throughput (FLOP/cycle)
        ax = axes[1, 1]
        fpc = [r.flops_theoretical / max(r.cycles, 1) for r in alex]
        ax.bar(x, fpc, color="#FF9800", alpha=0.85, edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("FLOP/Cycle"); ax.set_title("Per-Layer Throughput")

        plt.tight_layout()
        plt.savefig(f"{gdir}/10_alexnet_s{scale}_breakdown.png", dpi=150)
        plt.close()

    # --- 11. AlexNet scale comparison ---
    if len(alexnet_summaries) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("AlexNet Scaling Across Channel Widths", fontsize=14, fontweight="bold")
        scales = [s["scale"] for s in alexnet_summaries]
        total_cyc = [s["total_cycles"] for s in alexnet_summaries]
        total_instr = [s["total_instructions"] for s in alexnet_summaries]
        cos_sims = [s["cosine_sim"] for s in alexnet_summaries]

        ax1.plot(scales, total_cyc, "o-", color="#F44336", linewidth=2, markersize=8, label="Cycles")
        ax1_t = ax1.twinx()
        ax1_t.plot(scales, total_instr, "s--", color="#1976D2", linewidth=2, markersize=8, label="Instructions")
        ax1.set_xlabel("Channel Scale"); ax1.set_ylabel("Total Cycles", color="#F44336")
        ax1_t.set_ylabel("Total Instructions", color="#1976D2")
        ax1.set_title("Total Compute vs Scale")

        ax2.plot(scales, cos_sims, "D-", color="#4CAF50", linewidth=2, markersize=8)
        ax2.axhline(y=0.95, color="red", linestyle="--", alpha=0.5, label="Threshold (0.95)")
        ax2.set_xlabel("Channel Scale"); ax2.set_ylabel("Cosine Similarity")
        ax2.set_title("Output Accuracy vs Scale"); ax2.legend()
        ax2.set_ylim(-0.5, 1.1)

        plt.tight_layout(); plt.savefig(f"{gdir}/11_alexnet_scale_comparison.png", dpi=150); plt.close()

    # --- 12. Roofline Model ---
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("Roofline Model (Atalla)", fontsize=14, fontweight="bold")
    peak_flops_per_cycle = 32.0  # 32x32 systolic -> 32 MACs/cycle * 2 = 64 FLOP/cycle
    peak_bw = 32 * 32 * 2  # bytes per SDMA cycle
    if compute_results:
        for r in compute_results:
            bytes_moved = max(r.sdma_ops, 1) * 32 * 32 * 2
            ai = r.flops_theoretical / bytes_moved
            perf = r.flops_theoretical / max(r.cycles, 1)
            ax.scatter(ai, perf, color=ccolor(r.category), s=60, zorder=5, alpha=0.85)
            ax.annotate(r.name, (ai, perf), fontsize=5, alpha=0.7, rotation=15)

        # Roofline lines
        x = np.logspace(-2, 3, 200)
        roof_bw = x * peak_bw / 1  # bandwidth-limited
        roof_compute = np.full_like(x, peak_flops_per_cycle)
        ax.plot(x, np.minimum(roof_bw, roof_compute), "k-", linewidth=2, alpha=0.5, label="Roofline")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)")
        ax.set_ylabel("Performance (FLOP/Cycle)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{gdir}/12_roofline.png", dpi=150); plt.close()

    # --- 13. Wall Clock Time ---
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Wall Clock Execution Time", fontsize=14, fontweight="bold")
    timed = [r for r in all_results if r.wall_time_s > 0]
    if timed:
        names = [r.name for r in timed]
        colors = [ccolor(r.category) for r in timed]
        ax.bar(range(len(timed)), [r.wall_time_s * 1000 for r in timed], color=colors, alpha=0.85, edgecolor="white")
        ax.set_xticks(range(len(timed))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("Time (ms)")
    plt.tight_layout(); plt.savefig(f"{gdir}/13_wall_time.png", dpi=150); plt.close()

    # --- 14. Heatmap: Kernel x Metric ---
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle("Kernel Performance Heatmap", fontsize=14, fontweight="bold")
    hm_results = [r for r in all_results if r.cycles > 0]
    if hm_results:
        metric_names = ["cycles", "instructions", "gemm_ops", "sdma_ops", "branches", "mem_ops"]
        data = np.array([[getattr(r, m) for m in metric_names] for r in hm_results], dtype=float)
        # Normalize each column to [0,1]
        for col in range(data.shape[1]):
            mx = data[:, col].max()
            if mx > 0:
                data[:, col] /= mx
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_xticks(range(len(metric_names))); ax.set_xticklabels(metric_names)
        ax.set_yticks(range(len(hm_results))); ax.set_yticklabels([r.name for r in hm_results], fontsize=6)
        plt.colorbar(im, ax=ax, label="Normalized Value")
    plt.tight_layout(); plt.savefig(f"{gdir}/14_heatmap.png", dpi=150); plt.close()

    # --- 15. Packet Utilization ---
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Packet Utilization (Instructions per Packet)", fontsize=14, fontweight="bold")
    packed = [r for r in all_results if r.packets > 0 and r.instructions > 0]
    if packed:
        names = [r.name for r in packed]
        ipp = [r.instructions / r.packets for r in packed]
        colors = [ccolor(r.category) for r in packed]
        ax.bar(range(len(packed)), ipp, color=colors, alpha=0.85, edgecolor="white")
        ax.set_xticks(range(len(packed))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("Instructions / Packet"); ax.axhline(y=4.0, color="red", linestyle="--", alpha=0.5, label="Max (4)")
        ax.legend(); ax.set_ylim(0, 4.5)
    plt.tight_layout(); plt.savefig(f"{gdir}/15_packet_utilization.png", dpi=150); plt.close()

    print(f"  Generated 15 graphs in {gdir}/")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def write_summary(all_results: List[BenchResult], alexnet_summaries: List[Dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/summary.txt", "w") as f:
        f.write(f"{'Name':<35s} {'Cat':<12s} {'Cycles':>8s} {'Instrs':>8s} {'GEMM':>6s} "
                f"{'SDMA':>6s} {'Branch':>7s} {'MemOps':>7s} {'FLOP':>12s} {'FLOP/Cyc':>9s} "
                f"{'Wall(ms)':>9s}\n")
        f.write("-" * 130 + "\n")
        for r in all_results:
            fpc = r.flops_theoretical / max(r.cycles, 1) if r.cycles > 0 else 0
            f.write(f"{r.name:<35s} {r.category:<12s} {r.cycles:>8d} {r.instructions:>8d} "
                    f"{r.gemm_ops:>6d} {r.sdma_ops:>6d} {r.branches:>7d} {r.mem_ops:>7d} "
                    f"{r.flops_theoretical:>12.0f} {fpc:>9.2f} {r.wall_time_s*1000:>9.1f}\n")

        f.write("\n\n=== AlexNet Pipeline Summaries ===\n")
        for s in alexnet_summaries:
            f.write(f"\nScale={s['scale']}:\n")
            for k, v in s.items():
                f.write(f"  {k}: {v}\n")

    # Also write JSON for programmatic access
    data = {
        "results": [asdict(r) for r in all_results],
        "alexnet_summaries": alexnet_summaries,
    }
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"  Summary written to {out_dir}/summary.txt")
    print(f"  JSON data written to {out_dir}/results.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="out/bench")
    args = parser.parse_args()

    out = args.out
    os.makedirs(out, exist_ok=True)

    print("=" * 60)
    print("  Atalla Full Pipeline Benchmark Suite")
    print("=" * 60)

    # 1. Standalone kernel benchmarks
    print("\n[1/3] Running standalone kernel benchmarks...")
    t0 = time.time()
    kernel_results = run_kernel_benchmarks(f"{out}/kernels")
    print(f"  Completed {len(kernel_results)} kernel benchmarks in {time.time()-t0:.1f}s")

    # 2. AlexNet pipeline at multiple scales
    all_results = list(kernel_results)
    alexnet_summaries = []

    print("\n[2/3] Running AlexNet pipeline benchmarks...")
    for scale in [0.01, 0.02, 0.05, 0.1]:
        print(f"  AlexNet scale={scale}...", end=" ", flush=True)
        t0 = time.time()
        try:
            alex_results, summary = run_alexnet_pipeline(scale, f"{out}/alexnet_s{scale}")
            all_results.extend(alex_results)
            alexnet_summaries.append(summary)
            print(f"done ({time.time()-t0:.1f}s, {summary['total_cycles']} cycles, "
                  f"cos_sim={summary['cosine_sim']:.3f})")
        except Exception as e:
            print(f"FAILED: {e}")

    # 3. Generate graphs + summary
    print(f"\n[3/3] Generating graphs and summary...")
    write_summary(all_results, alexnet_summaries, out)
    generate_graphs(all_results, alexnet_summaries, out)

    print(f"\n{'=' * 60}")
    print(f"  Benchmark complete: {len(all_results)} total benchmarks")
    print(f"  Results in: {out}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
