"""Unified pipeline: PyTorch model -> Atalla schedule + emulator validation.

Uses Vihaan's graph front-end (lower_linear_modules, allocate_memory) with
our kernel back-end (c_emitter, build_compiler, functional_sim).

Modes:
    schedule  — emit graph_schedule.c (Vihaan's generate_schedule)
    validate  — per-node compile + emulate + compare vs PyTorch
    both      — do both

Usage:
    python run_graph.py --model basic --mode both
    python run_graph.py --model alexnet_small --mode validate --scale 0.01
"""
from __future__ import annotations

import os
import sys
import struct
import time
import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

_FUNC_SIM = Path(__file__).resolve().parent.parent / "functional_sim"
if str(_FUNC_SIM) not in sys.path:
    sys.path.insert(0, str(_FUNC_SIM))

from src.functional_sim import run as run_emulator
from src.misc.memory import Memory
from src.components.scalar_register_file import ScalarRegisterFile
from src.components.vector_register_file import VectorRegisterFile
from src.components.execute import ExecuteUnit
from src.components.scpad import Scratchpad

from graph.lower_modules import lower_linear_modules
from graph.memoryallocator import allocate_memory
from graph.fx_capture import normalize_ops, get_node_shape
from graph.tile_planner import plan_tiles
from scripts.generate_schedule import emit as emit_schedule
from codegen.c_emitter import (
    emit_node, render_in_file, compile_and_assemble, LayerEmission, _to_bf16_array,
)
from codegen.dram_builder import extract_input_data


def bf16_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def _run_emulator(in_file: str, out_dir: str, tag: str):
    mem = Memory(in_file)
    sregs = ScalarRegisterFile()
    mregs = ScalarRegisterFile(num_regs=16)
    vregs = VectorRegisterFile()
    SP0 = Scratchpad(slots_per_bank=32)
    SP1 = Scratchpad(slots_per_bank=32)
    EU = ExecuteUnit()

    max_data_addr = max(mem.data_mem.keys()) if mem.data_mem else 0
    stack_base = ((max_data_addr + 0x1000) & ~0xFFF) + 0x1000
    sregs.write(2, stack_base)
    sregs.write(33, stack_base)
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{out_dir}/{tag}"

    run_emulator(
        mem, sregs, mregs, vregs, SP0, SP1, EU, 0, 4,
        f"{prefix}_mem.out",
        f"{prefix}_sregs.out",
        f"{prefix}_vregs.out",
        f"{prefix}_mregs.out",
        f"{prefix}_sp0.out",
        f"{prefix}_sp1.out",
        f"{prefix}_perf.out",
    )
    return mem, EU


def _read_bf16(mem: Memory, addr: int, count: int) -> np.ndarray:
    result = np.zeros(count, dtype=np.float32)
    for i in range(count):
        word = mem.read_data(addr + i * 2)
        result[i] = bf16_to_f32(word & 0xFFFF)
    return result


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    af, bf = a.flatten(), b.flatten()
    n = min(len(af), len(bf))
    af, bf = af[:n], bf[:n]
    d = np.dot(af, bf)
    return float(d / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))


# ── Front-end: shared across modes ──────────────────────────────────────────

def build_graph(model: nn.Module, example_input: torch.Tensor,
                verbose: bool = True):
    """Trace → lower → shape prop → normalize → plan tiles."""
    model = model.bfloat16().eval()
    example_bf16 = example_input.bfloat16()

    if verbose:
        print("FX trace + lower modules...")
    gm = symbolic_trace(model)
    gm = lower_linear_modules(gm)

    if verbose:
        print("Shape propagation...")
    ShapeProp(gm).propagate(example_bf16)

    if verbose:
        print("Normalize ops + plan tiles...")
    gm = normalize_ops(gm)
    gm = plan_tiles(gm)
    return gm


# ── Schedule mode ────────────────────────────────────────────────────────────

def run_schedule(gm, example_input: torch.Tensor, out_dir: str,
                 verbose: bool = True) -> str:
    """Emit graph_schedule.c using Vihaan's generate_schedule."""
    os.makedirs(out_dir, exist_ok=True)
    placeholder_data = {"x": example_input.bfloat16().clone()}
    gm_alloc = allocate_memory(gm, f"{out_dir}/dram.bin", placeholder_data)

    c_code = emit_schedule(gm_alloc)
    out_path = f"{out_dir}/graph_schedule.c"
    Path(out_path).write_text(c_code)

    kernel_calls = [l.strip() for l in c_code.splitlines() if "_kernel(" in l]
    if verbose:
        print(f"\nSchedule: {len(c_code.splitlines())} lines, "
              f"{len(kernel_calls)} kernel calls -> {out_path}")
        for k in kernel_calls[:8]:
            print(f"  {k[:100]}")
        if len(kernel_calls) > 8:
            print(f"  ... ({len(kernel_calls) - 8} more)")
    return c_code


# ── Validate mode ────────────────────────────────────────────────────────────

def run_validate(gm, model: nn.Module, example_input: torch.Tensor,
                 out_dir: str, verbose: bool = True) -> Dict:
    """Per-node compile → emulate → compare vs PyTorch golden."""
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)

    ref_activations = extract_input_data(gm, example_input.bfloat16())
    activation_cache: Dict[str, np.ndarray] = {}

    stats = {"total": 0, "emulated": 0, "numpy": 0, "passthrough": 0}
    kernel_metrics = []

    if verbose:
        print("\n--- Graph ---")
        for node in gm.graph.nodes:
            op = node.meta.get("atalla_op", "?")
            shape = get_node_shape(node)
            kt = node.meta.get("kernel_type", "-")
            print(f"  {node.name:30s}  {str(op):15s}  {str(kt):12s}  {shape}")
        print()

    for node in gm.graph.nodes:
        atalla_op = node.meta.get("atalla_op")

        if node.op == "output":
            continue

        stats["total"] += 1

        if node.op == "placeholder":
            activation_cache[node.name] = example_input.detach().float().cpu().numpy()
            if verbose:
                print(f"  [{node.name}] placeholder")
            continue

        if node.op == "get_attr":
            attr = gm
            for part in node.target.split("."):
                attr = getattr(attr, part)
            activation_cache[node.name] = (
                attr.detach().float().cpu().numpy()
                if isinstance(attr, torch.Tensor) else np.array(attr)
            )
            if verbose:
                print(f"  [{node.name}] get_attr")
            continue

        # Passthrough: flatten, dropout, transpose, unknown
        if atalla_op in ("flatten", "dropout", "transpose", None):
            if node.args and isinstance(node.args[0], torch.fx.Node):
                prev = node.args[0]
                if prev.name in activation_cache:
                    data = activation_cache[prev.name]
                    out_shape = get_node_shape(node)
                    if atalla_op == "transpose" and len(node.args) >= 3:
                        dims = [a if not isinstance(a, torch.fx.Node) else 0
                                for a in node.args[1:]]
                        try:
                            data = np.transpose(data, _resolve_transpose(data.ndim, dims))
                        except Exception:
                            data = data.T
                    elif out_shape:
                        try:
                            data = data.reshape(out_shape)
                        except ValueError:
                            data = data.flatten()
                    activation_cache[node.name] = data
                    stats["passthrough"] += 1
                    if verbose:
                        print(f"  [{node.name}] {atalla_op or 'passthrough'} -> {data.shape}")
                    continue

            if node.name in ref_activations:
                activation_cache[node.name] = ref_activations[node.name]
                stats["passthrough"] += 1
                if verbose:
                    print(f"  [{node.name}] passthrough (ref)")
                continue

        emission = emit_node(node, gm, activation_cache)

        if emission is None:
            if node.name in ref_activations:
                activation_cache[node.name] = ref_activations[node.name]
            elif node.args and isinstance(node.args[0], torch.fx.Node):
                prev_name = node.args[0].name
                if prev_name in activation_cache:
                    activation_cache[node.name] = activation_cache[prev_name]
            stats["passthrough"] += 1
            if verbose:
                print(f"  [{node.name}] {atalla_op} -> passthrough")
            continue

        if emission.skip_emulator:
            activation_cache[node.name] = emission.numpy_result
            stats["numpy"] += 1
            r = emission.numpy_result
            km = {"name": node.name, "op": atalla_op, "backend": "numpy",
                  "shape": list(r.shape), "elems": int(r.size)}
            ref = ref_activations.get(node.name)
            if ref is not None:
                km["cos_sim"] = _cos_sim(ref, r)
            kernel_metrics.append(km)
            if verbose:
                print(f"  [{node.name}] {atalla_op} -> NumPy {r.shape}")
            continue

        # Compile + emulate
        compile_and_assemble(emission, out_dir, node.name)
        in_file = f"{out_dir}/{node.name}.in"
        Path(in_file).write_text(render_in_file(emission))

        if verbose:
            print(f"  [{node.name}] {atalla_op} -> emulator "
                  f"(0x{emission.output_addr:X}, {emission.output_elements} elems)...",
                  end=" ", flush=True)

        mem, eu = _run_emulator(in_file, out_dir, node.name)
        result = _read_bf16(mem, emission.output_addr, emission.output_elements)

        if emission.output_shape:
            try:
                result = result.reshape(emission.output_shape)
            except ValueError:
                pass

        if emission.maxpool_post is not None:
            pp = emission.maxpool_post
            raw = result.reshape(pp["C"], pp["H_out"], pp["W"])
            out = np.empty((pp["C"], pp["H_out"], pp["W_out"]), dtype=np.float32)
            for c in range(pp["C"]):
                for oh in range(pp["H_out"]):
                    for ow in range(pp["W_out"]):
                        base = ow * pp["stride"]
                        out[c, oh, ow] = max(
                            float(raw[c, oh, base + p])
                            for p in range(pp["pool"])
                            if base + p < pp["W"]
                        )
            result = out.reshape(pp["final_shape"])

        activation_cache[node.name] = result
        stats["emulated"] += 1

        pm = eu.perf_metrics.metrics
        ref = ref_activations.get(node.name)
        km = {"name": node.name, "op": atalla_op, "backend": "emulator",
              "shape": list(result.shape), "elems": int(result.size),
              "packets": int(pm.get("packets", 0)),
              "instructions": int(pm.get("instructions", 0))}
        if ref is not None:
            km["cos_sim"] = _cos_sim(ref, result)
            km["max_diff"] = float(np.max(np.abs(
                ref.flatten()[:result.size] - result.flatten()[:result.size])))
        kernel_metrics.append(km)

        if verbose:
            cos = km.get("cos_sim", "?")
            print(f"done (cos={cos:.4f})" if isinstance(cos, float) else "done")

    # Final output comparison
    output_node = next((n for n in gm.graph.nodes if n.op == "output"), None)
    emu_out = None
    if output_node and output_node.args:
        out_arg = output_node.args[0]
        if isinstance(out_arg, torch.fx.Node) and out_arg.name in activation_cache:
            emu_out = activation_cache[out_arg.name]
        elif isinstance(out_arg, (tuple, list)):
            for a in out_arg:
                if isinstance(a, torch.fx.Node) and a.name in activation_cache:
                    emu_out = activation_cache[a.name]
                    break

    ref_out = ref_activations.get("output")
    elapsed = time.time() - t0

    if verbose:
        print(f"\n--- Results ({elapsed:.2f}s) ---")
        print(f"  Nodes: {stats['total']} total, {stats['emulated']} emulated, "
              f"{stats['numpy']} numpy, {stats['passthrough']} passthrough")

    if emu_out is not None and ref_out is not None:
        cos = _cos_sim(ref_out, emu_out)
        ef, rf = emu_out.flatten(), ref_out.flatten()
        n = min(len(ef), len(rf))
        max_d = float(np.max(np.abs(ef[:n] - rf[:n])))
        mean_d = float(np.mean(np.abs(ef[:n] - rf[:n])))
        if verbose:
            print(f"  Output: emu={emu_out.shape} ref={ref_out.shape}")
            print(f"  Cosine sim: {cos:.6f}")
            print(f"  Max diff:   {max_d:.6f}")
            print(f"  Mean diff:  {mean_d:.6f}")
            if cos > 0.95:
                print("  PASS")
            else:
                print("  WARN: divergence (expected with BF16)")

    return {"stats": stats, "elapsed_s": elapsed, "kernel_metrics": kernel_metrics,
            "emulator_output": emu_out, "reference_output": ref_out}


def _resolve_transpose(ndim: int, args) -> tuple:
    """Convert transpose method args (dim0, dim1) to numpy axes tuple."""
    if len(args) >= 2:
        d0, d1 = int(args[0]), int(args[1])
        if d0 < 0:
            d0 += ndim
        if d1 < 0:
            d1 += ndim
        axes = list(range(ndim))
        axes[d0], axes[d1] = axes[d1], axes[d0]
        return tuple(axes)
    return tuple(range(ndim))


# ── CLI ──────────────────────────────────────────────────────────────────────

def load_model(name: str, scale: float = 0.01):
    if name == "basic":
        from model.basic import BasicModule
        return BasicModule(dim=32, depth=2), torch.randn(1, 32)
    elif name == "alexnet_small":
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "alexnet_small",
            str(Path(__file__).resolve().parent / "_backup" / "model" / "alexnet.py"))
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        return _mod.AlexNetSmall(scale=scale, num_classes=10), torch.randn(1, 3, 32, 32)
    else:
        raise ValueError(f"Unknown model: {name}")


def main():
    parser = argparse.ArgumentParser(description="Unified Atalla graph pipeline")
    parser.add_argument("--model", default="basic",
                        choices=["basic", "alexnet_small"])
    parser.add_argument("--mode", default="both",
                        choices=["schedule", "validate", "both"])
    parser.add_argument("--scale", type=float, default=0.01)
    parser.add_argument("--out-dir", default="out/graph")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    model, example_input = load_model(args.model, args.scale)
    verbose = not args.quiet

    gm = build_graph(model, example_input, verbose=verbose)

    if args.mode in ("validate", "both"):
        run_validate(gm, model, example_input, args.out_dir, verbose=verbose)

    if args.mode in ("schedule", "both"):
        import copy
        gm_sched = copy.deepcopy(gm)
        run_schedule(gm_sched, example_input, args.out_dir, verbose=verbose)


if __name__ == "__main__":
    main()
