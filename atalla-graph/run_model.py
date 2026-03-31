"""End-to-end orchestrator: PyTorch model -> Atalla emulator.

1. Capture FX graph from a PyTorch nn.Module
2. Normalize ops, remove BN/dropout
3. Plan tiles and assign DRAM addresses
4. For each compute node: emit assembly, run emulator, extract output
5. Compare final output vs PyTorch reference

Usage:
    python run_model.py --model alexnet --scale 0.01
    python run_model.py --model basic
"""
from __future__ import annotations

import os
import sys
import struct
import math
import time
import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

# Add functional_sim to path — must import emulator components BEFORE c_emitter
# because c_emitter adds aihw-ppci-compiler/emulator to sys.path, whose src/
# package shadows functional_sim/src/.
_FUNC_SIM = Path(__file__).resolve().parent.parent / "functional_sim"
sys.path.insert(0, str(_FUNC_SIM))

from src.functional_sim import run as run_emulator
from src.misc.memory import Memory
from src.components.scalar_register_file import ScalarRegisterFile
from src.components.vector_register_file import VectorRegisterFile
from src.components.execute import ExecuteUnit
from src.components.scpad import Scratchpad

from graph.fx_capture import capture, normalize_ops, get_node_shape
from graph.remove_ops import remove_ops
from graph.tile_planner import plan_tiles, TileConfig
from codegen.c_emitter import (
    emit_node, render_in_file, compile_and_assemble, LayerEmission, _to_bf16_array,
)
from codegen.dram_builder import extract_input_data


def bf16_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def run_on_emulator(in_file: str, out_dir: str, tag: str):
    """Run the emulator and return (Memory, ExecuteUnit)."""
    mem = Memory(in_file)
    sregs = ScalarRegisterFile()
    mregs = ScalarRegisterFile(num_regs=16)
    vregs = VectorRegisterFile()
    SP0 = Scratchpad(slots_per_bank=32)
    SP1 = Scratchpad(slots_per_bank=32)
    EU = ExecuteUnit()

    # ppci compiler uses x2 as scalar stack pointer and x33 as vector
    # spill base.  Place the stack above all DRAM data so sw.s writes
    # don't clobber matrix data loaded by DRAMWriter.
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


def read_bf16_from_memory(mem: Memory, addr: int, count: int) -> np.ndarray:
    """Read `count` bf16 values from emulator Memory at `addr`.

    Memory stores 32-bit words at 2-byte stride (little-endian).
    Each bf16 occupies the lower 16 bits of the word at its even address.
    """
    result = np.zeros(count, dtype=np.float32)
    for i in range(count):
        byte_addr = addr + i * 2
        word = mem.read_data(byte_addr)
        bits = word & 0xFFFF
        result[i] = bf16_to_f32(bits)
    return result


def run_pipeline(model: nn.Module, example_input: torch.Tensor,
                 out_dir: str = "out/pipeline", verbose: bool = True) -> Dict:
    """Full pipeline: capture -> plan -> emit -> run -> validate."""
    t0 = time.time()

    # Phase 1: Capture
    if verbose:
        print("Phase 1: FX capture + op normalization...")
    gm = capture(model, example_input)
    gm = remove_ops(gm)

    # Phase 2: Tile planning
    if verbose:
        print("Phase 2: Tile planning...")
    gm = plan_tiles(gm)

    # Print graph summary
    if verbose:
        print("\n--- Graph Summary ---")
        for node in gm.graph.nodes:
            op = node.meta.get("atalla_op") or "?"
            shape = get_node_shape(node)
            kt = node.meta.get("kernel_type") or "-"
            print(f"  {node.name:30s}  op={op:15s}  kernel={kt:10s}  shape={shape}")
        print("---\n")

    # Phase 3+4: Emit and run each node
    if verbose:
        print("Phase 3+4: Assembly emission + emulator execution...")

    activation_cache: Dict[str, np.ndarray] = {}
    os.makedirs(out_dir, exist_ok=True)

    # Pre-populate placeholder activations
    ref_activations = extract_input_data(gm, example_input.bfloat16())

    stats = {"nodes_total": 0, "nodes_emulated": 0, "nodes_numpy": 0, "nodes_passthrough": 0}
    kernel_metrics = []  # per-kernel detailed metrics

    for node in gm.graph.nodes:
        atalla_op = node.meta.get("atalla_op")

        if node.op == "output":
            continue

        stats["nodes_total"] += 1

        # Handle placeholders (input)
        if node.op == "placeholder":
            activation_cache[node.name] = example_input.detach().float().cpu().numpy()
            if verbose:
                print(f"  [{node.name}] placeholder -> cached")
            continue

        # Handle get_attr (weights/params)
        if node.op == "get_attr":
            attr = gm
            for part in node.target.split("."):
                attr = getattr(attr, part)
            activation_cache[node.name] = attr.detach().float().cpu().numpy() if isinstance(attr, torch.Tensor) else np.array(attr)
            if verbose:
                print(f"  [{node.name}] get_attr -> cached")
            continue

        # Flatten/reshape/dropout: just pass through the activation
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
                    stats["nodes_passthrough"] += 1
                    if verbose:
                        print(f"  [{node.name}] {atalla_op or 'passthrough'} -> reshaped {data.shape}")
                    continue

            # If we reach here, use reference activations
            if node.name in ref_activations:
                activation_cache[node.name] = ref_activations[node.name]
                stats["nodes_passthrough"] += 1
                if verbose:
                    print(f"  [{node.name}] passthrough (from ref)")
                continue

        # Emit assembly for this node
        emission = emit_node(node, gm, activation_cache)

        if emission is None:
            # No emission = passthrough
            if node.name in ref_activations:
                activation_cache[node.name] = ref_activations[node.name]
            elif node.args and isinstance(node.args[0], torch.fx.Node):
                prev_name = node.args[0].name
                if prev_name in activation_cache:
                    activation_cache[node.name] = activation_cache[prev_name]
            stats["nodes_passthrough"] += 1
            if verbose:
                print(f"  [{node.name}] {atalla_op} -> passthrough")
            continue

        if emission.skip_emulator:
            activation_cache[node.name] = emission.numpy_result
            stats["nodes_numpy"] += 1
            r = emission.numpy_result
            ref = ref_activations.get(node.name)
            km = {"name": node.name, "op": atalla_op, "backend": "numpy",
                  "shape": list(r.shape), "elems": int(r.size),
                  "min": float(r.min()), "max": float(r.max())}
            if ref is not None:
                rf, ef = ref.flatten(), r.flatten()
                ml = min(len(rf), len(ef))
                km["max_diff"] = float(np.max(np.abs(rf[:ml] - ef[:ml])))
                km["cos_sim"] = float(np.dot(rf[:ml], ef[:ml]) /
                    (np.linalg.norm(rf[:ml]) * np.linalg.norm(ef[:ml]) + 1e-12))
            kernel_metrics.append(km)
            if verbose:
                print(f"  [{node.name}] {atalla_op} -> NumPy ({r.shape}, "
                      f"range=[{r.min():.4f}, {r.max():.4f}])")
            continue

        # Compile C -> asm -> .in, then run emulator
        compile_and_assemble(emission, out_dir, node.name)
        in_file = f"{out_dir}/{node.name}.in"
        in_content = render_in_file(emission)
        Path(in_file).write_text(in_content)

        if verbose:
            print(f"  [{node.name}] {atalla_op} -> emulator "
                  f"(out_addr=0x{emission.output_addr:X}, "
                  f"elems={emission.output_elements})...", end=" ", flush=True)

        mem, eu = run_on_emulator(in_file, out_dir, node.name)
        result = read_bf16_from_memory(mem, emission.output_addr, emission.output_elements)

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
        stats["nodes_emulated"] += 1

        pm = eu.perf_metrics.metrics
        ref = ref_activations.get(node.name)
        km = {"name": node.name, "op": atalla_op, "backend": "emulator",
              "shape": list(result.shape) if hasattr(result, 'shape') else [],
              "elems": int(result.size),
              "min": float(result.min()), "max": float(result.max()),
              "cycles": int(pm.get("cycles", 0)),
              "instructions": int(pm.get("instructions", 0)),
              "gemm_ops": int(pm.get("gemm_ops", 0)),
              "sdma_ops": int(pm.get("sdma_ops", 0)),
              "mem_ops": int(pm.get("mem_ops", 0)),
              "has_nan": bool(np.any(np.isnan(result)))}
        if ref is not None:
            rf, ef = ref.flatten(), result.flatten()
            ml = min(len(rf), len(ef))
            km["max_diff"] = float(np.max(np.abs(rf[:ml] - ef[:ml])))
            d = np.dot(rf[:ml], ef[:ml])
            n1, n2 = np.linalg.norm(rf[:ml]), np.linalg.norm(ef[:ml])
            km["cos_sim"] = float(d / (n1 * n2 + 1e-12))
        kernel_metrics.append(km)

        if verbose:
            print(f"done (range=[{result.min():.4f}, {result.max():.4f}])")

    # Extract final output
    output_node = None
    for node in gm.graph.nodes:
        if node.op == "output":
            output_node = node
            break

    emulator_output = None
    if output_node and output_node.args:
        out_arg = output_node.args[0]
        if isinstance(out_arg, torch.fx.Node) and out_arg.name in activation_cache:
            emulator_output = activation_cache[out_arg.name]
        elif isinstance(out_arg, (tuple, list)):
            for a in out_arg:
                if isinstance(a, torch.fx.Node) and a.name in activation_cache:
                    emulator_output = activation_cache[a.name]
                    break

    # Phase 5: Validate
    if verbose:
        print("\n--- Validation ---")
    ref_output = ref_activations.get("output")

    elapsed = time.time() - t0
    results = {
        "stats": stats,
        "elapsed_s": elapsed,
        "emulator_output": emulator_output,
        "reference_output": ref_output,
        "kernel_metrics": kernel_metrics,
    }

    if emulator_output is not None and ref_output is not None:
        emu_flat = emulator_output.flatten()
        ref_flat = ref_output.flatten()
        min_len = min(len(emu_flat), len(ref_flat))
        emu_flat = emu_flat[:min_len]
        ref_flat = ref_flat[:min_len]

        max_diff = np.max(np.abs(emu_flat - ref_flat))
        mean_diff = np.mean(np.abs(emu_flat - ref_flat))

        dot = np.dot(emu_flat, ref_flat)
        norm_e = np.linalg.norm(emu_flat)
        norm_r = np.linalg.norm(ref_flat)
        cos_sim = dot / (norm_e * norm_r + 1e-12)

        results["max_diff"] = float(max_diff)
        results["mean_diff"] = float(mean_diff)
        results["cosine_sim"] = float(cos_sim)

        if verbose:
            print(f"  Output shape: emulator={emulator_output.shape}, ref={ref_output.shape}")
            print(f"  Max diff:  {max_diff:.6f}")
            print(f"  Mean diff: {mean_diff:.6f}")
            print(f"  Cosine sim: {cos_sim:.6f}")
            if cos_sim > 0.95:
                print("  PASS: outputs match within BF16 tolerance")
            else:
                print("  WARN: outputs diverge (expected with BF16 accumulation)")

    if verbose:
        print(f"\nPipeline complete in {elapsed:.2f}s")
        print(f"  Nodes: {stats['nodes_total']} total, "
              f"{stats['nodes_emulated']} emulated, "
              f"{stats['nodes_numpy']} numpy, "
              f"{stats['nodes_passthrough']} passthrough")

    return results


def main():
    parser = argparse.ArgumentParser(description="PyTorch -> Atalla emulator pipeline")
    parser.add_argument("--model", type=str, default="basic",
                        choices=["basic", "alexnet"],
                        help="Model to run")
    parser.add_argument("--scale", type=float, default=0.01,
                        help="Channel scale for AlexNet")
    parser.add_argument("--out-dir", type=str, default="out/pipeline")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    if args.model == "basic":
        from model.basic import BasicModule
        model = BasicModule(dim=32, depth=2)
        example_input = torch.randn(1, 32)
    elif args.model == "alexnet":
        from model.alexnet import AlexNetSmall
        model = AlexNetSmall(scale=args.scale, num_classes=10)
        example_input = torch.randn(1, 3, 32, 32)

    results = run_pipeline(model, example_input,
                           out_dir=args.out_dir,
                           verbose=not args.quiet)

    return results


if __name__ == "__main__":
    main()
