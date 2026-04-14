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
    python run_graph.py --model basic --mode validate --metrics-json poster_metrics.json
    python run_graph.py --model alexnet_small --mode validate --validate-inputs oracle
    python run_graph.py --model layernorm_smoke --mode validate --validate-inputs oracle
    python run_graph.py --model vit_micro --mode validate --validate-inputs oracle
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Set

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
from src.components.scalar_register_file import ScalarRegisterFile, mask_register_file
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


DEFAULT_KERNEL_BUNDLE_OPS: FrozenSet[str] = frozenset(
    {"conv", "relu", "maxpool", "matmul", "add", "layernorm", "gelu"}
)

# Ops that run in the emulator; oracle mode refreshes activations from PyTorch refs before emit.
_EMU_ATALLA_OPS: FrozenSet[str] = frozenset(
    {"conv", "relu", "maxpool", "matmul", "add", "layernorm", "gelu"}
)


def bf16_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def _run_emulator(in_file: str, out_dir: str, tag: str):
    mem = Memory(in_file)
    sregs = ScalarRegisterFile()
    mregs = mask_register_file()
    vregs = VectorRegisterFile()
    SP0 = Scratchpad(slots_per_bank=32)
    SP1 = Scratchpad(slots_per_bank=32)
    EU = ExecuteUnit()

    max_data_addr = max(mem.data_mem.keys()) if mem.data_mem else 0
    stack_base = ((max_data_addr + 0x1000) & ~0xFFF) + 0x1000
    sregs.write(2, stack_base)
    # x33-64/-128 vector spills must not alias the frame (sdma_ctl_* live near x8+72).
    sregs.write(33, stack_base + 0x1000)
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
    """Read ``count`` BF16 values from byte ``addr`` (packed two per 32-bit word in .data)."""
    result = np.zeros(count, dtype=np.float32)
    for i in range(count):
        bits = mem.read_bf16_le(addr + i * 2)
        result[i] = bf16_to_f32(bits)
    return result


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    af, bf = a.flatten(), b.flatten()
    n = min(len(af), len(bf))
    af, bf = af[:n], bf[:n]
    d = np.dot(af, bf)
    return float(d / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))


def _nz_count(a: np.ndarray, eps: float = 1e-8) -> int:
    return int(np.count_nonzero(np.abs(a) > eps))


def _apply_oracle_inputs(
    activation_cache: Dict[str, np.ndarray],
    ref_activations: Dict[str, np.ndarray],
) -> None:
    """Overwrite cache with PyTorch activations so each kernel sees matched inputs."""
    for name, arr in ref_activations.items():
        activation_cache[name] = np.asarray(arr, dtype=np.float32).copy()


def _layer_compare_metrics(
    ref: np.ndarray, result: np.ndarray, eps: float = 1e-12
) -> Dict[str, float]:
    """Compare ref vs result: cos (direction-only), abs errors, rel ℓ₂, rel max error."""
    rf = np.asarray(ref, dtype=np.float64).flatten()
    ef = np.asarray(result, dtype=np.float64).flatten()
    n = min(len(rf), len(ef))
    if n == 0:
        return {
            "cos_sim": 1.0,
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "rmse": 0.0,
            "rel_l2_error": 0.0,
            "ref_max_abs": 0.0,
            "rel_max_abs_error": 0.0,
        }
    rf, ef = rf[:n], ef[:n]
    diff = ef - rf
    ref_norm = float(np.linalg.norm(rf))
    err_norm = float(np.linalg.norm(diff))
    ref_max_abs = float(np.max(np.abs(rf)))
    max_abs = float(np.max(np.abs(diff)))
    return {
        "cos_sim": _cos_sim(
            rf.astype(np.float32, copy=False),
            ef.astype(np.float32, copy=False),
        ),
        "max_abs_error": max_abs,
        "mean_abs_error": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "rel_l2_error": float(err_norm / (ref_norm + eps)),
        "ref_max_abs": ref_max_abs,
        "rel_max_abs_error": float(max_abs / (ref_max_abs + eps)),
    }


def _aggregate_kernel_metrics(kernel_metrics: list) -> Dict:
    """Roll up poster / roofline metrics across all emulated layers."""
    emus = [k for k in kernel_metrics if k.get("backend") == "emulator"]
    tot_pkt = sum(int(k.get("sched_packets", 0)) for k in emus)
    tot_slot = sum(int(k.get("sched_slots_filled", 0)) for k in emus)
    tot_bl = sum(int(k.get("bytes_loaded", 0)) for k in emus)
    tot_bw = sum(int(k.get("bytes_written", 0)) for k in emus)
    tot_flops = sum(float(k.get("flops_total", 0.0)) for k in emus)
    emu_pkts = sum(
        int(
            k.get("packets_executed", k.get("packets", 0)),
        )
        for k in emus
    )
    emu_ins = sum(
        int(
            k.get("instructions_executed", k.get("instructions", 0)),
        )
        for k in emus
    )
    mem_b = tot_bl + tot_bw
    hist: Counter = Counter()
    for k in emus:
        for sk, sv in k.get("sched_slot_histogram", {}).items():
            hist[int(sk)] += int(sv)
    n_empty_pkt = int(hist.get(0, 0))
    n_nonempty_pkt = tot_pkt - n_empty_pkt
    return {
        "emulated_layer_count": len(emus),
        # Includes empty scheduled packets (padding / alignment) in the denominator.
        "aggregate_static_slot_efficiency": (tot_slot / (tot_pkt * 4.0)) if tot_pkt else 0.0,
        # Excludes zero-fill packets — closer to “how full are packets that actually issue work”.
        "aggregate_static_slot_efficiency_nonempty": (
            (tot_slot / (n_nonempty_pkt * 4.0)) if n_nonempty_pkt else 0.0
        ),
        "aggregate_dynamic_slot_efficiency": (emu_ins / (emu_pkts * 4.0)) if emu_pkts else 0.0,
        "total_sched_packets": tot_pkt,
        "total_sched_slots_filled": tot_slot,
        "total_emu_packets": emu_pkts,
        "total_emu_instructions_retired": emu_ins,
        "total_bytes_loaded": tot_bl,
        "total_bytes_written": tot_bw,
        "total_bytes_memory_traffic": mem_b,
        "total_flops": tot_flops,
        "aggregate_arithmetic_intensity": (tot_flops / mem_b) if mem_b > 0 else 0.0,
        "aggregate_sched_slot_histogram": {str(k): v for k, v in sorted(hist.items())},
        "sched_packets_empty": n_empty_pkt,
        "sched_packets_nonempty": n_nonempty_pkt,
    }


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

def run_validate(
    gm,
    model: nn.Module,
    example_input: torch.Tensor,
    out_dir: str,
    verbose: bool = True,
    *,
    kernel_bundle_dir: Optional[str] = None,
    kernel_bundle_ops: Optional[FrozenSet[str]] = None,
    metrics_json_path: Optional[str] = None,
    validate_inputs: str = "chained",
) -> Dict:
    """Per-node compile → emulate → compare vs PyTorch golden.

    ``validate_inputs``:
      * ``chained`` — each layer consumes the previous layer's emulator output
        (end-to-end integration). Per-layer cos mixes accumulated drift with the
        local kernel error; the meaningful end-to-end signal is the final output
        metrics.
      * ``oracle`` — before each emulated op, ``_apply_oracle_inputs`` copies
        every tensor from ``ref_activations`` into the cache so that kernel
        inputs match PyTorch. Per-layer output is still compared to ref for that
        node; **final output cos can be perfect while matmul (etc.) layers still
        show bad cos/rmse**, because later ops are fed ref activations again, not
        the bad emulator outputs.

    If ``kernel_bundle_dir`` is set, the first emulated node for each op in
    ``kernel_bundle_ops`` (default: conv, relu, maxpool, matmul, add) copies
    ``.c`` / ``.s`` / ``.in``, and ``verify.json`` (metrics)
    into a subdirectory.
    """
    if validate_inputs not in ("chained", "oracle"):
        raise ValueError("validate_inputs must be 'chained' or 'oracle'")
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    bundle_ops = kernel_bundle_ops or DEFAULT_KERNEL_BUNDLE_OPS
    bundle_exported: Optional[Set[str]] = set() if kernel_bundle_dir else None

    ref_activations = extract_input_data(gm, example_input.bfloat16())
    activation_cache: Dict[str, np.ndarray] = {}

    stats = {"total": 0, "emulated": 0, "numpy": 0, "passthrough": 0}
    kernel_metrics = []

    if verbose:
        print(f"\n--- Graph (validate_inputs={validate_inputs}) ---")
        print(
            "  Note: cos similarity is direction-only (ignores scale). Prefer "
            "rmse, rel_l2, rel_max_abs (max_abs / max|ref|) for element error. "
            "Refs use the traced BF16 model (numpy float), i.e. BF16 matmul semantics."
        )
        if validate_inputs == "chained":
            print(
                "  chained: true pipeline — bad early layers corrupt later "
                "per-layer cos."
            )
        else:
            print(
                "  oracle: ref activations reloaded before each emulated op — "
                "good final output does NOT imply every kernel matched ref."
            )
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
            activation_cache[node.name] = (
                example_input.detach().bfloat16().float().cpu().numpy()
            )
            if verbose:
                print(f"  [{node.name}] placeholder")
            continue

        if node.op == "get_attr":
            attr = gm
            for part in node.target.split("."):
                attr = getattr(attr, part)
            activation_cache[node.name] = (
                attr.detach().bfloat16().float().cpu().numpy()
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

        if validate_inputs == "oracle" and atalla_op in _EMU_ATALLA_OPS:
            _apply_oracle_inputs(activation_cache, ref_activations)

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
                km.update(_layer_compare_metrics(ref, r))
            kernel_metrics.append(km)
            if verbose:
                extra = ""
                if "cos_sim" in km:
                    extra = (
                        f" (cos={km['cos_sim']:.6f}, rmse={km['rmse']:.6f}, "
                        f"rel_l2={km['rel_l2_error']:.6f}, "
                        f"relmax={km['rel_max_abs_error']:.6f})"
                    )
                print(f"  [{node.name}] {atalla_op} -> NumPy {r.shape}{extra}")
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

        if emission.conv_post is not None:
            cp = emission.conv_post
            raw = result.reshape(cp["Ho"], cp["Wo"], cp["C"])
            result = raw.transpose(2, 0, 1).reshape(cp["final_shape"])

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

        eu.perf_metrics.update_derived_metrics()
        pm = eu.perf_metrics.metrics
        ref = ref_activations.get(node.name)
        pk = int(pm.get("packets_executed", pm.get("packets", 0)))
        ins = int(pm.get("instructions_executed", pm.get("instructions", 0)))
        km = {
            "name": node.name,
            "op": atalla_op,
            "backend": "emulator",
            "shape": list(result.shape),
            "elems": int(result.size),
            "packets": pk,
            "instructions": ins,
            "packets_executed": pk,
            "instructions_executed": ins,
            "sched_packets": int(emission.sched_packets),
            "sched_slots_filled": int(emission.sched_slots_filled),
            "sched_slot_efficiency": float(emission.sched_slot_efficiency),
            "sched_slot_histogram": dict(emission.sched_slot_histogram),
            "bytes_loaded": int(pm.get("bytes_loaded", 0)),
            "bytes_written": int(pm.get("bytes_written", 0)),
            "flops_total": float(pm.get("flops_total", 0.0)),
            "arithmetic_intensity": float(pm.get("arithmetic_intensity", 0.0)),
            "arithmetic_intensity_loads": float(pm.get("arithmetic_intensity_loads", 0.0)),
        }
        if ref is not None:
            cmpm = _layer_compare_metrics(ref, result)
            km.update(cmpm)
            km["max_diff"] = cmpm["max_abs_error"]
            km["emu_norm"] = float(np.linalg.norm(result.flatten()))
            km["ref_norm"] = float(np.linalg.norm(ref.flatten()))
            km["emu_nz"] = _nz_count(result)
            km["ref_nz"] = _nz_count(ref)
        kernel_metrics.append(km)

        if (
            bundle_exported is not None
            and atalla_op in bundle_ops
            and atalla_op not in bundle_exported
        ):
            bundle_exported.add(atalla_op)
            bdir = Path(kernel_bundle_dir) / f"{atalla_op}_{node.name}"
            bdir.mkdir(parents=True, exist_ok=True)
            tag = node.name
            for ext in (".c", ".s", ".in"):
                src = Path(out_dir) / f"{tag}{ext}"
                if src.is_file():
                    shutil.copy(src, bdir / src.name)
            (bdir / "verify.json").write_text(json.dumps(km, indent=2) + "\n")

        if verbose:
            cos = km.get("cos_sim")
            if isinstance(cos, float):
                extra = (
                    f", max_abs={km['max_abs_error']:.6f}, "
                    f"relmax={km['rel_max_abs_error']:.6f}, "
                    f"rmse={km['rmse']:.6f}, rel_l2={km['rel_l2_error']:.6f}, "
                    f"emu_norm={km['emu_norm']:.6f}, ref_norm={km['ref_norm']:.6f}, "
                    f"emu_nz={km['emu_nz']}, ref_nz={km['ref_nz']}"
                    if "max_abs_error" in km else ""
                )
                se = km.get("sched_slot_efficiency", 0.0)
                print(
                    f"done (cos={cos:.6f}{extra}, "
                    f"slot_eff={se:.3f}, bytes_ld={km['bytes_loaded']}, bytes_st={km['bytes_written']})"
                )
            else:
                print("done")

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

    aggregate_metrics = _aggregate_kernel_metrics(kernel_metrics)

    if emu_out is not None and ref_out is not None:
        end_cmp = _layer_compare_metrics(ref_out, emu_out)
        cos = end_cmp["cos_sim"]
        if verbose:
            print(f"  Output: emu={emu_out.shape} ref={ref_out.shape}")
            print(
                f"  cos={cos:.6f}  max_abs={end_cmp['max_abs_error']:.6f}  "
                f"relmax={end_cmp['rel_max_abs_error']:.6f}  "
                f"rmse={end_cmp['rmse']:.6f}  rel_l2={end_cmp['rel_l2_error']:.6f}"
            )
            if validate_inputs == "oracle" and cos > 0.95:
                print(
                    "  PASS (output cos threshold) — in oracle mode, scan "
                    "per-layer relmax/rmse for kernels that still disagree."
                )
            elif cos > 0.95:
                print("  PASS (cos threshold)")
            if validate_inputs == "oracle":
                print(
                    "\n  Oracle caveat: inputs are reset from PyTorch before each "
                    "emulated op, so the last add can be exact even when matmul "
                    "outputs vs ref are poor."
                )

    if verbose:
        am = aggregate_metrics
        print("\n--- Aggregate metrics (all emulated layers) ---")
        print(f"  Static slot efficiency (all scheduled packets): {am['aggregate_static_slot_efficiency']:.4f}")
        print(f"  Static slot efficiency (non-empty packets only): {am['aggregate_static_slot_efficiency_nonempty']:.4f}")
        print(f"  Dynamic slot efficiency (emulator retired): {am['aggregate_dynamic_slot_efficiency']:.4f}")
        print(f"  Total bytes loaded / written: {am['total_bytes_loaded']} / {am['total_bytes_written']}")
        print(f"  Total FLOPs (model counters): {am['total_flops']:.0f}")
        print(f"  Aggregate arithmetic intensity (FLOPs / byte moved): {am['aggregate_arithmetic_intensity']:.4f}")

    out = {
        "stats": stats,
        "elapsed_s": elapsed,
        "validate_inputs": validate_inputs,
        "kernel_metrics": kernel_metrics,
        "aggregate_metrics": aggregate_metrics,
        "emulator_output": emu_out,
        "reference_output": ref_out,
    }
    if metrics_json_path:
        Path(metrics_json_path).parent.mkdir(parents=True, exist_ok=True)
        # JSON-serializable snapshot for posters / spreadsheets
        blob = {
            "model": getattr(model, "__class__", type(model)).__name__,
            "validate_inputs": validate_inputs,
            "aggregate_metrics": aggregate_metrics,
            "kernel_metrics": kernel_metrics,
            "stats": stats,
            "elapsed_s": elapsed,
        }
        Path(metrics_json_path).write_text(json.dumps(blob, indent=2) + "\n")
        if verbose:
            print(f"\nWrote metrics JSON: {metrics_json_path}")

    return out


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
        from model.alexnet_small import AlexNetSmall
        return AlexNetSmall(scale=scale, num_classes=10), torch.randn(1, 3, 32, 32)
    elif name == "layernorm_smoke":
        from model.layernorm_smoke import LayerNormSmoke
        return LayerNormSmoke(dim=32), torch.randn(1, 32)
    elif name == "vit_micro":
        from model.vit_micro import ViTMicro
        m = ViTMicro(dim=32, n_tokens=4)
        return m, torch.randn(1, 4, 32)
    else:
        raise ValueError(f"Unknown model: {name}")


def main():
    parser = argparse.ArgumentParser(description="Unified Atalla graph pipeline")
    parser.add_argument("--model", default="basic",
                        choices=["basic", "alexnet_small", "layernorm_smoke", "vit_micro"])
    parser.add_argument("--mode", default="both",
                        choices=["schedule", "validate", "both"])
    parser.add_argument("--scale", type=float, default=0.01)
    parser.add_argument("--out-dir", default="out/graph")
    parser.add_argument(
        "--export-kernel-bundle",
        metavar="DIR",
        default=None,
        help=(
            "During validate, copy first conv/relu/maxpool/matmul/add artifacts "
            "(.c, .s, .in, verify.json) into DIR/<op>_<node>/"
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--metrics-json",
        metavar="PATH",
        default=None,
        help="After validate, write per-layer + aggregate poster metrics to PATH (JSON).",
    )
    parser.add_argument(
        "--validate-inputs",
        choices=("chained", "oracle"),
        default="chained",
        help=(
            "chained: prior emu output feeds the next op (end-to-end; per-layer cos is mixed). "
            "oracle: copy all ref activations into cache before each emulated op — "
            "per-layer output still vs ref, but final output can match even if matmuls do not."
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    model, example_input = load_model(args.model, args.scale)
    verbose = not args.quiet

    gm = build_graph(model, example_input, verbose=verbose)

    if args.mode in ("validate", "both"):
        run_validate(
            gm,
            model,
            example_input,
            args.out_dir,
            verbose=verbose,
            kernel_bundle_dir=args.export_kernel_bundle,
            metrics_json_path=args.metrics_json,
            validate_inputs=args.validate_inputs,
        )

    if args.mode in ("schedule", "both"):
        import copy
        gm_sched = copy.deepcopy(gm)
        run_schedule(gm_sched, example_input, args.out_dir, verbose=verbose)


if __name__ == "__main__":
    main()
