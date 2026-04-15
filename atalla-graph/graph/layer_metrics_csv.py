"""Write per-layer metrics CSV from run_graph.validate ``kernel_metrics`` list."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Keep in sync with functional_sim/src/misc/instruction_retire.py RETIRED_BUCKET_NAMES.
RETIRED_BUCKET_NAMES: Tuple[str, ...] = (
    "branch_control",
    "sdma",
    "scalar_mem",
    "scalar_alu",
    "vector_mem",
    "vector_alu",
    "gemm_systolic",
    "move_convert",
)

_DYN_RETIRED_COLS = [f"dyn_retired_{n}" for n in RETIRED_BUCKET_NAMES]
_PCT_DYN_RETIRED_COLS = [f"pct_dyn_retired_{n}" for n in RETIRED_BUCKET_NAMES]

# One row per graph node that produced metrics; emulator rows carry mapping + perf detail.
CSV_FIELDS: List[str] = [
    "name",
    "op",
    "backend",
    "shape",
    "elems",
    "packets_executed",
    "instructions_executed",
    "sched_packets",
    "packets_per_static_row",
    "bytes_loaded",
    "bytes_loaded_sp0",
    "bytes_loaded_sp1",
    "bytes_written",
    "bytes_stored_sp0",
    "bytes_stored_sp1",
    "bytes_memory_total",
    "flops_total",
    "flops_matmul",
    "flops_non_matmul",
    "flops_vector",
    "flops_scalar",
    "moveconvert_ops",
    *_DYN_RETIRED_COLS,
    *_PCT_DYN_RETIRED_COLS,
    "ai_load_store",
    "ai_loads_only",
    "static_sched_slot_efficiency",
    "dynamic_slot_efficiency",
    "packet_share_of_emulated",
    "map_kind",
    "map_M",
    "map_N",
    "map_K",
    "map_TILE",
    "map_M_tiles",
    "map_N_tiles",
    "map_K_tiles",
    "map_k_stride",
    "bytes_est_activation",
    "bytes_est_input",
    "bytes_est_weight",
    "bytes_est_output",
    "bytes_est_Z_tile",
    "bytes_est_a",
    "bytes_est_b",
    "bytes_est_in",
    "bytes_est_out",
    "bytes_est_io_inplace",
    "map_rows",
    "map_width",
    "map_num_rows",
    "map_row_len",
    "map_M_rows",
    "map_D",
    "bytes_est_gamma",
    "bytes_est_beta",
    "map_reuse_note",
    "pct_est_bytes_activation",
    "pct_est_bytes_weight",
    "pct_est_bytes_output",
    "flops_per_packet",
    "flops_per_instruction",
    "reuse_R_A_gemm",
    "reuse_R_W_gemm",
    "cos_sim",
    "max_abs_error",
    "rmse",
    "rel_l2_error",
]

TEMPLATE_SUMMARY_FIELDS = [
    "template_key",
    "map_kind",
    "map_M",
    "map_N",
    "map_K",
    "n_layers",
    "layer_names",
    "sum_packets_executed",
    "sum_instructions_executed",
    "sum_sched_packets",
    "sum_flops_total",
    "sum_flops_matmul",
    "mean_packet_share",
    "sum_bytes_loaded",
    "sum_bytes_written",
    "sum_bytes_loaded_sp0",
    "sum_bytes_loaded_sp1",
    "sum_bytes_stored_sp0",
    "sum_bytes_stored_sp1",
    "reuse_R_A_gemm",
    "reuse_R_W_gemm",
    *[f"sum_dyn_retired_{n}" for n in RETIRED_BUCKET_NAMES],
]


def _cell(v: Any) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return f"{v:.6f}".rstrip("0").rstrip(".") or "0"
    if isinstance(v, (list, tuple)):
        return str(list(v))
    return str(v)


def _gemm_estimated_byte_pcts(raw: Dict[str, Any]) -> Tuple[str, str, str]:
    """For gemm / conv_as_gemm, % of (act + weight + out) from emitter estimates."""
    mk = raw.get("map_kind")
    if mk not in ("gemm", "conv_as_gemm"):
        return "", "", ""
    ba = int(raw.get("bytes_est_activation") or 0)
    bw = int(raw.get("bytes_est_weight") or 0)
    bo = int(raw.get("bytes_est_output") or 0)
    t = ba + bw + bo
    if t <= 0:
        return "", "", ""
    return (
        _cell(100.0 * ba / t),
        _cell(100.0 * bw / t),
        _cell(100.0 * bo / t),
    )


def _gemm_reuse_from_sdma_split(raw: Dict[str, Any]) -> None:
    """FLOPs / SDMA bytes on SP0 vs SP1. For graph GEMM kernels: SP0≈A, SP1≈W (see writeup)."""
    mk = str(raw.get("map_kind") or "")
    if mk not in ("gemm", "conv_as_gemm"):
        raw["reuse_R_A_gemm"] = ""
        raw["reuse_R_W_gemm"] = ""
        return
    ft = float(raw.get("flops_total") or 0)
    b0 = int(raw.get("bytes_loaded_sp0") or 0)
    b1 = int(raw.get("bytes_loaded_sp1") or 0)
    raw["reuse_R_A_gemm"] = (ft / b0) if b0 else ""
    raw["reuse_R_W_gemm"] = (ft / b1) if b1 else ""


def enrich_derived_csv_fields(raw: Dict[str, Any]) -> None:
    """Mutates ``raw`` with presentation helpers (also useful if JSON is re-exported)."""
    pa, pw, po = _gemm_estimated_byte_pcts(raw)
    raw["pct_est_bytes_activation"] = pa
    raw["pct_est_bytes_weight"] = pw
    raw["pct_est_bytes_output"] = po
    _gemm_reuse_from_sdma_split(raw)
    ft = float(raw.get("flops_total") or 0)
    pk = int(raw.get("packets_executed") or 0)
    ins = int(raw.get("instructions_executed") or 0)
    raw["flops_per_packet"] = (ft / pk) if pk else ""
    raw["flops_per_instruction"] = (ft / ins) if ins else ""
    for n in RETIRED_BUCKET_NAMES:
        c = int(raw.get(f"dyn_retired_{n}") or 0)
        raw[f"pct_dyn_retired_{n}"] = (100.0 * c / ins) if ins else ""


def annotate_packet_shares(kernel_metrics: List[Dict[str, Any]]) -> None:
    tot = sum(
        int(k.get("packets_executed") or 0)
        for k in kernel_metrics
        if k.get("backend") == "emulator"
    )
    for k in kernel_metrics:
        if k.get("backend") != "emulator":
            k["packet_share_of_emulated"] = ""
            continue
        pk = int(k.get("packets_executed") or 0)
        k["packet_share_of_emulated"] = (pk / tot) if tot else 0.0


def write_layer_metrics_csv(path: Path, kernel_metrics: List[Dict[str, Any]]) -> None:
    annotate_packet_shares(kernel_metrics)
    for raw in kernel_metrics:
        enrich_derived_csv_fields(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for raw in kernel_metrics:
            row: Dict[str, str] = {}
            for key in CSV_FIELDS:
                if key == "shape":
                    row[key] = _cell(raw.get("shape"))
                elif key == "bytes_memory_total":
                    bl = int(raw.get("bytes_loaded") or 0)
                    bw = int(raw.get("bytes_written") or 0)
                    row[key] = str(bl + bw)
                elif key == "flops_non_matmul":
                    ft = float(raw.get("flops_total") or 0)
                    fm = float(raw.get("flops_matmul") or 0)
                    row[key] = _cell(ft - fm)
                elif key == "packets_per_static_row":
                    pk = int(raw.get("packets_executed") or 0)
                    sp = int(raw.get("sched_packets") or 0)
                    row[key] = _cell((pk / sp) if sp else "")
                elif key == "dynamic_slot_efficiency":
                    ins = int(raw.get("instructions_executed") or 0)
                    pk = int(raw.get("packets_executed") or 0)
                    row[key] = _cell((ins / (pk * 4.0)) if pk else "")
                elif key == "ai_load_store":
                    row[key] = _cell(raw.get("arithmetic_intensity"))
                elif key == "ai_loads_only":
                    row[key] = _cell(raw.get("arithmetic_intensity_loads"))
                elif key == "static_sched_slot_efficiency":
                    row[key] = _cell(raw.get("sched_slot_efficiency"))
                elif key.startswith("pct_dyn_retired_"):
                    row[key] = _cell(raw.get(key))
                elif key in (
                    "pct_est_bytes_activation",
                    "pct_est_bytes_weight",
                    "pct_est_bytes_output",
                    "flops_per_packet",
                    "flops_per_instruction",
                    "reuse_R_A_gemm",
                    "reuse_R_W_gemm",
                ):
                    row[key] = _cell(raw.get(key))
                else:
                    row[key] = _cell(raw.get(key))
            w.writerow(row)


def _template_group_key(k: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """4-tuple for grouping; GEMM uses M,N,K — other kinds use their logical dims."""
    mk = str(k.get("map_kind") or "")
    if mk in ("gemm", "conv_as_gemm"):
        return (
            mk,
            str(k.get("map_M") or ""),
            str(k.get("map_N") or ""),
            str(k.get("map_K") or ""),
        )
    if mk == "add":
        return (mk, str(k.get("map_rows") or ""), str(k.get("map_width") or ""), "")
    if mk == "layernorm":
        return (mk, str(k.get("map_M_rows") or ""), str(k.get("map_D") or ""), "")
    if mk == "softmax":
        return (
            mk,
            str(k.get("map_num_rows") or ""),
            str(k.get("map_row_len") or ""),
            "",
        )
    if mk == "relu":
        return (mk, str(k.get("map_rows") or ""), str(k.get("map_width") or ""), "")
    if mk == "maxpool":
        return (
            mk,
            str(k.get("map_C") or ""),
            f"{k.get('map_H', '')}x{k.get('map_W', '')}",
            str(k.get("map_pool") or ""),
        )
    return (mk, str(k.get("op") or ""), str(k.get("name") or ""), "")


def write_template_summary_csv(path: Path, kernel_metrics: List[Dict[str, Any]]) -> None:
    """Roll up emulator layers by logical template (GEMM M×N×K, add rows×width, etc.)."""
    annotate_packet_shares(kernel_metrics)
    for raw in kernel_metrics:
        enrich_derived_csv_fields(raw)

    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for k in kernel_metrics:
        if k.get("backend") != "emulator":
            continue
        groups[_template_group_key(k)].append(k)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TEMPLATE_SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for key in sorted(groups.keys(), key=lambda t: (t[0], t[1], t[2], t[3])):
            mk, m, n, kk = key
            rows = groups[key]
            names = ";".join(r.get("name", "") for r in rows)
            spk = sum(int(r.get("packets_executed") or 0) for r in rows)
            sins = sum(int(r.get("instructions_executed") or 0) for r in rows)
            ssched = sum(int(r.get("sched_packets") or 0) for r in rows)
            sft = sum(float(r.get("flops_total") or 0) for r in rows)
            sfm = sum(float(r.get("flops_matmul") or 0) for r in rows)
            shares = [float(r.get("packet_share_of_emulated") or 0) for r in rows]
            mean_share = sum(shares) / len(shares) if shares else 0.0
            sbl = sum(int(r.get("bytes_loaded") or 0) for r in rows)
            sbw = sum(int(r.get("bytes_written") or 0) for r in rows)
            sbl0 = sum(int(r.get("bytes_loaded_sp0") or 0) for r in rows)
            sbl1 = sum(int(r.get("bytes_loaded_sp1") or 0) for r in rows)
            sbs0 = sum(int(r.get("bytes_stored_sp0") or 0) for r in rows)
            sbs1 = sum(int(r.get("bytes_stored_sp1") or 0) for r in rows)
            if mk in ("gemm", "conv_as_gemm"):
                tkey = f"{mk}|M{m}|N{n}|K{kk}"
            elif mk == "add":
                tkey = f"{mk}|rows{m}|width{n}"
            elif mk == "layernorm":
                tkey = f"{mk}|M{m}|D{n}"
            elif mk == "softmax":
                tkey = f"{mk}|rows{m}|len{n}"
            elif mk == "relu":
                tkey = f"{mk}|rows{m}|width{n}"
            elif mk == "maxpool":
                tkey = f"{mk}|C{m}|HW{n}|pool{kk}"
            else:
                tkey = "|".join(key)
            ra_g = (sft / sbl0) if mk in ("gemm", "conv_as_gemm") and sbl0 else ""
            rw_g = (sft / sbl1) if mk in ("gemm", "conv_as_gemm") and sbl1 else ""
            retire_sums = {
                f"sum_dyn_retired_{n}": sum(
                    int(r.get(f"dyn_retired_{n}") or 0) for r in rows
                )
                for n in RETIRED_BUCKET_NAMES
            }
            w.writerow(
                {
                    "template_key": tkey,
                    "map_kind": mk,
                    "map_M": m,
                    "map_N": n,
                    "map_K": kk,
                    "n_layers": str(len(rows)),
                    "layer_names": names,
                    "sum_packets_executed": str(spk),
                    "sum_instructions_executed": str(sins),
                    "sum_sched_packets": str(ssched),
                    "sum_flops_total": _cell(sft),
                    "sum_flops_matmul": _cell(sfm),
                    "mean_packet_share": _cell(mean_share),
                    "sum_bytes_loaded": str(sbl),
                    "sum_bytes_written": str(sbw),
                    "sum_bytes_loaded_sp0": str(sbl0),
                    "sum_bytes_loaded_sp1": str(sbl1),
                    "sum_bytes_stored_sp0": str(sbs0),
                    "sum_bytes_stored_sp1": str(sbs1),
                    "reuse_R_A_gemm": _cell(ra_g),
                    "reuse_R_W_gemm": _cell(rw_g),
                    **{k: str(v) for k, v in retire_sums.items()},
                }
            )
