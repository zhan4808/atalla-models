#!/usr/bin/env python3
"""Generate figures + LaTeX table fragments for the ViT-Micro Atalla writeup.

Run from ``atalla-graph/``::

    python3 docs/vit_micro/generate_writeup_figures.py
    cd docs/vit_micro && pdflatex vit_micro_writeup.tex   # twice for longtable

Reads pipeline outputs from ``out/graph/`` (CSV + JSON). Writes figure PDFs and
generated ``.tex`` fragments into this directory (next to ``vit_micro_writeup.tex``).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# This file lives in docs/vit_micro/; pipeline artifacts live in ../../out/graph/
_SCRIPT_DIR = Path(__file__).resolve().parent
_ATALLA_GRAPH = _SCRIPT_DIR.parent.parent
GRAPH_OUT = _ATALLA_GRAPH / "out" / "graph"
OUT_DIR = _SCRIPT_DIR

CSV_PATH = str(GRAPH_OUT / "vit_micro_layer_metrics.csv")
TMPL_PATH = str(GRAPH_OUT / "vit_micro_layer_metrics_template_summary.csv")
JSON_PATH = str(GRAPH_OUT / "vit_micro_metrics.json")
TABLE_RETIRE_TEX = str(OUT_DIR / "vit_micro_writeup_table_retire.tex")
TABLE_SCHED_TEX = str(OUT_DIR / "vit_micro_writeup_table_schedule.tex")
TABLE_PER_LAYER_TEX = str(OUT_DIR / "vit_micro_writeup_table_per_layer.tex")

OP_COLORS = {
    "matmul": "#4C72B0",
    "add": "#55A868",
    "layernorm": "#DD8452",
    "softmax": "#C44E52",
    "atalla_sdpa": "#A35F8A",
    "gelu": "#8172B3",
    "mul": "#a89f91",
    "relu": "#937860",
}

# Order for stacked mix figure + aggregate table (sync with instruction_retire buckets).
RETIRED_TABLE_ORDER = (
    "gemm_systolic",
    "vector_alu",
    "vector_mem",
    "sdma",
    "move_convert",
    "scalar_alu",
    "scalar_mem",
    "branch_control",
)
RETIRED_DISPLAY = {
    "branch_control": "Branch / control",
    "sdma": "SDMA (scpad)",
    "scalar_mem": "Scalar GMEM (lw/sw)",
    "scalar_alu": "Scalar ALU (other)",
    "vector_mem": "Vector mem (vreg.ld/st)",
    "vector_alu": "Vector ALU (.vv/.vi/.vs)",
    "gemm_systolic": "GEMM / systolic",
    "move_convert": "Move / convert",
}
RETIRED_STACK_COLORS = (
    "#4C72B0",
    "#55A868",
    "#C44E52",
    "#DD8452",
    "#8172B3",
    "#937860",
    "#8C8C8C",
    "#333333",
)

FRIENDLY_TEMPLATE = {
    "layernorm|M32|D32": "LayerNorm (M32, D32)",
    "layernorm|M4|D32": "LayerNorm (M4, D32)",
    "gemm|M32|N32|K32": "GEMM (32,32,32)",
    "gemm|M32|N64|K32": "GEMM FF1 (32,64,32)",
    "gemm|M32|N32|K64": "GEMM FF2 (32,32,64)",
    "gemm|M4|N32|K32": "GEMM (4,32,32) [legacy T=4]",
    "gemm|M4|N64|K32": "GEMM FF1 (4,64,32)",
    "gemm|M4|N32|K64": "GEMM FF2 (4,32,64)",
    "gemm|M4|N4|K32": "GEMM attn score (4,4,32)",
    "gemm|M4|N32|K4": "GEMM attn combine (4,32,4)",
    "add|rows32|width32": "Add (32x32, residual et al.)",
    "add|rows4|width32": "Add (4x32)",
    "add|rows8|width32": "Add (8x32)",
    "atalla_sdpa|atalla_sdpa|sdpa|": "Fused SDPA (flash, N=D=32)",
    "softmax|rows4|len4": "Softmax (4x4)",
}


def tex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def friendly_template_label(template_key: str) -> str:
    return FRIENDLY_TEMPLATE.get(template_key, template_key.replace("|", " · "))


def read_layer_csv() -> list[dict]:
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


def read_template_csv() -> list[dict]:
    with open(TMPL_PATH) as f:
        return list(csv.DictReader(f))


def read_json() -> dict:
    with open(JSON_PATH) as f:
        return json.load(f)


def logical_map_tex(r: dict) -> str:
    """LaTeX math for logical geometry; avoid \\t in f-strings (tab escape)."""
    xm = r"\times"
    mk = (r.get("map_kind") or "").strip()
    if mk == "gemm":
        m, n, k = r.get("map_M"), r.get("map_N"), r.get("map_K")
        if m and n and k:
            return f"${m}{xm}{n}{xm}{k}$"
    if mk == "add":
        rr, w = r.get("map_rows"), r.get("map_width")
        if rr and w:
            return f"${rr}{xm}{w}$"
    if mk == "layernorm":
        m, d = r.get("map_M_rows"), r.get("map_D")
        if m and d:
            return f"LN ${m}{xm}{d}$"
    if mk == "softmax":
        nr, rl = r.get("map_num_rows"), r.get("map_row_len")
        if nr or rl:
            return f"${nr or '?'}{xm}{rl or '?'}$"
    return "---"


def _rollup_retire_to_aggregate(metrics: dict) -> None:
    """If JSON predates total_dyn_retired_*, sum per-layer emulator dyn_retired_*."""
    agg = metrics["aggregate_metrics"]
    if any(int(agg.get(f"total_dyn_retired_{b}", 0)) > 0 for b in RETIRED_TABLE_ORDER):
        return
    emus = [
        k
        for k in metrics.get("kernel_metrics", [])
        if k.get("backend") == "emulator"
    ]
    if not emus:
        return
    for b in RETIRED_TABLE_ORDER:
        agg[f"total_dyn_retired_{b}"] = sum(
            int(k.get(f"dyn_retired_{b}", 0)) for k in emus
        )


def emit_latex_tables(metrics: dict, rows: list[dict]) -> None:
    _rollup_retire_to_aggregate(metrics)
    agg = metrics["aggregate_metrics"]
    hist = agg.get("aggregate_sched_slot_histogram", {})
    total = int(agg["total_sched_packets"])
    empty = int(agg["sched_packets_empty"])
    nonempty = int(agg["sched_packets_nonempty"])
    pct_empty = 100.0 * empty / total if total else 0
    slot_all = 100.0 * float(agg["aggregate_static_slot_efficiency"])
    slot_ne = 100.0 * float(agg["aggregate_static_slot_efficiency_nonempty"])
    dyn_slot = 100.0 * float(agg["aggregate_dynamic_slot_efficiency"])
    emu_ins = int(agg.get("total_emu_instructions_retired", 0))

    hdr = (
        "% -*- tex -*- auto-generated by docs/vit_micro/generate_writeup_figures.py\n"
        "% Regenerate:  (from atalla-graph/) python3 docs/vit_micro/generate_writeup_figures.py\n\n"
    )

    retire_lines = [
        "\\begin{table}[H]",
        "\\raggedright",
        "\\small",
        "\\caption{\\textbf{Measured.} Dynamic retired instruction mix (all emulated kernels). "
        "One count per non-\\texttt{nop.s} op; buckets follow \\texttt{misc/instruction\\_retire.py} "
        "(mnemonic-based; not hardware-classified). "
        "Scalar ALU plus scalar GMEM dominate this run; GEMM/systolic is a small fraction.}",
        "\\begin{tabular}{@{}l r r@{}}",
        "\\toprule",
        "Class & Count & \\% of retired ops \\\\",
        "\\midrule",
    ]
    sum_buckets = 0
    for b in RETIRED_TABLE_ORDER:
        cnt = int(agg.get(f"total_dyn_retired_{b}", 0))
        sum_buckets += cnt
        pct = 100.0 * cnt / emu_ins if emu_ins else 0.0
        retire_lines.append(
            f"{tex_escape(RETIRED_DISPLAY[b])} & {cnt:,} & {pct:.2f}\\% \\\\"
        )
    if emu_ins and sum_buckets != emu_ins:
        retire_lines.append(
            "\\multicolumn{3}{p{0.92\\linewidth}}{\\small\\textit{Bucket sum "
            f"({sum_buckets:,}) $\\neq$ retired-op total ({emu_ins:,}); re-run validate.}} \\\\"
        )
    retire_lines.extend(
        [
            "\\midrule",
            f"\\textbf{{Total (retired ops)}} & \\textbf{{{emu_ins:,}}} & 100.00\\% \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\label{tab:retire-mix}",
            "\\end{table}",
        ]
    )
    Path(TABLE_RETIRE_TEX).write_text(hdr + "\n".join(retire_lines) + "\n", encoding="utf-8")
    print(f"wrote {TABLE_RETIRE_TEX}")

    sched_lines = [
        "\\begin{table}[H]",
        "\\raggedright",
        "\\small",
        "\\caption{Static instruction image (aggregated over emulated kernels). "
        "A packet row is four slots. "
        "\\textbf{Empty rows} have no real op before \\texttt{nop.s} fill. "
        "Last row: ISA-weighted FLOPs / SDMA BF16 bytes---\\textit{not} hardware roofline AI.}",
        "\\begin{tabular}{@{}l r@{}}",
        "\\toprule",
        "Metric & Value \\\\",
        "\\midrule",
        f"Total static packet rows & {total:,} \\\\",
        f"Rows with $\\geq 1$ real instruction & {nonempty:,} ({100.0 * nonempty / total:.1f}\\%) \\\\",
        f"\\textbf{{Empty rows}} & \\textbf{{{empty:,}}} "
        f"(\\textbf{{{pct_empty:.1f}\\%}}) \\\\",
        "\\midrule",
        f"Slot fill (all rows) & {slot_all:.2f}\\% \\\\",
        f"Slot fill (non-empty rows only) & {slot_ne:.2f}\\% \\\\",
        f"Dynamic slot fill (retired / dyn.\\ packets) & {dyn_slot:.2f}\\% \\\\",
        "\\midrule",
        f"Total FLOPs (model) & {float(agg['total_flops']):,.0f} \\\\",
        f"DMA bytes loaded & {int(agg['total_bytes_loaded']):,} \\\\",
        f"\\quad SP0 / SP1 loads & {int(agg.get('total_bytes_loaded_sp0', 0)):,} / "
        f"{int(agg.get('total_bytes_loaded_sp1', 0)):,} \\\\",
        f"DMA bytes written & {int(agg['total_bytes_written']):,} \\\\",
        f"\\quad SP0 / SP1 stores & {int(agg.get('total_bytes_stored_sp0', 0)):,} / "
        f"{int(agg.get('total_bytes_stored_sp1', 0)):,} \\\\",
        f"FLOPs / DMA (ld$+$st), model & {float(agg['aggregate_arithmetic_intensity']):.3f} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\label{tab:schedule-aggregate}",
        "\\end{table}",
    ]
    slot_lines = [
        "\\begin{table}[H]",
        "\\raggedright",
        "\\small",
        "\\caption{Histogram: how many real instructions occupy each static packet row (summed across kernels).}",
        "\\begin{tabular}{@{}c r r@{}}",
        "\\toprule",
        "\\# real ops in row & Count & Share \\\\",
        "\\midrule",
    ]
    for k in sorted(hist.keys(), key=lambda x: int(x)):
        c = int(hist[k])
        slot_lines.append(f"${k}$ & {c:,} & {100.0 * c / total:.2f}\\% \\\\")
    slot_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\label{tab:slot-hist}",
            "\\end{table}",
        ]
    )
    Path(TABLE_SCHED_TEX).write_text(
        hdr + "\n".join(sched_lines + ["", ""] + slot_lines) + "\n", encoding="utf-8"
    )
    print(f"wrote {TABLE_SCHED_TEX}")

    layer_lines = [
        "\\begin{flushleft}",
        "{\\footnotesize",
        "\\begin{longtable}{@{}l l c r r r r r r r@{}}",
        "\\caption[Per-node metrics]{Per-node metrics (emulator layers): "
        "Pkt\\% = share of dynamic packets; "
        "$\\eta_{\\mathrm{st}}$ = static slot fill (non-\\texttt{nop.s} / all static slots); "
        "AI$_{\\mathrm{ld}}$ = model FLOPs / SDMA bytes loaded---not hardware AI.\\protect\\footnotemark}"
        "\\label{tab:per-layer}\\\\",
        "\\toprule",
        "Name & Op & Map & Pkt\\% & $P_{\\mathrm{stat}}$ & $\\eta_{\\mathrm{st}}$ "
        "& Bytes & FLOPs & AI$_{\\mathrm{ld}}$ & $\\cos$ \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\caption[]{Per-node metrics (continued)}\\\\",
        "\\toprule",
        "Name & Op & Map & Pkt\\% & $P_{\\mathrm{stat}}$ & $\\eta_{\\mathrm{st}}$ "
        "& Bytes & FLOPs & AI$_{\\mathrm{ld}}$ & $\\cos$ \\\\",
        "\\midrule",
        "\\endhead",
        "\\midrule",
        "\\multicolumn{10}{r}{\\small\\itshape Continued on next page}\\\\",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
    ]
    for r in rows:
        if r.get("backend") != "emulator":
            continue
        name = tex_escape(r["name"])
        op = tex_escape(r["op"])
        mmap = logical_map_tex(r)
        pshare = 100.0 * float(r.get("packet_share_of_emulated") or 0)
        pstat = int(float(r.get("sched_packets") or 0))
        eta = 100.0 * float(r.get("static_sched_slot_efficiency") or 0)
        bmem = int(float(r.get("bytes_memory_total") or 0))
        flops = int(float(r.get("flops_total") or 0))
        ai_ld = float(r.get("ai_loads_only") or 0)
        cosv = float(r.get("cos_sim") or 0)
        layer_lines.append(
            f"{name} & {op} & {mmap} & {pshare:.2f} & {pstat} & {eta:.1f}\\% "
            f"& {bmem} & {flops} & {ai_ld:.2f} & {cosv:.5f} \\\\"
        )
    layer_lines.extend(["\\end{longtable}", "}", "\\end{flushleft}"])
    Path(TABLE_PER_LAYER_TEX).write_text(hdr + "\n".join(layer_lines) + "\n", encoding="utf-8")
    print(f"wrote {TABLE_PER_LAYER_TEX}")


def fig_template_rollup(tmpl_rows: list[dict]) -> None:
    for r in tmpl_rows:
        r["_cum_share"] = float(r["mean_packet_share"]) * int(r["n_layers"]) * 100
    tmpl_rows.sort(key=lambda r: r["_cum_share"], reverse=True)

    labels = [friendly_template_label(r["template_key"]) for r in tmpl_rows]
    cum_shares = [r["_cum_share"] for r in tmpl_rows]
    n_layers = [int(r["n_layers"]) for r in tmpl_rows]
    kinds = [r["map_kind"] for r in tmpl_rows]
    colors = [OP_COLORS.get(k, "#888888") for k in kinds]

    n = len(labels)
    _fs_y = 8.5
    _fs_axis = 9
    fig_h = max(2.75, 0.24 * n + 0.72)
    fig, ax = plt.subplots(figsize=(5.2, fig_h))
    y = np.arange(n)
    ax.barh(
        y,
        cum_shares,
        color=colors,
        edgecolor="white",
        linewidth=0.45,
        height=0.55,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=_fs_y)
    ax.invert_yaxis()
    ax.set_xlabel("Cumulative packet share (%)", fontsize=_fs_axis, labelpad=6)
    ax.set_title("Templates (dynamic packets)", fontsize=_fs_axis, pad=6)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlim(0, max(cum_shares) * 1.16 if cum_shares else 1)
    xmax = max(cum_shares) if cum_shares else 1
    for i, (v, nl) in enumerate(zip(cum_shares, n_layers)):
        ax.text(
            v + xmax * 0.012,
            i,
            f"{v:.1f}% ({nl})",
            va="center",
            fontsize=7.5,
        )
    fig.subplots_adjust(left=0.34, right=0.98, top=0.88, bottom=0.16)
    path = str(OUT_DIR / "fig_template_rollup.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_template_instruction_mix(tmpl_rows: list[dict]) -> None:
    """Stacked horizontal bars: % of retired (non-nop) ops per template."""
    rows = [
        r
        for r in tmpl_rows
        if int(r.get("sum_instructions_executed") or 0) > 0
    ]
    if not rows:
        print("skip fig_template_instruction_mix (no template rows)")
        return
    if not any(
        int(r.get(f"sum_dyn_retired_{b}") or 0)
        for r in rows
        for b in RETIRED_TABLE_ORDER
    ):
        print(
            "skip fig_template_instruction_mix "
            "(regenerate template CSV after dyn_retired_* in run_graph)"
        )
        path = str(OUT_DIR / "fig_template_instruction_mix.pdf")
        fig, ax = plt.subplots(figsize=(5.0, 0.85))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Placeholder: re-run vit_micro validate to populate sum_dyn_retired_*",
            ha="center",
            va="center",
            fontsize=8,
        )
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path} (placeholder)")
        return
    rows.sort(
        key=lambda r: int(r.get("sum_instructions_executed") or 0),
        reverse=True,
    )
    labels = [friendly_template_label(r["template_key"]) for r in rows]
    y = np.arange(len(labels))
    left = np.zeros(len(labels))
    _fs_y = 8.5
    _fs_axis = 9
    fig_h = max(2.55, 0.20 * len(labels) + 0.95)
    fig, ax = plt.subplots(figsize=(5.4, fig_h))
    for i, bucket in enumerate(RETIRED_TABLE_ORDER):
        vals = np.array(
            [
                100.0
                * int(r.get(f"sum_dyn_retired_{bucket}") or 0)
                / max(int(r.get("sum_instructions_executed") or 0), 1)
                for r in rows
            ]
        )
        ax.barh(
            y,
            vals,
            left=left,
            label=RETIRED_DISPLAY[bucket],
            color=RETIRED_STACK_COLORS[i],
            height=0.52,
            edgecolor="white",
            linewidth=0.3,
        )
        left = left + vals
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=_fs_y)
    ax.invert_yaxis()
    # xlabel above legend: padding keeps it clear of the legend block below
    ax.set_xlabel("Retired ops (%)", fontsize=_fs_axis, labelpad=10)
    ax.set_xlim(0, 100)
    ax.set_title("Retire mix by template", fontsize=_fs_axis, pad=6)
    ax.legend(
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        fontsize=7,
        frameon=False,
        columnspacing=0.85,
        handlelength=1.0,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.34, right=0.98, top=0.88, bottom=0.42)
    path = str(OUT_DIR / "fig_template_instruction_mix.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# --- Tiling roadmap figures (schematic; Level-1 AI anchored to measured aggregate) ---

# Relative reuse indices for a representative GEMM-style kernel (not measured per-template yet).
# Rows: Level 1 naive stream, Level 2 scpad-blocked, Level 3 weight-stationary inference.
TILING_REUSE_MATRIX = np.array(
    [
        [1.0, 1.0, 1.0, 1.0],  # A, W, C accum, SP round-trips (higher = more churn)
        [3.0, 3.0, 2.5, 0.42],
        [1.8, 14.0, 4.5, 0.32],
    ]
)

# Memory-hierarchy stress: % of naive-DRAM traffic still hitting each stage (analytical model).
HIERARCHY_STAGE_PCT = np.array(
    [
        [100, 100, 100, 100],
        [35, 35, 100, 100],
        [20, 20, 60, 100],
    ]
)


def fig_tiling_reuse_heatmap() -> None:
    """Relative activation / weight / output reuse vs scratchpad churn (schematic)."""
    data = TILING_REUSE_MATRIX
    rows = [
        r"L1: naive $32\times32$ stream",
        "L2: scpad-blocked submatrix",
        "L3: weight-stationary (inference)",
    ]
    cols = [
        "$R_A$\nact. reuse",
        "$R_W$\nweight reuse",
        "$R_C$\naccum reuse",
        "SP turns\n$\\downarrow$ better",
    ]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=15)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Relative index (L1 = 1)", fontsize=8)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.1f}x", ha="center", va="center", fontsize=9, color="black")
    ax.set_title(
        "Reuse hierarchy (schematic GEMM-centric indices; calibrate per kernel)",
        fontsize=9,
        pad=8,
    )
    fig.tight_layout()
    path = str(OUT_DIR / "fig_tiling_reuse_heatmap.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_memory_hierarchy_stages() -> None:
    """Traffic intensity at DRAM → SDMA edge → scratchpad ↔ VLS (normalized to L1 DRAM)."""
    data = HIERARCHY_STAGE_PCT
    rows = [
        "L1 naive tile stream",
        "L2 scpad-blocked",
        "L3 weight-stationary",
    ]
    cols = ["DRAM\nunique\nbytes", "SDMA\nedge\nbytes", "ScPad\nresident\npressure", "VLS\nmath\nthroughput"]

    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    im = ax.imshow(data, cmap="Blues_r", aspect="auto", vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("% of L1 DRAM stress", fontsize=8)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{int(data[i, j])}%", ha="center", va="center", fontsize=9, color="white" if data[i, j] < 55 else "black")
    ax.set_title(
        "Memory hierarchy stress (model; L1 DRAM = 100%)",
        fontsize=9,
        pad=8,
    )
    fig.tight_layout()
    path = str(OUT_DIR / "fig_memory_hierarchy_stages.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_residency_timeline() -> None:
    """Gantt-style residency: weight vs activation vs accumulator lifetimes (arbitrary time units)."""
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 4.2), sharex=True)
    ylabels = [r"$W$ tile", r"$A$ tile", r"$C$ accum"]
    colors = {"W": "#4C72B0", "A": "#DD8452", "C": "#55A868"}

    scenarios = [
        (
            [(0, 0.9, "W"), (1.0, 0.9, "W"), (2.0, 0.9, "W")],
            [(0, 0.85, "A"), (1.05, 0.85, "A"), (2.1, 0.85, "A")],
            [(0.2, 0.6, "C"), (1.2, 0.6, "C"), (2.2, 0.6, "C")],
        ),
        (
            [(0, 2.2, "W"), (2.4, 2.2, "W")],
            [(0.1, 1.0, "A"), (1.2, 1.0, "A"), (2.3, 1.0, "A"), (3.4, 1.0, "A")],
            [(0.3, 3.8, "C")],
        ),
        (
            [(0, 4.2, "W")],
            [(0.15, 0.7, "A"), (1.0, 0.7, "A"), (1.85, 0.7, "A"), (2.7, 0.7, "A"), (3.55, 0.7, "A")],
            [(0.25, 4.0, "C")],
        ),
    ]
    titles = [
        r"L1: reload $W$ each $K$-step; $A$ streams; short $C$",
        r"L2: larger $W{+}A$ resident; fewer DRAM rounds",
        r"L3: pinned $W$; stream $A$; long-lived $C$ (WS inference)",
    ]

    for ax, (segs_w, segs_a, segs_c), title in zip(axes, scenarios, titles):
        ax.set_title(title, fontsize=8, loc="left", pad=4)
        ax.set_ylim(-0.5, 2.5)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.grid(axis="x", ls=":", alpha=0.35)
        for yi, segs in enumerate([segs_w, segs_a, segs_c]):
            for start, w, k in segs:
                ax.broken_barh(
                    [(start, w)],
                    (yi - 0.35, 0.7),
                    facecolors=colors[k],
                    edgecolor="white",
                    linewidth=0.5,
                )
        ax.set_xlim(0, 5.0)
    axes[-1].set_xlabel("Time (arbitrary units; schematic)", fontsize=8)
    fig.suptitle("Scratchpad / tile residency (conceptual)", fontsize=9, y=1.02)
    fig.tight_layout()
    path = str(OUT_DIR / "fig_residency_timeline.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_ai_tiling_levels(metrics: dict) -> None:
    """Bar chart: measured load-side AI at current stack ≈ L1; L2/L3 from DRAM traffic model."""
    agg = metrics["aggregate_metrics"]
    flops = float(agg.get("total_flops", 0))
    bl = float(agg.get("total_bytes_loaded", 0))
    ai_ld = flops / bl if bl > 0 else 0.0
    # Assume same FLOPs, unique DRAM reads scale with stage-1 column of HIERARCHY_STAGE_PCT / 100
    f2, f3 = HIERARCHY_STAGE_PCT[1, 0] / 100.0, HIERARCHY_STAGE_PCT[2, 0] / 100.0
    ai2 = ai_ld / f2 if f2 > 0 else 0
    ai3 = ai_ld / f3 if f3 > 0 else 0
    labels = ["L1 (current\nmeasure)", "L2 (proj.\n$\\times$DRAM$^{-1}$)", "L3 (proj.\n$\\times$DRAM$^{-1}$)"]
    vals = [ai_ld, ai2, ai3]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(3)
    bars = ax.bar(x, vals, color=["#8c8c8c", "#5a9bd4", "#2e5aac"], edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("FLOPs / DMA byte loaded", fontsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_title(
        "Arithmetic intensity vs tiling level\n(L1 = aggregate emulated; L2/L3 assume Fig.\\,B DRAM scaling)",
        fontsize=8,
        pad=6,
    )
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    path = str(OUT_DIR / "fig_ai_tiling_levels.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_dma_bytes_by_layer(rows: list[dict]) -> None:
    emu = [r for r in rows if r["backend"] == "emulator"]
    emu.sort(key=lambda r: int(float(r.get("bytes_memory_total") or 0)), reverse=True)
    names = [r["name"] for r in emu]
    totals = [int(float(r.get("bytes_memory_total") or 0)) for r in emu]
    ops = [r["op"] for r in emu]
    colors = [OP_COLORS.get(o, "#888888") for o in ops]

    n = len(names)
    _fs_y = 8.5
    _fs_axis = 9
    fig_h = max(2.85, 0.118 * n + 0.58)
    fig, ax = plt.subplots(figsize=(5.2, fig_h))
    y = np.arange(n)
    ax.barh(y, totals, color=colors, height=0.48, edgecolor="white", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=_fs_y)
    ax.invert_yaxis()
    ax.set_xlabel("DMA bytes (ld+st, BF16)", fontsize=_fs_axis, labelpad=6)
    ax.set_title("DMA by emulated node", fontsize=_fs_axis, pad=6)
    ax.tick_params(axis="x", labelsize=8)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.88, bottom=0.16)
    path = str(OUT_DIR / "fig_dma_bytes_by_layer.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def print_flop_hints(metrics: dict) -> None:
    kms = [k for k in metrics.get("kernel_metrics", []) if k.get("backend") == "emulator"]
    total = sum(float(k.get("flops_total") or 0) for k in kms)
    if total <= 0:
        return
    by_name = {k["name"]: float(k.get("flops_total") or 0) for k in kms}
    ln = by_name.get("layer_norm", 0) + by_name.get("layer_norm_1", 0)
    ff = by_name.get("matmul_5", 0) + by_name.get("matmul_6", 0)
    print(
        f"[stats] aggregate FLOPs emulated: {total:.0f}; "
        f"LayerNorm pair: {ln:.0f} ({100 * ln / total:.1f}%); "
        f"FFN pair (matmul_5+6): {ff:.0f} ({100 * ff / total:.1f}%)"
    )


if __name__ == "__main__":
    rows = read_layer_csv()
    tmpl = read_template_csv()
    metrics = read_json()
    print_flop_hints(metrics)
    emit_latex_tables(metrics, rows)
    fig_template_rollup(tmpl)
    fig_template_instruction_mix(tmpl)
    fig_dma_bytes_by_layer(rows)
    print("done (tables + figures 1--3)")
