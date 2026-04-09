# Atalla graph / emulator correctness timeline

Short engineering record of major bugs fixed on the path to reliable
`run_graph` validation (oracle + chained) and operator goldens.

**For compiler / codegen reviewers:** see **`compiler_review_handoff.md`** — scoped list of PPCI paths, review questions, and what lives in Python vs the compiler.

## 1. GEMM RHS / matmul packing

- **Symptom:** Wide-N or layout-sensitive matmuls disagreed with PyTorch BF16.
- **Cause:** Global issue with RHS weight layout in DRAM vs what the emitter
  and kernel expected (`_write_gemm_rhs_weight` / transpose–pack alignment).
- **Fix:** Corrected RHS packing to match the tiled GEMM path; added
  `debug_matmul_golden.py` cases (including small-K and wide-N).

## 2. Small-K / wide-N GEMM corner cases

- **Symptom:** Errors when inner dimension K was below tile width or N was large.
- **Cause:** Tile bounds / stride padding around `K` and `N` (separate from
  transpose confusion).
- **Fix:** Tightened padding and bounds in the GEMM DRAM path; covered by
  harness cases `matmul1` (small K) and `matmul7` (wide N).

## 3. Maxpool masked merge (compiler / RA)

- **Symptom:** Vertical raw wrong (e.g. `all_losers` stuck on wrong “best”);
  `tiny_tie` could hide the bug.
- **Cause:** Masked `add.vv` uses prior `vd` on inactive lanes; codegen/RA
  could leave `vd` aliased or not seeded with the running best before the
  masked merge.
- **Fix:** Interference among `vd`/`vs1`/`vs2`, `merge_base` + seed `MOV`
  ordered before masked `ADD`, and `patt_add_vv` destination when `tree.value`
  is set; `debug_maxpool_golden.py` with BF16-consistent NumPy gold and gates.

## 4. Conv graph path: missing bias on `F.conv2d`

- **Symptom:** Oracle `conv2d_*` layers showed huge `rel_l2` while
  `debug_conv_golden.py` on the same tensors matched PyTorch.
- **Cause:** `emit_conv` read bias only from `node.args[2].meta["val"]`.
  FX `get_attr` nodes (e.g. `conv2.bias`) typically **do not** have `meta["val"]`,
  so bias was skipped and `C_GMEM` was zero-initialized.
- **Fix:** Fall back to `getattr(gm, …)` on `get_attr` bias nodes, same idea as
  weights. **Audit:** `c_emitter.py` only used `meta["val"]` in this conv
  branch; weights already had a getattr fallback.

## Harness → integration workflow

1. If oracle fails for an op but a golden passes, diff **emitter vs harness**
   (layout, bias, addr table, readback).
2. Use `debug_conv_golden.py --graph-node <name>` to replay **exact** graph
   tensors through the standalone path.
3. Run `scripts/run_operator_regressions.py` before large changes.

## Baselines

Capture all four runs (alexnet + vit_micro × oracle + chained) in one step:

`python scripts/capture_regression_baselines.py`

Or manually, from `atalla-graph` root:

`python run_graph.py --model alexnet_small --mode validate --validate-inputs oracle --metrics-json out/regression_baselines/alexnet_small_oracle.json --quiet`

Repeat for `--validate-inputs chained` and for `--model vit_micro`.

## Operator regression suite

`python scripts/run_operator_regressions.py`

Runs matmul (standard / small-K / wide-N with `--gate`), maxpool subset, and conv (`cin2` + graph `conv2d_1` with `--gate`).

## Performance (next phase)

After correctness is locked, primary lever is schedule / packet utilization (static vs dynamic slot efficiency from `aggregate_metrics` in the baseline JSONs). Targets: fuller packets, fewer empty schedule rows, better co-issue on conv/maxpool/relu-heavy graphs.
