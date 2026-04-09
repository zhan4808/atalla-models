# Handoff for compiler / codegen reviewers

This complements **`correctness_timeline.md`** (full journey). Use this file to see **what changed in the PPCI / Atalla backend** vs what stayed in the **Python graph emitter**.

## Repos / trees to review

| Area | Primary path (this work) | Notes |
|------|---------------------------|--------|
| AtallaC → IR → DAG → asm | `atalla-models/aihw-ppci-compiler/` | Default compiler for `atalla-graph` (`ATALLA_COMPILER_PATH`). |
| Mirror / standalone checkout | `aihw-ppci-compiler/` (repo root) | Keep in sync with `atalla-models/aihw-ppci-compiler` if you maintain both. |
| Graph, DRAM, golden scripts | `atalla-models/atalla-graph/` | Not the compiler; emits C + runs emulator. |

## Compiler-relevant changes (maxpool / masked vector ops)

1. **Masked merge semantics**  
   Masked `add.vv` effectively merges into `vd` (inactive lanes keep the **previous** `vd`). If `vd` is not the running “best” before the op, results are wrong.

2. **`merge_base` (AtallaC assignment)**  
   For `lhs = vec_op_masked('+', …)` on vectors, the frontend passes **`merge_base`** = loaded value of `lhs` so the backend can seed `vd` before the masked add.

   - `ppci/lang/atalla_c/codegenerator.py` — special-case in `gen_binop` for `=` + `VecOpMasked` `+`.
   - `ppci/irutils/builder.py` — `emit_vec_op_masked(..., merge_base=…)`.

3. **Selection DAG** (`ppci/codegen/irdag.py`)  
   When `merge_base` is set on `VecOpMasked` **ADD**: phi-style **`MOV(ty, mb, value=shared_vreg)`** (no extra data output on MOV → avoids `MOVVEC(MOVVEC(…))` in tree splitting), **MEMORY** edge into the ADD for ordering, **`add_n.value = shared`** so destination is explicit.

4. **Instruction selection**  
   - `ppci/arch/atalla/vector_instructions.py` — `patt_add_vv`: if `tree.value` is set, use it as **`d`** for `AddVv(d, …)` (same register as seed MOV).
   - `ppci/codegen/instructionselector.py` (and `registerallocator.py` `MiniCtx`) — **`move(dst, src)`** no-op when `dst is src` (avoids redundant move after in-place add).

5. **Register allocation**  
   - `ppci/codegen/interferencegraph.py` — for `add_vv` / `sub_vv` / `mul_vv`, interference among **`vd`, `vs1`, `vs2`** so they are not coalesced into one vreg (stops bad `vd == vs2` style patterns).

6. **`AddVv` liveness**  
   - `ppci/arch/atalla/vector_instructions.py` — **`AddVv`**: `vd_reads_old=True` (and paired with **`merge_base`** seeding where the IR assigns into an existing logical “best”).

**Review question for compilers:** Is the `MOV` + `MEMORY` ordering + shared `vreg` + `tree.value` pattern the right way to express “seed then masked add in-place,” or would you prefer a dedicated IR opcode?

## Not a compiler bug (graph Python)

- **`atalla-graph/codegen/c_emitter.py` `emit_conv`**: Bias for `F.conv2d` was skipped when FX `get_attr` nodes lacked `meta["val"]`. Fixed by **`getattr(gm, …)`** fallback for bias (same pattern as weights). Compiler team may still want to know FX graphs often omit `meta["val"]` on `get_attr`.

## How to validate (after pulling)

From `atalla-graph` root:

```bash
python scripts/run_operator_regressions.py
python scripts/debug_maxpool_golden.py --case all --seed 0
```

Compiler-only sanity: build any AtallaC tests you already use for vector + masked ops.

## What we are *not* asking the compiler team to own

- Emulator / functional_sim numeric details (unless a harness proves a spec mismatch).
- `maxpool_post` horizontal indexing in Python (separate from vertical masked merge).
- Performance / slot efficiency (follow-up; see `correctness_timeline.md`).

## Suggested commit split (for the person staging)

1. **PPCI / backend** — `irdag`, `builder`, `codegenerator`, `interferencegraph`, `vector_instructions`, `instructionselector`, `registerallocator` (as applicable).
2. **atalla-graph** — `c_emitter` bias fix, golden scripts, `run_operator_regressions.py`, `capture_regression_baselines.py`, docs.
3. Optionally **do not commit** large `out/` trees or generated `.json` baselines; regenerate with `scripts/capture_regression_baselines.py` or extend `.gitignore` for `out/`.
