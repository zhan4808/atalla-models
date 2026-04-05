# Contributing: how to add a new kernel

This guide walks through adding a compute kernel to the Atalla pipeline: standalone checks first, then AtallaC reference, then parameterized generator, then graph wiring.

## Prerequisites

- Clone **with submodules** (`functional_sim`, `aihw-ppci-compiler`). See `README.md`.
- Python env with **PyTorch**, **NumPy**.
- From repo root, emulator code lives in **`functional_sim/`** (submodule); graph code in **`atalla-graph/`**.

## Architecture (four layers)

```
Layer 1: functional_sim/build_*.py     — hand-written asm + DRAMWriter; run on emulator
Layer 2: aihw-ppci-compiler/atalla_tests/kernels/*.c — fixed-shape reference AtallaC
Layer 3: atalla-graph/kernels/*.py   — parameterized AtallaC strings for arbitrary shapes
Layer 4: atalla-graph/codegen/c_emitter.py — emit_* + DRAMWriter + graph plumbing
```

Work bottom-up: validate Layer 1 on the emulator, then promote.

## Step 0: Study `relu`

| Layer | File | Role |
|-------|------|------|
| 1 | `functional_sim/build_relu.py` | Small BF16 tile, mask+ReLU, `render_testfile` |
| 2 | `aihw-ppci-compiler/atalla_tests/kernels/relu.c` | Same idea in AtallaC |
| 3 | `atalla-graph/kernels/relu.py` | `relu_c(total, width)` generator |
| 4 | `c_emitter.py` → `emit_relu()` | DRAM + `relu_c()` + output addr |

Also read **`functional_sim/ASSEMBLY_SYNTAX.md`** for current **`scpad.ld` / `scpad.st`** (three scalar operands + packed metadata) and **`vreg.ld` / `vreg.st`** (five operands: vd/vs, base, row reg, `num_cols`, `sid`).

## Step 1: Ground-truth asm (`functional_sim/build_<op>.py`)

Hand-write a minimal kernel and DRAM image using **`build.py`**: `DRAMWriter`, `assemble_file` (or scheduled path via `build_compiler.compile_asm` for multi-slot packets), `emit_test_format`, `render_testfile`.

**Do not copy decade-old 5-operand `scpad.ld` text** from informal examples. Follow **`build_gemms.py`**, **`build_relu.py`**, or **`emit_sdma_metadata_asm()`** in `build.py` for SDMA setup, and **`ASSEMBLY_SYNTAX.md`** for mnemonics.

Sketch:

```python

import numpy as np
from build import DRAMWriter, assemble_file, emit_test_format, render_testfile

# 1) asm: use li_s / scpad.ld rs1, rs2, rs3 / vreg.ld vd, rs1, rs2, ncols, sid / …
# 2) DRAMWriter: ADDR_TABLE at 0x3C + bf16 payload
# 3) instrs = assemble_file(asm)
# 4) final = render_testfile(emit_test_format(instrs), img.render_data_mem())
```

Run:

```bash
cd functional_sim
python build_myop.py -o tests/myop.in
python run.py --input_file tests/myop.in
```

Compare emulator `.out` regions against NumPy.

## Step 2: Reference C (`aihw-ppci-compiler/atalla_tests/kernels/<op>.c`)

Use intrinsics consistent with **`atalla-graph/kernels/*.py`** and the compiler README.

| Intrinsic | Role |
|-----------|------|
| `scpad_load(sp, gmem, sdma_ctl)` | DMA DRAM → scratchpad (third arg is packed metadata, often via `li_s`) |
| `scpad_store(sp, gmem, sdma_ctl)` | Scratchpad → DRAM |
| `vector_load(sp_row, row_idx, num_cols_m1, sid)` | Load one row into a `vec` |
| `vector_store(v, sp_row, row_idx, num_cols_m1, sid)` | Store `vec` to scratchpad row |
| `vec_op_masked(...)`, `make_mask(...)` | Masked element ops |
| `load_weights(v)` | `lw.vi` path into systolic |
| `gemm(...)` | MAC into accumulator vector |
| `sqrt(x)` | Scalar BF16 sqrt (where supported) |

**ADDR_TABLE** base is **`0x3C` (decimal 60)** — same constant as `ADDR_TABLE` in `atalla-graph/kernels/common.py`.

**Packed SDMA control:** use `sdma_ctl_val` / `sdma_ctl_expr` from **`atalla-graph/kernels/common.py`** (or mirror the integer in `asm("li_s …")` in standalone C).

Compile:

```bash
cd aihw-ppci-compiler
./atalla_cc -S atalla_tests/kernels/myop.c
```

Schedule / encode the `.s`:

```bash
cd ../functional_sim
python -c "
import build_compiler as bc
s = open('../aihw-ppci-compiler/atalla_tests/kernels/myop.s').read()
text, _, _ = bc.compile_asm(s)
print(text[:800])
"
```

## Step 3: Generator (`atalla-graph/kernels/<op>.py`)

Return AtallaC source as a string. Match **argument order** used in **`gemm.py` / `relu.py`** for `vector_load` / `vector_store` / `scpad_load`.

Example shape (adjust ops and SDMA vars):

```python
# atalla-graph/kernels/myop.py
import math
from kernels.common import ADDR_TABLE, TILE, sdma_ctl_expr

def myop_c(total: int, width: int = 32) -> str:
    rows = math.ceil(total / width)
    w_m1 = width - 1
    sp_rows = min(rows, TILE)
    tile_count = math.ceil(rows / sp_rows)
    tile_bytes = sp_rows * width * 2
    sdma_s = sdma_ctl_expr("sdma_ctl", 0, sp_rows, width, width)

    return (
        "int main() {\n"
        f"    int cfg = {ADDR_TABLE};\n"
        "    int IN_GMEM;\n"
        "    int OUT_GMEM;\n"
        '    asm("lw_s %0, 0(%1)" : "=r"(IN_GMEM)  : "r"(cfg));\n'
        '    asm("lw_s %0, 4(%1)" : "=r"(OUT_GMEM) : "r"(cfg));\n'
        "\n"
        "    int sp = 0;\n"
        "    int ncols = 1;\n"
        f"{sdma_s}"
        "\n"
        "    int tile = 0;\n"
        f"    while (tile < {tile_count}) {{\n"
        "        scpad_load(sp, IN_GMEM, sdma_ctl);\n"
        "\n"
        "        int row = 0;\n"
        f"        while (row < {sp_rows}) {{\n"
        f"            vec v = vector_load(sp, row, {w_m1}, 0);\n"
        "\n"
        "            /* --- your op on v --- */\n"
        "\n"
        f"            vector_store(v, sp, row, {w_m1}, 0);\n"
        "            row = row + 1;\n"
        "        }\n"
        "\n"
        "        scpad_store(sp, OUT_GMEM, sdma_ctl);\n"
        f"        IN_GMEM = IN_GMEM + {tile_bytes};\n"
        f"        OUT_GMEM = OUT_GMEM + {tile_bytes};\n"
        "        tile = tile + 1;\n"
        "    }\n"
        "\n"
        '    asm("halt");\n'
        "    return 0;\n"
        "}\n"
    )
```

Register in **`atalla-graph/kernels/__init__.py`** if you use a central export.

## Step 4: Graph emitter (`codegen/c_emitter.py`)

Add **`emit_myop()`** building `LayerEmission` (`c_source`, `dram`, `output_addr`, shapes, flags). Dispatch from **`emit_node()`**. Map FX op → **`atalla_op`** string in **`graph/fx_capture.py`** (`_OP_MAP` / `_METHOD_MAP` / `_MODULE_MAP`).

## Step 5: Validate

```bash
cd atalla-graph
python run_graph.py --model basic --mode validate
python run_graph.py --model alexnet_small --mode validate

python run_graph.py --model basic --mode validate --metrics-json metrics_basic.json
```

Expect **`emulator`** (not `NumPy`) for your op; check per-node cosine / `max_diff` in verbose logs.
<!--

## Common issues

| Problem | What to check |
|---------|----------------|
| `Unknown mnemonic for scheduling` | `build_compiler.py` aliases / spelling vs `ASSEMBLY_SYNTAX.md` |
| Immediate / offset overflow | `asm("li_s %0, VAL")` or `sdma_ctl_expr` pattern |
| `Tree … not covered` / stack errors | Reduce live `vec` variables; compiler frame limits |
| `cos` ≈ 0 or garbage | BF16 layout, ADDR_TABLE, SDMA metadata, GEMM mask (`mv.stm`), spill vs GMEM in emulator |
| Op stuck as NumPy | `skip_emulator` / missing `emit_*` dispatch / unsupported `atalla_op` |
-->

## File quick reference

| Piece | Location |
|-------|----------|
| Emulator loop | `functional_sim/src/functional_sim.py` |
| Scratchpad / GMEM vector access | `functional_sim/src/components/scpad_ls.py` |
| Scheduler + encoder | `functional_sim/build_compiler.py` |
| DRAM + single-issue asm | `functional_sim/build.py` |
| Opcodes | `functional_sim/src/misc/opcode_table.py` |
| Compiler backend | `aihw-ppci-compiler/ppci/arch/atalla/` |
| AtallaC frontend | `aihw-ppci-compiler/ppci/lang/atalla_c/` |
| Deep dive | `pipeline_reference.md` |
