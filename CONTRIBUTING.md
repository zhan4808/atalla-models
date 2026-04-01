# Contributing: How to Add a New Kernel

This guide walks through adding a new compute kernel to the Atalla pipeline,
from standalone assembly verification up through the full graph-level integration.

## Architecture Overview

```
Layer 1: functional_sim/build_*.py    — hand-written ASM kernel + DRAM setup, run on emulator
Layer 2: aihw-ppci-compiler/atalla_tests/kernels/*.c  — reference AtallaC kernel (fixed tile size)
Layer 3: atalla-graph/kernels/*.py    — parameterised AtallaC generator (arbitrary tensor shapes)
Layer 4: atalla-graph/codegen/c_emitter.py  — graph-level emit function (wires DRAM + kernel)
```

Each layer builds on the one below. Start at Layer 1 and work up.

## Step 0: Understand the Existing Examples

Look at `relu` — it's the simplest kernel.

| Layer | File | What it does |
|-------|------|--------------|
| 1 | `functional_sim/build_relu.py` | Loads a 4×8 BF16 tile, applies max(x,0) via mask+zero, stores back |
| 2 | `aihw-ppci-compiler/atalla_tests/kernels/relu.c` | Same logic in AtallaC using intrinsics |
| 3 | `atalla-graph/kernels/relu.py` | Generates AtallaC string for arbitrary rows×width with tiling |
| 4 | `c_emitter.py :: emit_relu()` | Packs input tensor into DRAM, calls `relu_c()`, reads output |

## Step 1: Write the Assembly Kernel (`functional_sim/build_<op>.py`)

This is the ground truth. You hand-write Atalla assembly and set up DRAM data.

```python
# functional_sim/build_myop.py
import numpy as np
from build import DRAMWriter, assemble_file, emit_test_format, render_testfile

INPUT_BASE  = 0x00001000
OUTPUT_BASE = 0x00001040

# 1. Write your assembly
asm = """
    lw.s   $3, 0($0)            # input base
    lw.s   $8, 4($0)            # output base
    addi.s $9, $0, 0
    scpad.ld $9, $3, 8, 4, 0    # load tile SP0
    addi.s $255, $0, -1
    mv.stm 1, $255              # mask1 = all lanes

    # --- your operation here ---
    vreg.ld $4, $9, 8, 4, 0, 1, 0
    # ... process rows ...
    vreg.st $1, $9, 8, 4, 0, 1, 0

    scpad.st $9, $8, 8, 4, 0
    halt.s
"""

# 2. Set up DRAM data
tensor = np.random.randn(4, 8).astype(np.float32)
img = DRAMWriter()
img.u32(0x0, INPUT_BASE)
img.u32(0x4, OUTPUT_BASE)
addr = INPUT_BASE
for x in tensor.flatten():
    img.bf16(addr, float(x))
    addr += 2

# 3. Assemble and render
instrs = assemble_file(asm)
final = render_testfile(emit_test_format(instrs), img.render_data_mem())
```

Run:
```bash
cd functional_sim
python build_myop.py -o tests/myop.in
python run.py tests/myop.in
```

Read back the output region from the `.out` file and compare against NumPy.

## Step 2: Write the Reference C Kernel (`aihw-ppci-compiler/atalla_tests/kernels/<op>.c`)

Translate your assembly logic into AtallaC using intrinsics.

**Key intrinsics:**
| Intrinsic | What it does |
|-----------|-------------|
| `scpad_load(sp, gmem_addr, ctl)` | DMA: DRAM → scratchpad |
| `scpad_store(sp, gmem_addr, ctl)` | DMA: scratchpad → DRAM |
| `vector_load(row, ncols, width_m1, sid)` | Load row from scratchpad into vector register |
| `vector_store(v, row, ncols, width_m1, sid)` | Store vector register to scratchpad row |
| `vec_op_masked(op, a, b, mask)` | Element-wise op (`"+"`, `"-"`, `"*"`) under mask |
| `make_mask(cmp, a, b, mask)` | Compare vectors, return mask (`"<"`, `">"`) |
| `load_weights(v)` | Load weight vector from weight buffer |
| `sqrt(x)` | Scalar BF16 square root |

**DRAM config table:** The kernel reads its DRAM pointers from a config table at
address `0x3C` (decimal 60). Convention:
- `[0x3C + 0]`: input address A
- `[0x3C + 4]`: input address B (if binary op) or output address
- `[0x3C + 8]`: output address C (if binary op)

**SDMA control register:** Use inline asm `li_s` to load pre-computed bit-packed
values (the compiler can't handle large immediates directly):
```c
int sdma_ctl;
asm("li_s %0, 133169183" : "=r"(sdma_ctl));
```

Compute the value with `sdma_ctl_val(sid, num_rows, num_cols, full_cols)` from
`atalla-graph/kernels/common.py`.

**Compile test:**
```bash
cd aihw-ppci-compiler
python main.py atalla_tests/kernels/myop.c
```

This produces a `.s` file. Feed it through `build_compiler`:
```bash
cd functional_sim
python -c "
import build_compiler as bc
text, _, _ = bc.compile_asm(open('../aihw-ppci-compiler/atalla_tests/kernels/myop.s').read())
print(text[:500])
"
```

## Step 3: Write the Kernel Generator (`atalla-graph/kernels/<op>.py`)

Parameterise the C kernel for arbitrary tensor sizes. Your generator receives
`total` elements and `width` (capped at 32), and returns AtallaC source as a string.

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
        "    int all_mask = -1;\n"
        "    int ncols = 1;\n"
        f"{sdma_s}"
        "\n"
        "    int tile = 0;\n"
        f"    while (tile < {tile_count}) {{\n"
        "        scpad_load(sp, IN_GMEM, sdma_ctl);\n"
        "\n"
        "        int row = 0;\n"
        f"        while (row < {sp_rows}) {{\n"
        f"            vec v = vector_load(row, ncols, {w_m1}, 0);\n"
        "\n"
        "            /* --- your operation on v --- */\n"
        "\n"
        f"            vector_store(v, row, ncols, {w_m1}, 0);\n"
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

**Register it** in `atalla-graph/kernels/__init__.py`:
```python
from kernels.myop import myop_c
```

## Step 4: Wire into the Graph Emitter (`c_emitter.py`)

Add an `emit_myop()` function in `atalla-graph/codegen/c_emitter.py`:

```python
def emit_myop(node, input_data, tc):
    p = tc.params
    total = p["total_elements"]
    width = min(p["width"], 32)
    rows = math.ceil(total / width)

    flat = input_data.flatten()[:total]
    IN_GMEM = 0x1000
    OUT_GMEM = IN_GMEM + _align_data(rows * width * 2)

    img = DRAMWriter()
    img.u32(ADDR_TABLE + 0, IN_GMEM)
    img.u32(ADDR_TABLE + 4, OUT_GMEM)

    padded = np.zeros(rows * width, dtype=np.float32)
    padded[:len(flat)] = flat
    for i in range(rows * width):
        img.bf16(IN_GMEM + i * 2, float(padded[i]))

    em = LayerEmission()
    em.c_source = myop_c(total, width)
    em.dram = img
    em.output_addr = OUT_GMEM
    em.output_shape = get_node_shape(node) or (total,)
    em.output_elements = total
    return em
```

Then add the dispatch in `emit_node()`:
```python
elif atalla_op == "myop":
    return emit_myop(node, input_data, tc)
```

And map the op in `atalla-graph/graph/fx_capture.py`:
```python
_OP_MAP[F.my_torch_function] = "myop"
```

## Step 5: Validate

```bash
# Full model (basic module has all standard ops):
cd atalla-graph
python run_graph.py --model basic --mode validate

# AlexNet:
python run_graph.py --model alexnet_small --mode validate
```

Look for your op in the output — it should say `emulator` not `NumPy`:
```
[myop_node] myop -> emulator (0x3000, 32 elems)... done (cos=0.9998)
```

## Common Issues

| Problem | Fix |
|---------|-----|
| `Unknown mnemonic for scheduling` | Add alias in `functional_sim/build_compiler.py :: MNEMONIC_ALIASES` |
| `value X cannot be fit into 25 bits` | Use `asm("li_s %0, VAL")` for large immediates; `build_compiler` expands to `lui_s + addi_s` |
| `Tree ... not covered` | Reduce live vector variables; compiler stack frame may exceed 12-bit offset range |
| `cosine_sim = 0.0` (all zeros) | Check BF16 register convention in `functional_sim.py :: bf16_reg_to_f32` |
| Op runs as NumPy | Make sure `emit_myop` does NOT set `em.skip_emulator = True` and returns `em.c_source` |

## Kernel Ownership

| Kernel | Owner | Status |
|--------|-------|--------|
| conv | Robert | Done |
| gemm | Mary | Done |
| softmax | Soumil/Jiayi | Done |
| relu | Soumil/Jiayi | Done |
| sigmoid | Soumil/Jiayi | Done (functional_sim only) |
| maxpool | Robert | Done |
| add | Robert | Done |
| mul | Unassigned | Needed for BasicModule residual |
| layernorm | — | Done (functional_sim only) |

## File Quick Reference

| What | Where |
|------|-------|
| Emulator core | `functional_sim/src/functional_sim.py` |
| VLIW scheduler / encoder | `functional_sim/build_compiler.py` |
| DRAMWriter + assembler | `functional_sim/build.py` |
| Instruction opcodes | `functional_sim/src/misc/opcode_table.py` |
| Compiler backend | `aihw-ppci-compiler/ppci/arch/atalla/` |
| AtallaC frontend | `aihw-ppci-compiler/ppci/lang/atalla_c/` |
| ISA spec | `ISA Atalla Bit Spec Updated.csv` |
| Technical reference | `PIPELINE_TECHNICAL_REFERENCE.md` |
