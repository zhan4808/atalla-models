# atalla-models

End-to-end pipeline: PyTorch model → AtallaC → Atalla assembly → functional simulator.

## Setup

A vendored copy of `aihw-ppci-compiler` (branch `atalla_arch_emul_robert`) is included at
`atalla-models/aihw-ppci-compiler/`. No sibling clone is required.

To point at an external compiler instead:

```bash
export ATALLA_COMPILER_PATH=/path/to/aihw-ppci-compiler
```

## Pipeline Flow

```
PyTorch nn.Module
    |  torch.fx.symbolic_trace + ShapeProp
    v
FX Graph
    |  tile_planner.py
    v
Tiled FX Graph
    |  c_emitter.py -- AtallaC per node + DRAMWriter tensor data
    v
AtallaC .c
    |  ppci atalla_cc -S
    v
ppci .s
    |  build_compiler.compile_asm() -- notation conversion + scheduling + encoding
    v
.in test file (instruction packets + .data tensor payloads)
    |  functional_sim run.py
    v
Emulator output
```

## Directory Layout

```
atalla-models/
    atalla-graph/
        graph/             fx_capture.py, tile_planner.py, remove_ops.py
        codegen/           c_emitter.py (primary), asm_emitter.py (reference)
        model/             basic.py, alexnet.py
        run_model.py       orchestrator + per-kernel metrics
        collect_metrics.py per-kernel and end-to-end metrics collection
    functional_sim/
        src/               functional_sim.py (emulator core), components/, misc/
        build.py           DRAMWriter, render_testfile
        build_*.py         standalone kernel generators (used for testing, not by pipeline)
        run.py
        tests/
    aihw-ppci-compiler/    vendored compiler (atalla_arch_emul_robert branch)
        ppci/arch/atalla/  compiler backend
        emulator/          build_compiler.py: scheduler + encoder
        atalla_cc/         AtallaC frontend
        atalla_tests/      reference C programs
    PIPELINE_TECHNICAL_REFERENCE.md   full architecture + metrics documentation
```

## Usage

```bash
cd atalla-graph
python run_model.py --model basic
python run_model.py --model alexnet --scale 0.01
python collect_metrics.py   # full per-kernel metrics for both models
```

## Key Design Decisions

**C compiler path for all compute ops.** ReLU, Softmax, GEMM, Conv, and Linear generate
AtallaC via `c_emitter.py`, compiled through ppci to `.s`, then scheduled and encoded by
`build_compiler.compile_asm()` from `aihw-ppci-compiler/emulator/`. Non-compute ops
(MaxPool, elementwise add/mul, adaptive avg pool) fall back to NumPy.

**`build_compiler.compile_asm()` replaces the legacy `asm_converter.py`.** Handles notation
conversion (underscore to dot mnemonics, symbolic to $N registers), hazard scheduling, and
binary encoding in one pass. Supports updated ISA instruction formats (register-based
scpad_ld/st, 5-arg vreg_ld/st, sac on VV ops).

**DRAMWriter data section.** `c_emitter.py` writes tensor data (weights, inputs, outputs)
to a `DRAMWriter` keyed by byte address. `render_in_file()` merges instruction packets with
the `.data` section into a single `.in` file.

**Per-layer execution.** Each layer is a standalone emulator invocation. Activations pass
between layers via the Python orchestrator. Stack pointer (`x2`) is set dynamically above
all DRAM data to prevent compiler stack frames from corrupting tensor data.

## Validation Results

| Model | Emulated | NumPy | Passthrough | Cycles | Instructions | Final cos sim |
|-------|----------|-------|-------------|--------|-------------|---------------|
| BasicModule (dim=32, depth=2) | 5 | 4 | 0 | 5,582 | 3,846 | 0.868 |
| AlexNetSmall (scale=0.01) | 15 | 3 | 1 | 182,736 | 116,227 | 0.198 |

Both models produce **zero NaN** values. Cosine similarity degradation vs float32 reference
is expected BF16 accumulation drift (16-bit mantissa). Individual BasicModule kernels achieve
cos=1.0; AlexNet compounds errors through 19 layers. See `PIPELINE_TECHNICAL_REFERENCE.md`
§8 for full per-kernel metrics tables.

## Compiler Status

| Issue | Status |
|-------|--------|
| Notation mismatch (mnemonics, registers) | Fixed via build_compiler.compile_asm() |
| sac operand missing from VV instructions | Fixed in atalla_arch_emul_robert |
| MTS/STM token bit layout swapped | Fixed in atalla_arch_emul_robert |
| halt/nop opcode mismatch | Fixed in atalla_arch_emul_robert |
| Vector spill stores only 1/32 elements | Fixed: ty="VEC" on AtallaVectorRegister |
| rcp.bf not recognized | Fixed in atalla_arch_emul_robert |
| vreg_ld/st 7-arg format in C templates | Fixed in c_emitter.py (now 5-arg) |
| scpad_ld/st 5-arg format in C templates | Fixed in c_emitter.py (now 3-register) |
| No lw.vi compiler intrinsic | Workaround: inline asm `lw_vi` in C source |
| Compiler only colors v1/v2 vector registers | Workaround: inline asm for vector ops |

## Emulator Fixes

| Fix | File | Description |
|-----|------|-------------|
| m0 hardwired to all-ones | functional_sim.py | Patch `mregs.read(0)` → 0xFFFFFFFF |
| gemm.vv computation order | functional_sim.py | `gemm_weights @ v_in + v_acc` (W^T @ v_in) |
| lw.vi weight buffer reset | functional_sim.py | Reset `gemm_weights` between GEMM tiles |
| Stack/DRAM overlap | run_model.py | Dynamic stack base above max DRAM address |
