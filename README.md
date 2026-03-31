# atalla-models

End-to-end pipeline: PyTorch model → AtallaC → Atalla assembly → functional simulator.

## Setup

A vendored copy of `aihw-ppci-compiler` (based on `origin/master` + `isa-fixes` branch) is
included at `aihw-ppci-compiler/`. No sibling clone is required.

## Pipeline Flow

```
PyTorch nn.Module
    |  torch.fx.symbolic_trace + ShapeProp
    v
FX Graph
    |  tile_planner.py
    v
Tiled FX Graph
    |  c_emitter.py + kernels/*.py — AtallaC per node + DRAMWriter tensor data
    v
AtallaC .c
    |  ppci atalla_cc -S
    v
ppci .s
    |  build_compiler.compile_asm() — VLIW scheduling + encoding
    v
.in test file (instruction packets + .data tensor payloads)
    |  functional_sim
    v
Emulator output
```

## Directory Layout

```
atalla-models/
    atalla-graph/
        graph/             fx_capture.py, tile_planner.py, remove_ops.py
        codegen/           c_emitter.py (orchestrator), asm_converter.py, dram_builder.py
        kernels/           AtallaC kernel generators (gemm.py, relu.py, softmax.py, maxpool.py)
        model/             basic.py, alexnet.py
        run_model.py       orchestrator + per-kernel metrics
        collect_metrics.py per-kernel and end-to-end metrics collection
    functional_sim/
        src/               functional_sim.py (emulator core), components/, misc/
        build.py           DRAMWriter, render_testfile, assembler
        build_compiler.py  compile_asm(): VLIW scheduling + encoding for compiler output
        _asm_encoding.py   instruction encoding library (shared with build_compiler)
        build_*.py         standalone kernel generators (team reference, not used by pipeline)
        archive/           superseded conv scripts
        run.py             standalone emulator entry point
        tests/
    aihw-ppci-compiler/    vendored compiler (master + isa-fixes)
        ppci/arch/atalla/  compiler backend
        atalla_cc/         AtallaC frontend
        atalla_tests/
            kernels/       reference .c kernels + validation harness
            compile_and_convert.py   CLI: .c → ppci → build_compiler → .in
    PIPELINE_TECHNICAL_REFERENCE.md
```

## Usage

```bash
cd atalla-graph
python run_model.py --model basic
python run_model.py --model alexnet --scale 0.01
python collect_metrics.py
```

## Key Design Decisions

**C compiler path for all compute ops.** ReLU, Softmax, GEMM, Conv, Linear, and MaxPool
generate AtallaC via kernel generators in `atalla-graph/kernels/`, compiled through ppci
to `.s`, then scheduled and encoded by `build_compiler.compile_asm()`. MaxPool vertical
max is on-chip; horizontal stride-select is post-processed in Python.

**Kernel generators as a package.** Each kernel type (GEMM, ReLU, Softmax, MaxPool) has
its own generator in `atalla-graph/kernels/`. `c_emitter.py` calls these with per-layer
parameters. Reference `.c` files for fixed sizes live in `atalla_tests/kernels/`.

**`build_compiler.py` in `functional_sim/`.** Extended from Sahil's original to handle
ppci's 3-register SDMA and 5-arg vreg formats. `_asm_encoding.py` provides the instruction
encoding library. `compile_and_convert.py` is a CLI wrapper around the same engine.

**Per-layer execution.** Each layer is a standalone emulator invocation. Activations pass
between layers via the Python orchestrator. Stack pointer (`x2`) is set dynamically above
all DRAM data to prevent compiler stack frames from corrupting tensor data.

## Validation Results

| Model | Emulated | NumPy | Passthrough | Cycles | Instructions | Final cos sim |
|-------|----------|-------|-------------|--------|-------------|---------------|
| BasicModule (dim=32, depth=2) | 5 | 4 | 0 | 5,582 | 3,846 | 0.868 |
| AlexNetSmall (scale=0.01) | 18 | 0 | 1 | ~183K | ~116K | 0.086 |

All compute ops (including MaxPool) are emulated. Zero NaN values. Cosine similarity
degradation is expected BF16 accumulation drift. See `PIPELINE_TECHNICAL_REFERENCE.md` for
full per-kernel metrics.

## Compiler Status

| Issue | Status |
|-------|--------|
| Notation mismatch (mnemonics, registers) | Fixed via build_compiler.compile_asm() |
| sac operand on VV instructions | Fixed in isa-fixes branch |
| MTS/STM token bit layout | Fixed in isa-fixes branch |
| halt/nop opcode mismatch | Fixed in isa-fixes branch |
| Vector spill stores only 1/32 elements | Fixed on master (STRVEC/LDRVEC `31, 0` dims) |
| rcp.bf not recognized | Fixed on atalla_arch_emul_robert |
| vreg_ld/st format in C templates | Fixed (now 5-arg, handled by build_compiler) |
| scpad_ld/st format in C templates | Fixed (now 3-register, handled by build_compiler) |
| sqrti_vi / div_vs / shift_vi | Not yet supported in ppci ISA definitions |
| stbf.s / bfts.s emulator semantics | Emulator bug: treats IEEE hex as plain int |

## Emulator Fixes

| Fix | File | Description |
|-----|------|-------------|
| m0 hardwired to all-ones | functional_sim.py | Patch `mregs.read(0)` → 0xFFFFFFFF |
| gemm.vv computation order | functional_sim.py | `gemm_weights @ v_in + v_acc` (W^T @ v_in) |
| lw.vi weight buffer reset | functional_sim.py | Reset `gemm_weights` between GEMM tiles |
| Stack/DRAM overlap | run_model.py | Dynamic stack base above max DRAM address |
