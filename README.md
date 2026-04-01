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
        graph/             fx_capture.py, tile_planner.py, lower_modules.py, memoryallocator.py
        codegen/           c_emitter.py (orchestrator), dram_builder.py
        kernels/           AtallaC kernel generators (gemm.py, relu.py, softmax.py, maxpool.py, add.py)
        model/             basic.py, alexnet.py
        scripts/           generate_schedule.py (Vihaan's C schedule emitter)
        run_graph.py       unified entry point: validate (emulator) or schedule (C codegen)
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
python run_graph.py --model basic --mode validate
python run_graph.py --model alexnet_small --mode validate
python run_graph.py --model basic --mode schedule --out-dir out/basic
```

## Key Design Decisions

**C compiler path for all compute ops.** ReLU, Softmax, GEMM, Conv, Linear, MaxPool, and Add
generate AtallaC via kernel generators in `atalla-graph/kernels/`, compiled through ppci
to `.s`, then scheduled and encoded by `build_compiler.compile_asm()`. MaxPool vertical
max is on-chip; horizontal stride-select is post-processed in Python.

**Kernel generators as a package.** Each kernel type (GEMM, ReLU, Softmax, MaxPool, Add) has
its own generator in `atalla-graph/kernels/`. `c_emitter.py` calls these with per-layer
parameters. Reference `.c` files for fixed sizes live in `atalla_tests/kernels/`.

**`build_compiler.py` in `functional_sim/`.** Extended from Sahil's original to handle
ppci's 3-register SDMA and 5-arg vreg formats. `_asm_encoding.py` provides the instruction
encoding library. `compile_and_convert.py` is a CLI wrapper around the same engine.

**Per-layer execution.** Each layer is a standalone emulator invocation. Activations pass
between layers via the Python orchestrator. Stack pointer (`x2`) is set dynamically above
all DRAM data to prevent compiler stack frames from corrupting tensor data.

## Validation Results

| Model | Emulated | NumPy | Passthrough | Final cos sim |
|-------|----------|-------|-------------|---------------|
| BasicModule (dim=32, depth=2) | 9 | 1 (mul) | 1 | 0.9999 |
| AlexNetSmall (scale=0.01) | 21 | 0 | 4 | 0.969 |

AlexNet runs **fully on-chip** with zero NumPy fallbacks. BasicModule has 1 remaining
NumPy op (`mul` for residual scaling). Cosine degradation is expected BF16 drift.
See `PIPELINE_TECHNICAL_REFERENCE.md` for details and `CONTRIBUTING.md` for how to add kernels.

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
