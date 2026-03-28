# atalla-models

End-to-end pipeline: **PyTorch model → AtallaC → Atalla assembly → functional simulator**.

## Pipeline Flow

```
PyTorch nn.Module
    │  torch.fx.symbolic_trace + ShapeProp
    ▼
FX Graph (normalized ops: matmul, conv, relu, softmax, maxpool, add, …)
    │  tile_planner.py — assign DRAM addresses, compute tiling
    ▼
Tiled FX Graph
    │  c_emitter.py — generate AtallaC / direct assembly per node
    ▼
AtallaC .c  ──► ppci atalla_cc -S ──► .s assembly
                                          │  asm_converter.py (notation bridge)
                                          ▼
                              Emulator-compatible assembly
                                          │  build.py assemble + render
                                          ▼
                                     .in test file
                                          │  functional_sim run.py
                                          ▼
                                   Emulator output
```

## Directory Layout

```
atalla-models/
├── atalla-graph/          # PyTorch frontend + code generation
│   ├── graph/             # FX capture, op normalization, tile planning
│   │   ├── fx_capture.py        # symbolic_trace + ShapeProp + op mapping
│   │   ├── remove_ops.py        # BN folding, dropout removal
│   │   └── tile_planner.py      # tiling strategy + DRAM layout
│   ├── codegen/           # Code generation backends
│   │   ├── c_emitter.py         # AtallaC / hybrid asm emitter (primary)
│   │   ├── asm_emitter.py       # Direct assembly emitter (reference)
│   │   ├── asm_converter.py     # Compiler → emulator notation converter
│   │   └── dram_builder.py      # Tensor serialization to .in data format
│   ├── model/             # Model definitions
│   │   ├── basic.py             # BasicModule (matmul + relu)
│   │   └── alexnet.py           # AlexNetSmall (scaled-down AlexNet)
│   └── run_model.py       # End-to-end orchestrator
├── aihw-ppci-compiler/    # AtallaC → assembly compiler (ppci-based)
│   ├── ppci/              # Compiler internals (IR, codegen, targets)
│   ├── atalla_cc/         # Atalla C frontend
│   └── atalla_tests/      # Reference AtallaC programs
├── functional_sim/        # Atalla functional simulator
│   ├── src/               # Simulator core (execute, decode, memory, …)
│   ├── build.py           # Assembler + .in file builder
│   ├── build_*.py         # Kernel generators (GEMM, conv, relu, …)
│   ├── run.py             # Simulator entry point
│   └── tests/             # Pre-built .in test files
└── out/                   # Generated outputs and benchmarks
    └── bench/             # Benchmark results + graphs
```

## Usage

### Run a model through the pipeline

```bash
cd atalla-graph

# BasicModule (matmul + add + relu, smallest test)
python run_model.py --model basic

# AlexNet (5 conv, 6 relu, 3 FC, 3 maxpool)
python run_model.py --model alexnet --scale 0.01
```

### Run individual kernels on the simulator

```bash
cd functional_sim

# Build and run a tiled GEMM
python build_gemm_tiled.py
python run.py tests/gemm_tiled.in

# Build and run AlexNet layer-by-layer
python run_alexnet.py
```

## Key Design Decisions

- **Hybrid compilation strategy**: Compute-heavy kernels (GEMM, Conv, Linear, ReLU, Softmax) use direct, proven assembly from `build_*.py` generators rather than C-compiled code. The C compiler path (`c_emitter.py → ppci → asm_converter.py`) is wired and functional but the compiler has a vector register spill bug (stores only 1 of 32 vector elements) that corrupts multi-intrinsic kernels.

- **Assembly notation bridge (`asm_converter.py`)**: The ppci compiler outputs underscore-separated mnemonics (`add_s`, `vreg_ld`) with symbolic registers (`x0`, `v1`, `m3`). The emulator expects dot-separated mnemonics (`add.s`, `vreg.ld`) with `$N` registers and numeric mask values. The converter handles all known differences including operand reordering for `vreg.ld/st` and `scpad.ld/st`.

- **NumPy fallback for non-compute ops**: MaxPool, elementwise add/mul, and adaptive average pooling run in NumPy rather than on the emulator. These ops lack efficient hardware paths and don't benefit from acceleration.

- **Per-layer execution**: Each layer runs as a standalone emulator invocation with its own `.in` file. Activations are passed between layers via the Python orchestrator. This matches the emulator's single-kernel execution model.

- **BF16 throughout**: All weights and activations are quantized to bfloat16 before entering the pipeline. Accumulation drift vs. float32 PyTorch reference is expected and tracked via cosine similarity.

## Compiler Limitations Addressed

| Issue | Resolution |
|-------|------------|
| Compiler emits `_` mnemonics, emulator expects `.` | `asm_converter.py` regex transforms |
| Compiler uses `xN`/`vN`/`mK` registers, emulator uses `$N` | `asm_converter.py` register mapping |
| Compiler emits `halt`, emulator expects `halt.s` | `asm_converter.py` mnemonic remap |
| Compiler emits `nop`, emulator expects `nop.s` | `asm_converter.py` mnemonic remap |
| `vreg.ld/st` operand order differs | `asm_converter.py` operand swap |
| `scpad.ld/st` operand order differs | `asm_converter.py` operand swap |
| No `lw.vi` intrinsic for weight preload | Direct assembly for GEMM kernels |
| Vector spill saves only 1/32 elements | Direct assembly for vector-heavy kernels |
| No `div_vs` in inline assembler | Use `rcp_bf` + masked multiply workaround |
| Compiler emits `.section`/`.align` directives | `asm_converter.py` strips them |

## Validation

Both `BasicModule` and `AlexNetSmall` produce **identical outputs** (cosine similarity = 1.0) between the new `c_emitter` pipeline and the reference `asm_emitter` pipeline across all emulated kernels.
