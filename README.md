# atalla-models

End-to-end pipeline: PyTorch model → AtallaC → Atalla assembly → **atalla-functional-sim** emulator.

## Upstream sources

Two **Git submodules** plus an in-tree graph frontend (not a submodule):

| Path | Upstream | Branch |
|------|----------|--------|
| `aihw-ppci-compiler/` | [Purdue-SoCET/aihw-ppci-compiler](https://github.com/Purdue-SoCET/aihw-ppci-compiler) | `atalla-models` |
| `functional_sim/` | [Purdue-SoCET/atalla-functional-sim](https://github.com/Purdue-SoCET/atalla-functional-sim) | `main` |
| `atalla-graph/` | Based on [vihaanrc/atalla-graph](https://github.com/vihaanrc/atalla-graph); evolved in this repo | — |

Graph/compiler/emulator deltas are summarized in `pipeline_reference.md` §9.

## Clone and submodules

```bash
git clone --recurse-submodules git@github.com:Purdue-SoCET/atalla-models.git
cd atalla-models
git submodule update --init --recursive
```

You need **`functional_sim`** and **`aihw-ppci-compiler`** populated for validate mode. Nested submodules are not required for the graph path.


## Pipeline flow

```
PyTorch nn.Module
    |  torch.fx.symbolic_trace + ShapeProp + lower_linear_modules
    v
FX graph (normalized ops: conv, matmul, relu, softmax, maxpool, add, …)
    |  tile_planner.py
    v
Tiled FX graph
    |  c_emitter.py + kernels/*.py
    |  ├─ AtallaC source per node
    |  └─ DRAMWriter: config table + tensors as .data section
    v
AtallaC .c ─────────────────────────── DRAMWriter (data section)
    |  ppci atalla_cc -S                      |
    v                                         |
ppci .s                                       |
    |  functional_sim/build_compiler.compile_asm()
    |  (VLIW scheduling + encoding)           |
    v                                         |
instruction section ──── + ────── data section
                         |
                    render_testfile()
                         |
                         v
                 .in file (complete)
                         |
                    functional_sim (emulator)
                         |
                         v
                   layer output → activation_cache
```

## Directory layout

```
atalla-models/
    atalla-graph/
        graph/                 fx_capture.py, tile_planner.py (+ upstream export/lower/allocator)
        codegen/               c_emitter.py, dram_builder.py
        kernels/               AtallaC generators: gemm, relu, softmax, maxpool, add
        model/                 basic.py, alexnet.py, alexnet_small.py
        scripts/               generate_schedule.py
        run_graph.py           validate | schedule | both
    functional_sim/            Submodule (emulator + build.py + build_compiler + build_*.py)
    aihw-ppci-compiler/        Submodule (ppci / atalla_cc)
    pipeline_reference.md      Technical deep-dive
    CONTRIBUTING.md            How to add a kernel
```

## Usage

```bash
cd atalla-graph

python run_graph.py --model basic --mode validate
python run_graph.py --model alexnet_small --mode validate --scale 0.01


python run_graph.py --model basic --mode schedule --out-dir out/graph

python run_graph.py --model alexnet_small --mode validate --metrics-json metrics_alexnet.json
```

Models: `basic`, `alexnet_small`

## Validation snapshot (representative)

| Model | Emulated | NumPy | Passthrough | End-to-end cos sim |
|-------|----------|-------|-------------|-------------------|
| BasicModule (dim=32, depth=2) | 9 | 1 (`mul`) | 1 | ≈ **0.99999** |
| AlexNetSmall (`scale=0.01`) | 21 | 0 | 4 | ≈ **0.996** |


## Performance metrics (poster / roofline)


```bash
cd atalla-graph
python run_graph.py --model basic --mode validate --metrics-json metrics_basic.json
python run_graph.py --model alexnet_small --mode validate --metrics-json metrics_alexnet.json
```

Each emulated layer gets a row in `kernel_metrics`; the file also contains `aggregate_metrics` over **all** emulated layers. Tables below used **fixed seeds** (`torch`/`numpy` 42), **`scale=0.01`** for AlexNetSmall, current `main` submodules.

### Metrics from Akshath

| Metric | Meaning |
|--------|---------|
| **Static VLIW slot efficiency (scheduled)** | `sched_slots_filled / (sched_packets × 4)` from `build_compiler` packets. Measures how full each **4-slot** issue row is in the **static** `.in` program. |
| **Non-empty packet efficiency** | Same numerator; denominator uses only packets with **≥1** non-`nop` issue (excludes scheduler padding rows). Reported as `aggregate_static_slot_efficiency_nonempty`. |
| **Dynamic slot efficiency** | `instructions_retired / (emu_packets × 4)` from the functional model (counts what actually retired). Loops / control flow differ from static schedule. |
| **Bytes loaded / written** | BF16 traffic counted on **SDMA loads** (`scpad.ld`) and **stores** (`scpad.st`) in the emulator (2 bytes per element moved). Does not yet count every scalar/GMEM path. |
| **FLOPs (`flops_total`)** | Emulator counters: vector ALU + matmul contributions (approximate; use for relative comparisons). |
| **Arithmetic intensity** | `flops_total / (bytes_loaded + bytes_written)` after the layer run (“roofline-style” bytes moved for modeled DMA). **`arithmetic_intensity_loads`** uses **loads only** in the denominator. |
| **`sched_slot_histogram`** | Count of scheduled packets by **number of filled slots** (0–4). Explains padding-heavy schedules. |

### What these enable

- **Kernel comparison:** same op with handwritten `build_*.py` vs ppci output — compare static slot efficiency and bytes moved.
- **Tiling comparison:** baseline vs pipelined / different tile sizes — watch total bytes and intensity.
- **Per-layer insight:** e.g. large **conv** tiles should show high byte traffic; **matmul** depth shows in FLOPs vs bytes.

### Full model aggregates

| Model | Emul. layers | Static slot η (all pkts) | Static slot η (non-empty) | Dynamic slot η | Bytes L / W | Total bytes | FLOPs (cnt) | AI (FLOP/B) |
|-------|--------------|-------------------------|----------------------------|----------------|-------------|-------------|-------------|-------------|
| BasicModule | 9 | 0.037 | **0.269** | 0.063 | 7168 / 576 | 7744 | 1528 | 0.20 |
| AlexNetSmall (`scale=0.01`) | 21 | 0.043 | **0.271** | 0.097 | 94196 / 9276 | 103472 | 85559 | 0.83 |

### Per-layer samples (same runs)

| Node (model) | Op | Static slot η | Bytes load / store | FLOPs | AI (FLOP/B) |
|--------------|-----|---------------|--------------------|-------|-------------|
| `matmul` (Basic) | matmul | 0.042 | 2176 / 64 | 278 | 0.12 |
| `relu` (Basic) | relu | 0.051 | 64 / 64 | 159 | 1.24 |
| `add` (Basic) | add | 0.027 | 128 / 64 | 94 | 0.49 |
| `conv2d` (AlexNetSmall) | conv | 0.042 | 59072 / 2048 | 50504 | 0.83 |


## Design notes

- **Compiler path for compute ops.** Conv, Linear, Matmul, ReLU, Softmax, MaxPool, Add go through AtallaC → ppci → `build_compiler` → emulator. **`mul`** on the basic module may still use a NumPy fallback unless a kernel is added.
- **Passthrough ops** (flatten, transpose, dropout at inference): reshapes only — no `.in` file.
- **Graph owns `.data`.** `DRAMWriter` (ADDR_TABLE at `0x3C`) + `render_testfile()`; see `pipeline_reference.md` §6.2.
- **Per-layer emulator run.** Fresh emulator state per layer; `activation_cache` passes BF16-aligned activations.
- **Stack / DRAM.** Scalar `x2` placed above tensor data to avoid overlap with spills and globals.

See **`CONTRIBUTING.md`** to add a new op end-to-end.
