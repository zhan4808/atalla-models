# atalla-graph

This directory takes a PyTorch FX graph and lowers it into a tiled SDMA + compute schedule (`graph_schedule.c`).

## Main modes

- `schedule`: current path. Builds step plans (`LOAD/COMPUTE/STORE`) and emits C schedule.
- `validate`: older path. Per-node kernel/emulator validation from the previous flow (when tiles handled compute and data transfer)



## Compute kernels currently used by schedule path

- `matmul_kernel`
- `add_kernel`
- `relu_kernel`
- `softmax_kernel`
- `maxpool_kernel`
- `conv_kernel` (stub)

All are treated as compute-only tile kernels. SDMA movement is emitted around them.

## Current kernel assumptions

- `softmax`: currently tile-local softmax. This is only correct when the softmax axis fits inside one tile (no cross-tile reduction).
- `maxpool`: currently assumes one source tile per output tile (no cross-tile gather). This is only correct when pooling windows needed by an output tile do not cross input tile boundaries.
- `conv`: currently lowered in schedule mode to a stub tile-local conv call (no im2col+gemm lowering yet), so it is not complete.

## What happens for unsupported ops

Schedule lowering raises:

`Kernel does not exist for <op> (node <fx_node_name>)`

## Quick run

```bash
python3 atalla-graph/run_graph.py --model basic --mode schedule
python3 atalla-graph/run_graph.py --model basic --mode validate
python3 atalla-graph/run_graph.py --model basic --mode both
```

## Tests

- `atalla-graph/tests/test_op_steps.py`: unit-level operation tests on step execution (`matmul`, `add`, `relu`) with simulated DRAM/SCPAD dicts.
- `atalla-graph/tests/test_schedule_pipeline.py`: end-to-end schedule pipeline check (graph lowering, DRAM allocation/serialization, step generation, simulated DRAM/SCPAD execution, and output vs PyTorch reference).

`softmax` and `maxpool` are intentionally excluded from `test_op_steps.py` for now because current schedule policies are simplifications (tile-local softmax and single-source-tile maxpool).

Run:

```bash
python3 -m unittest -v atalla-graph/tests/test_op_steps.py
python3 -m unittest -v atalla-graph/tests/test_schedule_pipeline.py
```
