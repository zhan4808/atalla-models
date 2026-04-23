from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMPORT_ERROR: str | None = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.fx import Node, symbolic_trace
    from torch.fx.passes.shape_prop import ShapeProp

    from graph.fx_capture import normalize_ops
    from graph.lower_modules import lower_linear_modules
    from graph.memoryallocator import TILE_HEIGHT, TILE_WIDTH, allocate_memory, tensor_bytes, tensor_for_node
    from graph.tile_planner import plan_tiles
    from scripts import generate_schedule as gs
    from scripts.tile_manager import MAX_SCPAD_TILES, plan_tile_moves
    from scripts.tile_structures import OpPlan, StepKind

    TORCH_OK = True
except Exception as exc:
    IMPORT_ERROR = str(exc)
    TORCH_OK = False

TILE = 32
TILE_ELEMS = TILE * TILE
TILE_BYTES = TILE_ELEMS * 2


def _bf16_tile_from_blob(blob: bytes, addr: int) -> np.ndarray:
    chunk = blob[addr:addr + TILE_BYTES]
    if len(chunk) < TILE_BYTES:
        chunk = chunk + bytes(TILE_BYTES - len(chunk))
    u16 = np.frombuffer(chunk, dtype="<u2", count=TILE_ELEMS).astype(np.uint32)
    u32 = (u16 << 16).reshape(TILE, TILE)
    return u32.view(np.float32).copy()


def _tensor_from_tiles(shape: List[int], tiles_per_dim: List[int], addrs: List[int], dram_tiles: Dict[int, np.ndarray]) -> np.ndarray:
    if len(shape) == 1:
        cols = int(shape[0])
        out = np.zeros((1, cols), dtype=np.float32)
        col_tiles = int(tiles_per_dim[-1])
        for tc in range(col_tiles):
            idx = tc
            cs = tc * TILE
            c = min(TILE, cols - cs)
            out[0, cs:cs + c] = dram_tiles[addrs[idx]][0, :c]
        return out.reshape(cols)

    if len(shape) == 2:
        rows = int(shape[0])
        cols = int(shape[1])
        out = np.zeros((rows, cols), dtype=np.float32)
        row_tiles = int(tiles_per_dim[-2])
        col_tiles = int(tiles_per_dim[-1])
        idx = 0
        for tr in range(row_tiles):
            for tc in range(col_tiles):
                rs = tr * TILE
                cs = tc * TILE
                r = min(TILE, rows - rs)
                c = min(TILE, cols - cs)
                out[rs:rs + r, cs:cs + c] = dram_tiles[addrs[idx]][:r, :c]
                idx += 1
        return out

    raise ValueError(f"test helper only supports rank-1/rank-2 tensors, got shape={shape}")


def _execute_op_plans(op_plans: List[OpPlan], dram_tiles: Dict[int, np.ndarray]) -> int:
    scpad: Dict[int, np.ndarray] = {}
    slot_owner: Dict[int, str] = {}
    max_slot_seen = -1

    for plan in op_plans:
        tile_info = plan.attrs.get("tile_info", {})
        if not isinstance(tile_info, dict):
            raise AssertionError(f"plan {plan.node_name} missing tile_info")

        for step in plan.steps:
            if step.kind is StepKind.LOAD:
                tid = step.attrs.get("tile_id")
                if not isinstance(tid, str):
                    continue
                if tid not in tile_info:
                    raise AssertionError(f"load tile {tid} missing tile_info")
                info = tile_info[tid]
                slot = int(step.attrs.get("slot", -1))
                if slot < 0:
                    raise AssertionError("LOAD step missing slot mapping")
                max_slot_seen = max(max_slot_seen, slot)
                rows = int(info.get("rows", TILE))
                cols = int(info.get("cols", TILE))
                addr = int(info["addr"])
                src = dram_tiles.setdefault(addr, np.zeros((TILE, TILE), dtype=np.float32))
                tile = np.zeros((TILE, TILE), dtype=np.float32)
                tile[:rows, :cols] = src[:rows, :cols]
                scpad[slot] = tile
                slot_owner[slot] = tid
                continue

            if step.kind is StepKind.COMPUTE:
                if step.op == "matmul":
                    a_tid = str(step.attrs["a_tile_id"])
                    b_tid = str(step.attrs["b_tile_id"])
                    c_tid = str(step.attrs["c_tile_id"])
                    sa = int(step.attrs["slot_a"])
                    sb = int(step.attrs["slot_b"])
                    sc = int(step.attrs["slot_c"])
                    if slot_owner.get(sa) != a_tid or slot_owner.get(sb) != b_tid or slot_owner.get(sc) != c_tid:
                        raise AssertionError("SCPAD slot-owner mismatch for matmul")
                    m_rows = int(step.attrs["m_rows"])
                    n_cols = int(step.attrs["n_cols"])
                    k_cols = int(step.attrs["k_cols"])
                    a = scpad[sa][:m_rows, :k_cols]
                    b = scpad[sb][:k_cols, :n_cols]
                    scpad[sc][:m_rows, :n_cols] += (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)
                    continue

                if step.op == "add":
                    l_tid = str(step.attrs["lhs_tile_id"])
                    r_tid = str(step.attrs["rhs_tile_id"])
                    o_tid = str(step.attrs["out_tile_id"])
                    sl = int(step.attrs["slot_lhs"])
                    sr = int(step.attrs["slot_rhs"])
                    so = int(step.attrs["slot_out"])
                    if slot_owner.get(sl) != l_tid or slot_owner.get(sr) != r_tid or slot_owner.get(so) != o_tid:
                        raise AssertionError("SCPAD slot-owner mismatch for add")
                    rows = int(step.attrs["rows"])
                    cols = int(step.attrs["cols"])
                    scpad[so][:rows, :cols] = scpad[sl][:rows, :cols] + scpad[sr][:rows, :cols]
                    continue

                if step.op == "relu":
                    i_tid = str(step.attrs["in_tile_id"])
                    o_tid = str(step.attrs["out_tile_id"])
                    si = int(step.attrs["slot_in"])
                    so = int(step.attrs["slot_out"])
                    if slot_owner.get(si) != i_tid or slot_owner.get(so) != o_tid:
                        raise AssertionError("SCPAD slot-owner mismatch for relu")
                    rows = int(step.attrs["rows"])
                    cols = int(step.attrs["cols"])
                    scpad[so][:rows, :cols] = np.maximum(scpad[si][:rows, :cols], 0.0)
                    continue

                raise AssertionError(f"unsupported compute op in test executor: {step.op}")

            if step.kind is StepKind.STORE:
                tid = step.attrs.get("tile_id")
                if not isinstance(tid, str):
                    continue
                if tid not in tile_info:
                    raise AssertionError(f"store tile {tid} missing tile_info")
                info = tile_info[tid]
                slot = int(step.attrs.get("slot", -1))
                if slot < 0:
                    raise AssertionError("STORE step missing slot mapping")
                if slot_owner.get(slot) != tid:
                    raise AssertionError("SCPAD slot-owner mismatch for store")
                rows = int(info.get("rows", TILE))
                cols = int(info.get("cols", TILE))
                addr = int(info["addr"])
                dst = dram_tiles.setdefault(addr, np.zeros((TILE, TILE), dtype=np.float32))
                dst[:rows, :cols] = scpad[slot][:rows, :cols]
                dram_tiles[addr] = dst
                continue

            if step.kind is StepKind.RELEASE:
                continue

    return max_slot_seen


if TORCH_OK:
    class SchedulePipelineTests(unittest.TestCase):
        class TinyModel(nn.Module):
            def __init__(self, dim: int = 40):
                super().__init__()
                self.w = nn.Parameter(torch.randn(dim, dim) * 0.1)
                self.b = nn.Parameter(torch.randn(dim) * 0.1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = torch.matmul(x, self.w)
                y = y + self.b
                y = F.relu(y)
                return y

        def test_schedule_and_memory_mappings(self) -> None:
            torch.manual_seed(0)

            model = self.TinyModel(dim=40).bfloat16().eval()
            x = torch.randn(1, 40).bfloat16()
            placeholder_data = {"x": x.clone()}

            with tempfile.TemporaryDirectory() as td:
                gm = symbolic_trace(model)
                gm = lower_linear_modules(gm)
                ShapeProp(gm).propagate(x)
                gm = normalize_ops(gm)
                gm = plan_tiles(gm)

                dram_path = Path(td) / "dram.bin"
                gm_alloc = allocate_memory(gm, str(dram_path), placeholder_data)
                dram_blob = dram_path.read_bytes()

                # DRAM mapping validation: preinitialized tensors serialize exactly.
                for node in gm_alloc.graph.nodes:
                    tensor = tensor_for_node(node, gm_alloc, placeholder_data)
                    if tensor is None or not isinstance(tensor, torch.Tensor):
                        continue
                    if tensor.dtype != torch.bfloat16:
                        continue
                    addr = int(str(node.meta["dram_addr"]), 16)
                    size = int(node.meta["bytes"])
                    expected = tensor_bytes(tensor, size)
                    actual = dram_blob[addr:addr + size]
                    self.assertEqual(expected, actual, msg=f"DRAM payload mismatch at node {node.name}")

                attr_nodes = {
                    n.target: n
                    for n in gm_alloc.graph.nodes
                    if n.op == "get_attr"
                }
                tensor_specs, specs_by_node = gs._collect_tensor_specs(gm_alloc)
                raw_plans = gs._build_op_plans(gm_alloc, specs_by_node, attr_nodes)
                self.assertGreater(len(raw_plans), 0)

                op_plans = [plan_tile_moves(p) for p in raw_plans]
                self.assertTrue(any(p.op_type == "matmul" for p in op_plans))
                self.assertTrue(any(p.op_type == "add" for p in op_plans))
                self.assertTrue(any(p.op_type == "relu" for p in op_plans))

                dram_tiles: Dict[int, np.ndarray] = {}
                for spec in tensor_specs:
                    for addr in spec.tiles:
                        dram_tiles[addr] = _bf16_tile_from_blob(dram_blob, addr)

                max_slot = _execute_op_plans(op_plans, dram_tiles)
                self.assertGreaterEqual(max_slot, 0)
                self.assertLess(max_slot, MAX_SCPAD_TILES)

                output_fx_node = None
                for node in gm_alloc.graph.nodes:
                    if node.op == "output":
                        out_arg = node.args[0]
                        if isinstance(out_arg, Node):
                            output_fx_node = out_arg
                        elif isinstance(out_arg, (tuple, list)):
                            for item in out_arg:
                                if isinstance(item, Node):
                                    output_fx_node = item
                                    break
                        break
                self.assertIsNotNone(output_fx_node)

                out_spec = specs_by_node[output_fx_node]
                got = _tensor_from_tiles(out_spec.shape, out_spec.tiles_per_dim, out_spec.tiles, dram_tiles)
                ref = model(x).detach().float().cpu().numpy()
                np.testing.assert_allclose(got, ref, rtol=2e-2, atol=2e-2)
else:
    class SchedulePipelineTests(unittest.TestCase):
        @unittest.skip(f"torch + graph tooling required ({IMPORT_ERROR})")
        def test_schedule_and_memory_mappings(self) -> None:
            pass


if __name__ == "__main__":
    unittest.main()
