from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.policies import (
    build_add_plan,
    build_matmul_plan,
    build_relu_plan,
)
from scripts.tile_manager import plan_tile_moves
from scripts.tile_structures import OpPlan, StepKind

TILE = 32
TILE_BYTES = TILE * TILE * 2
GRID = 32
BIG = TILE * GRID


def _tiles_for_shape(shape: List[int]) -> List[int]:
    if len(shape) == 1:
        return [(shape[0] + TILE - 1) // TILE]
    if len(shape) == 2:
        return [
            (shape[0] + TILE - 1) // TILE,
            (shape[1] + TILE - 1) // TILE,
        ]
    raise ValueError("Only rank-1/rank-2 tensors are used in these tests")


def _alloc_addrs(count: int, base: int) -> List[int]:
    return [base + i * TILE_BYTES for i in range(count)]


def _alloc_regions(base: int, counts: List[int]) -> List[List[int]]:
    regions: List[List[int]] = []
    cursor = base
    for count in counts:
        regions.append(_alloc_addrs(count, cursor))
        cursor += count * TILE_BYTES + 0x1000
    return regions


def _matrix_to_tiles(mat: np.ndarray, rows: int, cols: int) -> List[np.ndarray]:
    row_tiles = (rows + TILE - 1) // TILE
    col_tiles = (cols + TILE - 1) // TILE
    out: List[np.ndarray] = []
    for tr in range(row_tiles):
        for tc in range(col_tiles):
            tile = np.zeros((TILE, TILE), dtype=np.float32)
            rs = tr * TILE
            cs = tc * TILE
            r = min(TILE, rows - rs)
            c = min(TILE, cols - cs)
            if r > 0 and c > 0:
                tile[:r, :c] = mat[rs:rs + r, cs:cs + c]
            out.append(tile)
    return out


def _tiles_to_matrix(tiles: List[np.ndarray], rows: int, cols: int) -> np.ndarray:
    row_tiles = (rows + TILE - 1) // TILE
    col_tiles = (cols + TILE - 1) // TILE
    out = np.zeros((rows, cols), dtype=np.float32)
    idx = 0
    for tr in range(row_tiles):
        for tc in range(col_tiles):
            rs = tr * TILE
            cs = tc * TILE
            r = min(TILE, rows - rs)
            c = min(TILE, cols - cs)
            out[rs:rs + r, cs:cs + c] = tiles[idx][:r, :c]
            idx += 1
    return out


def _execute_plan(plan: OpPlan, dram_tiles: Dict[int, np.ndarray]) -> None:
    tile_info = plan.attrs.get("tile_info", {})
    if not isinstance(tile_info, dict):
        raise ValueError("Plan missing tile_info")

    scpad: Dict[int, np.ndarray] = {}

    for step in plan.steps:
        if step.kind is StepKind.LOAD:
            tid = step.attrs.get("tile_id")
            if not isinstance(tid, str):
                continue
            info = tile_info[tid]
            slot = int(step.attrs.get("slot", 0))
            rows = int(info.get("rows", TILE))
            cols = int(info.get("cols", TILE))
            addr = int(info["addr"])
            src = dram_tiles[addr]
            tile = np.zeros((TILE, TILE), dtype=np.float32)
            tile[:rows, :cols] = src[:rows, :cols]
            scpad[slot] = tile
            continue

        if step.kind is StepKind.COMPUTE:
            if step.op == "add":
                slot_l = int(step.attrs.get("slot_lhs", 0))
                slot_r = int(step.attrs.get("slot_rhs", 0))
                slot_o = int(step.attrs.get("slot_out", 0))
                rows = int(step.attrs.get("rows", TILE))
                cols = int(step.attrs.get("cols", TILE))
                scpad[slot_o][:rows, :cols] = (
                    scpad[slot_l][:rows, :cols] + scpad[slot_r][:rows, :cols]
                )
                continue

            if step.op == "matmul":
                slot_a = int(step.attrs.get("slot_a", 0))
                slot_b = int(step.attrs.get("slot_b", 1))
                slot_c = int(step.attrs.get("slot_c", 2))
                m_rows = int(step.attrs.get("m_rows", TILE))
                n_cols = int(step.attrs.get("n_cols", TILE))
                k_cols = int(step.attrs.get("k_cols", TILE))
                a = scpad[slot_a][:m_rows, :k_cols]
                b = scpad[slot_b][:k_cols, :n_cols]
                scpad[slot_c][:m_rows, :n_cols] += (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)
                continue

            if step.op == "relu":
                slot_i = int(step.attrs.get("slot_in", 0))
                slot_o = int(step.attrs.get("slot_out", 0))
                rows = int(step.attrs.get("rows", TILE))
                cols = int(step.attrs.get("cols", TILE))
                scpad[slot_o][:rows, :cols] = np.maximum(scpad[slot_i][:rows, :cols], 0.0)
                continue

            raise ValueError(f"Unsupported compute op in test executor: {step.op}")

        if step.kind is StepKind.STORE:
            tid = step.attrs.get("tile_id")
            if not isinstance(tid, str):
                continue
            info = tile_info[tid]
            slot = int(step.attrs.get("slot", 0))
            rows = int(info.get("rows", TILE))
            cols = int(info.get("cols", TILE))
            addr = int(info["addr"])
            dst = dram_tiles[addr]
            dst[:rows, :cols] = scpad[slot][:rows, :cols]
            dram_tiles[addr] = dst
            continue

        if step.kind is StepKind.RELEASE:
            continue


class OperationStepTests(unittest.TestCase):
    def test_add(self) -> None:
        rng = np.random.default_rng(7)
        rows, cols = BIG, BIG
        lhs = rng.standard_normal((rows, cols), dtype=np.float32)
        rhs = rng.standard_normal((rows, cols), dtype=np.float32)
        expected = torch.add(torch.from_numpy(lhs), torch.from_numpy(rhs)).numpy()

        tiles_per_dim = _tiles_for_shape([rows, cols])
        count = tiles_per_dim[0] * tiles_per_dim[1]
        lhs_addrs, rhs_addrs, dst_addrs = _alloc_regions(0x10000000, [count, count, count])

        lhs_tiles = _matrix_to_tiles(lhs, rows, cols)
        rhs_tiles = _matrix_to_tiles(rhs, rows, cols)
        dst_tiles = _matrix_to_tiles(np.zeros((rows, cols), dtype=np.float32), rows, cols)

        dram: Dict[int, np.ndarray] = {}
        for i, addr in enumerate(lhs_addrs):
            dram[addr] = lhs_tiles[i].copy()
        for i, addr in enumerate(rhs_addrs):
            dram[addr] = rhs_tiles[i].copy()
        for i, addr in enumerate(dst_addrs):
            dram[addr] = dst_tiles[i].copy()

        plan = build_add_plan(
            node_name="add_test",
            lhs_tiles=lhs_addrs,
            lhs_shape=[rows, cols],
            lhs_tiles_per_dim=tiles_per_dim,
            rhs_tiles=rhs_addrs,
            rhs_shape=[rows, cols],
            rhs_tiles_per_dim=tiles_per_dim,
            dst_tiles=dst_addrs,
            dst_shape=[rows, cols],
            dst_tiles_per_dim=tiles_per_dim,
            tile_size=TILE,
        )
        plan = plan_tile_moves(plan)
        _execute_plan(plan, dram)

        got_tiles = [dram[a] for a in dst_addrs]
        got = _tiles_to_matrix(got_tiles, rows, cols)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

    def test_matmul(self) -> None:
        rng = np.random.default_rng(11)
        m_rows, k_cols, n_cols = BIG, TILE, BIG
        a = rng.standard_normal((m_rows, k_cols), dtype=np.float32)
        b = rng.standard_normal((k_cols, n_cols), dtype=np.float32)
        expected = torch.matmul(torch.from_numpy(a), torch.from_numpy(b)).numpy()

        m_tiles = GRID
        k_tiles = 1
        n_tiles = GRID
        lhs_addrs, rhs_addrs, dst_addrs = _alloc_regions(
            0x20000000,
            [m_tiles * k_tiles, k_tiles * n_tiles, m_tiles * n_tiles],
        )

        lhs_tiles = _matrix_to_tiles(a, m_rows, k_cols)
        rhs_tiles = _matrix_to_tiles(b, k_cols, n_cols)
        dst_tiles = _matrix_to_tiles(np.zeros((m_rows, n_cols), dtype=np.float32), m_rows, n_cols)

        dram: Dict[int, np.ndarray] = {}
        for i, addr in enumerate(lhs_addrs):
            dram[addr] = lhs_tiles[i].copy()
        for i, addr in enumerate(rhs_addrs):
            dram[addr] = rhs_tiles[i].copy()
        for i, addr in enumerate(dst_addrs):
            dram[addr] = dst_tiles[i].copy()

        plan = build_matmul_plan(
            node_name="matmul_test",
            lhs_shape=[m_rows, k_cols],
            rhs_shape=[k_cols, n_cols],
            lhs_tiles=lhs_addrs,
            rhs_tiles=rhs_addrs,
            dst_tiles=dst_addrs,
            m_tiles=m_tiles,
            k_tiles=k_tiles,
            n_tiles=n_tiles,
            tile_size=TILE,
        )
        plan = plan_tile_moves(plan)
        _execute_plan(plan, dram)

        got_tiles = [dram[a] for a in dst_addrs]
        got = _tiles_to_matrix(got_tiles, m_rows, n_cols)
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)

    def test_relu(self) -> None:
        rng = np.random.default_rng(21)
        rows, cols = BIG, BIG
        src = rng.standard_normal((rows, cols), dtype=np.float32)
        expected = torch.relu(torch.from_numpy(src)).numpy()

        src_tpd = _tiles_for_shape([rows, cols])
        dst_tpd = _tiles_for_shape([rows, cols])
        count = src_tpd[0] * src_tpd[1]
        src_addrs, dst_addrs = _alloc_regions(0x30000000, [count, count])

        src_tiles = _matrix_to_tiles(src, rows, cols)
        dst_tiles = _matrix_to_tiles(np.zeros((rows, cols), dtype=np.float32), rows, cols)

        dram: Dict[int, np.ndarray] = {}
        for i, addr in enumerate(src_addrs):
            dram[addr] = src_tiles[i].copy()
        for i, addr in enumerate(dst_addrs):
            dram[addr] = dst_tiles[i].copy()

        plan = build_relu_plan(
            node_name="relu_test",
            src_tiles=src_addrs,
            src_shape=[rows, cols],
            src_tiles_per_dim=src_tpd,
            dst_tiles=dst_addrs,
            dst_shape=[rows, cols],
            dst_tiles_per_dim=dst_tpd,
            tile_size=TILE,
        )
        plan = plan_tile_moves(plan)
        _execute_plan(plan, dram)

        got_tiles = [dram[a] for a in dst_addrs]
        got = _tiles_to_matrix(got_tiles, rows, cols)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
