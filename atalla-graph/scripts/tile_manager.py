from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Set, Tuple

from scripts.tile_structures import OpPlan, Step, StepKind

SCPAD_BYTES = 1048576
ATALLA_TILE = 32
BF16_BYTES = 2
TILE_BYTES = ATALLA_TILE * ATALLA_TILE * BF16_BYTES
MAX_SCPAD_TILES = SCPAD_BYTES // TILE_BYTES


class Planner:
    def __init__(self):
        self.max_tiles = MAX_SCPAD_TILES
        self.live_tiles: Dict[str, int] = {}
        self.slot_tiles: Dict[int, str] = {}
        self.dirty: Set[str] = set()
        self.pinned: Set[str] = set()
        self.next_use: Dict[str, int] = {}

    def _first_free_slot(self) -> Optional[int]:
        for slot in range(self.max_tiles):
            if slot not in self.slot_tiles:
                return slot
        return None

    def free_tile(self, protect: Set[str]) -> Tuple[str, int, bool]:
        victim_tile: Optional[str] = None
        victim_slot = -1
        victim_next = -1
        for tile_id, slot_id in self.live_tiles.items():
            if tile_id in protect or tile_id in self.pinned:
                continue
            nuse = self.next_use.get(tile_id, 1 << 30)
            if nuse > victim_next:
                victim_next = nuse
                victim_tile = tile_id
                victim_slot = slot_id
        if victim_tile is None:
            raise ValueError("No evictable tile available for allocation")
        spilled = victim_tile in self.dirty
        self.live_tiles.pop(victim_tile, None)
        self.slot_tiles.pop(victim_slot, None)
        self.dirty.discard(victim_tile)
        return victim_tile, victim_slot, spilled

    def need_tile(self, tile_id: str, mode: str) -> Tuple[int, Optional[str], bool]:
        if tile_id in self.live_tiles:
            return self.live_tiles[tile_id], None, False

        slot_id = self._first_free_slot()
        evicted_tile: Optional[str] = None
        spilled = False
        if slot_id is None:
            evicted_tile, slot_id, spilled = self.free_tile(protect={tile_id})

        self.live_tiles[tile_id] = slot_id
        self.slot_tiles[slot_id] = tile_id
        if mode == "pin":
            self.pinned.add(tile_id)
        return slot_id, evicted_tile, spilled

    def mark_dirty(self, tile_id: str) -> None:
        if tile_id in self.live_tiles:
            self.dirty.add(tile_id)


def _collect_positions(steps: List[Step]) -> Dict[str, List[int]]:
    positions: Dict[str, List[int]] = {}
    def add_pos(tile_id: object, idx: int) -> None:
        if isinstance(tile_id, str):
            positions.setdefault(tile_id, []).append(idx)

    for idx, step in enumerate(steps):
        add_pos(step.attrs.get("tile_id"), idx)
        add_pos(step.attrs.get("a_tile_id"), idx)
        add_pos(step.attrs.get("b_tile_id"), idx)
        add_pos(step.attrs.get("c_tile_id"), idx)
        add_pos(step.attrs.get("in_tile_id"), idx)
        add_pos(step.attrs.get("out_tile_id"), idx)
        add_pos(step.attrs.get("lhs_tile_id"), idx)
        add_pos(step.attrs.get("rhs_tile_id"), idx)
    return positions


def plan_tile_moves(op_plan: OpPlan) -> OpPlan:
    if not isinstance(op_plan.attrs.get("tile_info"), dict):
        return op_plan

    state = Planner()
    positions = _collect_positions(op_plan.steps)
    next_pos_idx: Dict[str, int] = {k: 0 for k in positions}
    tile_info = op_plan.attrs.get("tile_info", {})
    if not isinstance(tile_info, dict):
        tile_info = {}

    rewritten: List[Step] = []
    slot_map: Dict[str, int] = {}

    def touch_next_use(tile_id: str) -> None:
        if tile_id not in positions:
            return
        pidx = next_pos_idx[tile_id]
        next_pos_idx[tile_id] = pidx + 1
        pvals = positions[tile_id]
        state.next_use[tile_id] = pvals[pidx + 1] if (pidx + 1) < len(pvals) else (1 << 30)

    for step in op_plan.steps:
        for key in (
            "tile_id",
            "a_tile_id",
            "b_tile_id",
            "c_tile_id",
            "in_tile_id",
            "out_tile_id",
            "lhs_tile_id",
            "rhs_tile_id",
        ):
            tid = step.attrs.get(key)
            if isinstance(tid, str):
                touch_next_use(tid)

        if step.kind is StepKind.LOAD:
            tile_id = step.attrs.get("tile_id")
            if not isinstance(tile_id, str):
                rewritten.append(step)
                continue
            mode = "pin" if int(step.attrs.get("pin", 0)) else "read"
            slot_id, evicted_tile, spilled = state.need_tile(tile_id, mode=mode)
            if evicted_tile is not None and spilled:
                einfo = tile_info.get(evicted_tile, {})
                rewritten.append(
                    Step(
                        kind=StepKind.STORE,
                        op=step.op,
                        attrs={
                            "tile_id": evicted_tile,
                            "slot": slot_id,
                            "spill": 1,
                            "rows": int(einfo.get("rows", 1)),
                            "cols": int(einfo.get("cols", 1)),
                            "full_cols": int(einfo.get("full_cols", 1)),
                            "addr": int(einfo.get("addr", 0)),
                        },
                    )
                )
            slot_map[tile_id] = slot_id
            rewritten.append(
                replace(step, attrs={**step.attrs, "slot": slot_id})
            )
            continue

        if step.kind is StepKind.COMPUTE:
            a_tile = step.attrs.get("a_tile_id")
            b_tile = step.attrs.get("b_tile_id")
            c_tile = step.attrs.get("c_tile_id")
            if isinstance(c_tile, str):
                state.mark_dirty(c_tile)
                rewritten.append(
                    replace(
                        step,
                        attrs={
                            **step.attrs,
                            "slot_a": state.live_tiles.get(str(a_tile), 0),
                            "slot_b": state.live_tiles.get(str(b_tile), 0),
                            "slot_c": state.live_tiles.get(str(c_tile), 0),
                        },
                    )
                )
                continue

            lhs_tile = step.attrs.get("lhs_tile_id")
            rhs_tile = step.attrs.get("rhs_tile_id")
            out_tile_for_binary = step.attrs.get("out_tile_id")
            if isinstance(lhs_tile, str) and isinstance(rhs_tile, str) and isinstance(out_tile_for_binary, str):
                state.mark_dirty(out_tile_for_binary)
                rewritten.append(
                    replace(
                        step,
                        attrs={
                            **step.attrs,
                            "slot_lhs": state.live_tiles.get(lhs_tile, 0),
                            "slot_rhs": state.live_tiles.get(rhs_tile, 0),
                            "slot_out": state.live_tiles.get(out_tile_for_binary, 0),
                        },
                    )
                )
                continue

            in_tile = step.attrs.get("in_tile_id")
            out_tile = step.attrs.get("out_tile_id")
            if isinstance(out_tile, str):
                state.mark_dirty(out_tile)
                rewritten.append(
                    replace(
                        step,
                        attrs={
                            **step.attrs,
                            "slot_in": state.live_tiles.get(str(in_tile), 0),
                            "slot_out": state.live_tiles.get(str(out_tile), 0),
                        },
                    )
                )
                continue

            rewritten.append(step)
            continue

        if step.kind is StepKind.STORE:
            tile_id = step.attrs.get("tile_id")
            if not isinstance(tile_id, str):
                rewritten.append(step)
                continue
            slot_id = state.live_tiles.get(tile_id, slot_map.get(tile_id, 0))
            state.dirty.discard(tile_id)
            rewritten.append(
                replace(step, attrs={**step.attrs, "slot": slot_id})
            )
            continue

        if step.kind is StepKind.RELEASE:
            tile_id = step.attrs.get("tile_id")
            if isinstance(tile_id, str):
                state.pinned.discard(tile_id)
            rewritten.append(step)
            continue

        rewritten.append(step)

    return replace(
        op_plan,
        steps=rewritten,
        attrs={**op_plan.attrs, "slot_map": slot_map},
    )
