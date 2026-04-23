from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple


class StepKind(Enum):
    LOAD = "load"
    COMPUTE = "compute"
    STORE = "store"
    RELEASE = "release"


@dataclass(frozen=True)
class TileRef:
    tensor: str
    tile_index: int
    coord: Tuple[int, ...]
    valid_rows: int
    valid_cols: int


@dataclass
class Step:
    kind: StepKind
    op: str
    inputs: List[TileRef] = field(default_factory=list)
    outputs: List[TileRef] = field(default_factory=list)
    slots: Dict[str, int] = field(default_factory=dict)
    attrs: Dict[str, int | float | str] = field(default_factory=dict)


@dataclass
class OpPlan:
    node_name: str
    op_type: str
    steps: List[Step]
    attrs: Dict[str, int | float | str] = field(default_factory=dict)
