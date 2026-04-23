from __future__ import annotations

from typing import Dict, List, Tuple

from scripts.tile_structures import OpPlan, Step, StepKind


def _tile_dim(total: int, tile_i: int, tile_size: int) -> int:
    start = tile_i * tile_size
    rem = total - start
    if rem <= 0:
        return 0
    return rem if rem < tile_size else tile_size


def _tile_rows_cols(shape: List[int], tiles_per_dim: List[int], tile_idx: int, tile_size: int) -> Tuple[int, int]:
    rank = len(shape)
    if rank == 1:
        cols_tiles = max(1, tiles_per_dim[-1])
        tc = tile_idx % cols_tiles
        return 1, _tile_dim(shape[-1], tc, tile_size)
    rows_tiles = max(1, tiles_per_dim[-2])
    cols_tiles = max(1, tiles_per_dim[-1])
    tc = tile_idx % cols_tiles
    tr = (tile_idx // cols_tiles) % rows_tiles
    return _tile_dim(shape[-2], tr, tile_size), _tile_dim(shape[-1], tc, tile_size)


def build_relu_plan(
    node_name: str,
    src_tiles: List[int],
    src_shape: List[int],
    src_tiles_per_dim: List[int],
    dst_tiles: List[int],
    dst_shape: List[int],
    dst_tiles_per_dim: List[int],
    tile_size: int,
) -> OpPlan:
    steps: List[Step] = []
    tile_info: Dict[str, Dict[str, int]] = {}
    count = len(dst_tiles)
    for idx in range(count):
        sidx = idx if idx < len(src_tiles) else max(0, len(src_tiles) - 1)
        src_id = f"X:{sidx}"
        did = f"Y:{idx}"
        s_rows, s_cols = _tile_rows_cols(src_shape, src_tiles_per_dim, sidx, tile_size)
        d_rows, d_cols = _tile_rows_cols(dst_shape, dst_tiles_per_dim, idx, tile_size)
        tile_info[src_id] = {"addr": src_tiles[sidx], "rows": s_rows, "cols": s_cols, "full_cols": src_shape[-1]}
        tile_info[did] = {"addr": dst_tiles[idx], "rows": d_rows, "cols": d_cols, "full_cols": dst_shape[-1]}
        steps.append(Step(kind=StepKind.LOAD, op="relu", attrs={"tile_id": src_id}))
        steps.append(Step(kind=StepKind.LOAD, op="relu", attrs={"tile_id": did}))
        steps.append(Step(kind=StepKind.COMPUTE, op="relu", attrs={"in_tile_id": src_id, "out_tile_id": did, "rows": d_rows, "cols": d_cols}))
        steps.append(Step(kind=StepKind.STORE, op="relu", attrs={"tile_id": did}))
    return OpPlan(node_name=node_name, op_type="relu", steps=steps, attrs={"tile_info": tile_info})


def build_softmax_plan(
    node_name: str,
    src_tiles: List[int],
    src_shape: List[int],
    src_tiles_per_dim: List[int],
    dst_tiles: List[int],
    dst_shape: List[int],
    dst_tiles_per_dim: List[int],
    tile_size: int,
) -> OpPlan:
    steps: List[Step] = []
    tile_info: Dict[str, Dict[str, int]] = {}
    count = len(dst_tiles)
    for idx in range(count):
        sidx = idx if idx < len(src_tiles) else max(0, len(src_tiles) - 1)
        src_id = f"S:{sidx}"
        did = f"T:{idx}"
        s_rows, s_cols = _tile_rows_cols(src_shape, src_tiles_per_dim, sidx, tile_size)
        d_rows, d_cols = _tile_rows_cols(dst_shape, dst_tiles_per_dim, idx, tile_size)
        tile_info[src_id] = {"addr": src_tiles[sidx], "rows": s_rows, "cols": s_cols, "full_cols": src_shape[-1]}
        tile_info[did] = {"addr": dst_tiles[idx], "rows": d_rows, "cols": d_cols, "full_cols": dst_shape[-1]}
        steps.append(Step(kind=StepKind.LOAD, op="softmax", attrs={"tile_id": src_id}))
        steps.append(Step(kind=StepKind.LOAD, op="softmax", attrs={"tile_id": did}))
        steps.append(Step(kind=StepKind.COMPUTE, op="softmax", attrs={"in_tile_id": src_id, "out_tile_id": did, "rows": d_rows, "cols": d_cols}))
        steps.append(Step(kind=StepKind.STORE, op="softmax", attrs={"tile_id": did}))
    return OpPlan(node_name=node_name, op_type="softmax", steps=steps, attrs={"tile_info": tile_info})


def build_add_plan(
    node_name: str,
    lhs_tiles: List[int],
    lhs_shape: List[int],
    lhs_tiles_per_dim: List[int],
    rhs_tiles: List[int],
    rhs_shape: List[int],
    rhs_tiles_per_dim: List[int],
    dst_tiles: List[int],
    dst_shape: List[int],
    dst_tiles_per_dim: List[int],
    tile_size: int,
) -> OpPlan:
    steps: List[Step] = []
    tile_info: Dict[str, Dict[str, int]] = {}
    count = len(dst_tiles)
    for idx in range(count):
        lidx = idx if idx < len(lhs_tiles) else 0
        ridx = idx if idx < len(rhs_tiles) else 0
        lid = f"L:{lidx}"
        rid = f"R:{ridx}"
        did = f"D:{idx}"
        l_rows, l_cols = _tile_rows_cols(lhs_shape, lhs_tiles_per_dim, lidx, tile_size)
        r_rows, r_cols = _tile_rows_cols(rhs_shape, rhs_tiles_per_dim, ridx, tile_size)
        d_rows, d_cols = _tile_rows_cols(dst_shape, dst_tiles_per_dim, idx, tile_size)
        tile_info[lid] = {"addr": lhs_tiles[lidx], "rows": l_rows, "cols": l_cols, "full_cols": lhs_shape[-1]}
        tile_info[rid] = {"addr": rhs_tiles[ridx], "rows": r_rows, "cols": r_cols, "full_cols": rhs_shape[-1]}
        tile_info[did] = {"addr": dst_tiles[idx], "rows": d_rows, "cols": d_cols, "full_cols": dst_shape[-1]}
        steps.append(Step(kind=StepKind.LOAD, op="add", attrs={"tile_id": lid}))
        steps.append(Step(kind=StepKind.LOAD, op="add", attrs={"tile_id": rid}))
        steps.append(Step(kind=StepKind.LOAD, op="add", attrs={"tile_id": did}))
        steps.append(
            Step(
                kind=StepKind.COMPUTE,
                op="add",
                attrs={"lhs_tile_id": lid, "rhs_tile_id": rid, "out_tile_id": did, "rows": d_rows, "cols": d_cols},
            )
        )
        steps.append(Step(kind=StepKind.STORE, op="add", attrs={"tile_id": did}))
    return OpPlan(node_name=node_name, op_type="add", steps=steps, attrs={"tile_info": tile_info})


def build_maxpool_plan(
    node_name: str,
    src_tiles: List[int],
    src_shape: List[int],
    src_tiles_per_dim: List[int],
    dst_tiles: List[int],
    dst_shape: List[int],
    dst_tiles_per_dim: List[int],
    tile_size: int,
    kernel: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    ceil_mode: int,
) -> OpPlan:
    steps: List[Step] = []
    tile_info: Dict[str, Dict[str, int]] = {}
    k_h, k_w = kernel
    s_h, s_w = stride
    p_h, p_w = padding
    d_h, d_w = dilation
    count = len(dst_tiles)
    for idx in range(count):
        sidx = idx if idx < len(src_tiles) else max(0, len(src_tiles) - 1)
        src_id = f"M:{sidx}"
        did = f"N:{idx}"
        s_rows, s_cols = _tile_rows_cols(src_shape, src_tiles_per_dim, sidx, tile_size)
        d_rows, d_cols = _tile_rows_cols(dst_shape, dst_tiles_per_dim, idx, tile_size)
        tile_info[src_id] = {"addr": src_tiles[sidx], "rows": s_rows, "cols": s_cols, "full_cols": src_shape[-1]}
        tile_info[did] = {"addr": dst_tiles[idx], "rows": d_rows, "cols": d_cols, "full_cols": dst_shape[-1]}
        steps.append(Step(kind=StepKind.LOAD, op="maxpool", attrs={"tile_id": src_id}))
        steps.append(Step(kind=StepKind.LOAD, op="maxpool", attrs={"tile_id": did}))
        steps.append(
            Step(
                kind=StepKind.COMPUTE,
                op="maxpool",
                attrs={
                    "in_tile_id": src_id,
                    "out_tile_id": did,
                    "rows": d_rows,
                    "cols": d_cols,
                    "kernel_h": k_h,
                    "kernel_w": k_w,
                    "stride_h": s_h,
                    "stride_w": s_w,
                    "pad_h": p_h,
                    "pad_w": p_w,
                    "dilation_h": d_h,
                    "dilation_w": d_w,
                    "ceil_mode": ceil_mode,
                },
            )
        )
        steps.append(Step(kind=StepKind.STORE, op="maxpool", attrs={"tile_id": did}))
    return OpPlan(node_name=node_name, op_type="maxpool", steps=steps, attrs={"tile_info": tile_info})


def build_conv_plan(
    node_name: str,
    src_tiles: List[int],
    src_shape: List[int],
    src_tiles_per_dim: List[int],
    dst_tiles: List[int],
    dst_shape: List[int],
    dst_tiles_per_dim: List[int],
    tile_size: int,
    kernel: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
) -> OpPlan:
    steps: List[Step] = []
    tile_info: Dict[str, Dict[str, int]] = {}
    k_h, k_w = kernel
    s_h, s_w = stride
    p_h, p_w = padding
    d_h, d_w = dilation
    count = len(dst_tiles)
    for idx in range(count):
        sidx = idx if idx < len(src_tiles) else max(0, len(src_tiles) - 1)
        src_id = f"CI:{sidx}"
        did = f"CO:{idx}"
        s_rows, s_cols = _tile_rows_cols(src_shape, src_tiles_per_dim, sidx, tile_size)
        d_rows, d_cols = _tile_rows_cols(dst_shape, dst_tiles_per_dim, idx, tile_size)
        tile_info[src_id] = {"addr": src_tiles[sidx], "rows": s_rows, "cols": s_cols, "full_cols": src_shape[-1]}
        tile_info[did] = {"addr": dst_tiles[idx], "rows": d_rows, "cols": d_cols, "full_cols": dst_shape[-1]}
        steps.append(Step(kind=StepKind.LOAD, op="conv", attrs={"tile_id": src_id}))
        steps.append(Step(kind=StepKind.LOAD, op="conv", attrs={"tile_id": did}))
        steps.append(
            Step(
                kind=StepKind.COMPUTE,
                op="conv",
                attrs={
                    "in_tile_id": src_id,
                    "out_tile_id": did,
                    "rows": d_rows,
                    "cols": d_cols,
                    "kernel_h": k_h,
                    "kernel_w": k_w,
                    "stride_h": s_h,
                    "stride_w": s_w,
                    "pad_h": p_h,
                    "pad_w": p_w,
                    "dilation_h": d_h,
                    "dilation_w": d_w,
                    "groups": groups,
                },
            )
        )
        steps.append(Step(kind=StepKind.STORE, op="conv", attrs={"tile_id": did}))
    return OpPlan(node_name=node_name, op_type="conv", steps=steps, attrs={"tile_info": tile_info})


def build_matmul_plan(
    node_name: str,
    lhs_shape: List[int],
    rhs_shape: List[int],
    lhs_tiles: List[int],
    rhs_tiles: List[int],
    dst_tiles: List[int],
    m_tiles: int,
    k_tiles: int,
    n_tiles: int,
    tile_size: int,
    a_block_tiles: int = 6,
) -> OpPlan:
    """Weight-stationary GEMM policy: hold A(m,k) block, stream B(k,n), accumulate C(m,n)."""
    m_total = lhs_shape[-2] if len(lhs_shape) >= 2 else 1
    k_total = lhs_shape[-1] if len(lhs_shape) >= 1 else 0
    n_total = rhs_shape[-1] if len(rhs_shape) >= 1 else 0

    def tdim(total: int, ti: int) -> int:
        start = ti * tile_size
        rem = total - start
        if rem <= 0:
            return 0
        return rem if rem < tile_size else tile_size

    def a_tile_id(mi: int, ki: int) -> str:
        return f"A:{mi}:{ki}"

    def b_tile_id(ki: int, ni: int) -> str:
        return f"B:{ki}:{ni}"

    def c_tile_id(mi: int, ni: int) -> str:
        return f"C:{mi}:{ni}"

    tile_info: Dict[str, Dict[str, int]] = {}
    steps: List[Step] = []
    block = max(1, min(a_block_tiles, m_tiles))

    for ki in range(k_tiles):
        for m0 in range(0, m_tiles, block):
            mend = min(m0 + block, m_tiles)

            for mi in range(m0, mend):
                a_id = a_tile_id(mi, ki)
                a_rows = tdim(m_total, mi)
                a_cols = tdim(k_total, ki)
                a_idx = mi * k_tiles + ki
                tile_info[a_id] = {
                    "addr": lhs_tiles[a_idx],
                    "rows": a_rows,
                    "cols": a_cols,
                    "full_cols": k_total,
                }
                steps.append(
                    Step(
                        kind=StepKind.LOAD,
                        op="matmul",
                        attrs={"tile_id": a_id, "pin": 1},
                    )
                )

            for ni in range(n_tiles):
                b_id = b_tile_id(ki, ni)
                b_rows = tdim(k_total, ki)
                b_cols = tdim(n_total, ni)
                b_idx = ki * n_tiles + ni
                tile_info[b_id] = {
                    "addr": rhs_tiles[b_idx],
                    "rows": b_rows,
                    "cols": b_cols,
                    "full_cols": n_total,
                }
                steps.append(
                    Step(
                        kind=StepKind.LOAD,
                        op="matmul",
                        attrs={"tile_id": b_id},
                    )
                )

                for mi in range(m0, mend):
                    c_id = c_tile_id(mi, ni)
                    c_rows = tdim(m_total, mi)
                    c_cols = tdim(n_total, ni)
                    c_idx = mi * n_tiles + ni
                    tile_info[c_id] = {
                        "addr": dst_tiles[c_idx],
                        "rows": c_rows,
                        "cols": c_cols,
                        "full_cols": n_total,
                    }
                    steps.append(
                        Step(
                            kind=StepKind.LOAD,
                            op="matmul",
                            attrs={"tile_id": c_id},
                        )
                    )
                    steps.append(
                        Step(
                            kind=StepKind.COMPUTE,
                            op="matmul",
                            attrs={
                                "a_tile_id": a_tile_id(mi, ki),
                                "b_tile_id": b_id,
                                "c_tile_id": c_id,
                                "m_rows": c_rows,
                                "n_cols": c_cols,
                                "k_cols": tdim(k_total, ki),
                            },
                        )
                    )
                    steps.append(
                        Step(
                            kind=StepKind.STORE,
                            op="matmul",
                            attrs={"tile_id": c_id},
                        )
                    )

            for mi in range(m0, mend):
                steps.append(
                    Step(
                        kind=StepKind.RELEASE,
                        op="matmul",
                        attrs={"tile_id": a_tile_id(mi, ki)},
                    )
                )

    return OpPlan(
        node_name=node_name,
        op_type="matmul",
        steps=steps,
        attrs={
            "tile_info": tile_info,
        },
    )
