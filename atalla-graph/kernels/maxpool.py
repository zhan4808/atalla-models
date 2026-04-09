"""AtallaC maxpool kernel: per-channel 2D max pooling (spatial W <= 32).

Vertical max uses ``make_mask`` + ``vec_op_masked("+", zero_vec, row, gt)``.
  RA must not coalesce ``vd`` with either operand on masked ``add_vv`` (see
  ``interferencegraph``); operand order ``zero + row`` avoids ``vd`` sharing
  the candidate vreg with the running best slot.
Horizontal reduction over the pool window is applied in ``run_graph`` via
``maxpool_post`` (matches PyTorch / NumPy gold). A full on-chip horizontal
path (``RMAX`` + masked scatter, as in ``functional_sim/build_maxpool.py``) is
plausible now that ``mv_stm`` uses the DAG mask vreg in ppci, but the
generated sequence still disagrees with the reference in emulation; keep post
until that is debugged.
"""

from kernels.common import ADDR_TABLE, sdma_ctl_expr


def maxpool_c(H: int, W: int, C: int, pool: int, stride: int) -> str:
    H_out = (H - pool) // stride + 1
    w_m1 = W - 1
    sdma_in = sdma_ctl_expr("sdma_in", 0, H, W, W)
    sdma_out = sdma_ctl_expr("sdma_out", 0, H_out, W, W)
    in_ch_bytes = H * W * 2
    out_ch_bytes = H_out * W * 2

    vert = (
        f"            vec best = vector_load(sp, in_row, {w_m1}, 0);\n"
    )
    for p in range(1, pool):
        vert += (
            f"            int r{p} = in_row + {p};\n"
            f"            vec v{p} = vector_load(sp, r{p}, {w_m1}, 0);\n"
            f'            int gt{p} = make_mask(">", v{p}, best, all_mask);\n'
            f'            best = vec_op_masked("+", zero_vec, v{p}, gt{p});\n'
        )

    return (
        "int main() {\n"
        f"    int cfg = {ADDR_TABLE};\n"
        "    int IN_BASE;\n"
        "    int OUT_BASE;\n"
        '    asm("lw_s %0, 0(%1)" : "=r"(IN_BASE)  : "r"(cfg));\n'
        '    asm("lw_s %0, 4(%1)" : "=r"(OUT_BASE) : "r"(cfg));\n'
        "\n"
        "    int sp = 0;\n"
        "    int all_mask = -1;\n"
        "    int ncols = 1;\n"
        f"{sdma_in}"
        f"{sdma_out}"
        "\n"
        f"    vec zero_vec = vector_load(sp, 0, {w_m1}, 0);\n"
        '    zero_vec = vec_op_masked("*", zero_vec, 0.0, all_mask);\n'
        "\n"
        "    int ch = 0;\n"
        f"    while (ch < {C}) {{\n"
        f"        int in_ptr = IN_BASE + ch * {in_ch_bytes};\n"
        f"        int out_ptr = OUT_BASE + ch * {out_ch_bytes};\n"
        "        scpad_load(sp, in_ptr, sdma_in);\n"
        "\n"
        "        int oh = 0;\n"
        f"        while (oh < {H_out}) {{\n"
        f"            int in_row = oh * {stride};\n"
        f"{vert}"
        f"            vector_store(best, sp, oh, {w_m1}, 0);\n"
        "            oh = oh + 1;\n"
        "        }\n"
        "\n"
        "        scpad_store(sp, out_ptr, sdma_out);\n"
        "        ch = ch + 1;\n"
        "    }\n"
        "\n"
        '    asm("halt");\n'
        "    return 0;\n"
        "}\n"
    )
