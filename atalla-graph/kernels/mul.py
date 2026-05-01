"""AtallaC element-wise multiply: C = A * B (BF16, same layout as add)."""

import math
from .common import ADDR_TABLE, TILE, sdma_ctl_expr, sdma_ctl_val


def mul_c(total: int, width: int = 32) -> str:
    """Generate AtallaC for element-wise vector multiply."""
    rows = math.ceil(total / width)
    w_m1 = width - 1
    sp_rows = min(rows, TILE)
    tile_count = math.ceil(rows / sp_rows)
    tile_bytes = sp_rows * width * 2

    sdma_sp0 = sdma_ctl_expr("sdma_ctl_sp0", 0, sp_rows, width, width)
    sdma_sp1 = sdma_ctl_expr("sdma_ctl_sp1", 1, sp_rows, width, width)

    return (
        "int main() {\n"
        f"    int cfg = {ADDR_TABLE};\n"
        "    int A_GMEM;\n"
        "    int B_GMEM;\n"
        "    int C_GMEM;\n"
        '    asm("lw_s %0, 0(%1)" : "=r"(A_GMEM)  : "r"(cfg));\n'
        '    asm("lw_s %0, 4(%1)" : "=r"(B_GMEM)  : "r"(cfg));\n'
        '    asm("lw_s %0, 8(%1)" : "=r"(C_GMEM)  : "r"(cfg));\n'
        "\n"
        "    int sp = 0;\n"
        "    int all_mask = -1;\n"
        "    int ncols = 1;\n"
        f"{sdma_sp0}"
        f"{sdma_sp1}"
        "\n"
        "    int tile = 0;\n"
        f"    while (tile < {tile_count}) {{\n"
        "        scpad_load(sp, A_GMEM, sdma_ctl_sp0);\n"
        "        scpad_load(sp, B_GMEM, sdma_ctl_sp1);\n"
        "\n"
        "        int row = 0;\n"
        f"        while (row < {sp_rows}) {{\n"
        f"            vec a = vector_load(sp, row, {w_m1}, 0);\n"
        f"            vec b = vector_load(sp, row, {w_m1}, 1);\n"
        '            vec c = vec_op_masked("*", a, b, all_mask);\n'
        f"            vector_store(c, sp, row, {w_m1}, 0);\n"
        "            row = row + 1;\n"
        "        }\n"
        "\n"
        "        scpad_store(sp, C_GMEM, sdma_ctl_sp0);\n"
        f"        A_GMEM = A_GMEM + {tile_bytes};\n"
        f"        B_GMEM = B_GMEM + {tile_bytes};\n"
        f"        C_GMEM = C_GMEM + {tile_bytes};\n"
        "        tile = tile + 1;\n"
        "    }\n"
        "\n"
        '    asm("halt");\n'
        "    return 0;\n"
        "}\n"
    )
