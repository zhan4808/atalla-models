from __future__ import annotations

import os
from pathlib import Path
import argparse
import numpy as np

from build import (
    DRAMWriter, assemble_file, emit_test_format, render_testfile,
)


def make_add_asm(rows: int, width: int) -> str:
    w_m1 = width - 1
    h_m1 = rows - 1

    lines = []
    a = lines.append

    a(f"# Element-wise add: C = A + B, {rows}x{width}")
    a( "")
    a( "addi.s  $1, $0, 60")
    a( "lw.s    $2, 0($1)           # A_GMEM")
    a( "lw.s    $3, 4($1)           # B_GMEM")
    a( "lw.s    $4, 8($1)           # C_GMEM")
    a( "")
    a( "addi.s  $10, $0, 0          # sp_base = 0")
    a(f"scpad.ld $10, $2, {w_m1}, {h_m1}, 0   # A -> SP0")
    a(f"scpad.ld $10, $3, {w_m1}, {h_m1}, 1   # B -> SP1")
    a( "")
    a( "addi.s  $20, $0, -1")
    a( "mv.stm  1, $20              # mask1 = all lanes")
    a( "")

    for r in range(rows):
        a(f"vreg.ld $40, $10, {w_m1}, {h_m1}, 0, 1, {r}   # a_row{r}")
        a(f"vreg.ld $41, $10, {w_m1}, {h_m1}, 1, 1, {r}   # b_row{r}")
        a(f"add.vv  $42, $40, $41, 1                    # c = a + b")
        a(f"vreg.st $42, $10, {w_m1}, {h_m1}, 0, 1, {r}   # -> SP0 row{r}")
        a( "")

    a(f"scpad.st $10, $4, {w_m1}, {h_m1}, 0   # SP0 -> C_GMEM")
    a( "halt.s")

    return "\n".join(f"        {line}" if line else "" for line in lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", type=Path, default=Path("tests/add.in"))
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--width", type=int, default=8)
    args = ap.parse_args()

    rows, width = args.rows, args.width
    assert width <= 32

    ADDR_TABLE = 60
    A_GMEM = 0x1000
    B_GMEM = 0x1080
    C_GMEM = 0x1100

    np.random.seed(42)
    A = np.random.randn(rows, width).astype(np.float32)
    B = np.random.randn(rows, width).astype(np.float32)
    expected = A + B

    asm = make_add_asm(rows, width)
    instrs = assemble_file(asm)
    instr_text = emit_test_format(instrs)

    img = DRAMWriter()
    img.u32(ADDR_TABLE + 0, A_GMEM)
    img.u32(ADDR_TABLE + 4, B_GMEM)
    img.u32(ADDR_TABLE + 8, C_GMEM)

    for r in range(rows):
        for c in range(width):
            img.bf16(A_GMEM + (r * width + c) * 2, float(A[r, c]))
            img.bf16(B_GMEM + (r * width + c) * 2, float(B[r, c]))

    for i in range(rows * width):
        img.bf16(C_GMEM + i * 2, 0.0)

    print(f"Add: {rows}x{width}")
    print(f"\nA:\n{A}")
    print(f"\nB:\n{B}")
    print(f"\nExpected C:\n{expected}")

    data_text = img.render_data_mem(include_zeros=True)
    final = render_testfile(instr_text, data_text)

    if args.output is not None:
        os.makedirs(args.output.parent, exist_ok=True)
        args.output.write_text(final)
        print(f"\nWrote {args.output}")
    else:
        print(final)


if __name__ == "__main__":
    main()
