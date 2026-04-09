#!/usr/bin/env python3
"""Run matmul, maxpool, and conv golden harnesses as one regression gate.

From ``atalla-graph`` root::

  python scripts/run_operator_regressions.py
  python scripts/run_operator_regressions.py --quick   # skip slow alex_tile

Exits 0 only if every subprocess succeeds (including BF16 gates where enabled).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def _run(title: str, argv: list[str]) -> None:
    print(f"\n--- {title} ---")
    cmd = [PY, str(_ROOT / "scripts" / argv[0])] + argv[1:]
    r = subprocess.run(cmd, cwd=str(_ROOT))
    if r.returncode != 0:
        raise SystemExit(
            f"FAIL: {title} -> exit {r.returncode}\n  {' '.join(cmd)}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Skip maxpool alex_tile (larger compile/emulate).",
    )
    args = ap.parse_args()

    # Matmul: standard (matmul2), small-K (matmul1), wide-N (matmul7)
    _run(
        "matmul matmul2 (standard)",
        ["debug_matmul_golden.py", "--case", "matmul2", "--gate"],
    )
    _run(
        "matmul matmul1 (small K)",
        ["debug_matmul_golden.py", "--case", "matmul1", "--gate"],
    )
    _run(
        "matmul matmul7 (wide N)",
        ["debug_matmul_golden.py", "--case", "matmul7", "--gate"],
    )

    # Maxpool: regression subset (script already gates BF16-rounded gold)
    for case in ("tiny_tie", "all_losers", "two_channel"):
        _run(
            f"maxpool {case}",
            ["debug_maxpool_golden.py", "--case", case, "--seed", "0"],
        )
    if not args.quick:
        _run(
            "maxpool alex_tile",
            ["debug_maxpool_golden.py", "--case", "alex_tile", "--seed", "0"],
        )

    # Conv: multi-channel synthetic + graph-extracted alexnet node
    _run(
        "conv cin2 (multi Cin/Cout)",
        ["debug_conv_golden.py", "--case", "cin2", "--gate"],
    )
    _run(
        "conv graph conv2d_1",
        ["debug_conv_golden.py", "--graph-node", "conv2d_1", "--gate"],
    )

    print("\nOK: all operator regression steps passed.")


if __name__ == "__main__":
    main()
