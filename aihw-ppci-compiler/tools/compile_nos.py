"""Helper script to build nOS

nOS is a RTOS for microcontrollers.

https://github.com/jimtremblay/nOS

Usage:

- git clone the nOS sourcecode.
- Run this script

"""

import logging
import os
import time
from pathlib import Path

from ppci.api import cc
from ppci.common import CompilerError, logformat
from ppci.lang.c import COptions
from ppci.utils.reporting import html_reporter

logger = logging.getLogger("nos")
home = Path(os.environ["HOME"]).resolve()
nos_folder = home / "GIT" / "nOS"
nos_inc_folder = nos_folder / "inc"
nos_src_folder = nos_folder / "src"
this_dir = Path(__file__).resolve().parent
root_path = this_dir.parent
build_path = root_path / "build"
if not build_path.exists():
    build_path.mkdir(parents=True)
report_filename = build_path / "report_nos.html"
libc_path = root_path / "librt" / "libc"
arch = "msp430"
coptions = COptions()
coptions.enable("freestanding")
include_paths = [
    libc_path / "include",
    nos_inc_folder,
    nos_inc_folder / "port" / "GCC" / "MSP430",
]
coptions.add_include_paths(include_paths)


def do_compile(filename: Path, reporter):
    with filename.open() as f:
        obj = cc(f, arch, coptions=coptions, reporter=reporter)
    print(filename, "compiled into", obj)
    return obj


def main():
    t1 = time.monotonic()
    failed = 0
    passed = 0
    with html_reporter(report_filename) as reporter:
        for filename in nos_src_folder.glob("*.c"):
            logger.info(f"==> Compiling {filename}")
            try:
                do_compile(filename, reporter)
            except CompilerError as ex:
                logger.exception(f"Error: {ex.msg}, {ex.loc}")
                ex.print()
                failed += 1
            except Exception as ex:
                logger.exception(f"General exception: {ex}")
                failed += 1
            else:
                logger.info("Great success!")
                passed += 1

    t2 = time.monotonic()
    elapsed = t2 - t1
    logger.info(f"{passed} passed, {failed} failed in {elapsed} seconds")


if __name__ == "__main__":
    verbose = False
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=logformat)
    main()
