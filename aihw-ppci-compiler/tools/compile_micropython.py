"""Helper script to compile micropython.

Links to micropython: https://micropython.org/

"""

import logging
import os
import time
from pathlib import Path

from ppci.api import cc
from ppci.common import CompilerError, logformat
from ppci.lang.c import COptions

logger = logging.getLogger("compile_micropython")
home = Path(os.environ["HOME"])
micropython_path = home / "GIT" / "micropython"
this_dir = Path(__file__).resolve().parent
root_path = this_dir.parent
libc_path = root_path / "librt" / "libc"
libc_includes = libc_path / "include"
port_folder = micropython_path / "ports" / "unix"
arch = "arm"


def do_compile(filename: Path, coptions):
    # coptions.add_define('NORETURN')
    with filename.open() as f:
        obj = cc(f, arch, coptions=coptions)
    return obj


def main():
    t1 = time.monotonic()
    failed = 0
    passed = 0
    include_paths = [
        # os.path.join(newlib_folder, 'libc', 'include'),
        # TODO: not sure about the include path below for stddef.h:
        # '/usr/lib/gcc/x86_64-pc-linux-gnu/7.1.1/include'
        libc_includes,
        micropython_path,
        port_folder,
    ]
    coptions = COptions()
    coptions.add_include_paths(include_paths)
    coptions.enable("freestanding")
    coptions.add_define("NO_QSTR", "1")
    micropython_src_folder = micropython_path / "py"
    objs = []
    for filename in sorted(micropython_src_folder.glob("*.c")):
        logger.info(f"==> Compiling {filename}")
        try:
            obj = do_compile(filename, coptions)
        except CompilerError as ex:
            logger.exception(f"Error: {ex.msg} {ex.loc}")
            ex.print()
            failed += 1
        except Exception as ex:
            logger.exception(f"General exception: {ex}")
            failed += 1
        else:
            logger.info(f"Great success! {obj}")
            passed += 1
            objs.append(obj)

    t2 = time.monotonic()
    elapsed = t2 - t1
    logger.info(f"Passed: {passed} failed: {failed} in {elapsed} seconds")


if __name__ == "__main__":
    verbose = False
    loglevel = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=loglevel, format=logformat)
    main()
