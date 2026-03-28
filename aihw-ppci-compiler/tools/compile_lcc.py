"""Helper script to build lcc.

https://github.com/drh/lcc

Usage:

- Clone the lcc sourcecode.
- Set the environment variable LCC_FOLDER to the lcc source dir.
- Run this script

"""

import logging
import os
import sys
import time
from pathlib import Path

from ppci.api import cc, link
from ppci.common import CompilerError, logformat
from ppci.lang.c import COptions
from ppci.utils.reporting import html_reporter


def do_compile(filename, include_paths, arch, reporter):
    coptions = COptions()
    coptions.add_include_paths(include_paths)
    coptions.add_define("FPM_DEFAULT", "1")
    with open(filename) as f:
        obj = cc(f, arch, coptions=coptions, reporter=reporter)
    return obj


def main():
    logger = logging.getLogger("compile-lcc")
    environment_variable = "LCC_FOLDER"
    if environment_variable in os.environ:
        lcc_folder = os.environ[environment_variable]
    else:
        logger.error(
            "Please define %s to point to the lcc source folder",
            environment_variable,
        )
        return

    this_path = Path(__file__).resolve().parent
    root_path = this_path.parent
    report_filename = this_path / "report_lcc.html"
    libc_includes = root_path / "librt" / "libc"
    include_paths = [libc_includes]
    arch = "x86_64"

    t1 = time.monotonic()
    failed = 0
    passed = 0
    src_folder = lcc_folder / "src"
    objs = []
    with html_reporter(report_filename) as reporter:
        for filename in src_folder.glob("*.c"):
            logger.info("      ======================")
            logger.info("    ========================")
            logger.info(f"  ==> Compiling {filename}")
            try:
                obj = do_compile(filename, include_paths, arch, reporter)
                objs.append(obj)
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
    print("Passed:", passed, "failed:", failed, "in", elapsed, "seconds")
    obj = link(objs)
    print(obj)


if __name__ == "__main__":
    verbose = "-v" in sys.argv
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=logformat)
    main()
