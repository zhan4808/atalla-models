"""Helper script to build 8cc

8cc is a small c99 compiler written in c99.

https://github.com/rui314/8cc

Usage:

- git clone the 8cc sourcecode.
- Set the environment variable LIBMAD_FOLDER to the unzipped dir
- Run this script

"""

import argparse
import logging
import os
import time
from pathlib import Path

from ppci import api
from ppci.common import CompilerError, logformat
from ppci.format.elf import write_elf
from ppci.lang.c import COptions
from ppci.utils.reporting import html_reporter

logger = logging.getLogger("compile_8cc")
home = Path(os.environ["HOME"]).resolve()
_8cc_folder = home / "GIT" / "8cc"
this_dir = Path(__file__).resolve().parent
root_path = this_dir.parent
build_path = root_path / "build"
report_filename = build_path / "report_8cc.html"
libc_path = root_path / "librt" / "libc"
libc_includes = libc_path / "include"
linux_include_dir = "/usr/include"
arch = "x86_64"
coptions = COptions()
include_paths = [
    libc_includes,
    _8cc_folder,
    linux_include_dir,
]
coptions.add_include_paths(include_paths)
coptions.add_define("BUILD_DIR", f'"{_8cc_folder}"')


def do_compile(filename: Path, reporter):
    with filename.open() as f:
        obj = api.cc(f, arch, coptions=coptions, reporter=reporter)
    logger.info(f"{filename} compiled into {obj}")
    return obj


def main():
    t1 = time.monotonic()
    failed = 0
    passed = 0
    sources = [
        "cpp.c",
        "debug.c",
        "dict.c",
        "gen.c",
        "lex.c",
        "vector.c",
        "parse.c",
        "buffer.c",
        "map.c",
        "error.c",
        "path.c",
        "file.c",
        "set.c",
        "encoding.c",
    ]
    objs = []
    with html_reporter(report_filename) as reporter:
        for filename in sources:
            filename = _8cc_folder / filename
            logger.info(f"==> Compiling {filename}")
            try:
                obj = do_compile(filename, reporter)
            except CompilerError as ex:
                logger.exception(f"Error: {ex.msg}, {ex.loc}")
                ex.print()
                failed += 1
            except Exception as ex:
                logger.exception(f"General exception: {ex}")
                failed += 1
            else:
                objs.append(obj)
                logger.info("Great success!")
                passed += 1

    t2 = time.monotonic()
    elapsed = t2 - t1
    logger.info(f"{passed} passed, {failed} failed in {elapsed} seconds")

    obj = api.link(objs)
    exe_path = build_path / "8cc.exe"
    with exe_path.open("wb") as f:
        write_elf(obj, f, type="executable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args()
    level = logging.DEBUG if args.verbose > 0 else logging.INFO
    logging.basicConfig(level=level, format=logformat)
    main()
