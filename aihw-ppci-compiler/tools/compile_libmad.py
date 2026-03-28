"""Helper script to build libmad

Version: libmad-0.15.1b

Usage:

- Download the libmad sourcecode.
- Unzip the sourcecode
- Set the environment variable LIBMAD_FOLDER to the unzipped dir
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

logger = logging.getLogger("compile_libmad")


def do_compile(filename: Path, include_paths, arch, reporter):
    coptions = COptions()
    coptions.add_include_paths(include_paths)
    coptions.add_define("FPM_DEFAULT", "1")
    with filename.open() as f:
        obj = cc(f, arch, coptions=coptions, reporter=reporter)
    return obj


def main():
    environment_variable = "LIBMAD_FOLDER"
    if environment_variable in os.environ:
        libmad_folder = Path(os.environ[environment_variable])
    else:
        logger.error(
            "Please define %s to point to the libmad source folder",
            environment_variable,
        )
        return

    this_path = Path(__file__).resolve().parent
    root_path = this_path.parent
    build_path = root_path / "build"
    if not build_path.exists():
        build_path.mkdir(parents=True)
    report_filename = build_path / "report_libmad.html"
    libc_folder = root_path / "librt" / "libc"
    libc_includes = libc_folder / "include"
    include_paths = [libc_includes, libmad_folder]
    arch = "x86_64"

    t1 = time.monotonic()
    failed = 0
    passed = 0
    sources = [
        "layer3.c",
        "version.c",
        "fixed.c",
        "bit.c",
        "timer.c",
        "stream.c",
        "frame.c",
        "synth.c",
        "decoder.c",
        "layer12.c",
        "huffman.c",
    ]
    objs = []
    with html_reporter(report_filename) as reporter:
        for filename in sources:
            filename = libmad_folder / filename
            logger.info(f"  ==> Compiling {filename}")
            try:
                obj = do_compile(filename, include_paths, arch, reporter)
                objs.append(obj)
            except CompilerError as ex:
                logger.exception(f"Error: {ex.msg}, {ex.loc}")
                ex.print()
                failed += 1
            except Exception:
                logger.exception("General exception:")
                failed += 1
            else:
                logger.info("Great success!")
                passed += 1

    t2 = time.monotonic()
    elapsed = t2 - t1
    logger.info(f"Passed: {passed} failed: {failed} in {elapsed} seconds")
    obj = link(objs)
    print(obj)


if __name__ == "__main__":
    verbose = "-v" in sys.argv
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=logformat)
    main()
