"""Helper script to build links

Version: links-2.17

Usage:

- Download the links sourcecode from: http://links.twibright.com/download.php
- untar the sourcecode
- Set the environment variable LINKS_FOLDER to the unzipped dir
- Run this script

"""

import logging
import os
import sys
import time
from pathlib import Path
from traceback import print_exc

from ppci.api import cc, link
from ppci.common import CompilerError, logformat
from ppci.lang.c import COptions
from ppci.utils.reporting import html_reporter

logger = logging.getLogger("compile_links")
links_folder = Path(os.environ["LINKS_FOLDER"]).resolve()
this_dir = Path(__file__).resolve().parent
root_path = this_dir.parent
build_path = root_path / "build"
report_filename = build_path / "report_links.html"
libc_path = root_path / "librtlibc"
libc_includes = libc_path / "include"
arch = "x86_64"


def do_compile(filename: Path, coptions, reporter):
    with filename.open() as f:
        obj = cc(f, arch, coptions=coptions, reporter=reporter)
    return obj


def main():
    if not build_path.exists():
        build_path.mkdir(parents=True)
    t1 = time.monotonic()
    failed = 0
    passed = 0
    sources = sorted(links_folder.glob("*.c"))
    objs = []
    coptions = COptions()
    include_paths = [
        libc_includes,
        links_folder,
        "/usr/include",
    ]
    coptions.add_include_paths(include_paths)
    with html_reporter(report_filename) as reporter:
        for filename in sources:
            print("      ======================")
            print("    ========================")
            print("  ==> Compiling", filename)
            try:
                obj = do_compile(filename, coptions, reporter)
                objs.append(obj)
            except CompilerError as ex:
                print("Error:", ex.msg, ex.loc)
                ex.print()
                print_exc()
                failed += 1
            except Exception as ex:
                print("General exception:", ex)
                print_exc()
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
