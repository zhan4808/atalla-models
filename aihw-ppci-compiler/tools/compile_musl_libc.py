"""Helper script to help in compilation of the musl libc.

See for the musl library:
https://www.musl-libc.org/
"""

import logging
import os
import time
from pathlib import Path

from ppci.api import cc
from ppci.common import CompilerError, logformat
from ppci.lang.c import COptions

logger = logging.getLogger("compile_musl")
home = Path(os.environ["HOME"]).resolve()
musl_folder = home / "GIT" / "musl"
cache_filename = musl_folder / "ppci_build.cache"


def do_compile(filename: Path):
    include_paths = [
        musl_folder / "include",
        musl_folder / "src" / "internal",
        musl_folder / "obj" / "include",
        musl_folder / "arch" / "x86_64",
        musl_folder / "arch" / "generic",
    ]
    coptions = COptions()
    coptions.add_include_paths(include_paths)
    with filename.open() as f:
        obj = cc(f, "x86_64", coptions=coptions)
    return obj


def main():
    t1 = time.monotonic()
    logger.info(f"Using musl folder: {musl_folder}")
    # crypt_md5_c = os.path.join(musl_folder, "src", "crypt", "crypt_md5.c")
    failed = 0
    passed = 0
    # file_pattern = os.path.join(musl_folder, 'src', 'crypt', '*.c')
    # file_pattern = os.path.join(musl_folder, 'src', 'string', '*.c')
    src_folder = musl_folder / "src" / "regex"
    for filename in src_folder.glob("*.c"):
        logger.info(f"==> Compiling {filename}")
        try:
            do_compile(filename)
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
    logger.info(f"Passed: {passed} failed: {failed} in {elapsed} seconds")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=logformat)
    main()
