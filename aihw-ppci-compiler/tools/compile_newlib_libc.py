"""Helper script to help in compilation of the newlib libc."""

import logging
import os
import time
import traceback
from pathlib import Path

from ppci.api import cc
from ppci.common import CompilerError, logformat
from ppci.lang.c import COptions

logger = logging.getLogger("compile_newlib")
home = Path(os.environ["HOME"]).resolve()
newlib_folder = home / "GIT" / "newlib-cygwin" / "newlib"
cache_filename = os.path.join(newlib_folder, "ppci_build.cache")
this_dir = os.path.abspath(os.path.dirname(__file__))
libc_includes = os.path.join(this_dir, "..", "librt", "libc")
arch = "msp430"


def do_compile(filename: Path):
    include_paths = [
        newlib_folder / "libc" / "include",
        libc_includes,
    ]
    coptions = COptions()
    coptions.add_include_paths(include_paths)
    coptions.add_define("HAVE_CONFIG_H")
    coptions.add_define("__MSP430__")
    with open(filename) as f:
        obj = cc(f, arch, coptions=coptions)
    return obj


def main():
    t1 = time.monotonic()
    logger.info(f"Using newlib folder: {newlib_folder}")
    # crypt_md5_c = os.path.join(newlib_folder, "src", "crypt", "crypt_md5.c")
    failed = 0
    passed = 0
    # file_pattern = os.path.join(newlib_folder, 'src', 'crypt', '*.c')
    # file_pattern = os.path.join(newlib_folder, 'src', 'string', '*.c')
    src_folder = newlib_folder / "libc" / "string"
    for filename in src_folder.glob("*.c"):
        logger.info(f"==> Compiling {filename}")
        try:
            do_compile(filename)
        except CompilerError as ex:
            print("Error:", ex.msg, ex.loc)
            ex.print()
            traceback.print_exc()
            failed += 1
            break
        except Exception as ex:
            print("General exception:", ex)
            traceback.print_exc()
            failed += 1
            break
        else:
            logger.info("Great success!")
            passed += 1

    t2 = time.monotonic()
    elapsed = t2 - t1
    print("Passed:", passed, "failed:", failed, "in", elapsed, "seconds")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format=logformat)
    main()
