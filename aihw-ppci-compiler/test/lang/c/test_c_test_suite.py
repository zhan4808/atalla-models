"""Unit test adapter for c-testsuite

The c-testsuite is a collection of C test cases.

This is an adapter script to enable running of those snippets
as unittests.

Usage with pytest:

    $ export C_TEST_SUITE_DIR=/path/to/GIT/c-testsuite
    $ python -m pytest test_c_test_suite.py -v

Usage as a script:

    $ python test_c_test_suite.py /path/to/GIT/c-testsuite

See also:

https://github.com/c-testsuite/c-testsuite

"""

import argparse
import fnmatch
import io
import logging
import os
import subprocess
import unittest
from pathlib import Path

from ppci import api
from ppci.common import CompilerError, logformat
from ppci.format.elf import write_elf
from ppci.lang.c import COptions
from ppci.utils.reporting import html_reporter

this_dir = Path(__file__).resolve().parent
root_folder = this_dir.parent.parent.parent
build_folder = root_folder / "build" / "c_test_suite"
logger = logging.getLogger("c-test-suite")


def c_test_suite_populate(cls):
    """Enrich a unittest.TestCase with a function for each test snippet."""
    if "C_TEST_SUITE_DIR" in os.environ:
        suite_folder = Path(os.environ["C_TEST_SUITE_DIR"]).resolve()

        for filename in get_test_snippets(suite_folder):
            create_test_function(cls, filename)
    else:

        def test_stub(self):
            self.skipTest(
                "Please specify C_TEST_SUITE_DIR for the C test suite"
            )

        setattr(cls, "test_stub", test_stub)
    return cls


def get_test_snippets(suite_folder: Path, name_filter="*"):
    snippet_folder = suite_folder / "tests" / "single-exec"

    # Check if we have a folder:
    if not snippet_folder.is_dir():
        raise ValueError(f"{snippet_folder} is not a directory")

    for filename in sorted(snippet_folder.glob("*.c")):
        if fnmatch.fnmatch(filename.stem, name_filter):
            yield filename


def create_test_function(cls, filename: Path):
    """Create a test function for a single snippet"""
    test_name = filename.stem
    test_name = test_name.replace(".", "_").replace("-", "_")
    test_function_name = "test_" + test_name

    def test_function(self):
        perform_test(filename)

    if hasattr(cls, test_function_name):
        raise ValueError(f"Duplicate test case {test_function_name}")
    setattr(cls, test_function_name, test_function)


def perform_test(filename: Path):
    """Try to compile the given snippet."""
    logger.info("Step 1: Compile %s!", filename)
    march = "x86_64"

    build_folder.mkdir(parents=True, exist_ok=True)

    html_report = build_folder / (filename.stem + "_report.html")

    coptions = COptions()
    libc_folder = root_folder / "librt" / "libc"
    libc_include = libc_folder / "include"
    coptions.add_include_path(libc_include)

    # TODO: this should be injected elsewhere?
    coptions.add_define("__LP64__", "1")
    # coptions.enable('freestanding')

    with html_reporter(html_report) as reporter, filename.open() as f:
        try:
            obj1 = api.cc(f, march, coptions=coptions, reporter=reporter)
        except CompilerError as ex:
            ex.print()
            raise
    logger.info("Compilation complete, %s", obj1)

    obj0 = api.asm(io.StringIO(STARTERCODE), march)
    obj2 = api.c3c([io.StringIO(BSP_C3_SRC)], [], march)
    with (libc_folder / "lib.c").open() as f:
        obj3 = api.cc(f, march, coptions=coptions)

    # with (libc_folder / "src" / "string" / "string.c").open() as f:
    #     obj4 = api.cc(f, march, coptions=coptions)

    objs = [obj0, obj1, obj2, obj3]
    obj = api.link(objs, layout=io.StringIO(ARCH_MMAP))

    logger.info("Step 2: Run it!")

    exe_filename = build_folder / (filename.stem + "_executable.elf")
    with exe_filename.open("wb") as f:
        write_elf(obj, f, type="executable")
    api.chmod_x(exe_filename)

    logger.info("Running %s", exe_filename)
    test_prog = subprocess.Popen(exe_filename, stdout=subprocess.PIPE)
    exit_code = test_prog.wait()
    assert exit_code == 0
    captured_stdout = test_prog.stdout.read().decode("ascii")

    expected_filename = filename.parent / (filename.stem + ".c.expected")
    with expected_filename.open() as f:
        expected_stdout = f.read()

    # Compare stdout:
    assert captured_stdout == expected_stdout


STARTERCODE = """
global bsp_exit
global bsp_syscall
global main
global start

start:
    call main
    mov rdi, rax
    call bsp_exit

bsp_syscall:
    mov rax, rdi ; abi param 1
    mov rdi, rsi ; abi param 2
    mov rsi, rdx ; abi param 3
    mov rdx, rcx ; abi param 4
    syscall
    ret
"""

ARCH_MMAP = """
ENTRY(start)
MEMORY code LOCATION=0x40000 SIZE=0x10000 {
    SECTION(code)
}
MEMORY ram LOCATION=0x20000000 SIZE=0xA000 {
    SECTION(data)
}
"""

BSP_C3_SRC = """
module bsp;

public function void putc(byte c)
{
    syscall(1, 1, cast<int64_t>(&c), 1);
}

public function void exit(int64_t code)
{
    syscall(60, code, 0, 0);
}

function void syscall(int64_t nr, int64_t a, int64_t b, int64_t c);

"""


@c_test_suite_populate
class CTestSuiteTestCase(unittest.TestCase):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="count", default=0)
    parser.add_argument(
        "--folder",
        help="the folder with the c test suite.",
    )
    parser.add_argument(
        "--filter", help="Apply filtering on the test cases", default="*"
    )
    args = parser.parse_args()

    loglevel = logging.DEBUG if args.verbose else logging.INFO

    logging.basicConfig(level=loglevel, format=logformat)

    if args.folder is not None:
        suite_folder = Path(args.folder)
    elif "C_TEST_SUITE_DIR" in os.environ:
        suite_folder = Path(os.environ["C_TEST_SUITE_DIR"])
    else:
        parser.print_help()
        logger.error("ERROR: Specify where the c test suite is located!")
        return 1

    for filename in get_test_snippets(suite_folder, name_filter=args.filter):
        perform_test(filename)

    logger.info("OK.")


if __name__ == "__main__":
    main()
