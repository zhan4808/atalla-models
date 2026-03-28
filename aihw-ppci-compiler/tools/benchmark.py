"""This contains a set of benchmarks to benchmark the ppci
compiler internals.

This can be used to verify a certain change yields a performance
improvement or not.

Run the benchmarks with pytest with the benchmark plugin:

python -m pytest benchmark.py

"""

import logging
import os
from glob import glob
from pathlib import Path

from ppci import api
from ppci.lang.c import COptions

this_path = Path(__file__).resolve().parent
root_path = this_path.parent


def test_nos_on_riscv(benchmark):
    benchmark(compile_nos_for_riscv)


def test_compile_8cc(benchmark):
    benchmark(compile_8cc)


def compile_nos_for_riscv():
    """Compile nOS for riscv architecture."""
    logging.basicConfig(level=logging.INFO)
    murax_path = root_path / "examples" / "riscvmurax"
    arch = api.get_arch("riscv")

    # Gather sources:
    path = murax_path / "csrc" / "nos"
    folders, srcs = get_sources(path, "*.c")
    folders += [os.path.join(murax_path, "csrc")]
    print(srcs)

    coptions = COptions()
    coptions.add_include_paths(folders)

    # Build code:
    o1 = api.asm(murax_path / "start.s", arch)
    o2 = api.asm(murax_path / "nOSPortasm.s", arch)
    objs = [o1, o2]

    for src in srcs:
        with open(src) as f:
            objs.append(api.cc(f, "riscv", coptions=coptions, debug=True))

    # Link code:
    api.link(
        objs,
        murax_path / "firmware.mmap",
        use_runtime=True,
        debug=True,
    )


def get_sources(folder: Path, extension):
    resfiles = []
    resdirs = []
    for x in os.walk(folder):
        subfolder = x[0]
        for y in glob(os.path.join(subfolder, extension)):
            resfiles.append(y)
        resdirs.append(subfolder)
    return (resdirs, resfiles)


def compile_8cc():
    """Compile the 8cc compiler.

    8cc homepage:
    https://github.com/rui314/8cc
    """

    home = Path(os.environ["HOME"]).resolve()
    _8cc_folder = home / "GIT" / "8cc"
    libc_includes = root_path / "librt" / "libc" / "include"
    linux_include_dir = "/usr/include"
    arch = api.get_arch("x86_64")
    coptions = COptions()
    include_paths = [
        libc_includes,
        _8cc_folder,
        linux_include_dir,
    ]
    coptions.add_include_paths(include_paths)
    coptions.add_define("BUILD_DIR", f'"{_8cc_folder}"')

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
    for filename in sources:
        source_path = _8cc_folder / filename
        with source_path.open() as f:
            objs.append(api.cc(f, arch, coptions=coptions))

    # TODO: maybe link it?
