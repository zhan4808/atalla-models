"""Compile coremark benchmark.


https://github.com/eembc/coremark

"""

import io
import logging
import os
from pathlib import Path

from ppci import __version__ as ppci_version
from ppci import api
from ppci.common import CompilerError, logformat
from ppci.format.elf import write_elf
from ppci.lang.c import COptions

logger = logging.getLogger("coremark")
logging.basicConfig(level=logging.INFO, format=logformat)

# Custom provider for clock_gettime using linux syscall:
hacked_libc_extras = """

#include <time.h>

// Hack to enable clock_gettime in linux:

extern void bsp_syscall(int nr, int a, int b, int c);

int clock_gettime(clockid_t clockid, struct timespec *tp)
{
  // clock_gettime ==> syscall 228
  bsp_syscall(228, clockid, tp, 0);
  return 0;
}

// hack to route main_main to main:

extern main();
void main_main()
{
    main();
}

"""

home = Path(os.environ["HOME"])
core_mark_folder = home / "GIT" / "coremark"
this_dir = Path(__file__).resolve().parent
root_dir = this_dir.parent
build_dir = root_dir / "build"
if not build_dir.exists():
    build_dir.mkdir(parents=True)
port_folder = core_mark_folder / "linux64"
libc_folder = root_dir / "librt" / "libc"
linux64_folder = root_dir / "examples" / "linux64"

opt_level = 2
march = api.get_arch("x86_64")
coptions = COptions()
coptions.add_include_path(core_mark_folder)
coptions.add_include_path(port_folder)
coptions.add_include_path(libc_folder / "include")
coptions.add_define("COMPILER_VERSION", f'"ppci {ppci_version}"')
coptions.add_define("FLAGS_STR", f'"-O{opt_level}"')

# Prevent malloc / free usage:
coptions.add_define("MEM_METHOD", "MEM_STATIC")

# TODO: Hack to enable %f formatting:
coptions.add_define("__x86_64__", "1")

objs = []

crt0_asm = linux64_folder / "glue.asm"
crt0_c3 = linux64_folder / "bsp.c3"
linker_script = linux64_folder / "linux64.mmap"
objs.append(api.asm(crt0_asm, march))
objs.append(api.c3c([crt0_c3], [], march))
objs.append(api.cc(io.StringIO(hacked_libc_extras), march, coptions=coptions))

sources = list(core_mark_folder.glob("*.c"))
sources.extend(port_folder.glob("*.c"))
sources.extend(libc_folder.glob("*.c"))

for source_file in sources:
    logger.info(f"compiling {source_file}")
    try:
        with source_file.open() as f:
            obj = api.cc(f, march, coptions=coptions, opt_level=opt_level)
    except CompilerError as ex:
        logger.exceptio(f"ERROR! {ex}")
        ex.print()
    else:
        objs.append(obj)

print(objs)

full_obj = api.link(objs, layout=linker_script)

exe_filename = build_dir / "coremark.elf"
logger.info(f"Creating {exe_filename}")
with exe_filename.open("wb") as f:
    write_elf(full_obj, f)

api.chmod_x(exe_filename)
