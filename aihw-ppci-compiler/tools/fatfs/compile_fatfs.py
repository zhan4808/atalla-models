import argparse
import logging
from pathlib import Path

from ppci import api
from ppci.binutils.objectfile import merge_memories
from ppci.common import CompilerError
from ppci.lang.c.options import COptions, coptions_parser
from ppci.utils.reporting import html_reporter

logger = logging.getLogger("compile-fatfs")
this_path = Path(__file__).resolve().parent
root_path = this_path.parent.parent
libc_includes = root_path / "librt" / "libc" / "include"

parser = argparse.ArgumentParser(parents=[coptions_parser])
parser.add_argument("-v", action="count", default=0)
args = parser.parse_args()

coptions = COptions.from_args(args)
coptions.add_include_path(libc_includes)
coptions.add_include_path(this_path)
report_html = this_path / "compilation_report.html"
loglevel = logging.DEBUG if args.v > 0 else logging.INFO
logging.basicConfig(level=loglevel)
arch = api.get_arch("riscv")


def cc(filename: str, reporter):
    logger.info("Compiling %s", filename)
    filename = this_path / filename
    with filename.open() as f:
        try:
            obj = api.cc(f, arch, reporter=reporter, coptions=coptions)
            logger.info("Compiled %s into %s bytes", filename, obj.byte_size)
        except CompilerError as e:
            print(e)
            e.print()
            obj = None
    return obj


with html_reporter(report_html) as reporter:
    file_list = ["xprintf.c", "loader.c", "ff.c", "sdmm.c"]
    objs = [cc(f, reporter) for f in file_list]
    objs = [api.asm("start.s", arch)] + objs
    print(objs)
    layout = this_path / "firmware.mmap"
    obj = api.link(
        objs, layout, use_runtime=True, reporter=reporter, debug=True
    )
    tlf_filename = this_path / "firmware.tlf"
    with tlf_filename.open("w") as of:
        obj.save(of)
    api.objcopy(obj, "flash", "bin", this_path / "code.bin")
    api.objcopy(obj, "ram", "bin", this_path / "data.bin")
    cimg = obj.get_image("flash")
    dimg = obj.get_image("ram")
    img = merge_memories(cimg, dimg, "img")
    imgdata = img.data
    with open(this_path / "firmware.hex", "w") as f:
        size = 0x8000
        for i in range(size):
            if i < len(imgdata) // 4:
                w = imgdata[4 * i : 4 * i + 4]
                print(f"{w[3]:02x}{w[2]:02x}{w[1]:02x}{w[0]:02x}", file=f)
            else:
                print("00000000", file=f)
