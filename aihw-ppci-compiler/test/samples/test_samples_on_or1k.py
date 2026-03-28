import unittest

from ppci.format import uboot_image

from ..helper_util import (
    create_qemu_launch_script,
    do_long_tests,
    examples_path,
    has_qemu,
    make_filename,
    qemu,
)
from .sample_helpers import add_samples, build


@unittest.skipUnless(do_long_tests("or1k"), "skipping slow tests")
@add_samples("simple", "medium")
class OpenRiscSamplesTestCase(unittest.TestCase):
    march = "or1k"
    opt_level = 2

    def do(self, src, expected_output, lang="c3"):
        base_filename = make_filename(self.id())
        bsp_c3 = examples_path / "or1k" / "bsp.c3"
        crt0 = examples_path / "or1k" / "crt0.asm"
        mmap = examples_path / "or1k" / "layout.mmp"
        build(
            base_filename,
            src,
            bsp_c3,
            crt0,
            self.march,
            self.opt_level,
            mmap,
            lang=lang,
            bin_format="bin",
            code_image="flash",
        )
        binfile = base_filename.with_suffix(".bin")

        # Create a uboot application file:
        bindata = binfile.read_bytes()

        img_filename = base_filename.with_suffix(".img")
        with img_filename.open("wb") as f:
            uboot_image.write_uboot_image(
                f, bindata, arch=uboot_image.Architecture.OPENRISC
            )

        qemu_cmd = [
            "qemu-system-or1k",
            "-nographic",
            "-M",
            "or1k-sim",
            "-m",
            "16",
            "-kernel",
            img_filename,
        ]

        create_qemu_launch_script(base_filename.with_suffix(".sh"), qemu_cmd)
        if has_qemu():
            output = qemu(qemu_cmd)
            self.assertEqual(expected_output, output)
