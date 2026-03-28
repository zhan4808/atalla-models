import unittest

from ..helper_util import (
    do_long_tests,
    examples_path,
    has_qemu,
    make_filename,
    qemu,
)
from .sample_helpers import add_samples, build


@unittest.skipUnless(do_long_tests("microblaze"), "skipping slow tests")
@add_samples("simple", "medium")
class MicroblazeSamplesTestCase(unittest.TestCase):
    march = "microblaze"
    opt_level = 2

    def do(self, src, expected_output, lang="c3"):
        base_filename = make_filename(self.id())
        bsp_c3 = examples_path / "microblaze" / "bsp.c3"
        crt0 = examples_path / "microblaze" / "crt0.asm"
        mmap = examples_path / "microblaze" / "layout.mmp"
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
        bin_filename = base_filename.with_suffix(".bin")

        if has_qemu():
            output = qemu(
                [
                    "qemu-system-microblaze",
                    "-nographic",
                    "-kernel",
                    bin_filename,
                ]
            )
            self.assertEqual(expected_output, output)
