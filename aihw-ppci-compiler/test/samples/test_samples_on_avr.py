import unittest

from ..helper_util import (
    do_iverilog,
    do_long_tests,
    examples_path,
    has_avr_emulator,
    make_filename,
    run_avr,
)
from .sample_helpers import add_samples, build


@unittest.skipUnless(do_long_tests("avr"), "skipping slow tests")
@add_samples("8bit", "simple")
class TestSamplesOnAvr(unittest.TestCase):
    march = "avr"
    opt_level = 0

    def do(self, src, expected_output, lang="c3"):
        base_filename = make_filename(self.id())
        bsp_c3 = examples_path / "avr" / "bsp.c3"
        crt0 = examples_path / "avr" / "glue.asm"
        mmap = examples_path / "avr" / "avr.mmap"
        build(
            base_filename,
            src,
            bsp_c3,
            crt0,
            self.march,
            self.opt_level,
            mmap,
            lang=lang,
            bin_format="hex",
            code_image="flash",
        )
        hexfile = base_filename.with_suffix(".hex")
        if has_avr_emulator() and do_iverilog():
            res = run_avr(hexfile)
            self.assertEqual(expected_output, res)


# Avr Only works with optimization enabled...
class TestSamplesOnAvrO2(TestSamplesOnAvr):
    opt_level = 2
