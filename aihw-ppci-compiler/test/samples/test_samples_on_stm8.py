import unittest

from ..helper_util import examples_path, make_filename
from .sample_helpers import add_samples, build


@unittest.skip("TODO")
@add_samples("8bit")
class TestSamplesOnStm8(unittest.TestCase):
    march = "stm8"
    opt_level = 0

    def do(self, src, expected_output, lang="c3"):
        base_filename = make_filename(self.id())
        bsp_c3 = examples_path / "stm8" / "bsp.c3"
        crt0 = examples_path / "stm8" / "start.asm"
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
