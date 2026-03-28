import io
import unittest

from ppci import ir
from ppci.arch.example import ExampleArch
from ppci.irutils import verify_module
from ppci.lang.c import CBuilder, CSynthesizer
from ppci.lang.c.options import COptions


class CSynthesizerTestCase(unittest.TestCase):
    def test_hello(self):
        """Convert C to Ir, and then this IR to C"""
        src = r"""
        void printf(char*);
        void main(int b) {
          printf("Hello" "world\n");
        }
        """
        arch = ExampleArch()
        builder = CBuilder(arch.info, COptions())
        f = io.StringIO(src)
        ir_module = builder.build(f, None)
        assert isinstance(ir_module, ir.Module)
        verify_module(ir_module)
        synthesizer = CSynthesizer()
        synthesizer.syn_module(ir_module)


if __name__ == "__main__":
    unittest.main()
