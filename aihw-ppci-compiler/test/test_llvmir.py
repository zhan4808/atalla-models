import io
import unittest
from pathlib import Path

from ppci.common import CompilerError
from ppci.lang.llvmir import LlvmIrFrontend

from .helper_util import source_files, test_path


def create_test_function(source: Path):
    """Create a test function for a source file"""
    snippet = source.read_text()

    def test_func(slf):
        slf.do(snippet)

    return test_func


def add_samples(*folders):
    """Create a decorator function that adds tests in the given folders"""

    def deco(cls):
        for folder in folders:
            for source in source_files(test_path / "data" / folder, ".ll"):
                test_func = create_test_function(source)
                func_name = "test_" + source.stem
                assert not hasattr(cls, func_name)
                setattr(cls, func_name, test_func)
        return cls

    return deco


@unittest.skip("todo")
@add_samples("llvm")
class LlvmIrFrontendTestCase(unittest.TestCase):
    def do(self, src):
        f = io.StringIO(src)
        try:
            LlvmIrFrontend().compile(f)
        except CompilerError as e:
            lines = src.split("\n")
            e.render(lines)
            raise


if __name__ == "__main__":
    unittest.main()
