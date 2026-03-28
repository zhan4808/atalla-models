import unittest
from pathlib import Path

from ppci.api import construct

from .helper_util import do_long_tests, examples_path, has_qemu, run_qemu


@unittest.skipUnless(do_long_tests("any"), "skipping slow tests")
class EmulationTestCase(unittest.TestCase):
    """Tests the compiler driver"""

    def test_m3_bare(self):
        """Build bare m3 binary and emulate it"""
        path = examples_path / "lm3s6965evb" / "bare"
        recipe = path / "build.xml"
        construct(recipe)
        if has_qemu():
            bin_file = path / "bare.bin"
            data = run_qemu(bin_file)
            self.assertEqual("Hello worle", data)

    def test_a9_bare(self):
        """Build vexpress cortex-A9 binary and emulate it"""
        path = examples_path / "realview-pb-a8"
        recipe = path / "build.xml"
        construct(recipe)
        if has_qemu():
            bin_file = path / "hello.bin"
            data = run_qemu(bin_file, machine="realview-pb-a8")
            self.assertEqual("Hello worle", data)


def add_test(cls, filename: Path):
    """Create a new test function and add it to the class"""
    name2 = str(filename.relative_to(examples_path))
    test_name = "test_" + "".join(x if x.isalnum() else "_" for x in name2)

    def test_func(self):
        construct(filename)

    test_func.__doc__ = f"Try to build example {name2}"
    setattr(cls, test_name, test_func)


def add_examples(cls):
    """Add all build.xml files as a test case to the class"""
    for buildfile in sorted(examples_path.glob("**/build.xml")):
        add_test(cls, buildfile)
    return cls


@unittest.skipUnless(do_long_tests("any"), "skipping slow tests")
@add_examples
class ExampleProjectsTestCase(unittest.TestCase):
    """Check whether the example projects work"""

    pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
