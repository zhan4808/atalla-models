"""

This adapter can check the fortrans suite:
'1978 FORTRAN COMPILER VALIDATION SYSTEM' (fcvs21.tar.Z)

See:

http://www.itl.nist.gov/div897/ctg/fortran_form.htm

"""

import os
import unittest
from pathlib import Path

from ppci.api import fortrancompile, get_arch


def create_test_function(cls, filename: Path):
    """Create a test function for a single snippet"""
    test_function_name = "test_" + filename.stem.replace(".", "_")

    def test_function(self):
        march = get_arch("arm")
        with filename.open() as f:
            fortrancompile([f.read()], march)
        # TODO: check output for correct values:

    if hasattr(cls, test_function_name):
        raise ValueError(f"Duplicate test {test_function_name}")

    setattr(cls, test_function_name, test_function)


def populate(cls):
    if "FCVS_DIR" in os.environ:
        path = Path(os.environ["FCVS_DIR"]).resolve()
        for filename in sorted(path.glob("*.FOR")):
            create_test_function(cls, filename)
    return cls


@populate
class FCVSTestCase(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
