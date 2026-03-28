import unittest

from ppci.lang.c.castxml import CastXmlReader

from ...helper_util import test_path


class CastXmlTestCase(unittest.TestCase):
    """Try out cast xml parsing."""

    def test_test8(self):
        reader = CastXmlReader()
        reader.process(test_path / "data" / "c" / "test8.xml")


if __name__ == "__main__":
    unittest.main()
