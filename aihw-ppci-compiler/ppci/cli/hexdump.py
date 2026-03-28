"""Display file contents in hexadecimal"""

import argparse

from ..utils.hexdump import hexdump as dump
from .base import LogSetup, base_parser

parser = argparse.ArgumentParser(description=__doc__, parents=[base_parser])
parser.add_argument(
    "file",
    metavar="file",
    type=argparse.FileType("rb"),
    help="File to dump contents of",
)
parser.add_argument(
    "--width", default=16, type=int, help="Width of the hexdump."
)


def hexdump(args=None):
    """Display file contents in hexadecimal"""
    args = parser.parse_args(args)
    with LogSetup(args):
        contents = args.file.read()
        args.file.close()
        dump(contents, width=args.width)


if __name__ == "__main__":
    hexdump()
