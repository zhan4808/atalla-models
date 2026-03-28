"""Dump windows exe/dll file contents"""

import argparse

from ..format.exefile import read_exe
from .base import base_parser

parser = argparse.ArgumentParser(description=__doc__, parents=[base_parser])
parser.add_argument(
    "exe", type=argparse.FileType("rb"), help="The exe file to analyze"
)


def pedump(args=None):
    args = parser.parse_args(args)
    print(args)
    read_exe(args.exe)


if __name__ == "__main__":
    pedump()
