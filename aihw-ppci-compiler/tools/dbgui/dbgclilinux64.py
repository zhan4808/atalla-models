#!/usr/bin/python

import argparse

from linux64debugdriver import Linux64DebugDriver

# import logging
from ppci import api
from ppci.binutils.dbg import Debugger
from ppci.binutils.dbg.ptcli import PtDebugCli

if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument("dut")
    parser.add_argument("obj")
    args = parser.parse_args()
    dut = args.dut
    obj = args.obj

    linux_specific = Linux64DebugDriver()
    linux_specific.go_for_it([dut])
    debugger = Debugger(api.get_arch("x86_64"), linux_specific)
    debugger.load_symbols(obj)
    # cli = DebugCli(debugger)
    cli = PtDebugCli(debugger)
    cli.cmdloop()
