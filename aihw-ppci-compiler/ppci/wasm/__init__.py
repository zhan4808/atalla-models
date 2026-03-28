"""
Tools for representing, loading and exporting WASM (Web Assembly),
and for converting between PPCI-IR and WASM.
"""

from .arch import WasmArchitecture
from .components import (
    BlockInstruction,
    Custom,
    Data,
    Elem,
    Export,
    Func,
    Global,
    Import,
    Instruction,
    Memory,
    Module,
    Ref,
    Start,
    Table,
    Type,
)
from .execution import WasmTrapException, execute_wasm, instantiate
from .opcodes import I
from .ppci2wasm import ir_to_wasm
from .util import (
    export_wasm_example,
    has_node,
    run_wasm_in_node,
    run_wasm_in_notebook,
)
from .wasm2ppci import wasm_to_ir


def read_wasm(input) -> Module:
    """Read wasm in the form of a string, tuple, bytes or file object.
    Returns a wasm Module object.
    """
    return Module(input)


def read_wat(f) -> Module:
    """Read wasm module from file handle"""
    wat = f.read()
    return Module(wat)


def wasmify(func, target="native"):
    """Convert a Python function to a WASM function, compiled
    to native code. Assumes that all variables are floats.
    Can be used as a decorator, like Numba!
    """

    from ppci.lang.python import python_to_wasm

    def f64_print(x: float) -> None:
        print(x)

    wa = python_to_wasm(func)
    imports = {"env": {"f64_print": f64_print}}
    mod = instantiate(wa, imports=imports, target=target)
    wasmfunc = getattr(mod.exports, func.__name__)
    return wasmfunc


__all__ = [
    "ir_to_wasm",
    "wasm_to_ir",
    "instantiate",
    "execute_wasm",
    "read_wasm",
    "read_wat",
    "run_wasm_in_node",
    "export_wasm_example",
    "run_wasm_in_notebook",
    "has_node",
    "WasmArchitecture",
    "WASMComponent",
    "Instruction",
    "BlockInstruction",
    "Definition",
    "Module",
    "Ref",
    "Type",
    "Import",
    "Table",
    "Memory",
    "Global",
    "Export",
    "Start",
    "Func",
    "Elem",
    "Data",
    "Custom",
    "I",
    "WasmTrapException",
]
