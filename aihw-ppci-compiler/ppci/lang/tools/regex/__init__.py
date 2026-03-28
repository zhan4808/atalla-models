"""Regular expression routines.

Implement regular expressions using derivatives.

Largely copied from: https://github.com/MichaelPaddon/epsilon

Implementation of this logic:
https://en.wikipedia.org/wiki/Brzozowski_derivative

Another good resource on regular expressions:
https://swtch.com/~rsc/regexp/

"""

from .codegen import generate_code
from .compiler import compile
from .parser import parse
from .regex import EPSILON, NULL, Kleene, Symbol, SymbolSet
from .scanner import make_scanner, scan

__all__ = (
    "parse",
    "compile",
    "Symbol",
    "SymbolSet",
    "Kleene",
    "EPSILON",
    "NULL",
    "make_scanner",
    "scan",
    "generate_code",
)
