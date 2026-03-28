"""This is the c3 language front end.

For the front-end a recursive descent parser is created.

.. graphviz::

   digraph c3 {
   rankdir="LR"
   1 [label="source text"]
   10 [label="lexer" ]
   20 [label="parser" ]
   40 [label="code generation"]
   99 [label="IR-code object"]
   1 -> 10
   10 -> 20
   20 -> 40
   40 -> 99
   }

"""

from .builder import C3Builder, c3_to_ir
from .codegenerator import CodeGenerator
from .context import Context
from .lexer import Lexer
from .parser import Parser
from .visitor import AstPrinter, Visitor

__all__ = [
    "AstPrinter",
    "C3Builder",
    "CodeGenerator",
    "Context",
    "Lexer",
    "Parser",
    "Visitor",
    "c3_to_ir",
]
