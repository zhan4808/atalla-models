"""Pascal front-end"""

from .builder import PascalBuilder, pascal_to_ir
from .lexer import Lexer
from .parser import Parser

__all__ = ["pascal_to_ir", "PascalBuilder", "Parser", "Lexer"]
