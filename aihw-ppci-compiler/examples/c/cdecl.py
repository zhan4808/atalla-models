"""Implement alike logic as is done on www.cdecl.org

Try for example:

$ cdelc.py 'char **a;'

"""

import argparse
import io

from ppci.api import get_current_arch
from ppci.lang.c import CContext, CLexer, COptions, CParser, CSemantics
from ppci.lang.c.nodes import declarations, types
from ppci.lang.c.preprocessor import prepare_for_parsing

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("source", type=str)
args = parser.parse_args()
# print('Source:', args.source)

# Parse into ast:
arch = get_current_arch()
coptions = COptions()
ccontext = CContext(coptions, arch.info)
semantics = CSemantics(ccontext)
cparser = CParser(coptions, semantics)
clexer = CLexer(COptions())
f = io.StringIO(args.source)
tokens = clexer.lex(f, "<snippet>")
tokens = prepare_for_parsing(tokens, cparser.keywords)
cparser.init_lexer(tokens)
semantics.begin()
decl = cparser.parse_declarations()[0]


# Explain:
def explain(x):
    if isinstance(x, declarations.VariableDeclaration):
        return f"{x.name} is {explain(x.typ)}"
    elif isinstance(x, types.PointerType):
        return f"a pointer to {explain(x.element_type)}"
    elif isinstance(x, types.ArrayType):
        return f"an array of {explain(x.element_type)}"
    elif isinstance(x, types.BasicType):
        return f"{x.type_id}"
    else:
        print("???", x)


print(explain(decl))
