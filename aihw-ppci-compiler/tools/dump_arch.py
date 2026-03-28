"""Helper script to dump all information for an architecture"""

from pathlib import Path

from ppci import api
from ppci.arch import encoding
from ppci.utils.htmlgen import Document

this_path = Path(__file__).resolve().parent
build_path = this_path.parent / "build"
if not build_path.exists():
    build_path.mkdir(parents=True)
arch = api.get_arch("msp430")
arch = api.get_arch("x86_64")


def mkstr(s):
    if isinstance(s, str):
        return s
    elif isinstance(s, encoding.Operand):
        return f"${s._name}"
    else:
        raise NotImplementedError()


instructions = []
for i in arch.isa.instructions:
    if not i.syntax:
        continue
    syntax = "".join(mkstr(s) for s in i.syntax.syntax)
    instructions.append((syntax, i))

filename = build_path / "arch_info.html"
with filename.open("w") as f, Document(f) as doc, doc.body() as body:
    body.header("Instructions")
    body.paragraph(f"{len(instructions)} instructions available")
    with body.table() as table:
        table.header("synax", "Class")
        for syntax, ins_class in sorted(instructions, key=lambda x: x[0]):
            table.row(syntax, str(ins_class))

print(f"Created {filename}")
