"""Analyze legacy C sourcecode.

This tool should help in understanding existing legacy C code better.


"""

import argparse
import io
import logging
from pathlib import Path

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import CLexer

from ppci.api import get_arch
from ppci.common import CompilerError
from ppci.lang.c import COptions, create_ast
from ppci.lang.c.nodes import declarations
from ppci.utils.htmlgen import Document

this_path = Path(__file__).resolve().parent
root_path = this_path.parent
build_path = root_path / "build"
libc_includes = root_path / "librt" / "libc" / "include"
logger = logging.getLogger("c-analyzer")


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    args = parser.parse_args()
    defines = {}
    analyze_sources(args.source_dir, defines)


def analyze_sources(source_dir: Path, defines):
    """Analyze a directory with sourcecode"""

    # Phase 1: acquire ast's:
    asts = read_sources(source_dir, defines)

    # Phase 2: do some bad-ass analysis:
    global_variables = []
    functions = []
    for _source_filename, _source_code, ast in asts:
        for decl in ast.declarations:
            if isinstance(decl, declarations.VariableDeclaration):
                global_variables.append(decl)
            elif isinstance(decl, declarations.FunctionDeclaration):
                if decl.body is not None and decl.storage_class != "static":
                    functions.append(decl)

    functions.sort(key=lambda d: d.name)

    # Phase 3: generate html report?
    gen_report(functions, asts)


def read_sources(source_dir: Path, defines):
    arch_info = get_arch("x86_64").info
    coptions = COptions()
    # TODO: infer defines from code:
    coptions.add_define("FPM_DEFAULT", "1")
    coptions.add_include_path(source_dir)
    coptions.add_include_path(libc_includes)
    coptions.add_include_path("/usr/include")
    asts = []
    for source_filename in sorted(source_dir.glob("*.c")):
        logger.info("Processing %s", source_filename)
        source_code = source_filename.read_text()
        f = io.StringIO(source_code)
        # ast = parse_text(source_code)
        try:
            ast = create_ast(
                f, arch_info, filename=source_filename, coptions=coptions
            )
        except CompilerError as ex:
            logger.exception(f"Compiler error: {ex}")
        else:
            asts.append((source_filename, source_code, ast))
    logger.info("Got %s ast's", len(asts))
    return asts


def get_tag(filename: Path) -> str:
    return filename.stem


def gen_report(functions, asts):
    html_filename = "analyze_report.html"
    html_path = (
        build_path / html_filename
        if build_path.exists()
        else Path(html_filename)
    )
    logger.info(f"Creating {html_path}")
    with open(html_path, "w") as f, Document(f) as doc:
        formatter = HtmlFormatter(lineanchors="fubar", linenos="inline")
        with doc.head() as head:
            head.title("Analyzed C code")
            head.style(formatter.get_style_defs())

        with doc.body() as body:
            body.h1("Overview")
            with body.table() as table:
                table.header("Name", "Location")
                for func in functions:
                    tagname = get_tag(func.location.filename)
                    tagname = f"{tagname}-{func.location.row}"
                    name = f'<a href="#{tagname}">{func.name}</a>'
                    location = str(func.location)
                    table.row(name, location, escape=False)

            body.h1("Files")
            for source_filename, source_code, ast in asts:
                report_single_file(body, source_filename, source_code, ast)


def report_single_file(body, source_filename, source_code, ast):
    tagname = get_tag(source_filename)
    body.h2(str(source_filename))
    with body.table() as table:
        table.header("Name", "Location", "typ", "storage_class")
        for decl in ast.declarations:
            if isinstance(decl, declarations.VariableDeclaration):
                tp = "var"
            elif isinstance(decl, declarations.FunctionDeclaration):
                tp = "func"
            else:
                tp = "other"

            if source_filename == decl.location.filename:
                anchor = f"{tagname}-{decl.location.row}"
                name = f'<a href="#{anchor}">{decl.name}</a>'
            else:
                name = decl.name
            table.row(
                name,
                str(decl.location),
                tp,
                str(decl.storage_class),
                escape=False,
            )

    c_lexer = CLexer()
    formatter = HtmlFormatter(lineanchors=tagname, linenos="inline")
    with body.tag("div"):
        print(highlight(source_code, c_lexer, formatter), file=body._f)


if __name__ == "__main__":
    main()
