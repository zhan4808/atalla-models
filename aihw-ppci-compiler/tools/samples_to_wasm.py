"""
Compile test samples to WebAssembly, and embed in an HTML document.

Handy online wasm to text conversion:

https://cdn.rawgit.com/WebAssembly/wabt/7e56ca56/demo/wasm2wast/

https://cdn.rawgit.com/WebAssembly/wabt/fb986fbd/demo/wat2wasm/

https://github.com/WebAssembly/wabt

"""

import argparse
import io
import logging
import time
import traceback
from pathlib import Path

from ppci.api import c3_to_ir, c_to_ir, get_arch, optimize
from ppci.common import logformat
from ppci.irutils import ir_link
from ppci.lang.c import COptions
from ppci.utils.htmlgen import Document
from ppci.wasm import ir_to_wasm

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", "-v", action="count", default=0)
args = parser.parse_args()

loglevel = logging.DEBUG if args.verbose else logging.INFO
logging.basicConfig(format=logformat, level=loglevel)
logger = logging.getLogger("samples_to_wasm")

this_path = Path(__file__).resolve().parent
root_path = this_path.parent
build_path = root_path / "build" / "samples_to_wasm"
if not build_path.exists():
    logger.info(f"Creating folder {build_path}")
    build_path.mkdir(parents=True)
arch = get_arch("arm")  # TODO: use wasm arch!
coptions = COptions()
libc_dir = root_path / "librt" / "libc"
libc_include_path = libc_dir / "include"
coptions.add_include_path(libc_include_path)

libc_filename = root_path / "librt" / "libc" / "lib.c"
libio_filename = root_path / "librt" / "io.c3"


def c_to_wasm(filename: Path, verbose=False):
    """Compile c source to wasm"""
    with libc_filename.open() as f:
        ir_libc = c_to_ir(f, arch, coptions=coptions)

    optimize(ir_libc, level="2")

    with filename.open() as f:
        x = c_to_ir(f, arch, coptions=coptions)

    logger.info(f"Before optimization: {x.stats()}")
    optimize(x, level="2")
    logger.info(f"After optimization: {x.stats()}")

    if verbose:
        x.display()

    wasm_module = ir_to_wasm(ir_link([ir_libc, x]))
    return wasm_module


def c3_to_wasm(filename: Path, verbose=False):
    """Take c3 to wasm"""
    bsp = io.StringIO(
        """
       module bsp;
       public function void putc(byte c);
       """
    )
    ir_module = c3_to_ir([bsp, libio_filename, filename], [], arch)

    # ir_modules.insert(0, ir_modules.pop(-1))  # Shuffle bsp forwards
    if verbose:
        print(str(ir_module))
    # optimize(x, level='2')
    # print(x, x.stats())

    # x.display()

    wasm_module = ir_to_wasm(ir_module)
    return wasm_module


simple_sample_path = root_path / "test" / "samples" / "simple"
samples = sorted(simple_sample_path.glob("*.c"))
samples.extend(sorted(simple_sample_path.glob("*.c3")))

html_filename = build_path / "samples_in_wasm.html"
logger.info(f"Creating {html_filename}")
with open(html_filename, "w") as f, Document(f) as doc:
    with doc.head() as head:
        head.title("Samples")
        head.print("""<meta charset="utf-8">""")
    with doc.body() as body:
        body.paragraph(f"Sample generated on {time.ctime()}")
        body.paragraph(f"Generator script: <pre>{__file__}</pre>")

        fns = []
        for nr, sample in enumerate(samples, 1):
            logger.info(f"Processing sample {sample}")
            body.h1(f"Example #{nr}: {sample}")

            # Sourcecode:
            body.h2("Code")
            body.pre(sample.read_text())

            # Expected output:
            body.h2("Expected output")
            expected_output = sample.with_suffix(".out")
            body.pre(expected_output.read_text())

            # Actual wasm code:
            try:
                if sample.suffix == ".c3":
                    wasm_module = c3_to_wasm(sample, verbose=args.verbose)
                elif sample.suffix == ".c":
                    wasm_module = c_to_wasm(sample, verbose=args.verbose)
                else:
                    raise NotImplementedError(str(sample.suffix))
            except Exception:
                logger.exception("Error during compilation")
                print("Massive error!", file=f)
                with body.tag("pre"):
                    traceback.print_exc(file=f)
                continue
            else:
                logger.info(f"Completed generating wasm module {wasm_module}")

            if args.verbose:
                wasm_module.show()
                print(wasm_module.to_bytes())

            body.h2("Actual output")
            print(f'<pre id="wasm_output{nr}">', file=f)
            print("</pre>", file=f)

            wasm_filename = build_path / f"example_{nr}.wasm"
            logger.info(f"Saving {wasm_filename}")
            with wasm_filename.open("wb") as f3:
                wasm_module.to_file(f3)

            wasm_text = str(list(wasm_module.to_bytes()))
            print(
                f"""<script>
            function print_charcode{nr}(i) {{
            var c = String.fromCharCode(i);
            var el = document.getElementById('wasm_output{nr}');
            el.innerHTML += c;
            }}

            var providedfuncs{nr} = {{
            bsp_putc: print_charcode{nr},
            }};

            function compile_wasm{nr}() {{
            var wasm_data = new Uint8Array({wasm_text});
            var module = new WebAssembly.Module(wasm_data);
            var inst = new WebAssembly.Instance(
                module, {{js: providedfuncs{nr}}});
            inst.exports.main_main();
            console.log('calling' + {nr});
            }}
            </script>""",
                file=f,
            )
            fns.append(f"compile_wasm{nr}")

        print(
            """
        <script>
        function run_samples() {""",
            file=f,
        )
        for fn in fns:
            print(f"{fn}();", file=f)
        print(
            """}
        window.onload = run_samples;
        </script>
        """,
            file=f,
        )
