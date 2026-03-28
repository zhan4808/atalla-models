from pathlib import Path

from ppci import api
from ppci.graph import callgraph, cyclo
from ppci.graph.cfg import ir_function_to_graph

this_path = Path(__file__).resolve().parent
sources = [
    this_path / "src" / "hello" / "hello.c3",
    this_path.parent / "librt" / "io.c3",
    this_path / "linux64" / "bsp.c3",
]
m = api.c3_to_ir(sources, [], "arm")
print(m)
print(m.stats())

cg = callgraph.mod_to_call_graph(m)
print(cg)

# Print callgraph
for func in m.functions:
    cfg, _ = ir_function_to_graph(func)
    complexity = cyclo.cyclomatic_complexity(cfg)
    print(f"Function: {func.name}, Complexity: {complexity}")
