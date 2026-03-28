import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from graph.export_fx import render_graph
from graph.memoryallocator import allocate_memory
from model.basic import BasicModule


def main() -> None:
    module = BasicModule()
    module.bfloat16()

    gm = symbolic_trace(module)

    example_input = torch.ones((32, 32), dtype=torch.bfloat16)
    ShapeProp(gm).propagate(example_input)

    placeholder_data = {"x": example_input.clone()}

    gm = allocate_memory(gm, "model/images/dram.txt", placeholder_data)

    render_graph(gm, "model/images/graph.svg", meta_keys=("dram_addr", "bytes"))


def run_pipeline_demo() -> None:
    """Run the full PyTorch -> Atalla emulator pipeline on BasicModule."""
    from run_model import run_pipeline

    torch.manual_seed(42)
    model = BasicModule(dim=32, depth=2)
    example_input = torch.randn(1, 32)
    run_pipeline(model, example_input, out_dir="out/basic_pipeline")


def run_alexnet_demo(scale: float = 0.01) -> None:
    """Run the full PyTorch -> Atalla emulator pipeline on AlexNet."""
    from run_model import run_pipeline
    from model.alexnet import AlexNetSmall

    torch.manual_seed(42)
    model = AlexNetSmall(scale=scale, num_classes=10)
    example_input = torch.randn(1, 3, 32, 32)
    run_pipeline(model, example_input, out_dir="out/alexnet_pipeline")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "pipeline":
        run_pipeline_demo()
    elif len(sys.argv) > 1 and sys.argv[1] == "alexnet":
        scale = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
        run_alexnet_demo(scale)
    else:
        main()
