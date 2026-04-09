#!/usr/bin/env python3
"""Write run_graph validate metrics JSONs for regression baselines.

  python scripts/capture_regression_baselines.py

Writes under ``out/regression_baselines/`` (same seeds as ``run_graph.main``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from run_graph import build_graph, load_model, run_validate  # noqa: E402


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    out = _ROOT / "out" / "regression_baselines"
    out.mkdir(parents=True, exist_ok=True)
    work = _ROOT / "out" / "graph_baseline_runs"
    work.mkdir(parents=True, exist_ok=True)

    for model_name in ("alexnet_small", "vit_micro"):
        model, example_input = load_model(model_name, 0.01)
        gm = build_graph(model, example_input, verbose=False)
        for mode in ("oracle", "chained"):
            path = out / f"{model_name}_{mode}.json"
            print(f"Capturing {model_name} {mode} -> {path}")
            run_validate(
                gm,
                model,
                example_input,
                str(work),
                verbose=False,
                metrics_json_path=str(path),
                validate_inputs=mode,
            )
    print("Done.")


if __name__ == "__main__":
    main()
