#!/usr/bin/env python3
"""Small conv2d golden checks (same DRAM layout as ``emit_conv`` / im2col + GEMM).

Run from ``atalla-graph`` root:
  python scripts/debug_conv_golden.py --case all
  python scripts/debug_conv_golden.py --case cin2 --seed 0

Compares emulator output to ``torch.nn.functional.conv2d`` in BF16.

**Graph repro** (oracle tensors vs harness path)::

  python scripts/debug_conv_golden.py --list-graph-convs --model alexnet_small
  python scripts/debug_conv_golden.py --graph-node conv2d_1 --model alexnet_small
  python scripts/debug_conv_golden.py --graph-node conv2d_1 --dump-npz out/conv2d_1.npz
  python scripts/debug_conv_golden.py --load-npz out/conv2d_1.npz

If a failing oracle conv **passes** here, diff ``run_graph`` / ``emit_node`` layout vs this
script; if it **fails**, debug the shared im2col+GEMM path on those exact tensors.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import GraphModule, Node

_ROOT = Path(__file__).resolve().parent.parent
_FUNC_SIM = _ROOT.parent / "functional_sim"
for p in (_ROOT, _FUNC_SIM):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from build import DRAMWriter  # noqa: E402
from build_alexnet_layer import im2col  # noqa: E402
from codegen.c_emitter import (  # noqa: E402
    _align_data,
    _gemm_k_stride,
    _get_module,
    _padded_gemm_a,
    _to_bf16_array,
    _write_gemm_params,
    _write_gemm_rhs_weight,
    _write_matrix,
    _write_zeros,
    compile_and_assemble,
    render_in_file,
)
from codegen.dram_builder import extract_input_data  # noqa: E402
from graph.fx_capture import get_node_shape  # noqa: E402
from kernels.common import TILE  # noqa: E402
from kernels.gemm import gemm_c as _gemm_c  # noqa: E402
from run_graph import (  # noqa: E402
    _layer_compare_metrics,
    _read_bf16,
    _run_emulator,
    build_graph,
    load_model,
)


def _graph_get_attr_tensor(gm: GraphModule, n: Node) -> torch.Tensor:
    attr: Any = gm
    for part in str(n.target).split("."):
        attr = getattr(attr, part)
    if not isinstance(attr, torch.Tensor):
        raise TypeError(f"{n.target!r} is not a Tensor")
    return attr


def _conv_weight_bias_numpy(
    gm: GraphModule, node: Node
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Match ``emit_conv`` weight/bias resolution (call_module or F.conv2d)."""
    if node.op == "call_module":
        mod = _get_module(gm, node.target)
        if not isinstance(mod, nn.Conv2d):
            raise TypeError(f"expected Conv2d, got {type(mod)}")
        w = _to_bf16_array(mod.weight)
        b = (
            _to_bf16_array(mod.bias).reshape(-1)
            if getattr(mod, "bias", None) is not None
            else None
        )
        return w, b
    if node.op != "call_function":
        raise ValueError(f"unsupported conv op {node.op}")
    wn = node.args[1]
    if not isinstance(wn, Node):
        raise ValueError("conv weight arg is not a Node")
    if wn.op == "get_attr":
        w = _to_bf16_array(_graph_get_attr_tensor(gm, wn))
    else:
        raise ValueError(f"expected get_attr weight, got {wn.op}")
    b = None
    if len(node.args) >= 3 and node.args[2] is not None:
        bn = node.args[2]
        if isinstance(bn, Node) and bn.op == "get_attr":
            b = _to_bf16_array(_graph_get_attr_tensor(gm, bn)).reshape(-1)
        elif isinstance(bn, Node):
            raise ValueError(f"expected get_attr bias, got {bn.op}")
    return w, b


def list_graph_conv_nodes(
    gm: GraphModule,
) -> List[Tuple[str, str, Optional[Tuple[int, ...]]]]:
    out: List[Tuple[str, str, Optional[Tuple[int, ...]]]] = []
    for n in gm.graph.nodes:
        if n.meta.get("atalla_op") == "conv":
            in_nm = n.args[0].name if n.args and isinstance(n.args[0], Node) else "?"
            out.append((n.name, in_nm, get_node_shape(n)))
    return out


def extract_graph_conv_bundle(
    gm: GraphModule,
    ref_activations: Dict[str, np.ndarray],
    node_name: str,
) -> Dict[str, Any]:
    """Same tensors ``emit_conv`` / oracle would use for this FX node."""
    node = next((n for n in gm.graph.nodes if n.name == node_name), None)
    if node is None:
        raise ValueError(f"no graph node {node_name!r}")
    if node.meta.get("atalla_op") != "conv":
        raise ValueError(
            f"{node_name} atalla_op={node.meta.get('atalla_op')!r} (need 'conv')"
        )
    tc = node.meta.get("tile_config")
    if tc is None:
        raise ValueError("missing tile_config on node")
    p = tc.params
    stride, pad = int(p["stride"]), int(p["pad"])
    in_node = node.args[0]
    if not isinstance(in_node, Node):
        raise ValueError("conv input is not a Node")
    x = np.asarray(ref_activations[in_node.name], dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"expected NCHW activations, got shape {x.shape}")
    exp_c, exp_h, exp_w = int(p["C_in"]), int(p["H"]), int(p["W"])
    if x.shape[1] != exp_c or x.shape[2] != exp_h or x.shape[3] != exp_w:
        raise ValueError(
            f"activation shape {x.shape} vs tile_config (C_in,H,W)=({exp_c},{exp_h},{exp_w})"
        )
    w, bias = _conv_weight_bias_numpy(gm, node)
    return {
        "x": x,
        "w": w,
        "bias": bias,
        "stride": stride,
        "pad": pad,
        "params": p,
        "input_node": in_node.name,
        "node_name": node.name,
    }


def _ref_conv_bf16(
    x_nchw: np.ndarray,
    w_oihw: np.ndarray,
    bias: np.ndarray | None,
    stride: int,
    pad: int,
) -> np.ndarray:
    xt = torch.from_numpy(np.asarray(x_nchw, np.float32)).to(torch.bfloat16)
    wt = torch.from_numpy(np.asarray(w_oihw, np.float32)).to(torch.bfloat16)
    b = (
        torch.from_numpy(np.asarray(bias, np.float32)).to(torch.bfloat16)
        if bias is not None
        else None
    )
    y = F.conv2d(xt, wt, bias=b, stride=stride, padding=pad)
    return y.float().numpy()


def _run_conv(
    x_nchw: np.ndarray,
    w_oihw: np.ndarray,
    bias: np.ndarray | None,
    stride: int,
    pad: int,
    work_dir: str,
    tag: str,
) -> np.ndarray:
    """x: (1,Cin,H,W), w: (Cout,Cin,R,S), bias: (Cout,) or None."""
    x_nchw = np.asarray(x_nchw, dtype=np.float32)
    w_oihw = np.asarray(w_oihw, dtype=np.float32)
    _, Cin, H, W = x_nchw.shape
    Cout, Cin2, R, S = w_oihw.shape
    if Cin != Cin2:
        raise ValueError(f"Cin mismatch {Cin} vs {Cin2}")
    Ho = (H + 2 * pad - R) // stride + 1
    Wo = (W + 2 * pad - S) // stride + 1
    M = Ho * Wo
    N = Cout
    K = R * S * Cin
    input_nhwc = x_nchw.transpose(0, 2, 3, 1)
    A_mat = im2col(input_nhwc, 1, H, W, Cin, R, S, stride, pad)
    ks = _gemm_k_stride(K)
    A_dram = _padded_gemm_a(A_mat, K)
    weight_flat = (
        w_oihw.reshape(N, Cin, R, S).transpose(2, 3, 1, 0).reshape(K, N)
    )

    A_GMEM = 0x1000
    W_GMEM = A_GMEM + _align_data(M * ks * 2)
    C_GMEM = W_GMEM + _align_data(N * ks * 2)
    Z_GMEM = C_GMEM + M * N * 2

    img = DRAMWriter()
    _write_gemm_params(img, A_GMEM, W_GMEM, C_GMEM, M, N, K, Z_GMEM)
    _write_matrix(img, A_GMEM, A_dram, M, ks)
    _write_gemm_rhs_weight(img, W_GMEM, weight_flat)
    if bias is not None and bias.size == N:
        c_init = np.tile(np.asarray(bias, dtype=np.float32).reshape(1, N), (M, 1))
        _write_matrix(img, C_GMEM, c_init, M, N)
    else:
        _write_zeros(img, C_GMEM, M * N)
    _write_zeros(img, Z_GMEM, TILE)

    from codegen.c_emitter import LayerEmission  # noqa: E402

    em = LayerEmission()
    em.c_source = _gemm_c(M, N, K)
    em.dram = img
    em.output_addr = C_GMEM
    em.output_elements = M * N
    em.conv_post = {"Ho": Ho, "Wo": Wo, "C": N, "final_shape": (1, Cout, Ho, Wo)}

    os.makedirs(work_dir, exist_ok=True)
    compile_and_assemble(em, work_dir, tag)
    in_path = Path(work_dir) / f"{tag}.in"
    in_path.write_text(render_in_file(em))
    mem, _eu = _run_emulator(str(in_path), work_dir, tag)
    raw = _read_bf16(mem, C_GMEM, M * N).reshape(Ho, Wo, N)
    out = raw.transpose(2, 0, 1).reshape(1, Cout, Ho, Wo)
    return out


def _report(
    case: str,
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None,
    verbose: bool,
    *,
    work_dir: Optional[str] = None,
    tag: Optional[str] = None,
    **kw: Any,
) -> Dict[str, float]:
    work = work_dir or str(_ROOT / "out" / "debug_conv")
    safe_tag = tag or case.replace(" ", "_").replace("/", "_")
    emu = _run_conv(x, w, b, work_dir=work, tag=safe_tag, **kw)
    ref = _ref_conv_bf16(x, w, b, kw["stride"], kw["pad"])
    m = _layer_compare_metrics(ref, emu)
    print(f"\n=== {case} ===")
    print(
        f"  cos={m['cos_sim']:.6f}  rel_l2={m['rel_l2_error']:.6f}  "
        f"relmax={m['rel_max_abs_error']:.6f}  max_abs={m['max_abs_error']:.6f}"
    )
    if verbose:
        print(f"  out shape ref={ref.shape} emu={emu.shape}")
    return m


def save_graph_conv_npz(path: str, bundle: Dict[str, Any]) -> None:
    bias = bundle["bias"]
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    np.savez_compressed(
        path,
        x=bundle["x"],
        w=bundle["w"],
        bias=bias if bias is not None else np.zeros(0, np.float32),
        has_bias=np.asarray([bias is not None], dtype=np.bool_),
        stride=np.int32(bundle["stride"]),
        pad=np.int32(bundle["pad"]),
        node_name=np.array(bundle["node_name"]),
        input_node=np.array(bundle["input_node"]),
    )


def load_graph_conv_npz(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    has_bias = bool(z["has_bias"][0])
    bias_flat = z["bias"]
    bias = bias_flat if has_bias and bias_flat.size > 0 else None

    def _scalar_str(key: str) -> str:
        a = z[key]
        return str(a.item()) if isinstance(a, np.ndarray) and a.ndim == 0 else str(a)

    return {
        "x": np.asarray(z["x"], dtype=np.float32),
        "w": np.asarray(z["w"], dtype=np.float32),
        "bias": np.asarray(bias, dtype=np.float32) if bias is not None else None,
        "stride": int(z["stride"]),
        "pad": int(z["pad"]),
        "node_name": _scalar_str("node_name"),
        "input_node": _scalar_str("input_node"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case",
        choices=("3x3_s1", "cin2", "cout4", "bias", "all"),
        default="all",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--model",
        default="alexnet_small",
        help="Model for --list-graph-convs / --graph-node (see run_graph.load_model).",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=0.01,
        help="AlexNet channel scale (same as run_graph --scale).",
    )
    p.add_argument(
        "--list-graph-convs",
        action="store_true",
        help="Print conv FX node names and shapes (no emulator).",
    )
    p.add_argument(
        "--graph-node",
        metavar="NAME",
        default=None,
        help="Run harness on tensors from this conv node (matches oracle ref path).",
    )
    p.add_argument(
        "--dump-npz",
        metavar="PATH",
        default=None,
        help="With --graph-node, save x/w/bias/stride/pad for replay.",
    )
    p.add_argument(
        "--load-npz",
        metavar="PATH",
        default=None,
        help="Run harness from a file written by --dump-npz.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "--gate",
        action="store_true",
        help="Exit 1 if conv vs PyTorch BF16 ref fails (cos>=0.995, rel_l2<0.02).",
    )
    args = p.parse_args()

    def _gate_ok(m: Dict[str, float]) -> bool:
        return m["cos_sim"] >= 0.995 and m["rel_l2_error"] < 0.02

    if args.load_npz:
        b = load_graph_conv_npz(args.load_npz)
        m = _report(
            f"npz:{b['node_name']}",
            b["x"],
            b["w"],
            b["bias"],
            verbose=args.verbose,
            tag=f"npz_{b['node_name']}",
            stride=b["stride"],
            pad=b["pad"],
        )
        print("\nDone (from npz).")
        if args.gate and not _gate_ok(m):
            raise SystemExit(1)
        return

    if args.list_graph_convs or args.graph_node:
        torch.manual_seed(42)
        np.random.seed(42)
        model, example_input = load_model(args.model, args.scale)
        gm = build_graph(model, example_input, verbose=args.verbose)
        if args.list_graph_convs:
            rows = list_graph_conv_nodes(gm)
            print(f"Conv nodes ({args.model}, scale={args.scale}):")
            for name, src, sh in rows:
                print(f"  {name:20s}  input={src:20s}  out_shape={sh}")
            return
        assert args.graph_node is not None
        ref = extract_input_data(gm, example_input.bfloat16())
        bundle = extract_graph_conv_bundle(gm, ref, args.graph_node)
        if args.dump_npz:
            save_graph_conv_npz(args.dump_npz, bundle)
            print(f"Wrote {args.dump_npz}")
        if args.verbose:
            pr = bundle["params"]
            print(
                f"  tile_config: M={pr['M']} N={pr['N']} K={pr['K']} "
                f"R,S={pr['R']},{pr['S']} stride={bundle['stride']} pad={bundle['pad']} "
                f"H,W={pr['H']},{pr['W']} Cin={pr['C_in']}"
            )
            print(
                f"  x {bundle['x'].shape}  w {bundle['w'].shape}  "
                f"bias {None if bundle['bias'] is None else bundle['bias'].shape}  "
                f"<- activation[{bundle['input_node']!r}]"
            )
        m = _report(
            f"graph:{bundle['node_name']}",
            bundle["x"],
            bundle["w"],
            bundle["bias"],
            verbose=args.verbose,
            tag=f"g_{bundle['node_name']}",
            stride=bundle["stride"],
            pad=bundle["pad"],
        )
        print(
            "\nInterpret: if this **passes** but run_graph oracle for the same node fails, "
            "diff emit_conv vs this script (bias/weight get_attr without meta['val'] was one "
            "real bug). If the harness **fails**, debug im2col/GEMM on these tensors."
        )
        if args.gate and not _gate_ok(m):
            raise SystemExit(1)
        return

    rng = np.random.default_rng(args.seed)

    conv_gate_bad: List[str] = []

    def run(name: str, x: np.ndarray, w: np.ndarray, b: np.ndarray | None, **kw) -> None:
        m = _report(name, x, w, b, verbose=False, **kw)
        if args.gate and not _gate_ok(m):
            conv_gate_bad.append(name)

    if args.case in ("3x3_s1", "all"):
        x = rng.standard_normal((1, 1, 8, 8)).astype(np.float32) * 0.05
        w = rng.standard_normal((2, 1, 3, 3)).astype(np.float32) * 0.05
        run("3x3 s1p1 (1->2 ch)", x, w, None, stride=1, pad=1)
    if args.case in ("cin2", "all"):
        x = rng.standard_normal((1, 2, 6, 6)).astype(np.float32) * 0.05
        w = rng.standard_normal((3, 2, 3, 3)).astype(np.float32) * 0.05
        run("Cin=2 Cout=3 3x3 s1p1", x, w, None, stride=1, pad=1)
    if args.case in ("cout4", "all"):
        x = rng.standard_normal((1, 1, 7, 7)).astype(np.float32) * 0.05
        w = rng.standard_normal((4, 1, 3, 3)).astype(np.float32) * 0.05
        run("Cout=4 3x3 s1p1", x, w, None, stride=1, pad=1)
    if args.case in ("bias", "all"):
        x = rng.standard_normal((1, 1, 8, 8)).astype(np.float32) * 0.05
        w = rng.standard_normal((2, 1, 3, 3)).astype(np.float32) * 0.05
        b = rng.standard_normal((2,)).astype(np.float32) * 0.01
        run("bias 2 filters", x, w, b, stride=1, pad=1)

    print("\nDone. Artifacts under out/debug_conv/")
    print("Interpret: cos~1 & low rel_l2 => conv path matches PyTorch BF16 conv2d.")

    if args.gate and conv_gate_bad:
        print(f"\nGATE FAIL: {conv_gate_bad}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
