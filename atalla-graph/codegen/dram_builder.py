"""Serialize PyTorch weight/activation tensors to .in data format.

Provides utilities to extract all weight tensors from a traced GraphModule
and write them (plus placeholder activations) into the DRAM format expected
by the Atalla emulator.
"""
from __future__ import annotations

import struct
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.fx import GraphModule, Node


def bf16_from_float(x: float) -> int:
    """Convert a Python float to bfloat16 (top 16 bits of float32)."""
    bits = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    return (bits >> 16) & 0xFFFF


def float_from_bf16(bits: int) -> float:
    """Convert bfloat16 bits back to Python float."""
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def tensor_to_bf16_bytes(tensor: torch.Tensor) -> bytes:
    """Convert a PyTorch tensor to a byte string of bf16 values in little-endian."""
    arr = tensor.detach().float().cpu().numpy().flatten()
    out = bytearray(len(arr) * 2)
    for i, v in enumerate(arr):
        bf = bf16_from_float(float(v))
        out[i * 2] = bf & 0xFF
        out[i * 2 + 1] = (bf >> 8) & 0xFF
    return bytes(out)


def extract_weights(gm: GraphModule) -> Dict[str, torch.Tensor]:
    """Extract all get_attr weight tensors from the graph."""
    weights = {}
    for node in gm.graph.nodes:
        if node.op == "get_attr":
            attr = gm
            for part in node.target.split("."):
                attr = getattr(attr, part)
            if isinstance(attr, (torch.Tensor, torch.nn.Parameter)):
                weights[node.name] = attr.detach()
        elif node.op == "call_module":
            mod = gm
            for part in node.target.split("."):
                mod = getattr(mod, part)
            if hasattr(mod, 'weight') and mod.weight is not None:
                weights[f"{node.name}_weight"] = mod.weight.detach()
            if hasattr(mod, 'bias') and mod.bias is not None:
                weights[f"{node.name}_bias"] = mod.bias.detach()
    return weights


def extract_input_data(
    gm: GraphModule,
    example_input: torch.Tensor,
) -> Dict[str, np.ndarray]:
    """Run the real PyTorch model to get intermediate activations for validation."""
    activations: Dict[str, np.ndarray] = {}

    env: Dict[str, torch.Tensor] = {}
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            env[node.name] = example_input
            activations[node.name] = example_input.detach().float().cpu().numpy()
        elif node.op == "get_attr":
            attr = gm
            for part in node.target.split("."):
                attr = getattr(attr, part)
            env[node.name] = attr if isinstance(attr, torch.Tensor) else torch.tensor(attr)
        elif node.op == "call_function":
            args = tuple(env[a.name] if isinstance(a, Node) else a for a in node.args)
            kwargs = {k: env[v.name] if isinstance(v, Node) else v for k, v in node.kwargs.items()}
            env[node.name] = node.target(*args, **kwargs)
        elif node.op == "call_method":
            self_obj = env[node.args[0].name]
            args = tuple(env[a.name] if isinstance(a, Node) else a for a in node.args[1:])
            kwargs = {k: env[v.name] if isinstance(v, Node) else v for k, v in node.kwargs.items()}
            env[node.name] = getattr(self_obj, node.target)(*args, **kwargs)
        elif node.op == "call_module":
            mod = gm
            for part in node.target.split("."):
                mod = getattr(mod, part)
            args = tuple(env[a.name] if isinstance(a, Node) else a for a in node.args)
            kwargs = {k: env[v.name] if isinstance(v, Node) else v for k, v in node.kwargs.items()}
            env[node.name] = mod(*args, **kwargs)
        elif node.op == "output":
            args = node.args[0]
            if isinstance(args, Node):
                activations["output"] = env[args.name].detach().float().cpu().numpy()
            elif isinstance(args, (tuple, list)):
                for a in args:
                    if isinstance(a, Node):
                        activations["output"] = env[a.name].detach().float().cpu().numpy()
                        break

        if node.name in env and isinstance(env[node.name], torch.Tensor):
            activations[node.name] = env[node.name].detach().float().cpu().numpy()

    return activations
