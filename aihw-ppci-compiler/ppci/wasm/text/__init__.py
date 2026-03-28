"""Logic to process the text based representation of wasm."""

from .parser import load_from_s_tokens, load_s_expr, load_tuple

__all__ = ["load_tuple", "load_s_expr", "load_from_s_tokens"]
