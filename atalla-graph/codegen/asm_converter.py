"""Convert compiler-output assembly (.s) to emulator-compatible assembly.

Handles notation differences between the ppci AtallaC compiler output and
the functional_sim assembler input:
  - Mnemonic separators: _ -> . (add_s -> add.s)
  - Special mnemonics: nop -> nop.s, halt -> halt.s
  - Scalar registers: xN -> $N
  - Vector registers: vN -> $N
  - Memory operands: imm(xN) -> imm($N)
  - Mask registers in operands: mK -> K (integer)
  - Strip directives (.section, .align, global, type)
"""
from __future__ import annotations

import re
from typing import List

_SCALAR_REG_RE = re.compile(r'\bx(\d+)\b')
_VEC_REG_RE = re.compile(r'\bv(\d+)\b')
_MASK_REG_RE = re.compile(r'\bm(\d+)\b')
_MEM_RE = re.compile(r'(-?(?:0x[0-9a-fA-F]+|\d+))\(\s*x(\d+)\s*\)')
_LABEL_LINE_RE = re.compile(r'^(\s*[A-Za-z_]\w*\s*:\s*)$')
_LABEL_PREFIX_RE = re.compile(r'^(\s*[A-Za-z_]\w*\s*:\s*)(.*)')
_DIRECTIVE_RE = re.compile(r'^\s*(?:\.section|\.align|\.data|global|type)\b', re.IGNORECASE)

_MNEMONIC_REMAP = {
    "nop": "nop.s",
    "halt": "halt.s",
}


def _convert_mnemonic(mnem: str) -> str:
    low = mnem.strip().lower()
    if low in _MNEMONIC_REMAP:
        return _MNEMONIC_REMAP[low]
    if low in ("jal", "jalr", "ret"):
        return low
    return low.replace("_", ".")


def _convert_operand(op: str, mnemonic: str, idx: int, total: int) -> str:
    """Convert a single operand token."""
    s = op.strip()

    mem_m = _MEM_RE.match(s)
    if mem_m:
        return f"{mem_m.group(1)}(${mem_m.group(2)})"

    mask_m = _MASK_REG_RE.fullmatch(s)
    if mask_m:
        return mask_m.group(1)

    vec_m = _VEC_REG_RE.fullmatch(s)
    if vec_m:
        return f"${vec_m.group(1)}"

    scalar_m = _SCALAR_REG_RE.fullmatch(s)
    if scalar_m:
        return f"${scalar_m.group(1)}"

    return s


def convert_line(line: str) -> str | None:
    """Convert a single assembly line. Returns None if line should be dropped."""
    stripped = line.strip()
    if not stripped:
        return ""

    if _DIRECTIVE_RE.match(stripped):
        return None

    comment = ""
    if "#" in stripped:
        code_part, comment = stripped.split("#", 1)
        comment = " # " + comment.strip()
        stripped = code_part.strip()
        if not stripped:
            return comment

    label_prefix = ""
    m = _LABEL_PREFIX_RE.match(stripped)
    if m and m.group(2).strip():
        label_prefix = m.group(1)
        stripped = m.group(2).strip()
    elif _LABEL_LINE_RE.match(stripped):
        return stripped + comment

    if not stripped:
        return label_prefix.rstrip() + comment if label_prefix else comment

    parts = stripped.split(None, 1)
    raw_mnem = parts[0]
    ops_str = parts[1] if len(parts) > 1 else ""

    mnemonic = _convert_mnemonic(raw_mnem)

    if ops_str:
        ops = [o.strip() for o in ops_str.split(",")]
        converted_ops = [
            _convert_operand(o, mnemonic, i, len(ops))
            for i, o in enumerate(ops)
        ]
        result = f"{label_prefix}{mnemonic} {', '.join(converted_ops)}{comment}"
    else:
        result = f"{label_prefix}{mnemonic}{comment}"

    return result


def convert(asm_text: str) -> str:
    """Convert full compiler assembly output to emulator-compatible assembly."""
    out_lines: List[str] = []
    for line in asm_text.splitlines():
        converted = convert_line(line)
        if converted is not None:
            out_lines.append(converted)
    return "\n".join(out_lines)
