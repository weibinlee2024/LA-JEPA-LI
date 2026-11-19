"""Correct LIBERO-Plus instructions.

The released evaluator builds the instruction from the variant file name, so
8493 of 10030 variants receive the training sentence plus the perturbation's
parameters (``view ...``, ``light 1``, ``initstate 0``, ...). La-JEPA evaluates
non-language variants with the base task's training instruction verbatim.
Language variants keep their paraphrase: that paraphrase is the perturbation.

Official per-variant files were not in the empty upstream repository when
this reproduction was written. Drop a JSON map of ``variant_id -> instruction``
into ``assets/libero_plus_fixed_instructions/`` and it takes priority. Until
then the rule above is applied at evaluation time.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

LANGUAGE_CATEGORIES = {"language", "lang", "instruction"}

# Trailing parameter fragments appended by the released evaluator.
_SUFFIX = re.compile(
    r"(?:"
    r"\s+view(?:\s+-?\d+(?:\.\d+)?){1,8}"
    r"|\s+initstate\s+\d+"
    r"|\s+init_state\s+\d+"
    r"|\s+light\s+\d+"
    r"|\s+noise(?:\s+-?\d+(?:\.\d+)?){1,6}"
    r"|\s+background\s+\S+"
    r"|\s+robot\s+\S+"
    r"|\s+layout\s+\S+"
    r"|\s+table\s+\d+"
    r"|\s+camera\s+\S+"
    r"|\s+texture\s+\S+"
    r")+",
    re.IGNORECASE,
)


def strip_perturbation_suffix(instruction: str) -> str:
    """Remove a trailing chain of perturbation parameters, if one is present."""
    cleaned = _SUFFIX.sub("", instruction).strip()
    return re.sub(r"\s+", " ", cleaned)


def corrected_instruction(
    raw_instruction: str,
    category: str | None = None,
    base_instruction: str | None = None,
) -> str:
    """Return the instruction La-JEPA should condition on.

    Language perturbations keep ``raw_instruction``. Every other category uses
    the base-task training instruction when it is known, otherwise the raw
    string with the leaked parameter suffix removed.
    """
    if category is not None and category.lower() in LANGUAGE_CATEGORIES:
        return raw_instruction.strip()
    if base_instruction:
        return base_instruction.strip()
    return strip_perturbation_suffix(raw_instruction)


def load_instruction_overrides(directory: str | Path) -> dict[str, str]:
    root = Path(directory)
    overrides: dict[str, str] = {}
    if not root.exists():
        return overrides
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text())
        mapping = payload.get("instructions", payload)
        if isinstance(mapping, dict):
            for key, value in mapping.items():
                if key in {"source", "examples"} or not isinstance(value, str):
                    continue
                overrides[str(key)] = value
    return overrides


def resolve_instruction(
    variant_id: str,
    raw_instruction: str,
    category: str | None = None,
    base_instruction: str | None = None,
    overrides: dict[str, str] | None = None,
) -> str:
    if overrides and variant_id in overrides:
        return overrides[variant_id]
    return corrected_instruction(raw_instruction, category, base_instruction)
