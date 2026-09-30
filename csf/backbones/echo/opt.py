# SPDX-License-Identifier: Apache-2.0
"""Parser for the `opt.txt` that ECHO writes next to each checkpoint.

Mirrors `ECHO_CODE/generator/options/get_opt.py` but returns a fresh namespace
instead of updating an argparse one.  Values are coerced the same way:
"True"/"False" to bool, decimals to float, digit strings to int, comma lists
such as "[2, 2, 2, 2]" to list[int], anything else stays a string.  Keys the
file omits receive the robotv2 defaults that get_opt.py hard-codes.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

_ROBOTV2_DEFAULTS: dict[str, object] = {
    "joints_num": 30,
    "dim_pose": 38,
    "max_motion_length": 490,
    "fps": 50,
}

_SKIP_LINES = {
    "-------------- End ----------------",
    "------------ Options -------------",
    "",
}


def _is_float(value: str) -> bool:
    return bool(re.match(r"^[0-9]+\.[0-9]+$", value.strip().lstrip("+-")))


def _is_int(value: str) -> bool:
    return value.strip().lstrip("+-").isdigit()


def _coerce(value: str) -> object:
    value = value.strip()
    if value in ("True", "False"):
        return value == "True"
    if _is_float(value):
        return float(value)
    if _is_int(value):
        return int(value)
    if "," in value:
        return [int(item.strip()) for item in value.strip("[]").split(",")]
    return value


def parse_opt_txt(path: str) -> SimpleNamespace:
    """Parse an ECHO `opt.txt` (``key: value`` or ``key=value`` per line).

    Returns:
        A namespace of the coerced values, with robotv2 defaults filled in for
        any pipeline-required key the file omits.
    """
    opt: dict[str, object] = {}
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line in _SKIP_LINES:
                continue
            if ": " in line:
                key, _, value = line.partition(": ")
            elif "=" in line:
                key, _, value = line.partition("=")
            else:
                continue
            key = key.strip()
            if key:
                opt[key] = _coerce(value)
    for key, value in _ROBOTV2_DEFAULTS.items():
        opt.setdefault(key, value)
    return SimpleNamespace(**opt)
