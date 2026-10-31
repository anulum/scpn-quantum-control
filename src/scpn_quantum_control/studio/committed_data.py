# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — shared write and drift check of committed Studio data
"""Write, print and check the committed JSON data of Studio instruments.

Every instrument builds its data from source owners and commits the result.
This module holds what they share: the exact serialisation, the comparison of
a committed file with a fresh build under the frozen float64 budget, and the
``--write`` / ``--check`` command line.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path


def _within_budget(actual: float, reference: float) -> bool:
    """Frozen float64 budget: ``|actual - reference| <= 1e-10 + 1e-8 |reference|``."""
    return abs(actual - reference) <= 1e-10 + 1e-8 * abs(reference)


def differences(committed: object, fresh: object, path: str = "$") -> list[str]:
    """List where a committed payload departs from a fresh build.

    Floats agree within the frozen float64 budget, since library and platform
    numerics may differ in their last bits; counts, digests, labels and
    decisions must be equal.

    Parameters
    ----------
    committed
        Decoded committed payload.
    fresh
        Freshly built payload.
    path
        Location prefix used in the messages.

    Returns
    -------
    list[str]
        One message per difference, empty when the payloads agree.

    """
    if isinstance(fresh, float) and isinstance(committed, int | float):
        if isinstance(committed, bool) or not _within_budget(float(committed), fresh):
            return [f"{path}: {committed!r} is outside the budget of {fresh!r}"]
        return []
    if isinstance(fresh, dict) and isinstance(committed, dict):
        if set(fresh) != set(committed):
            return [f"{path}: keys {sorted(committed)} != {sorted(fresh)}"]
        return [
            message
            for key in sorted(fresh)
            for message in differences(committed[key], fresh[key], f"{path}.{key}")
        ]
    if isinstance(fresh, list) and isinstance(committed, list):
        if len(fresh) != len(committed):
            return [f"{path}: length {len(committed)} != {len(fresh)}"]
        return [
            message
            for index, (left, right) in enumerate(zip(committed, fresh, strict=True))
            for message in differences(left, right, f"{path}[{index}]")
        ]
    return [] if committed == fresh else [f"{path}: {committed!r} != {fresh!r}"]


def serialise(payload: Mapping[str, object]) -> str:
    """Serialise a payload exactly as it is committed.

    Parameters
    ----------
    payload
        Artefact payload.

    Returns
    -------
    str
        Sorted, indented JSON with a trailing newline.

    """
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def run_committed_data_cli(
    argv: Sequence[str] | None,
    *,
    build: Callable[[], Mapping[str, object]],
    default_path: Path,
    label: str,
    description: str | None,
) -> int:
    """Print, write or check one instrument's committed data.

    Parameters
    ----------
    argv
        Argument vector, or ``None`` for ``sys.argv[1:]``.
    build
        Builds the payload from its source owners.
    default_path
        Committed JSON path used when ``--json-path`` is absent.
    label
        Name of the data in the printed messages.
    description
        Command-line help text.

    Returns
    -------
    int
        ``0`` on success, ``1`` when the committed file differs from a fresh
        build, ``2`` when it cannot be read.

    """
    parser = argparse.ArgumentParser(description=description)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="write the committed artefact")
    mode.add_argument("--check", action="store_true", help="compare it with a fresh build")
    parser.add_argument(
        "--json-path", type=Path, default=default_path, help="committed JSON artefact path"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    fresh = serialise(build())
    if args.check:
        try:
            committed = json.loads(args.json_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            print(f"{label}: unreadable: {exc}", file=sys.stderr)
            return 2
        found = differences(committed, json.loads(fresh))
        if not found:
            print(f"{label}: current")
            return 0
        for message in found:
            print(f"{label}: drift: {message}", file=sys.stderr)
        return 1
    if args.write:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(fresh, encoding="utf-8")
        print(f"wrote {args.json_path}")
        return 0
    print(fresh, end="")
    return 0


__all__ = ["differences", "run_committed_data_cli", "serialise"]
