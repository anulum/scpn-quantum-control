#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S2 scaling row validator
"""Validate S2 scaling rows against the preregistered protocol."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from scpn_quantum_control.benchmarks.advantage_protocol import (
    default_s2_scaling_protocol,
    validate_scaling_rows,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "rows_json", type=Path, help="JSON file containing rows or {'rows': [...]}."
    )
    return parser.parse_args(argv)


def _rows_from_payload(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return payload["rows"]
    raise ValueError("rows JSON must be a list or an object with a 'rows' list")


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the S2 scaling row artefact against its schema."""
    args = _parse_args(argv)
    payload = json.loads(args.rows_json.read_text(encoding="utf-8"))
    rows = _rows_from_payload(payload)
    validation = validate_scaling_rows(default_s2_scaling_protocol(), rows)
    print(json.dumps(validation.to_dict(), indent=2))
    return 0 if validation.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
