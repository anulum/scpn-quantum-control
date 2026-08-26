#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — DLA/topology evidence runner
"""Regenerate or byte-check deterministic topology-control evidence."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from scpn_quantum_control.dla_topology_control import (
    build_dla_topology_control_evidence,
    write_dla_topology_control_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = REPO_ROOT / "data/dla_topology_control/evidence.json"
DEFAULT_MARKDOWN = REPO_ROOT / "data/dla_topology_control/evidence.md"


def build_parser() -> argparse.ArgumentParser:
    """Build the fail-closed local evidence-runner parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=540)
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-path", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build evidence and write or byte-check both canonical artefacts."""
    arguments = build_parser().parse_args(argv)
    evidence = build_dla_topology_control_evidence(
        n_qubits=arguments.n_qubits,
        seed=arguments.seed,
    )
    paths = write_dla_topology_control_evidence(
        evidence,
        json_path=arguments.json_path,
        markdown_path=arguments.markdown_path,
        check=arguments.check,
    )
    verb = "checked" if arguments.check else "wrote"
    for path in paths:
        print(f"{verb} {path}")
    print(f"content_digest={evidence.content_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
