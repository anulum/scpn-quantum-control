#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — topology-kernel product evidence runner
"""Regenerate or byte-check deterministic topology-kernel topology-kernel evidence."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from scpn_quantum_control.topology_kernel_product import (
    TopologyKernelConfig,
    build_topology_kernel_evidence,
    write_topology_kernel_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = REPO_ROOT / "data/topology_kernel_product/evidence.json"
DEFAULT_MARKDOWN = REPO_ROOT / "data/topology_kernel_product/evidence.md"


def build_parser() -> argparse.ArgumentParser:
    """Return the bounded local evidence-runner argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=880)
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-path", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build frozen evidence, then write or byte-check both artefacts."""
    arguments = build_parser().parse_args(argv)
    evidence = build_topology_kernel_evidence(
        config=TopologyKernelConfig(),
        seed=arguments.seed,
    )
    paths = write_topology_kernel_evidence(
        evidence,
        arguments.json_path,
        arguments.markdown_path,
        check=arguments.check,
    )
    verb = "checked" if arguments.check else "wrote"
    for path in paths:
        print(f"{verb} {path}")
    print(f"content_digest={evidence.content_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
