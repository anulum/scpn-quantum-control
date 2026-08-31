# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — chimera-control evidence runner
"""Generate or byte-check deterministic chimera multiscale control evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from scpn_quantum_control.chimera_control import (
    build_chimera_multiscale_evidence,
    write_chimera_multiscale_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = REPO_ROOT / "data/chimera_multiscale_control/evidence.json"
DEFAULT_MARKDOWN = REPO_ROOT / "data/chimera_multiscale_control/evidence.md"


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for local evidence generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if committed evidence differs; do not write files",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=64,
        help="oscillators per population (committed evidence uses 64)",
    )
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-path", type=Path, default=DEFAULT_MARKDOWN)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build and write or check evidence, returning a process exit code."""
    args = build_parser().parse_args(argv)
    evidence = build_chimera_multiscale_evidence(population_size=args.population_size)
    json_path, markdown_path = write_chimera_multiscale_evidence(
        evidence,
        json_path=args.json_path,
        markdown_path=args.markdown_path,
        check=args.check,
    )
    mode = "checked" if args.check else "wrote"
    print(f"{mode} {json_path}")
    print(f"{mode} {markdown_path}")
    print(f"content_digest={evidence.content_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
