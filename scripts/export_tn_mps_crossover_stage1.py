#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — TN/MPS crossover stage-1 export
"""Export the QWC-5.1 TN/MPS crossover stage-1 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from scpn_quantum_control.benchmarks.tn_mps_crossover_stage1 import (
    build_tn_mps_crossover_stage1,
    render_tn_mps_crossover_stage1_markdown,
)

DATE = "2026-07-08"
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "s2_advantage_scaling"
DOC_PATH = REPO_ROOT / "docs" / "tn_mps_crossover_stage1.md"


def _write_json(path: Path, payload: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8")
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse TN/MPS crossover stage-1 export options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DOC_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Write the QWC-5.1 TN/MPS crossover stage-1 artifacts."""
    args = parse_args(argv)
    report = build_tn_mps_crossover_stage1()
    json_path = args.out_dir / f"tn_mps_crossover_stage1_{DATE}.json"
    json_sha = _write_json(json_path, report.to_dict())
    markdown_sha = _write_text(args.doc_path, render_tn_mps_crossover_stage1_markdown(report))
    print(f"wrote {_display_path(json_path)} sha256={json_sha}")
    print(f"wrote {_display_path(args.doc_path)} sha256={markdown_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
