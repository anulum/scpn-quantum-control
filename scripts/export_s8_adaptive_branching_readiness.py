# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — export s8 adaptive branching readiness script
# scpn-quantum-control -- S8 adaptive-branching readiness export
"""Export the S8 adaptive-branching readiness artefacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scpn_quantum_control.control.adaptive_branching import (
    s8_adaptive_branching_markdown,
    s8_adaptive_branching_payload,
)

DATE = "2026-05-20"
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "s8_adaptive_branching"
DOC_PATH = REPO_ROOT / "docs" / "adaptive_branching.md"


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


def parse_args() -> argparse.Namespace:
    """Parse S8 readiness export options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DOC_PATH)
    return parser.parse_args()


def main() -> int:
    """Write the S8 adaptive-branching readiness JSON and Markdown artefacts."""

    args = parse_args()
    payload = s8_adaptive_branching_payload()
    json_path = args.out_dir / f"adaptive_branching_readiness_{DATE}.json"
    sha_json = _write_json(json_path, payload)
    sha_md = _write_text(args.doc_path, s8_adaptive_branching_markdown(payload))
    print(f"wrote {_display_path(json_path)} sha256={sha_json}")
    print(f"wrote {_display_path(args.doc_path)} sha256={sha_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
