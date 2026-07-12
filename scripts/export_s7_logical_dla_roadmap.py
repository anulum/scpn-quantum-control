# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — export s7 logical DLA roadmap script
# scpn-quantum-control -- S7 logical-DLA parity roadmap export
"""Export the S7 logical-DLA parity roadmap artefacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scpn_quantum_control.qec.logical_dla_parity import (
    logical_dla_parity_markdown,
    logical_dla_parity_payload,
)

DATE = "2026-05-20"
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "s7_logical_dla_parity"
DOC_PATH = REPO_ROOT / "docs" / "logical_dla_parity.md"


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
    """Return a concise path for repo-local output and a full path otherwise."""

    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    """Parse S7 roadmap export options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DOC_PATH)
    return parser.parse_args()


def main() -> int:
    """Write the S7 logical-DLA roadmap JSON and Markdown artefacts."""

    args = parse_args()
    payload = logical_dla_parity_payload()
    json_path = args.out_dir / f"logical_dla_parity_roadmap_{DATE}.json"
    sha_json = _write_json(json_path, payload)
    sha_md = _write_text(args.doc_path, logical_dla_parity_markdown(payload))
    print(f"wrote {_display_path(json_path)} sha256={sha_json}")
    print(f"wrote {_display_path(args.doc_path)} sha256={sha_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
