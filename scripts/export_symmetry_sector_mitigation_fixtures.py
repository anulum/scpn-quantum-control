#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- symmetry-sector mitigation fixture exporter
"""Export deterministic symmetry-sector mitigation planner fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path

from scpn_quantum_control.mitigation.symmetry_sector_fixtures import (
    DEFAULT_DOC_PATH,
    DEFAULT_OUT_DIR,
    fixture_markdown,
    fixture_payload,
    write_json,
    write_text,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Export planner fixtures."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / DEFAULT_OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=REPO_ROOT / DEFAULT_DOC_PATH)
    args = parser.parse_args()
    data = fixture_payload()
    json_path = args.out_dir / "symmetry_sector_mitigation_fixtures.json"
    json_digest = write_json(json_path, data)
    doc_digest = write_text(args.doc_path, fixture_markdown(data))
    print(f"wrote {json_path.relative_to(REPO_ROOT)} sha256={json_digest}")
    print(f"wrote {args.doc_path.relative_to(REPO_ROOT)} sha256={doc_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
