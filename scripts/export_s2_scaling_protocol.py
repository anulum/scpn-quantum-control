#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — export s2 scaling protocol script
# scpn-quantum-control -- S2 scaling protocol export
"""Export the S2 scaling benchmark protocol manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scpn_quantum_control.benchmarks.advantage_protocol import default_s2_scaling_protocol

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "s2_advantage_scaling"
DATE = "2026-05-06"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    """Write the S2 scaling protocol artefacts."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    protocol = default_s2_scaling_protocol().to_dict()
    protocol["export"] = {
        "date": DATE,
        "script": "scripts/export_s2_scaling_protocol.py",
        "hardware_submission": False,
        "advantage_claim": False,
    }
    json_path = OUT_DIR / f"s2_scaling_protocol_{DATE}.json"
    json_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(f"wrote_json={json_path}")
    print(f"sha256_json={_sha256(json_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
