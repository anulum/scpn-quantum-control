#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S3 pulse feasibility probe
"""Export no-submit S3 pulse feasibility decisions from provider metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27
from scpn_quantum_control.hardware.pulse_feasibility import (
    assess_pulse_provider_fleet,
    pulse_snapshot_from_metadata,
)
from scpn_quantum_control.phase.pulse_shaping import build_trotter_pulse_schedule

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "s3_pulse_ansatz_design"
DATE = "2026-05-06"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, help="Optional provider metadata JSON list.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_metadata() -> list[dict[str, Any]]:
    return [
        {
            "provider": "ibm_pulse",
            "backend_name": "ibm_pulse_metadata_template",
            "n_qubits": 127,
            "supports_pulse_control": True,
            "supports_native_xy": False,
            "min_time_step": 0.0001,
            "max_pulse_duration": 1.0,
            "max_pulses": 64,
            "supported_features": ["pulse_control", "calibrated_drives"],
        },
        {
            "provider": "neutral_atom_analog",
            "backend_name": "neutral_atom_xy_review_template",
            "n_qubits": 64,
            "supports_pulse_control": False,
            "supports_native_xy": True,
            "min_time_step": 0.001,
            "max_pulse_duration": 10.0,
            "max_pulses": 16,
            "supported_features": ["native_xy", "global_detuning"],
        },
        {
            "provider": "metadata_light",
            "backend_name": "unknown_pulse_target",
            "n_qubits": 8,
            "supports_pulse_control": False,
            "supports_native_xy": False,
            "supported_features": [],
        },
    ]


def _load_metadata(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return _default_metadata()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("metadata JSON must be a list")
    rows: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, Mapping):
            raise ValueError("metadata entries must be mappings")
        rows.append(dict(item))
    return rows


def _markdown(summary: dict[str, Any]) -> str:
    decisions = summary["decisions"]
    if not isinstance(decisions, list):
        raise TypeError("decisions must be a list")
    lines = [
        "# S3 Pulse Feasibility Probe",
        "",
        "Submission state: no provider session and no hardware submission.",
        "",
        "## Decisions",
    ]
    for decision in decisions:
        if not isinstance(decision, dict):
            raise TypeError("decision must be a mapping")
        lines.append(
            f"- `{decision['backend_name']}` ({decision['provider']}): {decision['status']} "
            f"({'; '.join(decision['reasons'])})"
        )
    lines.extend(["", "## Claim Boundary", str(summary["claim_boundary"])])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    schedule = build_trotter_pulse_schedule(4, build_knm_paper27(4), t_step=0.2)
    providers = tuple(pulse_snapshot_from_metadata(row) for row in _load_metadata(args.metadata))
    decisions = assess_pulse_provider_fleet(providers, schedule)
    summary = {
        "date": DATE,
        "script": "scripts/probe_s3_pulse_feasibility.py",
        "hardware_submission": False,
        "provider_session_opened": False,
        "schedule_family": "hypergeometric_trotter_step",
        "decisions": [decision.to_dict() for decision in decisions],
        "claim_boundary": (
            "This is a metadata-only feasibility probe. It does not calibrate pulses, "
            "open provider sessions, submit jobs, or establish hardware performance."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"s3_pulse_feasibility_summary_{DATE}.json"
    md_path = args.out_dir / f"s3_pulse_feasibility_summary_{DATE}.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(summary), encoding="utf-8")
    print(f"wrote_json={json_path}")
    print(f"wrote_markdown={md_path}")
    print(f"sha256_json={_sha256(json_path)}")
    print(f"sha256_markdown={_sha256(md_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
