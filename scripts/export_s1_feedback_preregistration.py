#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S1 preregistration export
"""Export the no-submission S1 feedback preregistration package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scpn_quantum_control.control.realtime_feedback import (
    RealtimeFeedbackConfig,
    RealtimeSyncFeedbackController,
)
from scpn_quantum_control.hardware.feedback_capability_probe import (
    BackendCapabilitySnapshot,
    assess_feedback_backend_fleet,
    required_s1_dynamic_features,
)
from scpn_quantum_control.hardware.feedback_dryrun import build_s1_feedback_dry_run_bundle
from scpn_quantum_control.hardware.feedback_submission import (
    build_s1_feedback_submission_package,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "s1_feedback_loop"
DATE = "2026-05-06"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _controller() -> RealtimeSyncFeedbackController:
    return RealtimeSyncFeedbackController(
        np.array(
            [
                [0.0, 0.35, 0.20],
                [0.35, 0.0, 0.25],
                [0.20, 0.25, 0.0],
            ],
            dtype=np.float64,
        ),
        np.array([0.1, 0.4, 0.7], dtype=np.float64),
        config=RealtimeFeedbackConfig(
            target_r=0.72,
            deadband=0.03,
            base_dt=0.025,
            trotter_steps=1,
            measurement_shots=128,
            base_gain=0.6,
            max_gain=2.0,
            monitor_strength=0.18,
            correction_angle=0.12,
        ),
    )


def _manifest() -> dict[str, object]:
    package = build_s1_feedback_submission_package(
        _controller(),
        experiment_id="s1_dynamic_feedback_preregistration_2026-05-06",
        n_rounds=3,
        circuits=2,
        shots_per_circuit=1024,
        repetitions=12,
        estimated_seconds_per_circuit=1.0,
    )
    manifest = package.to_dict()
    manifest["export"] = {
        "date": DATE,
        "script": "scripts/export_s1_feedback_preregistration.py",
        "submission_state": "prepared_no_submission",
        "hardware_submission": False,
        "credential_access": False,
    }
    manifest["arms"] = [
        {
            "label": "feedback",
            "description": "Monitored dynamic-circuit feedback arm with conditional reset and rotation.",
        },
        {
            "label": "matched_open_loop_control",
            "description": (
                "Same oscillator family, shots, repetitions, and layout target without "
                "feedback action; required to decide whether feedback improves target-error."
            ),
        },
    ]
    manifest["provider_dry_runs"] = [
        dry_run.to_dict() for dry_run in build_s1_feedback_dry_run_bundle(package)
    ]
    features = required_s1_dynamic_features(package)
    manifest["capability_probe_examples"] = [
        decision.to_dict()
        for decision in assess_feedback_backend_fleet(
            (
                BackendCapabilitySnapshot(
                    provider="ibm",
                    backend_name="ibm_dynamic_metadata_template",
                    n_qubits=100,
                    supported_features=features,
                    max_shots=4096,
                    max_circuits=16,
                ),
                BackendCapabilitySnapshot(
                    provider="analog",
                    backend_name="analog_native_review_template",
                    n_qubits=64,
                    supported_features=("cross_shot_batches",),
                    max_shots=4096,
                    max_circuits=16,
                ),
            ),
            package,
        )
    ]
    return manifest


def _markdown(manifest: dict[str, object]) -> str:
    dossier = _required_mapping(manifest, "dossier")
    lines = [
        "# S1 Feedback Preregistration Manifest",
        "",
        f"Experiment ID: `{manifest['experiment_id']}`",
        "",
        "Submission state: prepared, no hardware submission.",
        "",
        "## Circuit Summary",
    ]
    circuit = _required_mapping(manifest, "circuit")
    for key, value in circuit.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Budget"])
    budget = _required_mapping(manifest, "budget")
    for key, value in budget.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Platform Readiness"])
    readiness = _required_sequence(manifest, "platform_readiness")
    for entry in readiness:
        entry = _required_entry(entry, "platform_readiness entry")
        lines.append(f"- `{entry['platform']}`: {entry['status']} ({'; '.join(entry['reasons'])})")
    lines.extend(["", "## Provider Dry-runs"])
    dry_runs = _required_sequence(manifest, "provider_dry_runs")
    for entry in dry_runs:
        entry = _required_entry(entry, "provider_dry_runs entry")
        lines.append(f"- `{entry['provider']}`: submission_enabled={entry['submission_enabled']}")
    lines.extend(["", "## Capability Probe Examples"])
    probe_examples = _required_sequence(manifest, "capability_probe_examples")
    for entry in probe_examples:
        entry = _required_entry(entry, "capability_probe_examples entry")
        lines.append(
            f"- `{entry['backend_name']}`: {entry['status']} ({'; '.join(entry['reasons'])})"
        )
    lines.extend(
        [
            "",
            "## Hardware Job Dossier",
            "",
            f"Purpose: {dossier['purpose']}",
            "",
            f"Hypothesis: {dossier['hypothesis']}",
            "",
            f"Falsification condition: {dossier['falsification_condition']}",
            "",
            f"Claim boundary: {dossier['claim_boundary']}",
            "",
            "## Reproducibility Package",
        ]
    )
    reproducibility = _required_mapping(dossier, "reproducibility_package")
    for key, value in reproducibility.items():
        lines.append(f"- `{key}`: {value}")
    return "\n".join(lines) + "\n"


def _required_mapping(mapping: dict[str, object], key: str) -> dict[str, object]:
    value = mapping[key]
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a mapping")
    return value


def _required_sequence(mapping: dict[str, object], key: str) -> list[object]:
    value = mapping[key]
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list")
    return value


def _required_entry(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return value


def main() -> int:
    """Write the S1 feedback-loop preregistration artefacts."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _manifest()
    json_path = OUT_DIR / f"s1_feedback_preregistration_{DATE}.json"
    md_path = OUT_DIR / f"s1_feedback_preregistration_{DATE}.md"
    json_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(manifest), encoding="utf-8")
    print(f"wrote_json={json_path}")
    print(f"wrote_markdown={md_path}")
    print(f"sha256_json={_sha256(json_path)}")
    print(f"sha256_markdown={_sha256(md_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
