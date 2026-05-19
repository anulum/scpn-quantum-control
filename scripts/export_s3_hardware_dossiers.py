#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S3 hardware dossier export
"""Export no-submit hardware-job dossiers for promoted S3 candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scpn_quantum_control.hardware.job_dossier import HardwareJobDossier

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "s3_pulse_ansatz_design"
DATE = "2026-05-06"
ANSATZ_PATH = OUT_DIR / f"s3_ansatz_observable_validation_{DATE}.json"
PULSE_PATH = OUT_DIR / f"s3_pulse_feasibility_summary_{DATE}.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _best_ansatz_row(ansatz_summary: dict[str, Any]) -> dict[str, Any]:
    rows = ansatz_summary.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("ansatz summary must contain non-empty rows")
    mappings = [row for row in rows if isinstance(row, dict)]
    if len(mappings) != len(rows):
        raise ValueError("ansatz rows must be mappings")
    return min(mappings, key=lambda row: float(row["energy_absolute_error"]))


def _ready_pulse_targets(pulse_summary: dict[str, Any]) -> list[dict[str, Any]]:
    decisions = pulse_summary.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("pulse summary must contain decisions")
    ready = [
        decision
        for decision in decisions
        if isinstance(decision, dict) and decision.get("status") == "ready"
    ]
    if not ready:
        raise ValueError("pulse summary must contain at least one ready target")
    return ready


def _ansatz_dossier(row: dict[str, Any]) -> HardwareJobDossier:
    return HardwareJobDossier(
        job_id="s3_ansatz_observable_followup",
        title="S3 promoted ansatz observable follow-up run",
        purpose=(
            "Test whether the promoted structured Kuramoto-XY ansatz candidate retains "
            "its no-QPU observable behaviour under real backend transpilation, layout, and noise."
        ),
        hypothesis=(
            "If the structured candidate is hardware-usable, its measured observable proxies "
            "should remain directionally consistent with the exact statevector baseline under a "
            "matched simulator and backend layout package."
        ),
        falsification_condition=(
            "The candidate is falsified for hardware follow-up if transpilation depth, layout noise, "
            "or readout effects destroy the observed energy/synchronisation ordering relative to "
            "matched baseline candidates."
        ),
        expected_observables=(
            "energy expectation estimate",
            "synchronisation proxy",
            "transpiled depth and two-qubit gate count",
            "layout and readout metadata",
            "matched simulator comparison",
        ),
        circuit_summary={
            "candidate_label": row["candidate_label"],
            "n_qubits": row["n_qubits"],
            "depth": row["depth"],
            "size": row["size"],
            "two_qubit_gates": row["two_qubit_gates"],
            "parameters": row["parameters"],
        },
        qpu_budget={
            "status": "not_requested",
            "recommended_initial_scope": "transpile-only plus optional small shot-count observable run after approval",
            "estimated_execution_seconds": 0.0,
            "hardware_submission": False,
        },
        platform_fit={
            "gate_based": "manual_review_required",
            "pulse_level": "not_required_for_this_ansatz_dossier",
            "analogue": "separate native formulation required",
        },
        risks_and_confounds=(
            "The current observable row is not VQE optimisation.",
            "Energy estimation may require many Pauli measurements if promoted beyond simulation.",
            "The shallow resource winner may not be the best accuracy winner after noise.",
            "Backend layout and readout can dominate the observable proxy.",
        ),
        decision_tree={
            "positive": "Promote to a small hardware observable comparison against matched ansatz baselines.",
            "null": "Keep as software-only design evidence and improve candidate generation.",
            "negative": "Reject this resource-ranked candidate for hardware follow-up and use accuracy-ranked candidates instead.",
        },
        paper_impact=(
            "Would support the S3 design-methods narrative by connecting no-QPU candidate selection "
            "to a bounded hardware-readiness package."
        ),
        follow_up_avenue="Provider-specific transpilation probes, then small observable runs only after budget approval.",
        possibilities_opened=(
            "budget-justified ansatz hardware screening",
            "layout-aware candidate promotion",
            "observable-driven candidate rejection before expensive QPU use",
        ),
        claim_boundary=(
            "This dossier does not authorise submission and does not claim VQE improvement, pulse-level "
            "performance, quantum advantage, or backend-independent behaviour."
        ),
        reproducibility_package={
            "ansatz_validation": str(ANSATZ_PATH.relative_to(REPO_ROOT)),
            "dossier_script": "scripts/export_s3_hardware_dossiers.py",
            "bench_command": "scpn-bench s3-hardware-dossiers",
        },
        prerequisites=(
            "run transpile-only backend feasibility checks",
            "attach exact measurement grouping before hardware execution",
            "obtain explicit QPU-budget approval",
        ),
    )


def _pulse_dossier(targets: list[dict[str, Any]]) -> HardwareJobDossier:
    first = targets[0]
    schedule = first["schedule"]
    if not isinstance(schedule, dict):
        raise ValueError("pulse schedule summary must be a mapping")
    return HardwareJobDossier(
        job_id="s3_pulse_feasibility_followup",
        title="S3 pulse-schedule feasibility follow-up run",
        purpose=(
            "Test whether the hypergeometric Kuramoto-XY pulse schedule can be mapped to a "
            "provider-supported pulse-control or native-XY execution path after calibration review."
        ),
        hypothesis=(
            "If provider timing, pulse-count, and native interaction constraints are satisfied, a "
            "small calibrated pulse follow-up can test whether pulse shaping reduces digital "
            "Trotter overhead for the same observable family."
        ),
        falsification_condition=(
            "The pulse route is falsified for near-term hardware if provider calibration, timing "
            "granularity, pulse duration, or native interaction constraints cannot realise the schedule "
            "within the documented budget and error boundary."
        ),
        expected_observables=(
            "provider pulse feasibility status",
            "calibrated duration and sample spacing",
            "pulse-count and qubit-count fit",
            "matched digital ansatz observable comparison",
            "post-calibration error budget",
        ),
        circuit_summary={
            "schedule_family": "hypergeometric_trotter_step",
            "n_qubits": schedule["n_qubits"],
            "pulse_count": schedule["pulse_count"],
            "total_time": schedule["total_time"],
            "min_sample_spacing": schedule["min_sample_spacing"],
            "ready_targets": ", ".join(str(target["backend_name"]) for target in targets),
        },
        qpu_budget={
            "status": "not_requested",
            "recommended_initial_scope": "provider calibration review only; no pulse job before approval",
            "estimated_execution_seconds": 0.0,
            "hardware_submission": False,
        },
        platform_fit={str(target["backend_name"]): str(target["status"]) for target in targets},
        risks_and_confounds=(
            "Metadata readiness is not calibration readiness.",
            "Pulse schedules may require provider-specific waveform constraints not captured in metadata.",
            "Native-XY analogue execution is not equivalent to a gate-level pulse schedule without a separate mapping.",
            "The current analytic infidelity proxy is not a measured hardware error model.",
        ),
        decision_tree={
            "positive": "Prepare a provider-specific calibrated pulse preregistration package.",
            "manual_review": "Request platform-specific waveform constraints before any execution.",
            "negative": "Keep S3 pulse work as simulation-only and prioritise gate-based ansatz validation.",
        },
        paper_impact=(
            "Would define the bridge from S3 design methods to S4 multi-hardware pulse-level control."
        ),
        follow_up_avenue="S4 provider-specific pulse adapter and calibrated no-submit waveform package.",
        possibilities_opened=(
            "analogue/native XY follow-up planning",
            "pulse-vs-digital Trotter comparison",
            "provider-specific QPU budget justification",
        ),
        claim_boundary=(
            "This dossier is metadata-only and does not calibrate pulses, open provider sessions, submit jobs, "
            "or establish hardware performance."
        ),
        reproducibility_package={
            "pulse_feasibility": str(PULSE_PATH.relative_to(REPO_ROOT)),
            "dossier_script": "scripts/export_s3_hardware_dossiers.py",
            "bench_command": "scpn-bench s3-hardware-dossiers",
        },
        prerequisites=(
            "obtain provider-specific waveform and calibration constraints",
            "run no-submit waveform compilation",
            "obtain explicit QPU-budget approval",
        ),
    )


def _markdown(dossiers: list[HardwareJobDossier]) -> str:
    return "\n\n".join(dossier.to_markdown() for dossier in dossiers)


def main() -> int:
    """Write the S3 hardware dossier artefacts."""
    ansatz_summary = _load_json(ANSATZ_PATH)
    pulse_summary = _load_json(PULSE_PATH)
    dossiers = [
        _ansatz_dossier(_best_ansatz_row(ansatz_summary)),
        _pulse_dossier(_ready_pulse_targets(pulse_summary)),
    ]
    summary = {
        "date": DATE,
        "script": "scripts/export_s3_hardware_dossiers.py",
        "hardware_submission": False,
        "provider_session_opened": False,
        "dossiers": [dossier.to_dict() for dossier in dossiers],
        "claim_boundary": "Dossiers document readiness only; they do not authorise hardware execution.",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"s3_hardware_dossiers_{DATE}.json"
    md_path = OUT_DIR / f"s3_hardware_dossiers_{DATE}.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(dossiers), encoding="utf-8")
    print(f"wrote_json={json_path}")
    print(f"wrote_markdown={md_path}")
    print(f"sha256_json={_sha256(json_path)}")
    print(f"sha256_markdown={_sha256(md_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
