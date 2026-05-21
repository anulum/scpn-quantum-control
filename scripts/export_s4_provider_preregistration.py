# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S4 provider preregistration dossier export
"""Export the no-submit S4 IBM pulse-level calibration preregistration dossier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scpn_quantum_control.hardware.job_dossier import HardwareJobDossier
from scpn_quantum_control.hardware.provider_capability_discovery import (
    ProviderCapabilitySnapshot,
    build_openpulse_control_readiness,
)

DATE = "2026-05-06"
REPO_ROOT = Path(__file__).resolve().parents[1]
S4_DIR = REPO_ROOT / "data" / "s4_multi_hardware_control"
READINESS_PATH = S4_DIR / f"s4_multi_hardware_readiness_{DATE}.json"
JSON_PATH = S4_DIR / f"s4_ibm_pulse_preregistration_{DATE}.json"
MD_PATH = REPO_ROOT / "docs" / f"s4_ibm_pulse_preregistration_{DATE}.md"


def _load_readiness(path: Path = READINESS_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("S4 readiness payload must be a JSON object")
    if payload.get("hardware_submission") is not False:
        raise ValueError("S4 readiness payload must be non-submitting")
    if payload.get("cloud_contact") is not False:
        raise ValueError("S4 readiness payload must not contact cloud providers")
    return payload


def _provider_plan(payload: dict[str, Any], provider: str = "ibm_pulse") -> dict[str, Any]:
    plans = payload.get("provider_plans")
    if not isinstance(plans, list):
        raise ValueError("S4 readiness payload must contain provider_plans")
    for plan in plans:
        if isinstance(plan, dict) and plan.get("provider") == provider:
            return plan
    raise ValueError(f"S4 readiness payload does not contain provider {provider!r}")


def build_ibm_pulse_dossier(payload: dict[str, Any] | None = None) -> HardwareJobDossier:
    """Build the IBM pulse-level calibration-review preregistration dossier."""
    readiness = _load_readiness() if payload is None else payload
    plan = _provider_plan(readiness)
    summary = plan["programme_summary"]
    export = plan["export"]
    pulse_readiness = _openpulse_readiness_from_plan(plan)
    pulse_readiness_payload = pulse_readiness.to_dict()
    pulse_status = "ready" if pulse_readiness.ready else "blocked"
    return HardwareJobDossier(
        job_id="s4_ibm_pulse_calibration_review",
        title="S4 IBM pulse-level Kuramoto-XY calibration-review preregistration",
        purpose=(
            "Prepare the IBM pulse-level route for human calibration review before any "
            "pulse job, backend session, or QPU spend."
        ),
        hypothesis=(
            "If the exported circuit-QED pulse design matches a backend-supported IBM pulse "
            "control path after calibration review, it can become a bounded candidate for "
            "testing pulse-shaped Kuramoto-XY evolution against the digital Trotter route."
        ),
        falsification_condition=(
            "The IBM pulse route is rejected for near-term execution if the selected backend "
            "does not expose compatible pulse controls, calibrated channels, timing units, "
            "or safe amplitude/duration bounds for the exported schedule."
        ),
        expected_observables=(
            "backend pulse-control capability status",
            "drive/control channel mapping for each coupled pair",
            "dt or duration-unit compatibility",
            "amplitude and duration bounds for the exported envelope",
            "transpiled digital comparator depth and two-qubit gate count",
            "post-review QPU time estimate before any execution approval",
        ),
        circuit_summary={
            "provider": plan["provider"],
            "platform": plan["platform"],
            "native_schema": summary["native_schema"],
            "n_oscillators": summary["n_oscillators"],
            "n_couplers": summary["n_couplers"],
            "n_drives": summary["n_drives"],
            "duration": summary["duration"],
            "sdk_module": export["sdk_module"],
            "sdk_available": export["sdk_available"],
            "openpulse_readiness_status": pulse_status,
            "openpulse_blockers": list(pulse_readiness.blockers),
            "openpulse_warnings": list(pulse_readiness.warnings),
        },
        qpu_budget={
            "status": "not_requested",
            "hardware_submission": False,
            "cloud_contact": False,
            "recommended_initial_scope": "calibration metadata review only",
            "estimated_execution_seconds": 0.0,
            "future_optional_ceiling_seconds": 300.0,
        },
        platform_fit={
            "ibm_pulse": plan["readiness"],
            "ibm_openpulse_readiness": pulse_status,
            "gate_based_comparator": "required_before_execution",
            "neutral_atom": "separate_S4_route",
        },
        risks_and_confounds=(
            "Modern IBM Runtime access may not expose OpenPulse-style low-level controls on every backend.",
            "A design payload is not a calibrated Qiskit pulse Schedule or provider-accepted instruction.",
            "Pulse-shaped evolution and digital Trotter circuits are not equivalent without a matched observable protocol.",
            "Calibration drift can invalidate a pulse review before execution.",
            "Backend channel constraints can dominate any expected reduction in Trotter overhead.",
        ),
        decision_tree={
            "accepted": "Promote to live backend metadata capture and a no-submit channel-map artefact.",
            "manual_review": "Request backend-specific pulse constraints before choosing a hardware scope.",
            "fail": "Reject IBM pulse execution for this route and prioritise digital or neutral-atom S4 paths.",
        },
        paper_impact=(
            "A passed calibration review would support the S4 methods narrative by showing how the "
            "software stack moves from analogue design payloads to provider-specific hardware readiness."
        ),
        follow_up_avenue=(
            "After review, generate a backend-specific pulse channel map, then compare a tiny pulse-shaped "
            "candidate against a matched digital comparator only after explicit QPU-budget approval."
        ),
        possibilities_opened=(
            "pulse-vs-digital Trotter overhead comparison",
            "provider-specific calibration package for future QPU credit requests",
            "bounded IBM pulse-level route before non-IBM S4 hardware replication",
        ),
        claim_boundary=(
            "This preregistration does not create a pulse Schedule, contact IBM services, submit QPU jobs, "
            "or claim pulse-level performance. It only defines the calibration-review gate."
        ),
        reproducibility_package={
            "s4_readiness": str(READINESS_PATH.relative_to(REPO_ROOT)),
            "preregistration_script": "scripts/export_s4_provider_preregistration.py",
            "bench_command": "scpn-bench s4-provider-preregistration",
            "readiness_doc": f"docs/campaigns/s4_multi_hardware_readiness_{DATE}.md",
            "openpulse_readiness": json.dumps(pulse_readiness_payload, sort_keys=True),
        },
        prerequisites=(
            "select a concrete IBM backend that exposes compatible pulse metadata",
            "capture backend calibration metadata without submitting jobs",
            "record channel map, duration unit, amplitude bounds, and timing granularity",
            "estimate QPU time after the review and before any hardware approval",
        ),
    )


def _openpulse_readiness_from_plan(plan: dict[str, Any]) -> Any:
    summary = plan["programme_summary"]
    export = plan["export"]
    payload = export.get("payload", {})
    mode_frequencies = payload.get("mode_frequencies", [])
    n_qubits = int(summary["n_oscillators"])
    calibration_timestamp = payload.get("calibration_timestamp")

    ir_formats = (
        ("openqasm3", "qiskit_qpy", "qiskit") if bool(export["sdk_available"]) else ("openqasm3",)
    )
    native_features = (
        ("pulse_control", "drive_channel_access", "measure_channel_access", "dynamic_circuits")
        if bool(export["sdk_available"])
        else ("dynamic_circuits",)
    )
    snapshot = ProviderCapabilitySnapshot(
        route_id="direct/ibm_quantum",
        aggregator="direct",
        provider="ibm_quantum",
        backend_id="ibm_quantum",
        target_name="ibm_pulse_design_track",
        n_qubits=n_qubits,
        supported_ir_formats=ir_formats,
        basis_gates=("rz", "sx", "x", "cx", "measure", "reset"),
        native_features=native_features,
        online=None,
        simulator=False,
        no_submit=True,
        calibration_timestamp=calibration_timestamp
        if isinstance(calibration_timestamp, str)
        else None,
        metadata={
            "adapter": "s4_ibm_design_readiness",
            "mode_frequency_count": len(mode_frequencies)
            if isinstance(mode_frequencies, list)
            else 0,
        },
    )
    return build_openpulse_control_readiness(
        snapshot,
        qubit=0,
        dt=2.22e-10,
        amplitude_grid=(0.1, 0.2, 0.3, 0.4, 0.5),
        shots=4096,
    )


def _payload(dossier: HardwareJobDossier) -> dict[str, Any]:
    return {
        "schema": "s4_ibm_pulse_preregistration_v1",
        "date": DATE,
        "hardware_submission": False,
        "cloud_contact": False,
        "dossier": dossier.to_dict(),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8")
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    """Parse the S4 provider preregistration export arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-path", type=Path, default=READINESS_PATH)
    parser.add_argument("--json-path", type=Path, default=JSON_PATH)
    parser.add_argument("--md-path", type=Path, default=MD_PATH)
    return parser.parse_args()


def main() -> int:
    """Write the S4 IBM pulse-level preregistration dossier artefacts."""
    args = parse_args()
    readiness = _load_readiness(args.readiness_path)
    dossier = build_ibm_pulse_dossier(readiness)
    sha_json = _write_json(args.json_path, _payload(dossier))
    sha_md = _write_text(args.md_path, dossier.to_markdown())
    print(f"wrote {args.json_path.relative_to(REPO_ROOT)} sha256={sha_json}")
    print(f"wrote {args.md_path.relative_to(REPO_ROOT)} sha256={sha_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
