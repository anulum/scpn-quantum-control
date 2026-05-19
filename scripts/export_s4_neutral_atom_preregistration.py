# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S4 neutral-atom preregistration export
"""Export the no-submit S4 neutral-atom provider-object preregistration dossier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scpn_quantum_control.hardware.job_dossier import HardwareJobDossier

DATE = "2026-05-06"
REPO_ROOT = Path(__file__).resolve().parents[1]
S4_DIR = REPO_ROOT / "data" / "s4_multi_hardware_control"
READINESS_PATH = S4_DIR / f"s4_multi_hardware_readiness_{DATE}.json"
JSON_PATH = S4_DIR / f"s4_neutral_atom_preregistration_{DATE}.json"
MD_PATH = REPO_ROOT / "docs" / f"s4_neutral_atom_preregistration_{DATE}.md"


def _load_readiness(path: Path = READINESS_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("S4 readiness payload must be a JSON object")
    if payload.get("hardware_submission") is not False:
        raise ValueError("S4 readiness payload must be non-submitting")
    if payload.get("cloud_contact") is not False:
        raise ValueError("S4 readiness payload must not contact cloud providers")
    return payload


def _provider_plans(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    plans = payload.get("provider_plans")
    if not isinstance(plans, list):
        raise ValueError("S4 readiness payload must contain provider_plans")
    selected: dict[str, dict[str, Any]] = {}
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        provider = plan.get("provider")
        if provider in {"pulser", "bloqade"}:
            selected[str(provider)] = plan
    if set(selected) != {"pulser", "bloqade"}:
        raise ValueError("S4 readiness payload must contain pulser and bloqade plans")
    return selected


def build_neutral_atom_dossier(payload: dict[str, Any] | None = None) -> HardwareJobDossier:
    """Build the neutral-atom provider/emulator-object preregistration dossier."""
    readiness = _load_readiness() if payload is None else payload
    plans = _provider_plans(readiness)
    pulser = plans["pulser"]
    bloqade = plans["bloqade"]
    pulser_summary = pulser["programme_summary"]
    bloqade_summary = bloqade["programme_summary"]
    return HardwareJobDossier(
        job_id="s4_neutral_atom_provider_object_review",
        title="S4 neutral-atom Kuramoto-XY provider-object preregistration",
        purpose=(
            "Prepare Pulser and Bloqade neutral-atom object-construction routes for local "
            "or approved-emulator review before any cloud provider session or hardware spend."
        ),
        hypothesis=(
            "If the neutral-atom payload can be converted into provider SDK objects under local "
            "unit and geometry constraints, it becomes the most plausible S4 path for testing "
            "native XY-like evolution without digital Trotter overhead."
        ),
        falsification_condition=(
            "The route is rejected for near-term execution if the register geometry, Rydberg interaction "
            "constraints, SDK object construction, or emulator-only resource estimate cannot realise the "
            "n=4 Kuramoto-XY payload without changing the preregistered observable family."
        ),
        expected_observables=(
            "provider SDK availability and version",
            "register geometry and minimum spacing validity",
            "Rydberg or AHS interaction coefficient compatibility",
            "local provider-object construction status",
            "emulator-only resource estimate where an approved emulator exists",
            "matched digital comparator observable definition",
        ),
        circuit_summary={
            "providers": "pulser,bloqade",
            "platform": "neutral_atoms",
            "pulser_schema": pulser_summary["native_schema"],
            "bloqade_schema": bloqade_summary["native_schema"],
            "n_oscillators": pulser_summary["n_oscillators"],
            "n_couplers": pulser_summary["n_couplers"],
            "duration": pulser_summary["duration"],
            "pulser_sdk_available": pulser["export"]["sdk_available"],
            "bloqade_sdk_available": bloqade["export"]["sdk_available"],
        },
        qpu_budget={
            "status": "not_requested",
            "hardware_submission": False,
            "cloud_contact": False,
            "recommended_initial_scope": "local SDK-object construction or approved emulator only",
            "estimated_execution_seconds": 0.0,
            "future_optional_ceiling_seconds": 600.0,
        },
        platform_fit={
            "pulser": pulser["readiness"],
            "bloqade": bloqade["readiness"],
            "ibm_pulse": "separate_S4_route",
            "gate_based_comparator": "required_before_execution",
        },
        risks_and_confounds=(
            "Neutral-atom native interactions are not identical to the gate-model XY Hamiltonian without a mapped observable protocol.",
            "Register geometry and interaction signs can force a different effective coupling matrix.",
            "Provider SDK object construction is not cloud execution and cannot establish hardware performance.",
            "Local emulator success can be classically spoofable and must be separated from QPU evidence.",
            "Provider cost and credit access can dominate practical execution feasibility.",
        ),
        decision_tree={
            "accepted": "Promote to local provider-object construction with versioned SDK metadata.",
            "manual_review": "Request provider-specific geometry, units, and emulator constraints before promotion.",
            "fail": "Reject neutral-atom S4 execution for this payload and keep the route as design-only evidence.",
        },
        paper_impact=(
            "A passed provider-object review would support a future S4 hardware-methods section by showing "
            "that the analogue Kuramoto compiler can target neutral-atom SDKs before hardware allocation."
        ),
        follow_up_avenue=(
            "Construct local Pulser and Bloqade objects in emulator-only mode, then prepare a budgeted cloud "
            "provider request only if geometry, units, and comparator observables pass review."
        ),
        possibilities_opened=(
            "native neutral-atom Kuramoto-XY execution planning",
            "digital-vs-analogue Trotter-overhead comparison",
            "non-IBM S4 credit application evidence",
            "provider-neutral artefact package for cross-vendor reproducibility",
        ),
        claim_boundary=(
            "This preregistration does not import provider SDK constructors, run emulators, contact cloud "
            "providers, submit jobs, or claim neutral-atom hardware performance."
        ),
        reproducibility_package={
            "s4_readiness": str(READINESS_PATH.relative_to(REPO_ROOT)),
            "preregistration_script": "scripts/export_s4_neutral_atom_preregistration.py",
            "bench_command": "scpn-bench s4-neutral-atom-preregistration",
            "readiness_doc": f"docs/s4_multi_hardware_readiness_{DATE}.md",
        },
        prerequisites=(
            "install or verify provider SDK versions in an isolated environment",
            "construct provider objects locally without cloud credentials",
            "record register geometry and unit-conversion assumptions",
            "define matched digital comparator observables before hardware execution",
            "obtain provider credit and explicit QPU-budget approval before cloud submission",
        ),
    )


def _payload(dossier: HardwareJobDossier) -> dict[str, Any]:
    return {
        "schema": "s4_neutral_atom_preregistration_v1",
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
    """Parse the S4 neutral-atom preregistration export arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-path", type=Path, default=READINESS_PATH)
    parser.add_argument("--json-path", type=Path, default=JSON_PATH)
    parser.add_argument("--md-path", type=Path, default=MD_PATH)
    return parser.parse_args()


def main() -> int:
    """Write the S4 neutral-atom preregistration artefact."""
    args = parse_args()
    readiness = _load_readiness(args.readiness_path)
    dossier = build_neutral_atom_dossier(readiness)
    sha_json = _write_json(args.json_path, _payload(dossier))
    sha_md = _write_text(args.md_path, dossier.to_markdown())
    print(f"wrote {args.json_path.relative_to(REPO_ROOT)} sha256={sha_json}")
    print(f"wrote {args.md_path.relative_to(REPO_ROOT)} sha256={sha_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
