# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S4 multi-hardware readiness export
"""Export no-submit S4 multi-hardware and pulse-level readiness artefacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27
from scpn_quantum_control.hardware.analog_kuramoto import (
    AnalogKuramotoPlatform,
    compile_analog_kuramoto,
    export_provider_payload,
    prepare_provider_execution_plan,
)

DATE = "2026-05-06"
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "s4_multi_hardware_control"
DOC_PATH = REPO_ROOT / "docs" / f"s4_multi_hardware_readiness_{DATE}.md"


def _calibration(provider: str) -> dict[str, str]:
    if provider == "ibm_pulse":
        return {
            "calibration_id": "ibm-pulse-design-units-v1",
            "duration_unit": "dt",
            "coupling_unit": "arb",
            "detuning_unit": "arb",
        }
    return {
        "calibration_id": f"{provider}-local-design-units-v1",
        "duration_unit": "us",
        "coupling_unit": "rad/us",
        "detuning_unit": "rad/us",
    }


def _serialise_programme(platform: AnalogKuramotoPlatform) -> dict[str, Any]:
    n = 4
    k_nm = build_knm_paper27(n)
    omega = np.linspace(-0.3, 0.3, n, dtype=np.float64)
    programme = compile_analog_kuramoto(
        k_nm,
        omega,
        platform=platform,
        duration=0.2,
        coupling_scale=0.75,
        lambda_fim=0.0,
        metadata={
            "track": "S4",
            "purpose": "multi_hardware_readiness",
            "date": DATE,
        },
    )
    return programme.to_dict()


def _provider_plan(provider: str) -> dict[str, Any]:
    platform = (
        AnalogKuramotoPlatform.CIRCUIT_QED
        if provider == "ibm_pulse"
        else AnalogKuramotoPlatform.NEUTRAL_ATOMS
    )
    n = 4
    k_nm = build_knm_paper27(n)
    omega = np.linspace(-0.3, 0.3, n, dtype=np.float64)
    programme = compile_analog_kuramoto(
        k_nm,
        omega,
        platform=platform,
        duration=0.2,
        coupling_scale=0.75,
        lambda_fim=0.0,
        metadata={"track": "S4", "provider": provider, "date": DATE},
    )
    export = export_provider_payload(programme, provider)
    blocked_plan = prepare_provider_execution_plan(
        export,
        calibration=_calibration(provider),
        approved=False,
    )
    return {
        "provider": provider,
        "platform": platform.value,
        "programme_summary": {
            "n_oscillators": programme.n_oscillators,
            "n_couplers": programme.n_couplers,
            "n_drives": len(programme.drive_terms),
            "duration": programme.duration,
            "native_schema": programme.payload["schema"],
        },
        "export": export.to_dict(),
        "execution_plan": blocked_plan.to_dict(),
        "readiness": "blocked_until_sdk_calibration_approval"
        if not export.sdk_available
        else "blocked_until_explicit_execution_approval",
    }


def build_readiness_payload() -> dict[str, Any]:
    """Build the no-submit S4 readiness payload."""
    provider_plans = [_provider_plan(provider) for provider in ("pulser", "bloqade", "ibm_pulse")]
    return {
        "schema": "s4_multi_hardware_readiness_v1",
        "date": DATE,
        "track": "S4 Multi-hardware backend + pulse-level control",
        "hardware_submission": False,
        "cloud_contact": False,
        "qpu_budget_requested_seconds": 0.0,
        "scientific_question": (
            "Can the Kuramoto-XY pulse/analogue programme family be represented as "
            "provider-specific no-submit payloads before any cross-vendor hardware run?"
        ),
        "falsification_boundary": (
            "S4 provider promotion is blocked unless the selected provider SDK is available, "
            "calibration metadata are attached, live resource checks pass, and explicit budget "
            "approval is recorded."
        ),
        "programmes": {
            "neutral_atoms": _serialise_programme(AnalogKuramotoPlatform.NEUTRAL_ATOMS),
            "circuit_qed": _serialise_programme(AnalogKuramotoPlatform.CIRCUIT_QED),
        },
        "provider_plans": provider_plans,
        "promotion_gates": [
            "provider account or research-credit route exists",
            "provider SDK object construction succeeds locally or in an approved emulator",
            "provider calibration metadata are recorded",
            "live transpilation or provider resource estimate is below the preregistered ceiling",
            "QPU budget is approved for a named provider and backend",
            "raw data, job identifiers, and analysis scripts are assigned before submission",
        ],
        "blocked_claims": [
            "no non-IBM hardware result",
            "no pulse-level calibration result",
            "no cross-vendor replication claim",
            "no hardware performance improvement claim",
            "no quantum-advantage claim",
        ],
        "next_actions": [
            "select one provider route with available credits",
            "construct provider SDK object in emulator-only mode after approval",
            "prepare a provider-specific preregistration dossier",
            "run budget-gated hardware only after live readiness checks",
        ],
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# S4 Multi-hardware Readiness",
        "",
        "This is a no-submit readiness artefact. It does not contact cloud providers, "
        "reserve QPU time, or authorise hardware execution.",
        "",
        "## Scientific Question",
        str(payload["scientific_question"]),
        "",
        "## Falsification Boundary",
        str(payload["falsification_boundary"]),
        "",
        "## Provider Plans",
    ]
    for plan in payload["provider_plans"]:
        export = plan["export"]
        execution = plan["execution_plan"]
        summary = plan["programme_summary"]
        lines.extend(
            [
                "",
                f"### {plan['provider']}",
                f"- Platform: `{plan['platform']}`",
                f"- SDK module: `{export['sdk_module']}`",
                f"- SDK available: `{export['sdk_available']}`",
                f"- Native schema: `{summary['native_schema']}`",
                f"- Oscillators / couplers: `{summary['n_oscillators']}` / `{summary['n_couplers']}`",
                f"- Readiness: `{plan['readiness']}`",
                f"- Can submit: `{export['can_submit']}`",
                f"- Can execute: `{execution['can_execute']}`",
                f"- Reason: `{execution['reason']}`",
            ]
        )
    lines.extend(["", "## Promotion Gates"])
    lines.extend(f"- {item}" for item in payload["promotion_gates"])
    lines.extend(["", "## Blocked Claims"])
    lines.extend(f"- {item}" for item in payload["blocked_claims"])
    lines.extend(["", "## Next Actions"])
    lines.extend(f"- {item}" for item in payload["next_actions"])
    return "\n".join(lines) + "\n"


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
    """Parse the S4 multi-hardware readiness export arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DOC_PATH)
    return parser.parse_args()


def main() -> int:
    """Write the S4 multi-hardware readiness artefact."""
    args = parse_args()
    payload = build_readiness_payload()
    json_path = args.out_dir / f"s4_multi_hardware_readiness_{DATE}.json"
    sha_json = _write_json(json_path, payload)
    sha_doc = _write_text(args.doc_path, _markdown(payload))
    print(f"wrote {json_path.relative_to(REPO_ROOT)} sha256={sha_json}")
    print(f"wrote {args.doc_path.relative_to(REPO_ROOT)} sha256={sha_doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
