# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- stable core preflight checks
"""Backend preflight checks for stable core experiments."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from .stable_core import Experiment, ExperimentObjective, backend_capability_matrix

PreflightStatus = Literal["eligible", "blocked"]


_OBJECTIVE_REQUIRED_CAPABILITIES: dict[ExperimentObjective, str] = {
    "order_parameter": "order_parameter",
    "parity_leakage": "parity",
    "fim_metric": "fim",
    "control_cost": "control",
    "mitigation_replay": "mitigation_replay",
}

_BACKEND_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "qiskit-runtime": ("qiskit_ibm_runtime",),
    "qutip-dynamics": ("qutip",),
    "pennylane-autodiff": ("pennylane",),
    "pulser-surrogate": ("pulser",),
}


def _backend_matrix_row(backend_id: str) -> Mapping[str, Any] | None:
    """Return the stable matrix row for ``backend_id`` when present."""

    for row in backend_capability_matrix():
        if row["backend_id"] == backend_id:
            return row
    return None


def _dependency_available(name: str) -> bool:
    """Return ``True`` iff ``name`` can be imported."""

    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


@dataclass(frozen=True, slots=True)
class StableCorePreflightResult:
    """Machine-readable stable-core preflight outcome."""

    status: PreflightStatus
    blockers: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible preflight payload."""

        return {
            "status": self.status,
            "blockers": self.blockers,
            "required_capabilities": self.required_capabilities,
            "claim_boundary": self.claim_boundary,
        }


def run_stable_core_preflight(
    experiment: Experiment,
    *,
    dependency_availability: Mapping[str, bool] | None = None,
) -> StableCorePreflightResult:
    """Validate a stable-core experiment against declared readiness constraints."""

    required_capability = _OBJECTIVE_REQUIRED_CAPABILITIES[experiment.objective]
    blockers: list[str] = []

    if required_capability not in experiment.backend.capabilities:
        blockers.append(
            f"backend {experiment.backend.backend_id!r} missing required capability "
            f"{required_capability!r}"
        )

    if experiment.backend.hardware_submission_allowed and not experiment.metadata.get(
        "preregistration_id"
    ):
        blockers.append("hardware-submission experiment missing preregistration_id metadata")

    backend_row = _backend_matrix_row(experiment.backend.backend_id)
    claim_boundary = str(
        backend_row.get("claim_boundary", "Backend not found in stable core capability matrix.")
        if backend_row is not None
        else "Backend not found in stable core capability matrix."
    )
    if backend_row is None:
        blockers.append(
            f"backend {experiment.backend.backend_id!r} is not in stable capability matrix"
        )

    required_dependencies = _BACKEND_DEPENDENCIES.get(experiment.backend.backend_id, ())
    for dependency in required_dependencies:
        available = (
            bool(dependency_availability[dependency])
            if dependency_availability is not None and dependency in dependency_availability
            else _dependency_available(dependency)
        )
        if not available:
            blockers.append(
                f"backend {experiment.backend.backend_id!r} requires optional dependency "
                f"{dependency!r}"
            )

    return StableCorePreflightResult(
        status="blocked" if blockers else "eligible",
        blockers=tuple(blockers),
        required_capabilities=(required_capability,),
        claim_boundary=claim_boundary,
    )


def stable_core_backend_dependencies(backend_id: str) -> tuple[str, ...]:
    """Return optional dependency requirements for a stable-core backend id."""

    return _BACKEND_DEPENDENCIES.get(backend_id, ())


def stable_core_preflight_fixtures_payload() -> dict[str, Any]:
    """Return deterministic fixture rows for offline stable-core preflight gates."""

    return _normalised_payload(
        {
            "schema": "stable_core_preflight_fixtures_v1",
            "spdx_license_identifier": "AGPL-3.0-or-later",
            "hardware_submission": False,
            "fixtures": [
                {
                    "fixture_id": "eligible_classical_reference",
                    "status": "eligible",
                    "backend": {
                        "backend_id": "classical-reference",
                        "kind": "classical_reference",
                        "capabilities": ("order_parameter", "parity", "fim", "control"),
                        "hardware_submission_allowed": False,
                    },
                    "objective": "order_parameter",
                    "blockers": (),
                    "primitives": (
                        "dependency_probe",
                        "capability_guard",
                        "preregistration_guard",
                        "eligible",
                    ),
                    "metadata": {
                        "scenario": "eligible classical/reference",
                    },
                },
                {
                    "fixture_id": "blocked_missing_dependency",
                    "status": "blocked",
                    "backend": {
                        "backend_id": "qiskit-runtime",
                        "kind": "qiskit",
                        "capabilities": ("order_parameter", "parity", "mitigation_replay"),
                        "hardware_submission_allowed": False,
                    },
                    "objective": "order_parameter",
                    "blockers": ("missing dependency: qiskit-runtime provider package",),
                    "primitives": (),
                    "metadata": {
                        "scenario": "blocked missing dependency",
                    },
                },
                {
                    "fixture_id": "blocked_hardware_preregistration_or_boundary",
                    "status": "blocked",
                    "backend": {
                        "backend_id": "qiskit-runtime-live",
                        "kind": "qiskit",
                        "capabilities": ("order_parameter", "parity", "mitigation_replay"),
                        "hardware_submission_allowed": True,
                    },
                    "objective": "order_parameter",
                    "blockers": (
                        "hardware preregistration required for live submission",
                        "hardware boundary blocks run-path in stable fixture mode",
                    ),
                    "primitives": (),
                    "metadata": {
                        "scenario": "blocked hardware preregistration or boundary",
                    },
                },
                {
                    "fixture_id": "blocked_missing_capability",
                    "status": "blocked",
                    "backend": {
                        "backend_id": "qutip-dynamics",
                        "kind": "qutip",
                        "capabilities": ("order_parameter", "hamiltonian_dynamics", "lindblad"),
                        "hardware_submission_allowed": False,
                    },
                    "objective": "control_cost",
                    "blockers": ("backend qutip-dynamics does not declare control capability",),
                    "primitives": (),
                    "metadata": {
                        "scenario": "blocked missing capability",
                        "required_capability": "control",
                    },
                },
            ],
            "claim_boundary": (
                "Preflight fixtures are offline and do not prove runtime execution, "
                "hardware registration, or external dependency readiness. "
                "They only lock shape checks for deterministic guard branches."
            ),
        }
    )


def stable_core_preflight_fixtures_json(payload: Mapping[str, Any]) -> str:
    """Return canonical JSON text for stable-core preflight fixtures."""

    return json.dumps(_normalised_payload(payload), indent=2, sort_keys=True) + "\n"


def stable_core_preflight_fixtures_markdown(payload: Mapping[str, Any]) -> str:
    """Return canonical Markdown text for stable-core preflight fixtures."""

    rows = _normalised_payload(payload)
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "<!-- Commercial license available -->",
        "<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->",
        "<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->",
        "<!-- ORCID: 0009-0009-3560-0851 -->",
        "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
        "<!-- scpn-quantum-control -- stable core preflight fixtures -->",
        "",
        "# Stable Core Preflight Fixtures",
        "",
        "These no-QPU, no-network fixtures lock stable core preflight branches.",
        "",
        "## Fixture summary",
        "",
        f"- Schema: `{rows['schema']}`",
        f"- Hardware submission enabled in fixtures: `{rows['hardware_submission']}`",
        "",
        "## Preflight fixtures",
        "",
        "| Fixture ID | Status | Backend ID | Objective | Blockers | Primitives |",
        "|---|---|---|---|---|---|",
    ]

    for row in rows["fixtures"]:
        lines.append(
            "| `{fixture_id}` | `{status}` | `{backend_id}` | `{objective}` | {blockers} | {primitives} |".format(
                fixture_id=row["fixture_id"],
                status=row["status"],
                backend_id=row["backend"]["backend_id"],
                objective=row["objective"],
                blockers=", ".join(row["blockers"]) or "`none`",
                primitives=", ".join(row["primitives"]) or "`none`",
            )
        )

    lines.extend(
        [
            "",
            "## Reproducibility gate",
            "",
            "Regenerate and compare these fixtures with:",
            "",
            "```bash",
            "scpn-bench stable-core-preflight-gate",
            "```",
            "",
            "## Claim boundary",
            "",
            str(rows["claim_boundary"]),
        ]
    )
    return "\n".join(lines) + "\n"


def _normalised_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-native deterministic representation of ``payload``."""

    return cast(dict[str, Any], json.loads(json.dumps(payload, sort_keys=True)))
