# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — S3 pulse/ansatz design protocol
"""Claim-bounded S3 pulse and ansatz design scoring protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from scpn_quantum_control.control.structured_ansatz import StructuredAnsatz
from scpn_quantum_control.phase.pulse_shaping import build_trotter_pulse_schedule

S3_PROTOCOL_ID = "s3_ml_augmented_pulse_ansatz_design_2026-05-06"
DesignFamily = Literal["ansatz", "pulse"]


@dataclass(frozen=True)
class S3DesignCandidate:
    """One pulse or ansatz candidate in the S3 no-QPU design gate."""

    label: str
    family: DesignFamily
    parameters: Mapping[str, float | int | str]

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("label must be non-empty")
        if not self.parameters:
            raise ValueError("parameters must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Serialise the candidate."""
        return {
            "label": self.label,
            "family": self.family,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class S3DesignRow:
    """Scored S3 candidate row."""

    protocol_id: str
    candidate_label: str
    family: DesignFamily
    status: str
    score: float
    metrics: Mapping[str, float | int | str]
    claim_boundary: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise the scored row."""
        return {
            "protocol_id": self.protocol_id,
            "candidate_label": self.candidate_label,
            "family": self.family,
            "status": self.status,
            "score": self.score,
            "metrics": dict(self.metrics),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class S3DesignProtocol:
    """Protocol manifest for S3 design-ranking readiness."""

    protocol_id: str
    objective: str
    acceptance_rule: str
    forbidden_claims: tuple[str, ...]
    required_followups: tuple[str, ...]
    candidates: tuple[S3DesignCandidate, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the protocol manifest."""
        return {
            "protocol_id": self.protocol_id,
            "objective": self.objective,
            "acceptance_rule": self.acceptance_rule,
            "forbidden_claims": list(self.forbidden_claims),
            "required_followups": list(self.required_followups),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def default_s3_design_protocol() -> S3DesignProtocol:
    """Return the default S3 no-QPU candidate-ranking protocol."""
    return S3DesignProtocol(
        protocol_id=S3_PROTOCOL_ID,
        objective=(
            "Rank small Kuramoto-XY pulse and ansatz candidates by deterministic "
            "resource and analytic-error proxies before any ML training or QPU use."
        ),
        acceptance_rule=(
            "A candidate is promotable only when it has finite metrics, a finite "
            "score, no hardware submission, and an explicit follow-up validation path."
        ),
        forbidden_claims=(
            "No learned optimiser is demonstrated by this readiness gate.",
            "No pulse-level hardware improvement is established without provider calibration data.",
            "No quantum advantage or backend-independent performance claim is permitted.",
        ),
        required_followups=(
            "Train or evaluate the ML surrogate on generated candidate rows with held-out checks.",
            "Compare promoted ansatz candidates against VQE or observable targets.",
            "Run provider-specific pulse feasibility checks before any pulse submission.",
            "Attach hardware-job dossiers before QPU or pulse-level execution.",
        ),
        candidates=(
            S3DesignCandidate(
                label="ansatz_shallow_knm",
                family="ansatz",
                parameters={"trotter_depth": 2, "time_step": 0.12, "coupling_scale": 1.0},
            ),
            S3DesignCandidate(
                label="ansatz_mid_knm",
                family="ansatz",
                parameters={"trotter_depth": 4, "time_step": 0.08, "coupling_scale": 1.5},
            ),
            S3DesignCandidate(
                label="pulse_stirap_balanced",
                family="pulse",
                parameters={"alpha": 0.5, "beta": 0.5, "t_step": 0.2, "omega_0": 10.0},
            ),
            S3DesignCandidate(
                label="pulse_demkov_kunike",
                family="pulse",
                parameters={"alpha": 1.0, "beta": 0.5, "t_step": 0.2, "omega_0": 10.0},
            ),
        ),
    )


def score_s3_candidates(
    protocol: S3DesignProtocol,
    k_matrix: NDArray[np.float64],
    omega: NDArray[np.float64],
) -> tuple[S3DesignRow, ...]:
    """Score all S3 candidates against deterministic no-QPU proxies."""
    k = np.asarray(k_matrix, dtype=np.float64)
    w = np.asarray(omega, dtype=np.float64)
    _validate_problem(k, w)
    rows = tuple(_score_candidate(candidate, k, w) for candidate in protocol.candidates)
    validate_s3_design_rows(rows, protocol=protocol)
    return rows


def validate_s3_design_rows(
    rows: Sequence[S3DesignRow | Mapping[str, Any]],
    *,
    protocol: S3DesignProtocol | None = None,
) -> None:
    """Validate scored S3 rows before artefact promotion."""
    if not rows:
        raise ValueError("S3 design rows must be non-empty")
    expected_protocol = (protocol or default_s3_design_protocol()).protocol_id
    seen_labels: set[str] = set()
    for raw in rows:
        row = raw.to_dict() if isinstance(raw, S3DesignRow) else dict(raw)
        if row.get("protocol_id") != expected_protocol:
            raise ValueError("row protocol_id does not match S3 protocol")
        label = row.get("candidate_label")
        if not isinstance(label, str) or not label:
            raise ValueError("candidate_label must be non-empty text")
        if label in seen_labels:
            raise ValueError(f"duplicate candidate_label: {label}")
        seen_labels.add(label)
        if row.get("family") not in {"ansatz", "pulse"}:
            raise ValueError("family must be ansatz or pulse")
        if row.get("status") != "ok":
            raise ValueError("S3 readiness rows must have status='ok'")
        score = row.get("score")
        if not isinstance(score, int | float) or not np.isfinite(float(score)):
            raise ValueError("score must be finite")
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping) or not metrics:
            raise ValueError("metrics must be a non-empty mapping")
        if "hardware_submission" not in metrics or metrics["hardware_submission"] is not False:
            raise ValueError("S3 readiness rows must record hardware_submission=False")


def _score_candidate(
    candidate: S3DesignCandidate, k: NDArray[np.float64], omega: NDArray[np.float64]
) -> S3DesignRow:
    if candidate.family == "ansatz":
        return _score_ansatz(candidate, k, omega)
    return _score_pulse(candidate, k)


def _score_ansatz(
    candidate: S3DesignCandidate, k: NDArray[np.float64], omega: NDArray[np.float64]
) -> S3DesignRow:
    params = candidate.parameters
    ansatz = StructuredAnsatz.from_kuramoto(
        k,
        omega=omega,
        trotter_depth=int(params["trotter_depth"]),
        time_step=float(params["time_step"]),
        coupling_scale=float(params["coupling_scale"]),
    )
    circuit = ansatz.build_circuit()
    ops = circuit.count_ops()
    two_qubit = int(
        sum(count for gate, count in ops.items() if gate in {"cx", "cz", "rzz", "ecr"})
    )
    depth = int(circuit.depth())
    score = float(depth + 4 * two_qubit + 0.25 * circuit.size())
    return S3DesignRow(
        protocol_id=S3_PROTOCOL_ID,
        candidate_label=candidate.label,
        family="ansatz",
        status="ok",
        score=score,
        metrics={
            "n_qubits": int(k.shape[0]),
            "depth": depth,
            "size": int(circuit.size()),
            "two_qubit_gates": two_qubit,
            "parameters": int(circuit.num_parameters),
            "hardware_submission": False,
        },
        claim_boundary="Resource proxy only; not a trained ML or VQE performance claim.",
    )


def _score_pulse(candidate: S3DesignCandidate, k: NDArray[np.float64]) -> S3DesignRow:
    params = candidate.parameters
    schedule = build_trotter_pulse_schedule(
        int(k.shape[0]),
        k,
        t_step=float(params["t_step"]),
        omega_0=float(params["omega_0"]),
        alpha=float(params["alpha"]),
        beta=float(params["beta"]),
    )
    max_points = max((len(pulse.times) for pulse in schedule.pulses), default=0)
    score = float(schedule.infidelity_bound + 0.01 * len(schedule.pulses) + 0.0001 * max_points)
    return S3DesignRow(
        protocol_id=S3_PROTOCOL_ID,
        candidate_label=candidate.label,
        family="pulse",
        status="ok",
        score=score,
        metrics={
            "n_qubits": int(k.shape[0]),
            "pulse_count": int(len(schedule.pulses)),
            "max_points_per_pulse": int(max_points),
            "total_time": float(schedule.total_time),
            "infidelity_bound": float(schedule.infidelity_bound),
            "hardware_submission": False,
        },
        claim_boundary="Analytic pulse proxy only; provider pulse calibration is still required.",
    )


def _validate_problem(k: NDArray[np.float64], omega: NDArray[np.float64]) -> None:
    if k.ndim != 2 or k.shape[0] != k.shape[1]:
        raise ValueError("k_matrix must be square")
    if omega.shape != (k.shape[0],):
        raise ValueError("omega length must match k_matrix")
    if not np.all(np.isfinite(k)) or not np.all(np.isfinite(omega)):
        raise ValueError("S3 problem inputs must be finite")


def generate_s3_candidate_grid() -> tuple[S3DesignCandidate, ...]:
    """Generate a deterministic candidate grid for S3 surrogate rehearsal."""
    candidates: list[S3DesignCandidate] = []
    for depth in (1, 2, 3, 4):
        for coupling_scale in (0.75, 1.0, 1.5, 2.0):
            candidates.append(
                S3DesignCandidate(
                    label=f"ansatz_d{depth}_c{str(coupling_scale).replace('.', 'p')}",
                    family="ansatz",
                    parameters={
                        "trotter_depth": depth,
                        "time_step": 0.1,
                        "coupling_scale": coupling_scale,
                    },
                )
            )
    for alpha in (0.0, 0.5, 1.0):
        for beta in (0.5, 1.0):
            for t_step in (0.2, 0.3):
                candidates.append(
                    S3DesignCandidate(
                        label=(
                            "pulse_a"
                            f"{str(alpha).replace('.', 'p')}_b{str(beta).replace('.', 'p')}"
                            f"_t{str(t_step).replace('.', 'p')}"
                        ),
                        family="pulse",
                        parameters={
                            "alpha": alpha,
                            "beta": beta,
                            "t_step": t_step,
                            "omega_0": 10.0,
                        },
                    )
                )
    return tuple(candidates)


def grid_s3_design_protocol() -> S3DesignProtocol:
    """Return the expanded deterministic protocol used for surrogate rehearsal."""
    base = default_s3_design_protocol()
    return S3DesignProtocol(
        protocol_id=base.protocol_id,
        objective=base.objective,
        acceptance_rule=base.acceptance_rule,
        forbidden_claims=base.forbidden_claims,
        required_followups=base.required_followups,
        candidates=generate_s3_candidate_grid(),
    )
