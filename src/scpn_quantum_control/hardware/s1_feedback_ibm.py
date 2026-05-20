# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — S1 IBM feedback execution helpers
"""S1 IBM paired-arm submission and raw-count conversion contracts."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from qiskit import QuantumCircuit

from ..control.realtime_feedback import (
    RealtimeSyncFeedbackController,
    build_open_loop_feedback_control_circuit,
)
from .feedback_loop import FeedbackCommand, FeedbackResult
from .runner import _extract_counts

S1_FEEDBACK_ARM = "feedback"
S1_CONTROL_ARM = "matched_open_loop_control"


@dataclass(frozen=True)
class S1FeedbackArmCircuit:
    """One preregistered S1 IBM arm circuit before provider submission."""

    label: str
    circuit: QuantumCircuit
    shots: int
    repetitions: int
    estimated_qpu_seconds: float

    def __post_init__(self) -> None:
        if self.label not in {S1_FEEDBACK_ARM, S1_CONTROL_ARM}:
            raise ValueError("unsupported S1 arm label")
        if self.shots < 1:
            raise ValueError("shots must be positive")
        if self.repetitions < 1:
            raise ValueError("repetitions must be positive")
        if self.estimated_qpu_seconds < 0.0:
            raise ValueError("estimated_qpu_seconds must be non-negative")


def build_s1_feedback_arm_circuits(
    controller: RealtimeSyncFeedbackController,
    *,
    n_rounds: int,
    shots: int,
    repetitions: int,
    estimated_seconds_per_circuit: float = 1.0,
) -> tuple[S1FeedbackArmCircuit, S1FeedbackArmCircuit]:
    """Build the monitored feedback and matched open-loop S1 arm circuits."""
    if not isinstance(controller, RealtimeSyncFeedbackController):
        raise TypeError("controller must be a RealtimeSyncFeedbackController")
    if n_rounds < 1:
        raise ValueError("n_rounds must be positive")
    feedback = controller.build_monitored_circuit(n_rounds)
    control = build_open_loop_feedback_control_circuit(
        controller.K,
        controller.omega,
        config=controller.config,
        n_rounds=n_rounds,
        trotter_order=controller.trotter_order,
    )
    estimate = float(estimated_seconds_per_circuit) * repetitions
    return (
        S1FeedbackArmCircuit(S1_FEEDBACK_ARM, feedback, shots, repetitions, estimate),
        S1FeedbackArmCircuit(S1_CONTROL_ARM, control, shots, repetitions, estimate),
    )


def binary_phase_synchrony_from_counts(counts: Mapping[str, int], *, n_qubits: int) -> float:
    """Estimate binary phase synchrony from final computational-basis counts.

    Each measured bit is interpreted as a binary phase, ``0 -> +1`` and
    ``1 -> -1``. The per-shot synchrony is ``abs(mean(phase_i))`` and the
    returned value is the shot-weighted mean over the raw count dictionary.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    total = 0
    weighted = 0.0
    for raw_bitstring, raw_count in counts.items():
        if not isinstance(raw_bitstring, str) or not raw_bitstring:
            raise ValueError("count keys must be non-empty bitstrings")
        if not isinstance(raw_count, int) or raw_count < 0:
            raise ValueError(f"count for {raw_bitstring!r} must be a non-negative integer")
        bitstring = raw_bitstring.replace(" ", "")
        if len(bitstring) < n_qubits:
            raise ValueError("count bitstrings must include all measured system qubits")
        system_bits = bitstring[-n_qubits:]
        phase_sum = sum(1.0 if bit == "0" else -1.0 for bit in system_bits)
        synchrony = abs(phase_sum / n_qubits)
        weighted += raw_count * synchrony
        total += raw_count
    if total < 1:
        raise ValueError("counts must contain at least one shot")
    return weighted / total


def raw_count_package_from_feedback_results(
    *,
    experiment_id: str,
    target_r: float,
    n_qubits: int,
    feedback_result: FeedbackResult,
    control_result: FeedbackResult,
) -> dict[str, Any]:
    """Convert approved S1 arm results into the preregistered raw-count schema."""
    if not experiment_id:
        raise ValueError("experiment_id must be non-empty")
    if not isinstance(target_r, int | float) or not math.isfinite(target_r):
        raise ValueError("target_r must be finite")
    return {
        "experiment_id": experiment_id,
        "target_r": float(target_r),
        "job_ids": [_required_job_id(feedback_result), _required_job_id(control_result)],
        "observable": "binary_phase_synchrony_from_final_counts",
        "arms": [
            {
                "label": S1_FEEDBACK_ARM,
                "records": _records_from_result(feedback_result, S1_FEEDBACK_ARM, n_qubits),
            },
            {
                "label": S1_CONTROL_ARM,
                "records": _records_from_result(control_result, S1_CONTROL_ARM, n_qubits),
            },
        ],
    }


def build_s1_arm_command(
    arm: S1FeedbackArmCircuit,
    *,
    isa_circuits: Sequence[QuantumCircuit],
    timeout_s: float,
) -> FeedbackCommand:
    """Build an approval-gated scheduler command for one S1 IBM arm."""
    if len(isa_circuits) != arm.repetitions:
        raise ValueError("isa_circuits length must match arm repetitions")
    if timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive")
    return FeedbackCommand(
        payload={
            "arm": arm.label,
            "isa_circuits": list(isa_circuits),
            "shots": arm.shots,
            "timeout_s": float(timeout_s),
        },
        label=arm.label,
        estimated_qpu_seconds=arm.estimated_qpu_seconds,
    )


def run_ibm_sampler_arm(
    *,
    backend: Any,
    arm: S1FeedbackArmCircuit,
    isa_circuits: Sequence[QuantumCircuit],
    timeout_s: float,
    sampler_cls: Any | None = None,
) -> FeedbackResult:
    """Submit one S1 arm through IBM SamplerV2 and preserve per-pub counts."""
    if len(isa_circuits) != arm.repetitions:
        raise ValueError("isa_circuits length must match arm repetitions")
    runtime_sampler_cls = sampler_cls
    if runtime_sampler_cls is None:
        from qiskit_ibm_runtime import SamplerV2

        runtime_sampler_cls = SamplerV2

    sampler = runtime_sampler_cls(mode=backend)
    sampler.options.default_shots = arm.shots
    started = time.time()
    job = sampler.run(list(isa_circuits))
    job_id = _job_id(job)
    result = job.result(timeout=timeout_s)
    wall = time.time() - started
    per_pub_counts: list[dict[str, int]] = []
    records: list[dict[str, Any]] = []
    for index, pub_result in enumerate(result):
        counts = _normalise_counts(_extract_counts(pub_result))
        per_pub_counts.append(counts)
        records.append({"source_index": index, "counts": counts})
    return FeedbackResult(
        counts=_aggregate_counts(per_pub_counts),
        job_id=job_id,
        qpu_seconds=wall,
        metadata={
            "arm": arm.label,
            "records": records,
            "shots": arm.shots,
            "repetitions": arm.repetitions,
            "wall_time_s": wall,
        },
    )


def _required_job_id(result: FeedbackResult) -> str:
    if not result.job_id:
        raise ValueError("S1 hardware result must preserve provider job_id")
    return result.job_id


def _records_from_result(
    result: FeedbackResult,
    expected_arm: str,
    n_qubits: int,
) -> list[dict[str, Any]]:
    arm = result.metadata.get("arm")
    if arm != expected_arm:
        raise ValueError(f"expected arm {expected_arm!r}, got {arm!r}")
    records = result.metadata.get("records")
    if not isinstance(records, Sequence) or isinstance(records, str) or not records:
        raise ValueError(f"arm {expected_arm!r} must contain non-empty records")
    converted: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("S1 result records must be mappings")
        counts = record.get("counts")
        if not isinstance(counts, Mapping):
            raise ValueError("S1 result records must preserve counts")
        clean_counts = {str(bitstring): int(count) for bitstring, count in counts.items()}
        converted.append(
            {
                "r_live": binary_phase_synchrony_from_counts(clean_counts, n_qubits=n_qubits),
                "counts": clean_counts,
                "source_index": int(record.get("source_index", index)),
            }
        )
    return converted


def _job_id(job: Any) -> str:
    value = job.job_id if hasattr(job, "job_id") else None
    resolved = value() if callable(value) else value
    if not resolved:
        raise RuntimeError("IBM Runtime job did not expose a job_id")
    return str(resolved)


def _normalise_counts(counts: Mapping[Any, Any]) -> dict[str, int]:
    normalised: dict[str, int] = {}
    for bitstring, count in counts.items():
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"count for {bitstring!r} must be a non-negative integer")
        normalised[str(bitstring)] = count
    if sum(normalised.values()) < 1:
        raise ValueError("counts must contain at least one shot")
    return normalised


def _aggregate_counts(rows: Sequence[Mapping[str, int]]) -> dict[str, int]:
    aggregate: dict[str, int] = {}
    for counts in rows:
        for bitstring, count in counts.items():
            aggregate[bitstring] = aggregate.get(bitstring, 0) + int(count)
    return aggregate
