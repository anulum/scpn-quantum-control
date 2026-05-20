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
    observable: str = "binary_phase_synchrony"

    def __post_init__(self) -> None:
        if self.label not in {S1_FEEDBACK_ARM, S1_CONTROL_ARM}:
            raise ValueError("unsupported S1 arm label")
        if self.shots < 1:
            raise ValueError("shots must be positive")
        if self.repetitions < 1:
            raise ValueError("repetitions must be positive")
        if self.estimated_qpu_seconds < 0.0:
            raise ValueError("estimated_qpu_seconds must be non-negative")
        _validate_observable_label(self.observable, n_qubits=None)


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


def build_s1_xy_observable_arm_circuits(
    controller: RealtimeSyncFeedbackController,
    *,
    observables: Sequence[str],
    n_rounds: int,
    shots: int,
    repetitions: int,
    estimated_seconds_per_circuit: float = 1.0,
) -> tuple[S1FeedbackArmCircuit, ...]:
    """Build S1b direct-XY observable variants for feedback and control arms."""
    if not observables:
        raise ValueError("observables must be non-empty")
    feedback, control = build_s1_feedback_arm_circuits(
        controller,
        n_rounds=n_rounds,
        shots=shots,
        repetitions=repetitions,
        estimated_seconds_per_circuit=estimated_seconds_per_circuit,
    )
    arms: list[S1FeedbackArmCircuit] = []
    for source in (feedback, control):
        for observable in observables:
            _validate_observable_label(observable, n_qubits=controller.n)
            arms.append(
                S1FeedbackArmCircuit(
                    source.label,
                    _with_final_pauli_basis(source.circuit, observable),
                    source.shots,
                    source.repetitions,
                    source.estimated_qpu_seconds,
                    observable,
                )
            )
    return tuple(arms)


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
        system_bits = bitstring[-n_qubits:].zfill(n_qubits)
        phase_sum = sum(1.0 if bit == "0" else -1.0 for bit in system_bits)
        synchrony = abs(phase_sum / n_qubits)
        weighted += raw_count * synchrony
        total += raw_count
    if total < 1:
        raise ValueError("counts must contain at least one shot")
    return weighted / total


def pauli_expectation_from_counts(
    counts: Mapping[str, int],
    *,
    observable: str,
    n_qubits: int,
) -> float:
    """Estimate a reduced Pauli expectation from rotated-basis counts."""
    _validate_observable_label(observable, n_qubits=n_qubits)
    active_indices = [index for index, pauli in enumerate(observable) if pauli != "I"]
    if not active_indices:
        raise ValueError("observable must contain at least one non-identity Pauli")
    total = 0
    weighted = 0.0
    for raw_bitstring, raw_count in counts.items():
        if not isinstance(raw_bitstring, str) or not raw_bitstring:
            raise ValueError("count keys must be non-empty bitstrings")
        if not isinstance(raw_count, int) or raw_count < 0:
            raise ValueError(f"count for {raw_bitstring!r} must be a non-negative integer")
        bitstring = raw_bitstring.replace(" ", "")
        system_bits = bitstring[-n_qubits:].zfill(n_qubits)
        eigenvalue = 1.0
        for index in active_indices:
            eigenvalue *= 1.0 if system_bits[index] == "0" else -1.0
        weighted += raw_count * eigenvalue
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


def raw_count_package_from_xy_observable_results(
    *,
    experiment_id: str,
    n_qubits: int,
    results: Sequence[FeedbackResult],
) -> dict[str, Any]:
    """Convert S1b XY-observable arm results into a grouped analysis package."""
    if not experiment_id:
        raise ValueError("experiment_id must be non-empty")
    if not results:
        raise ValueError("results must be non-empty")
    grouped: dict[str, dict[str, FeedbackResult]] = {}
    job_ids: list[str] = []
    for result in results:
        job_ids.append(_required_job_id(result))
        arm = result.metadata.get("arm")
        observable = result.metadata.get("observable")
        if arm not in {S1_FEEDBACK_ARM, S1_CONTROL_ARM}:
            raise ValueError("S1b result must preserve a supported arm label")
        if not isinstance(observable, str):
            raise ValueError("S1b result must preserve an observable label")
        _validate_observable_label(observable, n_qubits=n_qubits)
        grouped.setdefault(observable, {})[str(arm)] = result

    observables: list[dict[str, Any]] = []
    for observable, by_arm in sorted(grouped.items()):
        if S1_FEEDBACK_ARM not in by_arm or S1_CONTROL_ARM not in by_arm:
            raise ValueError(f"observable {observable!r} must contain both S1 arms")
        feedback_arm = _observable_arm_summary(
            by_arm[S1_FEEDBACK_ARM],
            expected_arm=S1_FEEDBACK_ARM,
            observable=observable,
            n_qubits=n_qubits,
        )
        control_arm = _observable_arm_summary(
            by_arm[S1_CONTROL_ARM],
            expected_arm=S1_CONTROL_ARM,
            observable=observable,
            n_qubits=n_qubits,
        )
        observables.append(
            {
                "basis": observable,
                "arms": [feedback_arm, control_arm],
                "feedback_minus_control": (
                    feedback_arm["mean_expectation"] - control_arm["mean_expectation"]
                ),
            }
        )
    return {
        "experiment_id": experiment_id,
        "job_ids": job_ids,
        "observable_family": "direct_xy_pauli_correlators",
        "observables": observables,
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
            "observable": arm.observable,
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


def _observable_arm_summary(
    result: FeedbackResult,
    *,
    expected_arm: str,
    observable: str,
    n_qubits: int,
) -> dict[str, Any]:
    records = _records_from_observable_result(result, expected_arm, observable, n_qubits)
    expectations = [float(record["expectation"]) for record in records]
    return {
        "label": expected_arm,
        "job_id": _required_job_id(result),
        "records": records,
        "mean_expectation": sum(expectations) / len(expectations),
        "repetitions": len(records),
        "total_shots": sum(sum(record["counts"].values()) for record in records),
    }


def _records_from_observable_result(
    result: FeedbackResult,
    expected_arm: str,
    observable: str,
    n_qubits: int,
) -> list[dict[str, Any]]:
    arm = result.metadata.get("arm")
    if arm != expected_arm:
        raise ValueError(f"expected arm {expected_arm!r}, got {arm!r}")
    if result.metadata.get("observable") != observable:
        raise ValueError(f"expected observable {observable!r}")
    records = result.metadata.get("records")
    if not isinstance(records, Sequence) or isinstance(records, str) or not records:
        raise ValueError(f"arm {expected_arm!r} must contain non-empty records")
    converted: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("S1b result records must be mappings")
        counts = record.get("counts")
        if not isinstance(counts, Mapping):
            raise ValueError("S1b result records must preserve counts")
        clean_counts = {str(bitstring): int(count) for bitstring, count in counts.items()}
        converted.append(
            {
                "expectation": pauli_expectation_from_counts(
                    clean_counts,
                    observable=observable,
                    n_qubits=n_qubits,
                ),
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


def _validate_observable_label(observable: str, *, n_qubits: int | None) -> None:
    if not isinstance(observable, str) or not observable:
        raise ValueError("observable must be non-empty text")
    if observable == "binary_phase_synchrony":
        return
    if any(pauli not in {"I", "X", "Y", "Z"} for pauli in observable):
        raise ValueError("observable may contain only I, X, Y, or Z")
    if n_qubits is not None and len(observable) != n_qubits:
        raise ValueError("observable length must match n_qubits")


def _with_final_pauli_basis(circuit: QuantumCircuit, observable: str) -> QuantumCircuit:
    """Return a copy of ``circuit`` with final readout rotated to ``observable``."""
    n_qubits = len(observable)
    qc = circuit.copy()
    for _ in range(n_qubits):
        if not qc.data or qc.data[-1].operation.name != "measure":
            raise ValueError("S1 circuit must end with system measurements")
        qc.data.pop()
    system_qubits = list(qc.qubits[:n_qubits])
    readout_bits = list(qc.clbits[-n_qubits:])
    for index, pauli in enumerate(observable):
        if pauli == "X":
            qc.h(system_qubits[index])
        elif pauli == "Y":
            qc.sdg(system_qubits[index])
            qc.h(system_qubits[index])
    qc.measure(system_qubits, readout_bits)
    return qc
