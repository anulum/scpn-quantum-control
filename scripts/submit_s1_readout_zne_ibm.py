#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S1 readout and ZNE IBM lane
"""Submit preregistered S1 readout-mitigation and local-folding ZNE lanes.

This runner intentionally creates a new lane. It does not retrofit mitigation
onto earlier S1b--S1f artefacts. Each submitted lane contains its own full
three-bit readout calibration circuits, folded main circuits, raw counts, and
reduced analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scpn_quantum_control.hardware.runner import _extract_counts  # noqa: E402
from scpn_quantum_control.hardware.s1_feedback_ibm import (  # noqa: E402
    S1_CONTROL_ARM,
    S1_FEEDBACK_ARM,
    S1FeedbackArmCircuit,
    build_s1_xy_observable_arm_circuits,
    pauli_expectation_from_counts,
)
from scpn_quantum_control.mitigation.readout_matrix import (  # noqa: E402
    build_readout_confusion_matrix,
    computational_basis_labels,
    mitigate_counts,
)
from scpn_quantum_control.mitigation.zne import zne_extrapolate  # noqa: E402
from scripts.prepare_s1_ibm_live_readiness import (  # noqa: E402
    DEFAULT_CREDENTIALS_VAULT,
    SEED_TRANSPILER,
    load_authenticated_backend,
)
from scripts.submit_s1b_ibm_xy_observable_pair import (  # noqa: E402
    DEFAULT_OBSERVABLES,
    _controller_from_policy,
)

EXPERIMENT_ID = "s1_readout_zne_direct_xy_extension_2026-05-20"
PARENT_EXPERIMENT_ID = "s1_dynamic_feedback_preregistration_2026-05-06"
DEFAULT_BACKEND = "ibm_fez"
DEFAULT_LANES = ("s1b", "s1c", "s1d", "s1e", "s1f")
DEFAULT_NOISE_SCALES = (1, 3, 5)
DEFAULT_SHOTS = 1024
DEFAULT_TIMEOUT_S = 7200.0
DEFAULT_MAX_DEPTH = 5000
DEFAULT_MAX_QPU_SECONDS = 1800.0
DATA_DIR = REPO_ROOT / "data" / "s1_feedback_loop"
SYSTEM_QUBITS = 3
TOTAL_QUBITS = 4


@dataclass(frozen=True)
class S1LaneConfig:
    """Concrete S1 direct-observable lane configuration."""

    lane: str
    observables: tuple[str, ...]
    policies: tuple[dict[str, Any], ...]
    repetitions: int


@dataclass(frozen=True)
class CircuitEntry:
    """One circuit and the metadata required for reduction."""

    block: str
    meta: dict[str, Any]
    circuit: QuantumCircuit


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--instance")
    parser.add_argument("--credentials-vault", type=Path, default=DEFAULT_CREDENTIALS_VAULT)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--parent-experiment-id", default=PARENT_EXPERIMENT_ID)
    parser.add_argument("--lanes", nargs="+", default=list(DEFAULT_LANES))
    parser.add_argument("--noise-scales", nargs="+", type=int, default=list(DEFAULT_NOISE_SCALES))
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--max-qpu-seconds", type=float, default=DEFAULT_MAX_QPU_SECONDS)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-budget", action="store_true")
    return parser.parse_args(argv)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return _sha256(path)


def _lane_config(lane: str) -> S1LaneConfig:
    common_observables = tuple(DEFAULT_OBSERVABLES)
    if lane == "s1b":
        return S1LaneConfig(
            lane=lane,
            observables=common_observables,
            repetitions=3,
            policies=(
                {
                    "policy_variant": "s1b_original_body",
                    "n_rounds": 3,
                    "correction_angle": 0.12,
                    "base_gain": 0.8,
                },
            ),
        )
    if lane == "s1c":
        return S1LaneConfig(
            lane=lane,
            observables=common_observables,
            repetitions=3,
            policies=(
                {
                    "policy_variant": "s1c_shallow_positive",
                    "n_rounds": 1,
                    "correction_angle": 0.06,
                    "base_gain": 0.4,
                },
            ),
        )
    if lane in {"s1d", "s1e"}:
        return S1LaneConfig(
            lane=lane,
            observables=common_observables,
            repetitions=5 if lane == "s1e" else 3,
            policies=(
                {
                    "policy_variant": "current_shallow_positive",
                    "n_rounds": 1,
                    "correction_angle": 0.06,
                    "base_gain": 0.4,
                },
                {
                    "policy_variant": "polarity_flipped",
                    "n_rounds": 1,
                    "correction_angle": -0.06,
                    "base_gain": 0.4,
                },
                {
                    "policy_variant": "weak_positive",
                    "n_rounds": 1,
                    "correction_angle": 0.03,
                    "base_gain": 0.2,
                },
            ),
        )
    if lane == "s1f":
        return S1LaneConfig(
            lane=lane,
            observables=("XXI", "YYI", "XYI", "YXI", "ZZI"),
            repetitions=5,
            policies=(
                {
                    "policy_variant": "current_shallow_positive_quadrature_check",
                    "n_rounds": 1,
                    "correction_angle": 0.06,
                    "base_gain": 0.4,
                },
            ),
        )
    raise ValueError(f"unsupported S1 lane: {lane}")


def _validate_scales(scales: Sequence[int]) -> tuple[int, ...]:
    if not scales:
        raise ValueError("noise scales must be non-empty")
    resolved = tuple(int(scale) for scale in scales)
    for scale in resolved:
        if scale < 1 or scale % 2 == 0:
            raise ValueError("noise scales must be odd positive integers")
    return resolved


def _locally_fold_dynamic_circuit(circuit: QuantumCircuit, scale: int) -> QuantumCircuit:
    """Fold invertible quantum operations while preserving dynamic operations."""
    if scale < 1 or scale % 2 == 0:
        raise ValueError("scale must be an odd positive integer")
    folded = circuit.copy_empty_like(name=f"{circuit.name}_lf{scale}")
    qubit_map = {bit: folded.qubits[index] for index, bit in enumerate(circuit.qubits)}
    clbit_map = {bit: folded.clbits[index] for index, bit in enumerate(circuit.clbits)}
    folds = (scale - 1) // 2
    for instruction in circuit.data:
        operation = instruction.operation
        qubits = [qubit_map[bit] for bit in instruction.qubits]
        clbits = [clbit_map[bit] for bit in instruction.clbits]
        folded.append(operation.copy(), qubits, clbits)
        if folds == 0 or clbits or operation.name in {"measure", "reset", "barrier", "delay"}:
            continue
        try:
            inverse = operation.inverse()
        except Exception:
            continue
        for _ in range(folds):
            folded.append(inverse.copy(), qubits, [])
            folded.append(operation.copy(), qubits, [])
    return folded


def _build_lane_entries(
    config: S1LaneConfig, scales: Sequence[int], shots: int
) -> list[CircuitEntry]:
    entries: list[CircuitEntry] = []
    for policy in config.policies:
        controller = _controller_from_policy(
            base_gain=float(policy["base_gain"]),
            correction_angle=float(policy["correction_angle"]),
        )
        arms = build_s1_xy_observable_arm_circuits(
            controller,
            observables=config.observables,
            n_rounds=int(policy["n_rounds"]),
            shots=shots,
            repetitions=config.repetitions,
        )
        for arm in arms:
            for scale in scales:
                folded = _locally_fold_dynamic_circuit(arm.circuit, scale)
                for repetition in range(config.repetitions):
                    entries.append(
                        CircuitEntry(
                            block="main",
                            meta={
                                "lane": config.lane,
                                "policy_variant": policy["policy_variant"],
                                "n_rounds": int(policy["n_rounds"]),
                                "correction_angle": float(policy["correction_angle"]),
                                "base_gain": float(policy["base_gain"]),
                                "arm": arm.label,
                                "observable": arm.observable,
                                "zne_noise_scale": int(scale),
                                "repetition": repetition,
                                "shots": int(shots),
                            },
                            circuit=folded.copy(
                                name=(
                                    f"{config.lane}_{policy['policy_variant']}_{arm.label}_"
                                    f"{arm.observable}_s{scale}_r{repetition}"
                                )
                            ),
                        )
                    )
    return entries


def _basis_labels() -> tuple[str, ...]:
    return computational_basis_labels(SYSTEM_QUBITS)


def _build_calibration_entries(config: S1LaneConfig, shots: int) -> list[CircuitEntry]:
    entries: list[CircuitEntry] = []
    for prepared in _basis_labels():
        qc = QuantumCircuit(TOTAL_QUBITS, SYSTEM_QUBITS, name=f"{config.lane}_readout_{prepared}")
        for label_index, bit in enumerate(prepared):
            if bit == "1":
                qc.x(SYSTEM_QUBITS - 1 - label_index)
        qc.measure([0, 1, 2], [0, 1, 2])
        entries.append(
            CircuitEntry(
                block="readout_calibration",
                meta={
                    "lane": config.lane,
                    "prepared": prepared,
                    "shots": int(shots),
                    "n_qubits": SYSTEM_QUBITS,
                },
                circuit=qc,
            )
        )
    return entries


def _representative_arm(config: S1LaneConfig) -> S1FeedbackArmCircuit:
    policy = config.policies[0]
    controller = _controller_from_policy(
        base_gain=float(policy["base_gain"]),
        correction_angle=float(policy["correction_angle"]),
    )
    return build_s1_xy_observable_arm_circuits(
        controller,
        observables=(config.observables[0],),
        n_rounds=int(policy["n_rounds"]),
        shots=DEFAULT_SHOTS,
        repetitions=1,
    )[0]


def _select_physical_layout(backend: Any, config: S1LaneConfig) -> list[int]:
    arm = _representative_arm(config)
    isa = transpile(
        arm.circuit,
        backend=backend,
        optimization_level=1,
        seed_transpiler=SEED_TRANSPILER,
    )
    physical_by_logical: list[int] = []
    physical_bits = isa.layout.initial_layout.get_physical_bits()
    for logical_qubit in arm.circuit.qubits[:TOTAL_QUBITS]:
        matches = [
            physical for physical, virtual in physical_bits.items() if virtual == logical_qubit
        ]
        if len(matches) != 1:
            raise RuntimeError("could not extract selected S1 physical layout")
        physical_by_logical.append(int(matches[0]))
    return physical_by_logical


def _transpile_entries(
    backend: Any,
    entries: Sequence[CircuitEntry],
    *,
    initial_layout: Sequence[int],
) -> list[QuantumCircuit]:
    return [
        transpile(
            entry.circuit,
            backend=backend,
            initial_layout=list(initial_layout),
            optimization_level=1,
            seed_transpiler=SEED_TRANSPILER,
        )
        for entry in entries
    ]


def _operation_counts(circuit: QuantumCircuit) -> dict[str, int]:
    return {str(name): int(count) for name, count in circuit.count_ops().items()}


def _expectation_from_probabilities(
    probabilities: np.ndarray,
    labels: Sequence[str],
    observable: str,
) -> float:
    active_indices = [index for index, pauli in enumerate(observable) if pauli != "I"]
    total = 0.0
    for label, probability in zip(labels, probabilities, strict=True):
        eigenvalue = 1.0
        for index in active_indices:
            eigenvalue *= 1.0 if label[index] == "0" else -1.0
        total += eigenvalue * float(probability)
    return float(total)


def _stderr(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(stdev(values) / (len(values) ** 0.5))


def _extract_result_rows(
    result: Any,
    entries: Sequence[CircuitEntry],
) -> tuple[str, list[dict[str, Any]]]:
    job_id_attr = result.job_id if hasattr(result, "job_id") else None
    job_id = job_id_attr() if callable(job_id_attr) else job_id_attr
    if not job_id:
        raise RuntimeError("IBM Runtime job did not expose a job id")
    rows: list[dict[str, Any]] = []
    payload = result.result(timeout=None)
    for index, pub_result in enumerate(payload):
        counts = {str(key): int(value) for key, value in _extract_counts(pub_result).items()}
        rows.append(
            {
                "block": entries[index].block,
                "meta": entries[index].meta,
                "counts": counts,
                "job_id": str(job_id),
                "source_index": index,
            }
        )
    return str(job_id), rows


def _readout_calibration_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    calibration: dict[str, dict[str, int]] = {}
    for row in rows:
        if row["block"] != "readout_calibration":
            continue
        prepared = str(row["meta"]["prepared"])
        calibration[prepared] = {str(key): int(value) for key, value in row["counts"].items()}
    return calibration


def _readout_rows(
    calibration_counts: Mapping[str, Mapping[str, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prepared, counts in sorted(calibration_counts.items()):
        total = sum(int(value) for value in counts.values())
        retained = int(counts.get(prepared, 0))
        rows.append(
            {
                "prepared": prepared,
                "total_shots": total,
                "retention": retained / total if total else None,
                "counts": dict(counts),
            }
        )
    return rows


def _analyse_rows(
    *,
    args: argparse.Namespace,
    lane: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    calibration_counts = _readout_calibration_counts(rows)
    readout_model = build_readout_confusion_matrix(calibration_counts, SYSTEM_QUBITS)
    labels = readout_model.labels

    grouped: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["block"] != "main":
            continue
        meta = row["meta"]
        key = (
            str(meta["policy_variant"]),
            str(meta["observable"]),
            str(meta["arm"]),
            int(meta["zne_noise_scale"]),
        )
        grouped[key].append(row)

    scale_rows: list[dict[str, Any]] = []
    for key, records in sorted(grouped.items()):
        raw_values: list[float] = []
        mitigated_values: list[float] = []
        for record in records:
            counts = {str(k): int(v) for k, v in record["counts"].items()}
            raw_values.append(
                pauli_expectation_from_counts(
                    counts,
                    observable=key[1],
                    n_qubits=SYSTEM_QUBITS,
                )
            )
            mitigated = mitigate_counts(counts, readout_model)
            mitigated_values.append(_expectation_from_probabilities(mitigated, labels, key[1]))
        scale_rows.append(
            {
                "policy_variant": key[0],
                "observable": key[1],
                "arm": key[2],
                "zne_noise_scale": key[3],
                "n_repetitions": len(records),
                "mean_expectation": float(mean(raw_values)),
                "stderr_expectation": _stderr(raw_values),
                "readout_mitigated_mean_expectation": float(mean(mitigated_values)),
                "readout_mitigated_stderr_expectation": _stderr(mitigated_values),
            }
        )

    paired_by_channel: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scale_rows:
        paired_by_channel[(row["policy_variant"], row["observable"])].append(row)

    channel_rows: list[dict[str, Any]] = []
    for key, scale_group in sorted(paired_by_channel.items()):
        by_scale_arm = {(int(row["zne_noise_scale"]), str(row["arm"])): row for row in scale_group}
        scale_values: list[int] = []
        raw_deltas: list[float] = []
        mitigated_deltas: list[float] = []
        for scale in sorted({int(row["zne_noise_scale"]) for row in scale_group}):
            feedback = by_scale_arm.get((scale, S1_FEEDBACK_ARM))
            control = by_scale_arm.get((scale, S1_CONTROL_ARM))
            if feedback is None or control is None:
                continue
            scale_values.append(scale)
            raw_deltas.append(
                float(feedback["mean_expectation"]) - float(control["mean_expectation"])
            )
            mitigated_deltas.append(
                float(feedback["readout_mitigated_mean_expectation"])
                - float(control["readout_mitigated_mean_expectation"])
            )
        linear = zne_extrapolate(scale_values, raw_deltas, order=1)
        linear_mitigated = zne_extrapolate(scale_values, mitigated_deltas, order=1)
        quadratic = None
        quadratic_mitigated = None
        if len(scale_values) >= 3:
            quadratic = zne_extrapolate(scale_values, raw_deltas, order=2)
            quadratic_mitigated = zne_extrapolate(scale_values, mitigated_deltas, order=2)
        channel_rows.append(
            {
                "policy_variant": key[0],
                "observable": key[1],
                "noise_scales": scale_values,
                "scale_feedback_minus_control": raw_deltas,
                "readout_mitigated_scale_feedback_minus_control": mitigated_deltas,
                "scale1_feedback_minus_control": raw_deltas[0],
                "linear_zne_feedback_minus_control": linear.zero_noise_estimate,
                "linear_zne_fit_residual": linear.fit_residual,
                "readout_mitigated_linear_zne_feedback_minus_control": (
                    linear_mitigated.zero_noise_estimate
                ),
                "readout_mitigated_linear_zne_fit_residual": linear_mitigated.fit_residual,
                "quadratic_zne_feedback_minus_control": (
                    None if quadratic is None else quadratic.zero_noise_estimate
                ),
                "quadratic_zne_fit_residual": None
                if quadratic is None
                else quadratic.fit_residual,
                "readout_mitigated_quadratic_zne_feedback_minus_control": (
                    None
                    if quadratic_mitigated is None
                    else quadratic_mitigated.zero_noise_estimate
                ),
                "readout_mitigated_quadratic_zne_fit_residual": (
                    None if quadratic_mitigated is None else quadratic_mitigated.fit_residual
                ),
            }
        )

    return {
        "schema": "scpn_s1_readout_zne_analysis_v1",
        "experiment_id": args.experiment_id,
        "parent_experiment_id": args.parent_experiment_id,
        "backend": args.backend,
        "lane": lane,
        "readout_model": {
            "n_qubits": readout_model.n_qubits,
            "labels": list(readout_model.labels),
            "condition_number": readout_model.condition_number,
            "shots_by_prepared_state": readout_model.shots_by_prepared_state,
            "calibration_rows": _readout_rows(calibration_counts),
        },
        "scale_rows": scale_rows,
        "channel_rows": channel_rows,
        "mean_abs_scale1_feedback_minus_control": float(
            mean(abs(row["scale1_feedback_minus_control"]) for row in channel_rows)
        ),
        "mean_abs_linear_zne_feedback_minus_control": float(
            mean(abs(row["linear_zne_feedback_minus_control"]) for row in channel_rows)
        ),
        "mean_abs_readout_mitigated_linear_zne_feedback_minus_control": float(
            mean(
                abs(row["readout_mitigated_linear_zne_feedback_minus_control"])
                for row in channel_rows
            )
        ),
        "claim_boundary": (
            "Dedicated S1 readout-mitigation and local-folding ZNE lane. It is "
            "not a retroactive mitigation of earlier S1b--S1f jobs and does "
            "not establish backend-general feedback control."
        ),
    }


def _readiness_payload(
    *,
    args: argparse.Namespace,
    config: S1LaneConfig,
    entries: Sequence[CircuitEntry],
    isa_circuits: Sequence[QuantumCircuit],
    physical_layout: Sequence[int],
) -> dict[str, Any]:
    depths = [int(circuit.depth()) for circuit in isa_circuits]
    estimated = len(isa_circuits) * 0.55
    return {
        "schema": "scpn_s1_readout_zne_readiness_v1",
        "timestamp_utc": _timestamp(),
        "experiment_id": args.experiment_id,
        "parent_experiment_id": args.parent_experiment_id,
        "backend": args.backend,
        "lane": config.lane,
        "hardware_submission": False,
        "noise_scales": list(_validate_scales(args.noise_scales)),
        "shots": int(args.shots),
        "policy_variants": list(config.policies),
        "observables": list(config.observables),
        "repetitions": int(config.repetitions),
        "physical_layout": {
            "system_qubits": list(physical_layout[:SYSTEM_QUBITS]),
            "monitor_qubit": int(physical_layout[SYSTEM_QUBITS]),
            "logical_order": ["sys_0", "sys_1", "sys_2", "monitor"],
        },
        "n_main_circuits": sum(1 for entry in entries if entry.block == "main"),
        "n_readout_calibration_circuits": sum(
            1 for entry in entries if entry.block == "readout_calibration"
        ),
        "n_total_circuits": len(entries),
        "max_depth": int(args.max_depth),
        "max_qpu_seconds": float(args.max_qpu_seconds),
        "estimated_qpu_seconds": estimated,
        "depth_summary": {
            "min": min(depths),
            "max": max(depths),
            "mean": float(mean(depths)),
        },
        "operation_counts_first_circuit": _operation_counts(isa_circuits[0]),
        "status": (
            "ready_for_submission"
            if max(depths) <= int(args.max_depth) and estimated <= float(args.max_qpu_seconds)
            else "blocked"
        ),
        "reasons": [
            "local gate folding preserves mid-circuit measurement and conditional structure",
            "full three-bit readout calibration is included in the same submitted lane",
            "all circuits are pinned to the selected S1 system/monitor physical layout",
        ],
    }


def _run_lane(args: argparse.Namespace, backend: Any, lane: str) -> int:
    config = _lane_config(lane)
    scales = _validate_scales(args.noise_scales)
    entries = _build_lane_entries(config, scales, int(args.shots))
    entries.extend(_build_calibration_entries(config, int(args.shots)))
    physical_layout = _select_physical_layout(backend, config)
    isa_circuits = _transpile_entries(backend, entries, initial_layout=physical_layout)
    readiness = _readiness_payload(
        args=args,
        config=config,
        entries=entries,
        isa_circuits=isa_circuits,
        physical_layout=physical_layout,
    )
    timestamp = readiness["timestamp_utc"]
    readiness_path = args.out_dir / f"{lane}_readout_zne_readiness_{args.backend}_{timestamp}.json"
    readiness_sha = _write_json(readiness_path, readiness)
    print(f"{lane}: readiness={readiness['status']}")
    print(f"{lane}: readiness_json={readiness_path}")
    print(f"{lane}: readiness_sha256={readiness_sha}")
    if readiness["status"] != "ready_for_submission":
        return 3
    if not args.submit:
        return 0
    if not args.confirm_budget:
        print("ERROR: --submit requires --confirm-budget", file=sys.stderr)
        return 2

    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = int(args.shots)
    started = time.time()
    job = sampler.run(isa_circuits)
    job_id, rows = _extract_result_rows(job, entries)
    wall = time.time() - started

    raw_payload = {
        "schema": "scpn_s1_readout_zne_raw_counts_v1",
        "experiment_id": args.experiment_id,
        "parent_experiment_id": args.parent_experiment_id,
        "backend": args.backend,
        "lane": lane,
        "timestamp_utc": timestamp,
        "job_ids": [job_id],
        "wall_time_s": wall,
        "readiness_json": str(readiness_path.relative_to(REPO_ROOT)),
        "readiness_sha256": readiness_sha,
        "physical_layout": readiness["physical_layout"],
        "noise_scales": list(scales),
        "shots": int(args.shots),
        "circuits": rows,
    }
    raw_path = args.out_dir / f"{lane}_readout_zne_raw_counts_{args.backend}_{timestamp}.json"
    raw_sha = _write_json(raw_path, raw_payload)
    analysis = _analyse_rows(args=args, lane=lane, rows=rows)
    analysis["raw_counts_json"] = str(raw_path.relative_to(REPO_ROOT))
    analysis["raw_counts_sha256"] = raw_sha
    analysis_path = args.out_dir / f"{lane}_readout_zne_analysis_{args.backend}_{timestamp}.json"
    analysis_sha = _write_json(analysis_path, analysis)
    print(f"{lane}: hardware_submission=true")
    print(f"{lane}: job_id={job_id}")
    print(f"{lane}: raw_counts_json={raw_path}")
    print(f"{lane}: raw_counts_sha256={raw_sha}")
    print(f"{lane}: analysis_json={analysis_path}")
    print(f"{lane}: analysis_sha256={analysis_sha}")
    print(
        f"{lane}: mean_abs_scale1_feedback_minus_control="
        f"{analysis['mean_abs_scale1_feedback_minus_control']}"
    )
    print(
        f"{lane}: mean_abs_linear_zne_feedback_minus_control="
        f"{analysis['mean_abs_linear_zne_feedback_minus_control']}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.submit and not args.confirm_budget:
        print("ERROR: --submit requires --confirm-budget", file=sys.stderr)
        return 2
    backend = load_authenticated_backend(args.backend, args.instance, args.credentials_vault)
    status = 0
    for lane in args.lanes:
        status = max(status, _run_lane(args, backend, lane))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
