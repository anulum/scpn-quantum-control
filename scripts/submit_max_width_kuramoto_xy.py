#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — maximum-width Kuramoto-XY campaign submission (WIDTH-1)
"""Approval-gated maximum-width Kuramoto-XY campaign submitter (WIDTH-1).

Executes the campaign preregistered in
``docs/campaigns/max_width_kuramoto_xy_prereg_2026-07-16.md``: shallow
first-order-Trotter Kuramoto-XY chain circuits at widths 32 → 64 → 104 →
device-max on an error-aware 1-D chain selected from fresh calibration data,
two measurement settings (all-X, all-Y) per width and Trotter depth, per-qubit
readout calibration, and the order parameter
``R = n⁻¹ · sqrt[(Σ_k <X_k>)² + (Σ_k <Y_k>)²]`` reported mitigated and
unmitigated against an exact matrix-product-state baseline of the identical
circuits.

The workload family is the repository standard: the Paper 27
exponential-decay ``K_nm`` profile restricted to nearest-neighbour chain
couplings, natural frequencies from ``omega_for_oscillators`` (the canonical
16-frequency table with its documented periodic extension), and the
``build_trotter_circuit`` initial-state and evolution conventions.

Claim boundary (preregistered): workload-width engineering and noise
characterisation only — these shallow one-dimensional circuits are
classically simulable by construction, so NO advantage claim of any kind is
available from this campaign, and a degraded ``R`` at large width is a
publishable outcome of equal standing.

The QPU-seconds estimate uses the empirical anchor from the 2026-05-05 B-C
scaling run: 280 circuits cost 305 IBM usage seconds ≈ 1.09 s per circuit
(ledger row ``ibm-run-1f46ebd0da8912ff``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scpn_quantum_control.bridge.knm_hamiltonian import (  # noqa: E402
    build_knm_paper27,
    knm_to_hamiltonian,
    omega_for_oscillators,
)
from scpn_quantum_control.hardware._count_integrity import (  # noqa: E402
    strict_fixed_width_bitstring_key,
    strict_non_negative_count,
    strict_provider_job_id,
    strict_shot_conservation,
)
from scpn_quantum_control.hardware.dynq_layout_pass import (  # noqa: E402
    calibration_from_target,
)
from scpn_quantum_control.hardware.error_aware_chain import (  # noqa: E402
    ChainSelection,
    longest_error_aware_chain,
    select_error_aware_chain,
)
from scpn_quantum_control.hardware.feedback_hardware_scheduler import (  # noqa: E402
    HardwareApprovalRecord,
    hash_package_manifest,
)

EXPERIMENT_ID = "max_width_kuramoto_xy_2026-07-16"
PREREGISTRATION_PATH = "docs/campaigns/max_width_kuramoto_xy_prereg_2026-07-16.md"
SCHEMA_PREFIX = "scpn_max_width_kuramoto_xy"

PREREGISTERED_WIDTHS = (32, 64, 104)
TROTTER_REPS = (1, 2)
EVOLUTION_TIME = 0.1
SETTINGS = ("x", "y")

DEFAULT_BACKEND = "ibm_fez"
DEFAULT_SHOTS = 4096
DEFAULT_CAL_SHOTS = 8192
DEFAULT_MAX_QPU_SECONDS = 120.0
DEFAULT_MEDIAN_EDGE_ERROR_ABORT = 0.03
ESTIMATED_SECONDS_PER_CIRCUIT = 1.1
SEED_TRANSPILER = 20260716
DATA_DIR = REPO_ROOT / "data" / "max_width_kuramoto_xy"


def chain_coupling(n: int) -> np.typing.NDArray[np.float64]:
    """Paper 27 exponential-decay coupling restricted to the chain edges."""
    full = build_knm_paper27(L=n)
    chain = np.zeros_like(full)
    for index in range(n - 1):
        chain[index, index + 1] = chain[index + 1, index] = full[index, index + 1]
    return chain


def evolution_body(n: int, reps: int) -> QuantumCircuit:
    """Initial state plus first-order Trotter evolution, no measurement.

    Matches ``hardware.circuit_export.build_trotter_circuit`` (per-qubit
    ``ry(ω_k mod 2π)`` preparation, ``PauliEvolutionGate`` with ``LieTrotter``)
    on the chain-restricted coupling.
    """
    coupling = chain_coupling(n)
    omega = omega_for_oscillators(n)
    hamiltonian = knm_to_hamiltonian(coupling, omega)
    evolution = PauliEvolutionGate(
        hamiltonian, time=EVOLUTION_TIME, synthesis=LieTrotter(reps=reps)
    )
    circuit = QuantumCircuit(n, name=f"width{n}_reps{reps}")
    for qubit in range(n):
        circuit.ry(float(omega[qubit]) % (2.0 * math.pi), qubit)
    circuit.append(evolution, range(n))
    return circuit


def measured_setting_circuit(body: QuantumCircuit, setting: str) -> QuantumCircuit:
    """Append the all-X or all-Y basis rotation and measure every qubit."""
    if setting not in SETTINGS:
        raise ValueError(f"unknown measurement setting {setting!r}")
    circuit = body.copy(name=f"{body.name}_{setting}")
    circuit.barrier()
    for qubit in range(circuit.num_qubits):
        if setting == "y":
            circuit.sdg(qubit)
        circuit.h(qubit)
    circuit.measure_all()
    return circuit


def width_calibration_circuits(n: int) -> tuple[QuantumCircuit, QuantumCircuit]:
    """Per-qubit readout calibration: all-zeros and all-ones preparations."""
    zeros = QuantumCircuit(n, name=f"width{n}_cal0")
    zeros.measure_all()
    ones = QuantumCircuit(n, name=f"width{n}_cal1")
    ones.x(range(n))
    ones.measure_all()
    return zeros, ones


@dataclass(frozen=True)
class WidthPlan:
    """One width point: chain layout plus its transpiled circuit block."""

    width: int
    chain: ChainSelection
    circuits: tuple[QuantumCircuit, ...]
    labels: tuple[str, ...]
    isa_depths: tuple[int, ...]


def build_width_plan(
    width: int,
    chain: ChainSelection,
    backend: Any,
) -> WidthPlan:
    """Build and transpile every circuit for one width on its chain layout."""
    circuits: list[QuantumCircuit] = []
    labels: list[str] = []
    for reps in TROTTER_REPS:
        body = evolution_body(width, reps)
        for setting in SETTINGS:
            circuits.append(measured_setting_circuit(body, setting))
            labels.append(f"main_reps{reps}_{setting}")
    zeros, ones = width_calibration_circuits(width)
    circuits.extend([zeros, ones])
    labels.extend(["cal0", "cal1"])
    isa_circuits = transpile(
        circuits,
        backend=backend,
        initial_layout=list(chain.qubits),
        optimization_level=1,
        seed_transpiler=SEED_TRANSPILER,
    )
    return WidthPlan(
        width=width,
        chain=chain,
        circuits=tuple(isa_circuits),
        labels=tuple(labels),
        isa_depths=tuple(int(circuit.depth()) for circuit in isa_circuits),
    )


def coerce_counts(
    raw_counts: Mapping[str, object],
    *,
    width: int,
    expected_shots: int,
    field_name: str,
) -> dict[str, int]:
    """Fail-closed coercion of provider counts to exact-width validated counts."""
    coerced: dict[str, int] = {}
    for key, value in raw_counts.items():
        bitstring = strict_fixed_width_bitstring_key(
            key, width=width, field_name=f"{field_name} key"
        )
        count = strict_non_negative_count(value, field_name=f"{field_name} value")
        if bitstring in coerced:
            raise ValueError(f"{field_name} carries duplicate bitstring {bitstring!r}")
        coerced[bitstring] = count
    strict_shot_conservation(coerced, expected_shots=expected_shots, field_name=field_name)
    return coerced


def per_qubit_z_expectations(counts: Mapping[str, int], width: int) -> list[float]:
    """Little-endian per-qubit ``<Z_k>`` marginals from one count table."""
    totals = [0.0] * width
    shots = 0
    for bitstring, count in counts.items():
        shots += count
        for qubit in range(width):
            if bitstring[-1 - qubit] == "0":
                totals[qubit] += count
            else:
                totals[qubit] -= count
    if shots <= 0:
        raise ValueError("no shots recorded in count table")
    return [total / shots for total in totals]


def readout_error_pairs(
    cal0: Mapping[str, int],
    cal1: Mapping[str, int],
    width: int,
) -> list[tuple[float, float]]:
    """Per-qubit ``(ε₀, ε₁)``: P(read 1 | prepared 0) and P(read 0 | prepared 1)."""
    z_prepared_zero = per_qubit_z_expectations(cal0, width)
    z_prepared_one = per_qubit_z_expectations(cal1, width)
    pairs: list[tuple[float, float]] = []
    for qubit in range(width):
        epsilon_zero = (1.0 - z_prepared_zero[qubit]) / 2.0
        epsilon_one = (1.0 + z_prepared_one[qubit]) / 2.0
        pairs.append((epsilon_zero, epsilon_one))
    return pairs


def mitigate_z_expectation(raw: float, epsilon_zero: float, epsilon_one: float) -> float:
    """Invert the per-qubit binary readout channel on one ``<Z>`` value.

    With ``p₁ = ε₀(1 − q₁) + (1 − ε₁)q₁`` for true excited-state probability
    ``q₁``, the inversion is ``q₁ = (p₁ − ε₀) / (1 − ε₀ − ε₁)``. A calibration
    with ``ε₀ + ε₁ ≥ 1`` carries no invertible information and fails closed.
    """
    denominator = 1.0 - epsilon_zero - epsilon_one
    if denominator <= 0.0:
        raise ValueError("readout calibration is not invertible (epsilon0 + epsilon1 >= 1)")
    observed_p1 = (1.0 - raw) / 2.0
    true_p1 = (observed_p1 - epsilon_zero) / denominator
    return 1.0 - 2.0 * true_p1


def order_parameter(x_expectations: Sequence[float], y_expectations: Sequence[float]) -> float:
    """Kuramoto order parameter ``R`` from per-qubit X and Y expectations."""
    if len(x_expectations) != len(y_expectations) or not x_expectations:
        raise ValueError("X and Y expectation vectors must be equal-length and non-empty")
    n = len(x_expectations)
    return math.hypot(sum(x_expectations), sum(y_expectations)) / n


def exact_baseline_expectations(width: int, reps: int) -> tuple[list[float], list[float]]:
    """Exact per-qubit ``<X_k>``/``<Y_k>`` of the identical circuit via Aer MPS."""
    from qiskit_aer import AerSimulator

    # Expand the evolution gate through its attached LieTrotter definition
    # first: re-synthesis during transpilation would reorder the Trotter
    # exponentials and silently change the simulated unitary at O(t²). The
    # basis translation is exact, and passing basis_gates instead of the
    # simulator target avoids Aer's 63-qubit default coupling-map cap (the
    # MPS method itself has no such width limit). The save instructions are
    # appended AFTER translation — they are Aer-native and have no
    # equivalence rules in the standard library.
    simulation = transpile(
        evolution_body(width, reps).decompose(),
        basis_gates=["cx", "rz", "ry", "rx", "h", "x", "sdg", "id"],
        optimization_level=0,
    )
    for qubit in range(width):
        simulation.save_expectation_value(
            _single_pauli(width, qubit, "X"), list(range(width)), label=f"x{qubit}"
        )
        simulation.save_expectation_value(
            _single_pauli(width, qubit, "Y"), list(range(width)), label=f"y{qubit}"
        )
    simulator = AerSimulator(method="matrix_product_state")
    data = simulator.run(simulation, shots=1).result().data(0)
    x_values = [float(data[f"x{qubit}"]) for qubit in range(width)]
    y_values = [float(data[f"y{qubit}"]) for qubit in range(width)]
    return x_values, y_values


def _single_pauli(width: int, qubit: int, axis: str) -> Any:
    from qiskit.quantum_info import SparsePauliOp

    label = ["I"] * width
    label[width - 1 - qubit] = axis
    return SparsePauliOp("".join(label))


def analyse_width(
    width: int,
    reps: int,
    x_counts: Mapping[str, int],
    y_counts: Mapping[str, int],
    cal0: Mapping[str, int],
    cal1: Mapping[str, int],
    *,
    include_baseline: bool = True,
) -> dict[str, Any]:
    """Hardware R (mitigated and unmitigated) against the exact baseline."""
    raw_x = per_qubit_z_expectations(x_counts, width)
    raw_y = per_qubit_z_expectations(y_counts, width)
    epsilon_pairs = readout_error_pairs(cal0, cal1, width)
    mitigated_x = [
        mitigate_z_expectation(value, *epsilon_pairs[qubit]) for qubit, value in enumerate(raw_x)
    ]
    mitigated_y = [
        mitigate_z_expectation(value, *epsilon_pairs[qubit]) for qubit, value in enumerate(raw_y)
    ]
    result: dict[str, Any] = {
        "width": width,
        "trotter_reps": reps,
        "r_unmitigated": order_parameter(raw_x, raw_y),
        "r_mitigated": order_parameter(mitigated_x, mitigated_y),
        "per_qubit": {
            "x_unmitigated": raw_x,
            "y_unmitigated": raw_y,
            "x_mitigated": mitigated_x,
            "y_mitigated": mitigated_y,
            "readout_epsilon_pairs": [list(pair) for pair in epsilon_pairs],
        },
    }
    if include_baseline:
        exact_x, exact_y = exact_baseline_expectations(width, reps)
        result["r_exact_mps"] = order_parameter(exact_x, exact_y)
        result["baseline_per_qubit"] = {"x_exact": exact_x, "y_exact": exact_y}
    return result


def estimate_qpu_seconds(n_circuits: int) -> float:
    """Estimate IBM usage seconds from the empirical per-circuit anchor."""
    return float(n_circuits) * ESTIMATED_SECONDS_PER_CIRCUIT


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
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return _sha256(path)


def _pub_counts(pub_result: Any) -> dict[str, Any]:
    """Extract raw counts from a SamplerV2 pub result."""
    data = pub_result.data
    for register_name in ("meas", "c", "cr", "c0"):
        register = getattr(data, register_name, None)
        if register is not None and hasattr(register, "get_counts"):
            counts: dict[str, Any] = register.get_counts()
            return counts
    raise RuntimeError("pub result carries no classical register with counts")


def plan_widths(
    target: Any,
    *,
    median_edge_error_abort: float,
) -> tuple[list[tuple[int, ChainSelection]], list[dict[str, Any]]]:
    """Resolve every preregistered width to a chain, recording skips honestly."""
    gate_errors, readout_errors = calibration_from_target(target)
    longest = longest_error_aware_chain(gate_errors, readout_errors)
    if longest is None:
        raise ValueError("target reports no usable calibrated chain at all")
    resolved: list[tuple[int, ChainSelection]] = []
    skipped: list[dict[str, Any]] = []
    widths: list[int] = []
    for width in PREREGISTERED_WIDTHS:
        if width < longest.length:
            widths.append(width)
        else:
            skipped.append(
                {
                    "width": width,
                    "reason": (
                        f"exceeds or equals the device-max usable chain "
                        f"({longest.length} qubits); covered by the device-max point"
                        if width == longest.length
                        else f"exceeds the device-max usable chain ({longest.length} qubits)"
                    ),
                }
            )
    for width in widths:
        chain = select_error_aware_chain(gate_errors, readout_errors, width)
        if chain is None:
            skipped.append({"width": width, "reason": "no chain of this width is reachable"})
            continue
        if chain.median_edge_error > median_edge_error_abort:
            skipped.append(
                {
                    "width": width,
                    "reason": (
                        f"median chain edge error {chain.median_edge_error:.5f} exceeds "
                        f"the abort threshold {median_edge_error_abort:.5f}"
                    ),
                }
            )
            continue
        resolved.append((width, chain))
    if longest.median_edge_error > median_edge_error_abort:
        skipped.append(
            {
                "width": longest.length,
                "reason": (
                    f"device-max chain median edge error {longest.median_edge_error:.5f} "
                    f"exceeds the abort threshold {median_edge_error_abort:.5f}"
                ),
            }
        )
    else:
        resolved.append((longest.length, longest))
    return resolved, skipped


def build_readiness(
    *,
    backend_name: str,
    plans: Sequence[WidthPlan],
    skipped: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Assemble the pre-submission readiness document with abort checks."""
    n_circuits = sum(len(plan.circuits) for plan in plans)
    estimated = estimate_qpu_seconds(n_circuits)
    budget_ok = estimated <= args.max_qpu_seconds
    any_width = bool(plans)
    return {
        "schema": f"{SCHEMA_PREFIX}_readiness_v1",
        "timestamp_utc": _timestamp(),
        "experiment_id": EXPERIMENT_ID,
        "preregistration": PREREGISTRATION_PATH,
        "backend": backend_name,
        "hardware_submission": False,
        "status": "ready_for_submission" if budget_ok and any_width else "blocked",
        "widths": [
            {
                "width": plan.width,
                "chain_qubits": list(plan.chain.qubits),
                "median_edge_error": plan.chain.median_edge_error,
                "worst_readout_error": max(plan.chain.readout_errors),
                "labels": list(plan.labels),
                "transpiled_depths": list(plan.isa_depths),
            }
            for plan in plans
        ],
        "skipped_widths": [dict(entry) for entry in skipped],
        "shots": args.shots,
        "calibration_shots": args.cal_shots,
        "n_circuits": n_circuits,
        "median_edge_error_abort": args.median_edge_error_abort,
        "estimated_qpu_seconds": estimated,
        "max_qpu_seconds": args.max_qpu_seconds,
        "seed_transpiler": SEED_TRANSPILER,
        "claim_boundary": (
            "workload-width engineering and noise characterisation only; the "
            "circuits are classically simulable by construction and no "
            "advantage claim is available"
        ),
        "reasons": [
            (
                "at least one preregistered width is executable"
                if any_width
                else "no width survived the calibration abort checks"
            ),
            (
                "estimate within the approved QPU-second cap"
                if budget_ok
                else "estimate exceeds the approved QPU-second cap"
            ),
        ],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--instance")
    parser.add_argument("--credentials-vault", type=Path)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--cal-shots", type=int, default=DEFAULT_CAL_SHOTS)
    parser.add_argument("--max-qpu-seconds", type=float, default=DEFAULT_MAX_QPU_SECONDS)
    parser.add_argument(
        "--median-edge-error-abort", type=float, default=DEFAULT_MEDIAN_EDGE_ERROR_ABORT
    )
    parser.add_argument("--timeout-s", type=float, default=7200.0)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-budget", action="store_true")
    parser.add_argument("--approval-id", default="max_width_kuramoto_xy_2026-07-16")
    parser.add_argument("--approver", default="Miroslav Sotek")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point."""
    args = _parse_args(argv)
    if args.submit and not args.confirm_budget:
        print("ERROR: --submit requires --confirm-budget", file=sys.stderr)
        return 2

    from scripts.prepare_s1_ibm_live_readiness import (
        DEFAULT_CREDENTIALS_VAULT,
        load_authenticated_backend,
    )

    vault = args.credentials_vault or DEFAULT_CREDENTIALS_VAULT
    backend = load_authenticated_backend(args.backend, args.instance, vault)
    resolved, skipped = plan_widths(
        backend.target, median_edge_error_abort=args.median_edge_error_abort
    )
    plans = [build_width_plan(width, chain, backend) for width, chain in resolved]

    readiness = build_readiness(backend_name=args.backend, plans=plans, skipped=skipped, args=args)
    timestamp = readiness["timestamp_utc"]
    readiness_path = args.out_dir / f"max_width_readiness_{args.backend}_{timestamp}.json"
    readiness_sha = _write_json(readiness_path, readiness)
    print(f"readiness={readiness['status']}")
    print(f"readiness_json={readiness_path}")
    print(f"readiness_sha256={readiness_sha}")
    for plan in plans:
        print(
            f"width={plan.width} median_edge_error={plan.chain.median_edge_error:.5f} "
            f"max_depth={max(plan.isa_depths)}"
        )
    for entry in skipped:
        print(f"skipped_width={entry['width']} reason={entry['reason']}")
    if readiness["status"] != "ready_for_submission":
        return 3
    if not args.submit:
        print("hardware_submission=false")
        print("Re-run with --submit --confirm-budget to submit the WIDTH-1 job.")
        return 0

    approval = HardwareApprovalRecord(
        approval_id=args.approval_id,
        approver=args.approver,
        package_hash=hash_package_manifest(readiness),
        max_qpu_seconds=args.max_qpu_seconds,
        allowed_provider="ibm_runtime",
        approved=True,
        notes="explicit command-line --submit --confirm-budget approval (owner GO)",
    )

    from qiskit_ibm_runtime import SamplerV2 as Sampler

    sampler = Sampler(mode=backend)
    pubs: list[tuple[Any, Any, int]] = []
    for plan in plans:
        for label, circuit in zip(plan.labels, plan.circuits, strict=True):
            shots = args.cal_shots if label.startswith("cal") else args.shots
            pubs.append((circuit, None, shots))
    job = sampler.run(pubs)
    job_id = strict_provider_job_id(job.job_id())
    print(f"job_id={job_id}")
    result = job.result(timeout=args.timeout_s)

    width_rows: list[dict[str, Any]] = []
    raw_results: list[dict[str, Any]] = []
    pub_index = 0
    for plan in plans:
        counts_by_label: dict[str, dict[str, int]] = {}
        for label in plan.labels:
            expected = args.cal_shots if label.startswith("cal") else args.shots
            counts_by_label[label] = coerce_counts(
                _pub_counts(result[pub_index]),
                width=plan.width,
                expected_shots=expected,
                field_name=f"width {plan.width} {label} counts",
            )
            raw_results.append(
                {
                    "pub_index": pub_index,
                    "width": plan.width,
                    "label": label,
                    "counts": counts_by_label[label],
                }
            )
            pub_index += 1
        for reps in TROTTER_REPS:
            width_rows.append(
                analyse_width(
                    plan.width,
                    reps,
                    counts_by_label[f"main_reps{reps}_x"],
                    counts_by_label[f"main_reps{reps}_y"],
                    counts_by_label["cal0"],
                    counts_by_label["cal1"],
                    include_baseline=not args.skip_baseline,
                )
            )

    raw_package = {
        "schema": f"{SCHEMA_PREFIX}_raw_counts_v1",
        "experiment_id": EXPERIMENT_ID,
        "preregistration": PREREGISTRATION_PATH,
        "backend": args.backend,
        "job_id": job_id,
        "timestamp_utc": timestamp,
        "width_layouts": [
            {"width": plan.width, "chain_qubits": list(plan.chain.qubits)} for plan in plans
        ],
        "skipped_widths": [dict(entry) for entry in skipped],
        "shots": args.shots,
        "calibration_shots": args.cal_shots,
        "seed_transpiler": SEED_TRANSPILER,
        "results": raw_results,
        "approval": {
            "approval_id": approval.approval_id,
            "approver": approval.approver,
            "package_hash": approval.package_hash,
            "max_qpu_seconds": approval.max_qpu_seconds,
            "allowed_provider": approval.allowed_provider,
        },
    }
    raw_path = args.out_dir / f"max_width_raw_counts_{args.backend}_{timestamp}.json"
    raw_sha = _write_json(raw_path, raw_package)

    analysis = {
        "schema": f"{SCHEMA_PREFIX}_analysis_v1",
        "experiment_id": EXPERIMENT_ID,
        "preregistration": PREREGISTRATION_PATH,
        "baseline": (
            "exact matrix-product-state simulation of the identical circuits"
            if not args.skip_baseline
            else "baseline deferred (--skip-baseline)"
        ),
        "width_rows": width_rows,
        "claim_boundary": readiness["claim_boundary"],
    }
    analysis_path = args.out_dir / f"max_width_analysis_{args.backend}_{timestamp}.json"
    analysis_sha = _write_json(analysis_path, analysis)

    print("hardware_submission=true")
    print(f"raw_counts_json={raw_path}")
    print(f"raw_counts_sha256={raw_sha}")
    print(f"analysis_json={analysis_path}")
    print(f"analysis_sha256={analysis_sha}")
    for row in width_rows:
        baseline_note = f" R_exact={row['r_exact_mps']:.4f}" if "r_exact_mps" in row else ""
        print(
            f"width={row['width']} reps={row['trotter_reps']} "
            f"R_raw={row['r_unmitigated']:.4f} R_mit={row['r_mitigated']:.4f}"
            f"{baseline_note}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
