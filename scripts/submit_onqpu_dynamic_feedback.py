#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — on-QPU dynamic-circuit feedback submission (RC-1)
"""Approval-gated on-QPU dynamic-circuit feedback demo submitter (RC-1).

Executes the campaign preregistered in
``docs/campaigns/onqpu_dynamic_feedback_prereg_2026-07-16.md``: the
repository's monitored Kuramoto feedback template
(``build_monitored_feedback_circuit`` — per-round Trotter evolution,
monitor-ancilla interaction, mid-circuit monitor measurement, conditional
ancilla reset, conditional system correction) against its circuit-family
matched open-loop control (``build_open_loop_feedback_control_circuit``) at
``n_rounds ∈ {2, 3}``, submitted as vendor-executed dynamic circuits.

Honest latency labelling (preregistered): controller decisions execute
inside the vendor runtime. This is NOT external-FPGA-in-the-loop control and
no latency claim of any kind is derived from queue or backend wall time. A
backend rejection of the conditional blocks, or a feedback-on result
indistinguishable from feedback-off, is a publishable outcome of equal
standing.

The QPU-seconds estimate uses the empirical anchor from the 2026-05-05 B-C
scaling run (≈ 1.09 s per circuit) with an explicit ×3 dynamic-circuit
multiplier for the mid-circuit measurement and conditional blocks.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scpn_quantum_control.control.realtime_feedback import (  # noqa: E402
    RealtimeFeedbackConfig,
    build_monitored_feedback_circuit,
    build_open_loop_feedback_control_circuit,
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
from scpn_quantum_control.hardware.feedback_hardware_scheduler import (  # noqa: E402
    HardwareApprovalRecord,
    hash_package_manifest,
)

EXPERIMENT_ID = "onqpu_dynamic_feedback_2026-07-16"
PREREGISTRATION_PATH = "docs/campaigns/onqpu_dynamic_feedback_prereg_2026-07-16.md"
SCHEMA_PREFIX = "scpn_onqpu_dynamic_feedback"

SYSTEM_QUBITS = 3
TOTAL_QUBITS = SYSTEM_QUBITS + 1
ROUND_SETTINGS = (2, 3)
ARMS = ("feedback_on", "feedback_off")

#: The S1 template coupling and natural frequencies (the tested shape).
K_COUPLING = np.array([[0.0, 0.35, 0.20], [0.35, 0.0, 0.25], [0.20, 0.25, 0.0]], dtype=np.float64)
OMEGA_NATURAL = np.array([0.1, 0.4, 0.7], dtype=np.float64)

DEFAULT_BACKEND = "ibm_fez"
DEFAULT_SHOTS = 4096
DEFAULT_CAL_SHOTS = 8192
DEFAULT_MAX_QPU_SECONDS = 120.0
DEFAULT_MAX_DEPTH_RATIO = 2.0
ESTIMATED_SECONDS_PER_CIRCUIT = 1.1
DYNAMIC_CIRCUIT_COST_MULTIPLIER = 3.0
SEED_TRANSPILER = 20260716
DATA_DIR = REPO_ROOT / "data" / "onqpu_dynamic_feedback"

REQUIRED_DYNAMIC_OPERATIONS = ("if_else", "measure", "reset")


def arm_circuit(arm: str, n_rounds: int) -> QuantumCircuit:
    """Build one campaign arm on the S1 template shape."""
    config = RealtimeFeedbackConfig()
    if arm == "feedback_on":
        circuit = build_monitored_feedback_circuit(
            K_COUPLING, OMEGA_NATURAL, config=config, n_rounds=n_rounds
        )
    elif arm == "feedback_off":
        circuit = build_open_loop_feedback_control_circuit(
            K_COUPLING, OMEGA_NATURAL, config=config, n_rounds=n_rounds
        )
    else:
        raise ValueError(f"unknown campaign arm {arm!r}")
    circuit.name = f"rc1_{arm}_rounds{n_rounds}"
    return circuit


def calibration_circuits() -> tuple[QuantumCircuit, QuantumCircuit]:
    """Context readout calibration on the four campaign qubits."""
    zeros = QuantumCircuit(TOTAL_QUBITS, name="rc1_cal0")
    zeros.measure_all()
    ones = QuantumCircuit(TOTAL_QUBITS, name="rc1_cal1")
    ones.x(range(TOTAL_QUBITS))
    ones.measure_all()
    return zeros, ones


def dynamic_support_missing(target: Any) -> list[str]:
    """Names of required dynamic-circuit operations the target lacks."""
    supported = set(target.operation_names)
    return [name for name in REQUIRED_DYNAMIC_OPERATIONS if name not in supported]


@dataclass(frozen=True)
class StarSelection:
    """A monitor qubit with its three system neighbours."""

    monitor: int
    system: tuple[int, int, int]
    edge_errors: tuple[float, float, float]
    readout_errors: tuple[float, float, float, float]

    @property
    def initial_layout(self) -> list[int]:
        """Physical layout ordered as the template registers (sys then monitor)."""
        return [*self.system, self.monitor]


def select_star_layout(target: Any) -> StarSelection:
    """Choose a monitor qubit with the three best-calibrated neighbours.

    The monitored-feedback template couples the ancilla to every system
    qubit, so the layout needs a degree-3 star; the winning centre minimises
    the summed edge score (gate error + mean endpoint readout error).
    """
    gate_errors, readout_errors = calibration_from_target(target)
    adjacency: dict[int, list[tuple[float, int]]] = {}
    for (first, second), gate_error in gate_errors.items():
        if first not in readout_errors or second not in readout_errors:
            continue
        score = gate_error + (readout_errors[first] + readout_errors[second]) / 2.0
        adjacency.setdefault(first, []).append((score, second))
        adjacency.setdefault(second, []).append((score, first))
    best: tuple[float, int, list[tuple[float, int]]] | None = None
    for centre, neighbours in adjacency.items():
        if len(neighbours) < SYSTEM_QUBITS:
            continue
        chosen = sorted(neighbours)[:SYSTEM_QUBITS]
        total = sum(score for score, _ in chosen)
        if best is None or total < best[0]:
            best = (total, centre, chosen)
    if best is None:
        raise ValueError("target has no qubit with three calibrated neighbours")
    _, centre, chosen = best
    system = tuple(qubit for _, qubit in chosen)
    edge_scores = tuple(gate_errors[(min(centre, qubit), max(centre, qubit))] for qubit in system)
    return StarSelection(
        monitor=centre,
        system=(system[0], system[1], system[2]),
        edge_errors=(edge_scores[0], edge_scores[1], edge_scores[2]),
        readout_errors=(
            readout_errors[system[0]],
            readout_errors[system[1]],
            readout_errors[system[2]],
            readout_errors[centre],
        ),
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


def total_variation_distance(
    counts_a: Mapping[str, int],
    counts_b: Mapping[str, int],
) -> dict[str, Any]:
    """TVD between two count tables with a multinomial error estimate.

    σ is the propagated multinomial standard error
    ``sqrt(Σ_b [p_a(1−p_a)/N_a + p_b(1−p_b)/N_b]) / 2`` — an approximation
    that treats bins as independent, stated as such in the output.
    """
    shots_a = sum(counts_a.values())
    shots_b = sum(counts_b.values())
    if shots_a <= 0 or shots_b <= 0:
        raise ValueError("both count tables need at least one shot")
    tvd = 0.0
    variance = 0.0
    per_bin: list[dict[str, Any]] = []
    for bitstring in sorted(set(counts_a) | set(counts_b)):
        p_a = counts_a.get(bitstring, 0) / shots_a
        p_b = counts_b.get(bitstring, 0) / shots_b
        delta = p_a - p_b
        sigma = math.sqrt(p_a * (1.0 - p_a) / shots_a + p_b * (1.0 - p_b) / shots_b)
        tvd += abs(delta) / 2.0
        variance += sigma * sigma
        per_bin.append(
            {
                "bitstring": bitstring,
                "p_feedback_on": p_a,
                "p_feedback_off": p_b,
                "delta": delta,
                "sigma": sigma,
                "z_score": delta / sigma if sigma > 0.0 else 0.0,
            }
        )
    return {
        "tvd": tvd,
        "tvd_sigma_multinomial_approximation": math.sqrt(variance) / 2.0,
        "sigma_note": "independent-bin multinomial approximation",
        "per_bin": per_bin,
    }


def monitor_trigger_rates(monitor_counts: Mapping[str, int], n_rounds: int) -> list[float]:
    """Per-round P(monitor = 1) marginals from the monitor-bit register."""
    shots = sum(monitor_counts.values())
    if shots <= 0:
        raise ValueError("monitor count table needs at least one shot")
    rates = []
    for round_index in range(n_rounds):
        triggered = sum(
            count
            for bitstring, count in monitor_counts.items()
            if bitstring[-1 - round_index] == "1"
        )
        rates.append(triggered / shots)
    return rates


def estimate_qpu_seconds(n_main: int, n_calibration: int) -> float:
    """Empirical anchor with the dynamic-circuit multiplier on main arms."""
    return (
        n_main * ESTIMATED_SECONDS_PER_CIRCUIT * DYNAMIC_CIRCUIT_COST_MULTIPLIER
        + n_calibration * ESTIMATED_SECONDS_PER_CIRCUIT
    )


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


def _register_counts(pub_result: Any, register_name: str) -> dict[str, Any]:
    """Extract raw counts from one named classical register of a pub result."""
    register = getattr(pub_result.data, register_name, None)
    if register is None or not hasattr(register, "get_counts"):
        raise RuntimeError(f"pub result carries no classical register {register_name!r}")
    counts: dict[str, Any] = register.get_counts()
    return counts


def build_readiness(
    *,
    backend_name: str,
    star: StarSelection,
    missing_operations: Sequence[str],
    logical_depths: Mapping[str, int],
    isa_depths: Mapping[str, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Assemble the pre-submission readiness document with abort checks."""
    n_main = len(ARMS) * len(ROUND_SETTINGS)
    estimated = estimate_qpu_seconds(n_main, 2)
    dynamic_ok = not missing_operations
    ratio_by_label = {
        label: (isa_depths[label] / logical_depths[label] if logical_depths[label] else math.inf)
        for label in logical_depths
    }
    depth_ok = all(ratio <= args.max_depth_ratio for ratio in ratio_by_label.values())
    budget_ok = estimated <= args.max_qpu_seconds
    return {
        "schema": f"{SCHEMA_PREFIX}_readiness_v1",
        "timestamp_utc": _timestamp(),
        "experiment_id": EXPERIMENT_ID,
        "preregistration": PREREGISTRATION_PATH,
        "backend": backend_name,
        "hardware_submission": False,
        "status": ("ready_for_submission" if dynamic_ok and depth_ok and budget_ok else "blocked"),
        "layout": {
            "system_qubits": list(star.system),
            "monitor_qubit": star.monitor,
            "edge_errors": list(star.edge_errors),
            "readout_errors": list(star.readout_errors),
        },
        "dynamic_operations_missing": list(missing_operations),
        "logical_depths": dict(logical_depths),
        "transpiled_depths": dict(isa_depths),
        "depth_ratios": ratio_by_label,
        "max_depth_ratio": args.max_depth_ratio,
        "shots": args.shots,
        "calibration_shots": args.cal_shots,
        "estimated_qpu_seconds": estimated,
        "max_qpu_seconds": args.max_qpu_seconds,
        "seed_transpiler": SEED_TRANSPILER,
        "claim_boundary": (
            "on-QPU vendor-executed feedback evidence only; no external-FPGA "
            "control, no latency numbers from queue or backend wall time, no "
            "synchronisation-protection claim beyond the executed observable"
        ),
        "reasons": [
            (
                "backend reports every required dynamic-circuit operation"
                if dynamic_ok
                else "backend lacks required dynamic-circuit operations"
            ),
            (
                "every arm satisfies the transpiled-depth ratio ceiling"
                if depth_ok
                else "an arm exceeds the transpiled-depth ratio ceiling"
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
    parser.add_argument("--max-depth-ratio", type=float, default=DEFAULT_MAX_DEPTH_RATIO)
    parser.add_argument("--timeout-s", type=float, default=7200.0)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-budget", action="store_true")
    parser.add_argument("--approval-id", default="onqpu_dynamic_feedback_2026-07-16")
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
    missing_operations = dynamic_support_missing(backend.target)
    star = select_star_layout(backend.target)

    labels: list[str] = []
    circuits: list[QuantumCircuit] = []
    logical_depths: dict[str, int] = {}
    for n_rounds in ROUND_SETTINGS:
        for arm in ARMS:
            circuit = arm_circuit(arm, n_rounds)
            labels.append(circuit.name)
            circuits.append(circuit)
    zeros, ones = calibration_circuits()
    labels.extend([zeros.name, ones.name])
    circuits.extend([zeros, ones])
    # The depth-ratio reference is the SAME circuit translated to the
    # backend's own basis but WITHOUT coupling-map routing, so the ratio
    # measures routing overhead alone. Comparing routed ECR-basis depth to
    # the untranslated template depth would conflate basis translation
    # (an inherent ×4-8 for this template) with the pathological routing
    # blow-up the preregistration actually gates on.
    unrouted = transpile(
        circuits,
        basis_gates=list(backend.target.operation_names),
        optimization_level=0,
    )
    for label, circuit in zip(labels, unrouted, strict=True):
        logical_depths[label] = int(circuit.depth())

    isa_circuits = transpile(
        circuits,
        backend=backend,
        initial_layout=star.initial_layout,
        optimization_level=1,
        seed_transpiler=SEED_TRANSPILER,
    )
    isa_depths = {
        label: int(circuit.depth()) for label, circuit in zip(labels, isa_circuits, strict=True)
    }

    readiness = build_readiness(
        backend_name=args.backend,
        star=star,
        missing_operations=missing_operations,
        logical_depths=logical_depths,
        isa_depths=isa_depths,
        args=args,
    )
    timestamp = readiness["timestamp_utc"]
    readiness_path = args.out_dir / f"rc1_readiness_{args.backend}_{timestamp}.json"
    readiness_sha = _write_json(readiness_path, readiness)
    print(f"readiness={readiness['status']}")
    print(f"readiness_json={readiness_path}")
    print(f"readiness_sha256={readiness_sha}")
    if readiness["status"] != "ready_for_submission":
        return 3
    if not args.submit:
        print("hardware_submission=false")
        print("Re-run with --submit --confirm-budget to submit the RC-1 job.")
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
    n_main = len(ARMS) * len(ROUND_SETTINGS)
    pubs = [(circuit, None, args.shots) for circuit in isa_circuits[:n_main]]
    pubs += [(circuit, None, args.cal_shots) for circuit in isa_circuits[n_main:]]
    job = sampler.run(pubs)
    job_id = strict_provider_job_id(job.job_id())
    print(f"job_id={job_id}")
    result = job.result(timeout=args.timeout_s)

    raw_results: list[dict[str, Any]] = []
    readout_by_label: dict[str, dict[str, int]] = {}
    monitor_by_label: dict[str, dict[str, int]] = {}
    for pub_index, label in enumerate(labels[:n_main]):
        n_rounds = int(label.rsplit("rounds", 1)[1])
        readout = coerce_counts(
            _register_counts(result[pub_index], "readout"),
            width=SYSTEM_QUBITS,
            expected_shots=args.shots,
            field_name=f"{label} readout counts",
        )
        monitor = coerce_counts(
            _register_counts(result[pub_index], "monitor_bit"),
            width=n_rounds,
            expected_shots=args.shots,
            field_name=f"{label} monitor counts",
        )
        readout_by_label[label] = readout
        monitor_by_label[label] = monitor
        raw_results.append(
            {
                "pub_index": pub_index,
                "label": label,
                "readout_counts": readout,
                "monitor_counts": monitor,
            }
        )
    calibration_counts = {}
    for offset, label in enumerate(labels[n_main:]):
        counts = coerce_counts(
            _register_counts(result[n_main + offset], "meas"),
            width=TOTAL_QUBITS,
            expected_shots=args.cal_shots,
            field_name=f"{label} counts",
        )
        calibration_counts[label] = counts
        raw_results.append({"pub_index": n_main + offset, "label": label, "counts": counts})

    raw_package = {
        "schema": f"{SCHEMA_PREFIX}_raw_counts_v1",
        "experiment_id": EXPERIMENT_ID,
        "preregistration": PREREGISTRATION_PATH,
        "backend": args.backend,
        "job_id": job_id,
        "timestamp_utc": timestamp,
        "layout": dict(readiness["layout"]),
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
    raw_path = args.out_dir / f"rc1_raw_counts_{args.backend}_{timestamp}.json"
    raw_sha = _write_json(raw_path, raw_package)

    comparisons: list[dict[str, Any]] = []
    for n_rounds in ROUND_SETTINGS:
        on_label = f"rc1_feedback_on_rounds{n_rounds}"
        off_label = f"rc1_feedback_off_rounds{n_rounds}"
        comparison = total_variation_distance(
            readout_by_label[on_label], readout_by_label[off_label]
        )
        comparisons.append(
            {
                "n_rounds": n_rounds,
                "comparison": comparison,
                "monitor_trigger_rates": {
                    "feedback_on": monitor_trigger_rates(monitor_by_label[on_label], n_rounds),
                    "feedback_off": monitor_trigger_rates(monitor_by_label[off_label], n_rounds),
                },
            }
        )
    analysis = {
        "schema": f"{SCHEMA_PREFIX}_analysis_v1",
        "experiment_id": EXPERIMENT_ID,
        "preregistration": PREREGISTRATION_PATH,
        "round_settings": comparisons,
        "claim_boundary": readiness["claim_boundary"],
    }
    analysis_path = args.out_dir / f"rc1_analysis_{args.backend}_{timestamp}.json"
    analysis_sha = _write_json(analysis_path, analysis)

    print("hardware_submission=true")
    print(f"raw_counts_json={raw_path}")
    print(f"raw_counts_sha256={raw_sha}")
    print(f"analysis_json={analysis_path}")
    print(f"analysis_sha256={analysis_sha}")
    for row in comparisons:
        comparison = row["comparison"]
        print(
            f"rounds={row['n_rounds']} tvd={comparison['tvd']:.4f}"
            f"±{comparison['tvd_sigma_multinomial_approximation']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
