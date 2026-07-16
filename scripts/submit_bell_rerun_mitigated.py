#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — mitigated Bell re-run submission (KIMI-9)
"""Approval-gated mitigated Bell re-run submitter (KIMI-9 preregistration).

Executes the campaign preregistered in
``docs/campaigns/bell_rerun_mitigated_prereg_2026-07-16.md``: two independent
Bell pairs on four qubits, four CHSH analyser settings as separate committed
transpiled circuits, and a full-basis readout calibration set, submitted as a
single SamplerV2 job. Analysis applies the exact conventions of
``scripts/recompute_chsh_bell_test.py`` (little-endian pairs, minus sign on
setting 1, multinomial σ) to the new counts, with readout-matrix mitigation
reported on and off.

The QPU-seconds estimate uses the empirical anchor from the 2026-05-05 B-C
scaling run: 280 circuits cost 305 IBM usage seconds ≈ 1.09 s per circuit
(ledger row ``ibm-run-1f46ebd0da8912ff``).

Deviation note (recorded here and in every emitted document): the
preregistration names per-qubit readout-matrix circuits; this runner uses the
repository's exact FULL-BASIS confusion matrix (16 prepared states for 4
qubits, ``mitigation/readout_matrix.py``) because that committed module makes
the stronger correction with no tensored-independence assumption.
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

from qiskit import QuantumCircuit, transpile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scpn_quantum_control.hardware._count_integrity import (  # noqa: E402
    strict_fixed_width_bitstring_key,
    strict_non_negative_count,
    strict_provider_job_id,
    strict_shot_conservation,
)
from scpn_quantum_control.hardware.feedback_hardware_scheduler import (  # noqa: E402
    HardwareApprovalRecord,
    hash_package_manifest,
)
from scpn_quantum_control.mitigation.readout_matrix import (  # noqa: E402
    ReadoutConfusionMatrix,
    build_readout_confusion_matrix,
    computational_basis_labels,
    mitigate_counts,
)
from scripts.recompute_chsh_bell_test import (  # noqa: E402
    QUBIT_PAIRS,
    SETTING_SIGNS,
    chsh_for_pair,
)

EXPERIMENT_ID = "bell_rerun_mitigated_2026-07-16"
PREREGISTRATION_PATH = "docs/campaigns/bell_rerun_mitigated_prereg_2026-07-16.md"
PARENT_ARTIFACT = "results/ibm_hardware_2026-03-28/bell_test_4q.json"
SCHEMA_PREFIX = "scpn_bell_rerun_mitigated"

N_QUBITS = 4

#: CHSH analyser angles in the X–Z measurement plane (radians). For a
#: |Φ+⟩ pair the correlator is E = cos(a − b), so this setting order gives
#: the ideal values (+1/√2, −1/√2, +1/√2, +1/√2) and, with the recompute
#: convention S = E₀ − E₁ + E₂ + E₃, the ideal S = 2√2.
ANALYSER_A_ANGLES = (0.0, math.pi / 2.0)
ANALYSER_B_ANGLES = (math.pi / 4.0, 3.0 * math.pi / 4.0)
SETTING_ANGLES: tuple[tuple[float, float], ...] = (
    (ANALYSER_A_ANGLES[0], ANALYSER_B_ANGLES[0]),
    (ANALYSER_A_ANGLES[0], ANALYSER_B_ANGLES[1]),
    (ANALYSER_A_ANGLES[1], ANALYSER_B_ANGLES[0]),
    (ANALYSER_A_ANGLES[1], ANALYSER_B_ANGLES[1]),
)

DEFAULT_BACKEND = "ibm_fez"
DEFAULT_SHOTS = 4096
DEFAULT_CAL_SHOTS = 8192
DEFAULT_MAX_QPU_SECONDS = 60.0
DEFAULT_MAX_DEPTH = 60
DEFAULT_READOUT_ERROR_ABORT = 0.15
ESTIMATED_SECONDS_PER_CIRCUIT = 1.1
SEED_TRANSPILER = 20260716
DATA_DIR = REPO_ROOT / "data" / "bell_rerun_mitigated"


def chsh_setting_circuits() -> tuple[QuantumCircuit, ...]:
    """Build the four per-setting CHSH circuits on two Bell pairs.

    Logical layout: pair (0, 1) and pair (2, 3), each prepared in |Φ+⟩.
    Even-index qubits carry the A analyser, odd-index qubits the B analyser.
    Measuring after ``ry(-θ)`` measures the spin along the X–Z-plane
    direction θ.
    """
    circuits: list[QuantumCircuit] = []
    for setting_index, (angle_a, angle_b) in enumerate(SETTING_ANGLES):
        circuit = QuantumCircuit(N_QUBITS, name=f"bell_rerun_s{setting_index}")
        for pair in QUBIT_PAIRS:
            circuit.h(pair[0])
            circuit.cx(pair[0], pair[1])
        circuit.barrier()
        for pair in QUBIT_PAIRS:
            circuit.ry(-angle_a, pair[0])
            circuit.ry(-angle_b, pair[1])
        circuit.measure_all()
        circuits.append(circuit)
    return tuple(circuits)


def calibration_circuits() -> tuple[QuantumCircuit, ...]:
    """Build the 16 full-basis readout-calibration circuits.

    Labels follow ``computational_basis_labels``; bit ``k`` of a label
    (little-endian, position ``label[-1-k]``) sets qubit ``k``.
    """
    circuits: list[QuantumCircuit] = []
    for label in computational_basis_labels(N_QUBITS):
        circuit = QuantumCircuit(N_QUBITS, name=f"bell_rerun_cal_{label}")
        for qubit in range(N_QUBITS):
            if label[-1 - qubit] == "1":
                circuit.x(qubit)
        circuit.measure_all()
        circuits.append(circuit)
    return tuple(circuits)


@dataclass(frozen=True)
class LayoutSelection:
    """Physical-qubit layout chosen from fresh backend calibration data."""

    physical_qubits: tuple[int, int, int, int]
    edge_errors: tuple[float, float]
    readout_errors: tuple[float, float, float, float]

    @property
    def worst_readout_error(self) -> float:
        """Largest per-qubit readout assignment error in the layout."""
        return max(self.readout_errors)


def _two_qubit_edge_errors(target: Any) -> dict[tuple[int, int], float]:
    """Collect the best reported error for every calibrated two-qubit edge."""
    edges: dict[tuple[int, int], float] = {}
    for op_name in target.operation_names:
        try:
            props_map = target[op_name]
        except KeyError:  # pragma: no cover - defensive against target quirks
            continue
        if props_map is None:
            continue
        for qargs, props in props_map.items():
            if qargs is None or len(qargs) != 2:
                continue
            error = getattr(props, "error", None)
            if error is None:
                continue
            edge = (min(qargs), max(qargs))
            known = edges.get(edge)
            if known is None or float(error) < known:
                edges[edge] = float(error)
    return edges


def _readout_errors(target: Any) -> dict[int, float]:
    """Collect per-qubit readout assignment errors from the target."""
    errors: dict[int, float] = {}
    props_map = target["measure"]
    for qargs, props in props_map.items():
        if qargs is None or len(qargs) != 1:
            continue
        error = getattr(props, "error", None)
        if error is None:
            continue
        errors[int(qargs[0])] = float(error)
    return errors


def select_physical_qubits(target: Any) -> LayoutSelection:
    """Choose two disjoint coupled pairs minimising gate plus readout error.

    The score of an edge is its two-qubit gate error plus the mean readout
    error of its qubits; the two best disjoint edges become the layout, with
    logical pair (0, 1) on the better edge.
    """
    edge_errors = _two_qubit_edge_errors(target)
    readout = _readout_errors(target)
    scored: list[tuple[float, tuple[int, int]]] = []
    for edge, gate_error in edge_errors.items():
        if edge[0] not in readout or edge[1] not in readout:
            continue
        score = gate_error + (readout[edge[0]] + readout[edge[1]]) / 2.0
        scored.append((score, edge))
    scored.sort()
    if not scored:
        raise ValueError("target reports no fully calibrated two-qubit edges")
    _, first = scored[0]
    second: tuple[int, int] | None = None
    second_error = math.nan
    for _score, edge in scored[1:]:
        if set(edge).isdisjoint(first):
            second = edge
            second_error = edge_errors[edge]
            break
    if second is None:
        raise ValueError("target has no two disjoint calibrated two-qubit edges")
    physical = (first[0], first[1], second[0], second[1])
    return LayoutSelection(
        physical_qubits=physical,
        edge_errors=(edge_errors[first], second_error),
        readout_errors=(
            readout[physical[0]],
            readout[physical[1]],
            readout[physical[2]],
            readout[physical[3]],
        ),
    )


def coerce_counts(
    raw_counts: Mapping[str, object],
    *,
    expected_shots: int,
    field_name: str,
) -> dict[str, int]:
    """Fail-closed coercion of provider counts to exact-width validated counts."""
    coerced: dict[str, int] = {}
    for key, value in raw_counts.items():
        bitstring = strict_fixed_width_bitstring_key(
            key, width=N_QUBITS, field_name=f"{field_name} key"
        )
        count = strict_non_negative_count(value, field_name=f"{field_name} value")
        if bitstring in coerced:
            raise ValueError(f"{field_name} carries duplicate bitstring {bitstring!r}")
        coerced[bitstring] = count
    strict_shot_conservation(coerced, expected_shots=expected_shots, field_name=field_name)
    return coerced


def probability_pair_correlator(
    probabilities: Sequence[float],
    labels: Sequence[str],
    pair: tuple[int, int],
) -> float:
    """Correlator E from a (quasi-)probability vector over basis labels.

    Same parity convention as ``pair_correlator``: +1 when the little-endian
    bits of the pair agree, −1 otherwise. Quasi-probabilities from readout
    inversion may be negative; they enter the sum unclipped.
    """
    correlator = 0.0
    for probability, label in zip(probabilities, labels, strict=True):
        same = label[-1 - pair[0]] == label[-1 - pair[1]]
        correlator += float(probability) * (1.0 if same else -1.0)
    return correlator


def mitigated_pair_statistics(
    setting_counts: Sequence[Mapping[str, int]],
    confusion_matrix: ReadoutConfusionMatrix,
    pair: tuple[int, int],
    *,
    shots: int,
) -> dict[str, Any]:
    """CHSH statistics for one pair from readout-mitigated distributions.

    σ uses the same multinomial formula as the unmitigated path at the raw
    shot count; the inversion's own error is NOT propagated, and the value is
    labelled accordingly.
    """
    settings_e: list[float] = []
    s_value = 0.0
    variance = 0.0
    for sign, counts in zip(SETTING_SIGNS, setting_counts, strict=True):
        probabilities = mitigate_counts(counts, confusion_matrix).tolist()
        e_value = probability_pair_correlator(probabilities, confusion_matrix.labels, pair)
        settings_e.append(e_value)
        s_value += sign * e_value
        variance += (1.0 - e_value * e_value) / shots
    return {
        "label": f"q{pair[0]}q{pair[1]}",
        "settings_e": settings_e,
        "s_value": s_value,
        "sigma_multinomial_approximation": math.sqrt(max(variance, 0.0)),
        "sigma_note": (
            "multinomial formula at the raw shot count; readout-inversion error not propagated"
        ),
    }


def setting1_band_decision(settings_e: Sequence[float], *, shots: int) -> dict[str, Any]:
    """Evaluate the preregistered decision rule on mitigated correlators.

    The anomaly is explained as a readout/transpilation artefact when the
    mitigated setting-1 correlator magnitude reaches the band spanned by the
    other settings' magnitudes to within 2σ of its own multinomial error.
    """
    abs_e1 = abs(float(settings_e[1]))
    others = [abs(float(settings_e[index])) for index in (0, 2, 3)]
    sigma_e1 = math.sqrt(max(1.0 - abs_e1 * abs_e1, 0.0) / shots)
    band_low = min(others)
    within = abs_e1 >= band_low - 2.0 * sigma_e1
    return {
        "abs_setting1_e": abs_e1,
        "other_settings_abs_e_band": [band_low, max(others)],
        "sigma_setting1": sigma_e1,
        "within_2_sigma_of_band": within,
        "preregistered_reading": (
            "anomaly explained as a readout/transpilation artefact"
            if within
            else "asymmetry persists; documented as a property of the executed setting"
        ),
    }


def build_analysis(
    setting_counts: Sequence[Mapping[str, int]],
    calibration_counts: Mapping[str, Mapping[str, int]],
    *,
    shots: int,
) -> dict[str, Any]:
    """Assemble the mitigation-on/off CHSH analysis for both pairs."""
    confusion_matrix = build_readout_confusion_matrix(calibration_counts, N_QUBITS)
    setting_entries = [{"counts": dict(counts)} for counts in setting_counts]
    pairs: list[dict[str, Any]] = []
    for pair in QUBIT_PAIRS:
        unmitigated = chsh_for_pair(setting_entries, pair)
        mitigated = mitigated_pair_statistics(setting_counts, confusion_matrix, pair, shots=shots)
        pairs.append(
            {
                "pair": list(pair),
                "unmitigated": {
                    "settings_e": list(unmitigated.settings_e),
                    "s_value": unmitigated.s_value,
                    "sigma": unmitigated.sigma,
                    "significance": unmitigated.significance,
                },
                "mitigated": mitigated,
                "decision_rule": setting1_band_decision(mitigated["settings_e"], shots=shots),
            }
        )
    return {
        "schema": f"{SCHEMA_PREFIX}_analysis_v1",
        "experiment_id": EXPERIMENT_ID,
        "preregistration": PREREGISTRATION_PATH,
        "parent_artifact": PARENT_ARTIFACT,
        "conventions": (
            "scripts/recompute_chsh_bell_test.py: little-endian pairs, "
            "S = E0 - E1 + E2 + E3, multinomial sigma"
        ),
        "pairs": pairs,
        "claim_boundary": (
            "New dated record; no loophole-free or device-independent claim, "
            "no QKD viability claim; the published March 2026 record stays "
            "as it is."
        ),
    }


def estimate_qpu_seconds(n_circuits: int) -> float:
    """Estimate IBM usage seconds from the empirical per-circuit anchor."""
    return float(n_circuits) * ESTIMATED_SECONDS_PER_CIRCUIT


def build_readiness(
    *,
    backend_name: str,
    layout: LayoutSelection,
    isa_depths: Sequence[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Assemble the pre-submission readiness document with abort checks."""
    n_circuits = len(SETTING_ANGLES) + 2**N_QUBITS
    estimated = estimate_qpu_seconds(n_circuits)
    readout_ok = layout.worst_readout_error <= args.readout_error_abort
    depth_ok = max(isa_depths) <= args.max_depth
    budget_ok = estimated <= args.max_qpu_seconds
    reasons = [
        (
            "layout readout errors within the correctable abort threshold"
            if readout_ok
            else "a layout qubit exceeds the readout-error abort threshold"
        ),
        (
            "transpiled depths within the ceiling"
            if depth_ok
            else "a transpiled circuit exceeds the depth ceiling"
        ),
        (
            "estimate within the approved QPU-second cap"
            if budget_ok
            else "estimate exceeds the approved QPU-second cap"
        ),
    ]
    return {
        "schema": f"{SCHEMA_PREFIX}_readiness_v1",
        "timestamp_utc": _timestamp(),
        "experiment_id": EXPERIMENT_ID,
        "preregistration": PREREGISTRATION_PATH,
        "backend": backend_name,
        "hardware_submission": False,
        "status": ("ready_for_submission" if readout_ok and depth_ok and budget_ok else "blocked"),
        "layout": {
            "physical_qubits": list(layout.physical_qubits),
            "edge_errors": list(layout.edge_errors),
            "readout_errors": list(layout.readout_errors),
        },
        "setting_angles": [list(angles) for angles in SETTING_ANGLES],
        "shots": args.shots,
        "calibration_shots": args.cal_shots,
        "n_circuits": n_circuits,
        "transpiled_depths": list(isa_depths),
        "max_depth": args.max_depth,
        "readout_error_abort": args.readout_error_abort,
        "estimated_qpu_seconds": estimated,
        "max_qpu_seconds": args.max_qpu_seconds,
        "seed_transpiler": SEED_TRANSPILER,
        "mitigation_deviation_note": (
            "full-basis 16-state readout calibration used instead of the "
            "preregistered per-qubit circuits: the committed module "
            "(mitigation/readout_matrix.py) makes the stronger exact "
            "correction with no tensored-independence assumption"
        ),
        "reasons": reasons,
    }


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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--instance")
    parser.add_argument("--credentials-vault", type=Path)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--cal-shots", type=int, default=DEFAULT_CAL_SHOTS)
    parser.add_argument("--max-qpu-seconds", type=float, default=DEFAULT_MAX_QPU_SECONDS)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--readout-error-abort", type=float, default=DEFAULT_READOUT_ERROR_ABORT)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-budget", action="store_true")
    parser.add_argument("--approval-id", default="bell_rerun_mitigated_2026-07-16")
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
    layout = select_physical_qubits(backend.target)

    settings = chsh_setting_circuits()
    calibrations = calibration_circuits()
    circuits = list(settings) + list(calibrations)
    isa_circuits = transpile(
        circuits,
        backend=backend,
        initial_layout=list(layout.physical_qubits),
        optimization_level=1,
        seed_transpiler=SEED_TRANSPILER,
    )
    isa_depths = [int(circuit.depth()) for circuit in isa_circuits]

    readiness = build_readiness(
        backend_name=args.backend,
        layout=layout,
        isa_depths=isa_depths,
        args=args,
    )
    timestamp = readiness["timestamp_utc"]
    readiness_path = args.out_dir / f"bell_rerun_readiness_{args.backend}_{timestamp}.json"
    readiness_sha = _write_json(readiness_path, readiness)
    print(f"readiness={readiness['status']}")
    print(f"readiness_json={readiness_path}")
    print(f"readiness_sha256={readiness_sha}")
    if readiness["status"] != "ready_for_submission":
        return 3
    if not args.submit:
        print("hardware_submission=false")
        print("Re-run with --submit --confirm-budget to submit the Bell re-run job.")
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
    pubs = [(circuit, None, args.shots) for circuit in isa_circuits[: len(settings)]]
    pubs += [(circuit, None, args.cal_shots) for circuit in isa_circuits[len(settings) :]]
    job = sampler.run(pubs)
    job_id = strict_provider_job_id(job.job_id())
    print(f"job_id={job_id}")
    result = job.result(timeout=args.timeout_s)

    setting_counts = [
        coerce_counts(
            _pub_counts(result[index]),
            expected_shots=args.shots,
            field_name=f"setting {index} counts",
        )
        for index in range(len(settings))
    ]
    calibration_labels = computational_basis_labels(N_QUBITS)
    calibration_counts = {
        label: coerce_counts(
            _pub_counts(result[len(settings) + index]),
            expected_shots=args.cal_shots,
            field_name=f"calibration {label} counts",
        )
        for index, label in enumerate(calibration_labels)
    }

    raw_package = {
        "schema": f"{SCHEMA_PREFIX}_raw_counts_v1",
        "experiment_id": EXPERIMENT_ID,
        "preregistration": PREREGISTRATION_PATH,
        "parent_artifact": PARENT_ARTIFACT,
        "backend": args.backend,
        "job_id": job_id,
        "timestamp_utc": timestamp,
        "layout": dict(readiness["layout"]),
        "setting_angles": [list(angles) for angles in SETTING_ANGLES],
        "shots": args.shots,
        "calibration_shots": args.cal_shots,
        "seed_transpiler": SEED_TRANSPILER,
        "results": [
            {"pub_index": index, "counts": counts} for index, counts in enumerate(setting_counts)
        ],
        "calibration_counts": calibration_counts,
        "mitigation_deviation_note": readiness["mitigation_deviation_note"],
        "approval": {
            "approval_id": approval.approval_id,
            "approver": approval.approver,
            "package_hash": approval.package_hash,
            "max_qpu_seconds": approval.max_qpu_seconds,
            "allowed_provider": approval.allowed_provider,
        },
    }
    raw_path = args.out_dir / f"bell_rerun_raw_counts_{args.backend}_{timestamp}.json"
    raw_sha = _write_json(raw_path, raw_package)

    analysis = build_analysis(setting_counts, calibration_counts, shots=args.shots)
    analysis_path = args.out_dir / f"bell_rerun_analysis_{args.backend}_{timestamp}.json"
    analysis_sha = _write_json(analysis_path, analysis)

    print("hardware_submission=true")
    print(f"raw_counts_json={raw_path}")
    print(f"raw_counts_sha256={raw_sha}")
    print(f"analysis_json={analysis_path}")
    print(f"analysis_sha256={analysis_sha}")
    for pair_row in analysis["pairs"]:
        unmitigated = pair_row["unmitigated"]
        mitigated = pair_row["mitigated"]
        print(
            f"pair=q{pair_row['pair'][0]}q{pair_row['pair'][1]} "
            f"S_unmitigated={unmitigated['s_value']:.4f}±{unmitigated['sigma']:.4f} "
            f"S_mitigated={mitigated['s_value']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
