#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — QBER re-run submitter with committed basis metadata
"""Submit the preregistered matched-basis QBER re-run (KIMI-8).

The published 5.5%/5.8% QKD bit-error rate cannot be re-derived from the
committed March 2026 artefact because it carries no basis metadata. This
submitter re-measures the matched-basis mismatch rate of two Bell pairs
(BBM92-style, fixed per-pub bases — explicitly NOT a QKD protocol run: no
per-shot random basis choice, no sifting, no security claim) with the
basis of every pub committed in the pack, so the derivation is
reproducible from the committed counts alone. The analysis also recomputes
the March artefact's naive matched-basis sift under explicitly stated
assumed bases, next to the new values.

Preregistration: docs/campaigns/qkd_qber_basis_metadata_prereg_2026-07-17.md.
Default run is a dry readiness check; submission requires
``--submit --confirm-budget`` (owner GO) and stays behind hard abort gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit, transpile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scpn_quantum_control.hardware._count_integrity import (  # noqa: E402
    strict_provider_job_id,
)
from scpn_quantum_control.hardware.feedback_hardware_scheduler import (  # noqa: E402
    HardwareApprovalRecord,
    hash_package_manifest,
)
from scpn_quantum_control.mitigation.readout_matrix import (  # noqa: E402
    build_readout_confusion_matrix,
    computational_basis_labels,
    mitigate_counts,
)
from scripts.recompute_chsh_bell_test import QUBIT_PAIRS  # noqa: E402
from scripts.submit_bell_rerun_mitigated import (  # noqa: E402
    LayoutSelection,
    coerce_counts,
    select_physical_qubits,
)

EXPERIMENT_ID = "qkd_qber_basis_metadata_2026-07-17"
PREREGISTRATION_PATH = "docs/campaigns/qkd_qber_basis_metadata_prereg_2026-07-17.md"
PARENT_ARTIFACT = "results/ibm_hardware_2026-03-28/qkd_qber_4q.json"
SCHEMA_PREFIX = "scpn_qkd_qber_basis_metadata"

N_QUBITS = 4

#: Fixed matched measurement bases, one per main pub. Even-index qubits act
#: as the Alice side, odd-index qubits as the Bob side of each pair; both
#: pairs in a pub share the basis. There is deliberately no per-shot random
#: basis choice — see the preregistration's honesty boundary.
BASIS_SETTINGS: tuple[str, ...] = ("Z", "X")

#: Assumed pub bases for the March artefact's naive sift, stated explicitly
#: because the artefact itself records none (that gap is the whole point).
MARCH_ASSUMED_PUB_BASES: tuple[str, ...] = ("Z", "X")

DEFAULT_BACKEND = "ibm_fez"
DEFAULT_SHOTS = 4096
DEFAULT_CAL_SHOTS = 8192
DEFAULT_MAX_QPU_SECONDS = 60.0
DEFAULT_MAX_DEPTH = 60
DEFAULT_READOUT_ERROR_ABORT = 0.15
ESTIMATED_SECONDS_PER_CIRCUIT = 1.1
SEED_TRANSPILER = 20260717
DATA_DIR = REPO_ROOT / "data" / "qkd_qber_basis_metadata"


def matched_basis_circuits() -> tuple[QuantumCircuit, ...]:
    """Build one circuit per matched basis on two Bell pairs.

    Logical layout: pair (0, 1) and pair (2, 3), each prepared in |Φ+⟩.
    Basis Z measures directly; basis X applies H to every qubit first. For
    an ideal |Φ+⟩ both matched bases give perfectly agreeing pair bits.
    """
    circuits: list[QuantumCircuit] = []
    for basis in BASIS_SETTINGS:
        circuit = QuantumCircuit(N_QUBITS, name=f"qkd_qber_{basis.lower()}{basis.lower()}")
        for pair in QUBIT_PAIRS:
            circuit.h(pair[0])
            circuit.cx(pair[0], pair[1])
        circuit.barrier()
        if basis == "X":
            for qubit in range(N_QUBITS):
                circuit.h(qubit)
        circuit.measure_all()
        circuits.append(circuit)
    return tuple(circuits)


def calibration_circuits() -> tuple[QuantumCircuit, ...]:
    """Build the 16 full-basis readout-calibration circuits."""
    circuits: list[QuantumCircuit] = []
    for label in computational_basis_labels(N_QUBITS):
        circuit = QuantumCircuit(N_QUBITS, name=f"qkd_qber_cal_{label}")
        for qubit in range(N_QUBITS):
            if label[-1 - qubit] == "1":
                circuit.x(qubit)
        circuit.measure_all()
        circuits.append(circuit)
    return tuple(circuits)


def basis_metadata() -> list[dict[str, Any]]:
    """Committed per-pub basis metadata — the record the March artefact lacks."""
    return [
        {
            "pub_index": index,
            "circuit_name": f"qkd_qber_{basis.lower()}{basis.lower()}",
            "alice_basis": basis,
            "bob_basis": basis,
            "pairs": [list(pair) for pair in QUBIT_PAIRS],
            "per_shot_random_basis": False,
        }
        for index, basis in enumerate(BASIS_SETTINGS)
    ]


def pair_mismatch_probability(counts: Mapping[str, int], pair: tuple[int, int]) -> float:
    """Fraction of shots whose little-endian pair bits disagree."""
    total = 0
    mismatched = 0
    for bitstring, count in counts.items():
        total += count
        if bitstring[-1 - pair[0]] != bitstring[-1 - pair[1]]:
            mismatched += count
    if total <= 0:
        raise ValueError("counts carry no shots")
    return mismatched / total


def quasi_probability_mismatch(
    probabilities: Sequence[float], labels: Sequence[str], pair: tuple[int, int]
) -> float:
    """Mismatch rate from a readout-mitigated quasi-probability vector.

    Quasi-probabilities from the exact inversion may be slightly negative;
    they enter the sum unclipped and the result is labelled accordingly.
    """
    mismatch = 0.0
    for probability, label in zip(probabilities, labels, strict=True):
        if label[-1 - pair[0]] != label[-1 - pair[1]]:
            mismatch += float(probability)
    return mismatch


def binomial_sigma(rate: float, shots: int) -> float:
    """Binomial standard error of a rate at the raw shot count."""
    clamped = min(max(rate, 0.0), 1.0)
    return math.sqrt(clamped * (1.0 - clamped) / shots)


def march_naive_sift(artifact_path: Path) -> dict[str, Any]:
    """Recompute the March artefact's naive matched-basis sift.

    Pure arithmetic on the committed counts under the explicitly assumed
    pub bases (the artefact records none — that gap is the derivability
    caveat). The third pub is treated as an unmatched control and skipped.
    """
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    results = {entry["pub_index"]: entry["counts"] for entry in payload["results"]}
    sift: list[dict[str, Any]] = []
    for pub_index, basis in enumerate(MARCH_ASSUMED_PUB_BASES):
        counts = results[pub_index]
        shots = sum(int(v) for v in counts.values())
        for pair in QUBIT_PAIRS:
            rate = pair_mismatch_probability(counts, pair)
            sift.append(
                {
                    "pub_index": pub_index,
                    "assumed_basis": basis,
                    "pair": list(pair),
                    "mismatch_rate": rate,
                    "sigma_binomial": binomial_sigma(rate, shots),
                    "shots": shots,
                }
            )
    try:
        artifact_reference = str(artifact_path.relative_to(REPO_ROOT))
    except ValueError:
        artifact_reference = str(artifact_path)
    return {
        "artifact": artifact_reference,
        "assumed_pub_bases": list(MARCH_ASSUMED_PUB_BASES),
        "assumption_note": (
            "the committed artefact carries no basis metadata; these pub-to-"
            "basis assignments are an assumption stated for reproducibility, "
            "not a record"
        ),
        "published_qber": {"zz": 0.055, "xx": 0.058},
        "entries": sift,
    }


def build_analysis(
    basis_counts: Sequence[Mapping[str, int]],
    calibration_counts: Mapping[str, Mapping[str, int]],
    *,
    shots: int,
    march_sift: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the mitigation-on/off matched-basis error-rate analysis."""
    confusion_matrix = build_readout_confusion_matrix(calibration_counts, N_QUBITS)
    measurements: list[dict[str, Any]] = []
    for metadata, counts in zip(basis_metadata(), basis_counts, strict=True):
        mitigated_vector = mitigate_counts(counts, confusion_matrix).tolist()
        for pair in QUBIT_PAIRS:
            raw_rate = pair_mismatch_probability(counts, pair)
            mitigated_rate = quasi_probability_mismatch(
                mitigated_vector, confusion_matrix.labels, pair
            )
            measurements.append(
                {
                    "pub_index": metadata["pub_index"],
                    "basis": metadata["alice_basis"],
                    "pair": list(pair),
                    "mismatch_rate": raw_rate,
                    "sigma_binomial": binomial_sigma(raw_rate, shots),
                    "mitigated_mismatch_rate": mitigated_rate,
                    "mitigated_sigma_note": (
                        "binomial sigma at the raw shot count; readout-inversion "
                        "error not propagated, quasi-probabilities unclipped"
                    ),
                }
            )
    return {
        "schema": f"{SCHEMA_PREFIX}_analysis_v1",
        "experiment_id": EXPERIMENT_ID,
        "preregistration": PREREGISTRATION_PATH,
        "parent_artifact": PARENT_ARTIFACT,
        "basis_metadata": basis_metadata(),
        "measurements": measurements,
        "march_naive_sift": dict(march_sift),
        "decision_rule": decision_rule(measurements, march_sift),
        "claim_boundary": (
            "Matched-basis mismatch rate of the entangled source only: no QKD "
            "security, key-rate, or viability claim; the March record's "
            "non-derivability caveat stands regardless; the published March "
            "2026 record stays as it is."
        ),
    }


def decision_rule(
    measurements: Sequence[Mapping[str, Any]], march_sift: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate the preregistered comparison against the March naive sift.

    The new rates support the "published values overstate the artefact"
    reading when every mitigated rate sits within 2σ of the March naive-sift
    band; otherwise the higher-rate reading is recorded. Both readings keep
    the derivability caveat in place.
    """
    march_rates = [float(entry["mismatch_rate"]) for entry in march_sift["entries"]]
    band_low, band_high = min(march_rates), max(march_rates)
    within = True
    for measurement in measurements:
        rate = float(measurement["mitigated_mismatch_rate"])
        sigma = float(measurement["sigma_binomial"])
        if not (band_low - 2.0 * sigma) <= rate <= (band_high + 2.0 * sigma):
            within = False
            break
    return {
        "march_naive_sift_band": [band_low, band_high],
        "all_mitigated_rates_within_2_sigma_of_band": within,
        "preregistered_reading": (
            "new rates match the March naive sift: the published 5.5%/5.8% "
            "overstate what the artefact supports; this dated record is the "
            "citable error rate"
            if within
            else "new rates sit outside the March naive-sift band: the March "
            "analysis plausibly included unrecorded effects; the caveat "
            "stands and this dated record is the citable error rate"
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
    n_circuits = len(BASIS_SETTINGS) + 2**N_QUBITS
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
        "basis_metadata": basis_metadata(),
        "shots": args.shots,
        "calibration_shots": args.cal_shots,
        "n_circuits": n_circuits,
        "transpiled_depths": list(isa_depths),
        "max_depth": args.max_depth,
        "readout_error_abort": args.readout_error_abort,
        "estimated_qpu_seconds": estimated,
        "max_qpu_seconds": args.max_qpu_seconds,
        "seed_transpiler": SEED_TRANSPILER,
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
    parser.add_argument("--march-artifact", type=Path, default=REPO_ROOT / PARENT_ARTIFACT)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-budget", action="store_true")
    parser.add_argument("--approval-id", default="qkd_qber_basis_metadata_2026-07-17")
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

    mains = matched_basis_circuits()
    calibrations = calibration_circuits()
    circuits = list(mains) + list(calibrations)
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
    readiness_path = args.out_dir / f"qkd_qber_readiness_{args.backend}_{timestamp}.json"
    readiness_sha = _write_json(readiness_path, readiness)
    print(f"readiness={readiness['status']}")
    print(f"readiness_json={readiness_path}")
    print(f"readiness_sha256={readiness_sha}")
    if readiness["status"] != "ready_for_submission":
        return 3
    if not args.submit:
        print("hardware_submission=false")
        print("Re-run with --submit --confirm-budget to submit the QBER re-run job.")
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
    pubs = [(circuit, None, args.shots) for circuit in isa_circuits[: len(mains)]]
    pubs += [(circuit, None, args.cal_shots) for circuit in isa_circuits[len(mains) :]]
    job = sampler.run(pubs)
    job_id = strict_provider_job_id(job.job_id())
    print(f"job_id={job_id}")
    result = job.result(timeout=args.timeout_s)

    basis_counts = [
        coerce_counts(
            _pub_counts(result[index]),
            expected_shots=args.shots,
            field_name=f"pub {index} ({BASIS_SETTINGS[index]}{BASIS_SETTINGS[index]})",
        )
        for index in range(len(mains))
    ]
    calibration_counts = {
        label: coerce_counts(
            _pub_counts(result[len(mains) + index]),
            expected_shots=args.cal_shots,
            field_name=f"calibration {label}",
        )
        for index, label in enumerate(computational_basis_labels(N_QUBITS))
    }

    usage: dict[str, Any] = {}
    try:
        metrics = job.metrics()
        if isinstance(metrics, Mapping):
            usage = dict(metrics.get("usage", {}) or {})
    except Exception:
        usage = {"note": "job metrics unavailable at retrieval time"}

    raw_payload = {
        "schema": f"{SCHEMA_PREFIX}_raw_v1",
        "experiment_id": EXPERIMENT_ID,
        "preregistration": PREREGISTRATION_PATH,
        "backend": args.backend,
        "job_id": job_id,
        "timestamp_utc": timestamp,
        "approval": {
            "approval_id": approval.approval_id,
            "approver": approval.approver,
            "package_hash": approval.package_hash,
        },
        "layout": readiness["layout"],
        "basis_metadata": basis_metadata(),
        "shots": args.shots,
        "calibration_shots": args.cal_shots,
        "usage": usage,
        "basis_counts": [dict(counts) for counts in basis_counts],
        "calibration_counts": {label: dict(c) for label, c in calibration_counts.items()},
    }
    raw_path = args.out_dir / f"qkd_qber_raw_counts_{args.backend}_{timestamp}.json"
    raw_sha = _write_json(raw_path, raw_payload)
    print(f"raw_json={raw_path}")
    print(f"raw_sha256={raw_sha}")

    march_sift = march_naive_sift(args.march_artifact)
    analysis = build_analysis(
        basis_counts,
        calibration_counts,
        shots=args.shots,
        march_sift=march_sift,
    )
    analysis["raw_counts_sha256"] = raw_sha
    analysis_path = args.out_dir / f"qkd_qber_analysis_{args.backend}_{timestamp}.json"
    analysis_sha = _write_json(analysis_path, analysis)
    print(f"analysis_json={analysis_path}")
    print(f"analysis_sha256={analysis_sha}")
    for measurement in analysis["measurements"]:
        print(
            f"pair q{measurement['pair'][0]}q{measurement['pair'][1]} "
            f"{measurement['basis']}{measurement['basis']}: "
            f"raw={measurement['mismatch_rate']:.4f} "
            f"mitigated={measurement['mitigated_mismatch_rate']:.4f} "
            f"sigma={measurement['sigma_binomial']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
