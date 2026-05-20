#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S1b IBM direct-XY observable extension
"""Approval-gated S1b IBM direct-XY observable runner.

S1b is an extension of the S1 paper, not a separate manuscript. It preserves
the same dynamic feedback and matched open-loop bodies, but replaces the
saturated binary synchrony proxy with final direct Pauli-basis measurements of
the XY-sector edge correlators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
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
    RealtimeSyncFeedbackController,
)
from scpn_quantum_control.hardware.feedback_hardware_scheduler import (  # noqa: E402
    ApprovalGatedFeedbackHardwareScheduler,
    HardwareApprovalRecord,
    hash_package_manifest,
)
from scpn_quantum_control.hardware.feedback_submission import (  # noqa: E402
    build_s1_feedback_submission_package,
)
from scpn_quantum_control.hardware.s1_feedback_ibm import (  # noqa: E402
    S1FeedbackArmCircuit,
    build_s1_arm_command,
    build_s1_xy_observable_arm_circuits,
    raw_count_package_from_xy_observable_results,
    run_ibm_sampler_arm,
)
from scripts.prepare_s1_ibm_live_readiness import (  # noqa: E402
    DEFAULT_CREDENTIALS_VAULT,
    SEED_TRANSPILER,
    load_authenticated_backend,
)

EXPERIMENT_ID = "s1b_dynamic_feedback_xy_observable_extension_2026-05-20"
PARENT_EXPERIMENT_ID = "s1_dynamic_feedback_preregistration_2026-05-06"
DEFAULT_LANE = "s1b"
DEFAULT_BACKEND = "ibm_kingston"
DEFAULT_OBSERVABLES = ("XXI", "YYI", "IXX", "IYY")
DEFAULT_N_ROUNDS = 3
SYSTEM_QUBITS = 3
DEFAULT_SHOTS = 1024
DEFAULT_REPETITIONS = 3
DEFAULT_QPU_CEILING = 120.0
DEFAULT_MAX_DEPTH = 1400
DATA_DIR = REPO_ROOT / "data" / "s1_feedback_loop"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", default=DEFAULT_LANE)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--parent-experiment-id", default=PARENT_EXPERIMENT_ID)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--instance")
    parser.add_argument("--credentials-vault", type=Path, default=DEFAULT_CREDENTIALS_VAULT)
    parser.add_argument("--n-rounds", type=int, default=DEFAULT_N_ROUNDS)
    parser.add_argument("--correction-angle", type=float, default=0.12)
    parser.add_argument("--base-gain", type=float, default=0.8)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--observables", nargs="+", default=list(DEFAULT_OBSERVABLES))
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--max-qpu-seconds", type=float, default=DEFAULT_QPU_CEILING)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-budget", action="store_true")
    parser.add_argument("--approval-id", default="s1b_ibm_xy_observable_extension_2026-05-20")
    parser.add_argument("--approver", default="Miroslav Sotek")
    return parser.parse_args(argv)


def _controller(args: argparse.Namespace) -> RealtimeSyncFeedbackController:
    return RealtimeSyncFeedbackController(
        np.array(
            [[0.0, 0.35, 0.20], [0.35, 0.0, 0.25], [0.20, 0.25, 0.0]],
            dtype=np.float64,
        ),
        np.array([0.1, 0.4, 0.7], dtype=np.float64),
        config=RealtimeFeedbackConfig(
            base_gain=float(args.base_gain),
            correction_angle=float(args.correction_angle),
        ),
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


def _operation_counts(circuit: QuantumCircuit) -> dict[str, int]:
    return {str(name): int(count) for name, count in circuit.count_ops().items()}


def _transpile_arm(backend: Any, arm: S1FeedbackArmCircuit) -> list[QuantumCircuit]:
    isa = transpile(
        arm.circuit,
        backend=backend,
        optimization_level=1,
        seed_transpiler=SEED_TRANSPILER,
    )
    name = f"{arm.label}_{arm.observable}"
    return [isa.copy(name=f"{name}_{index:02d}") for index in range(arm.repetitions)]


def _arm_summary(arm: S1FeedbackArmCircuit, isa: Sequence[QuantumCircuit]) -> dict[str, Any]:
    depths = [int(circuit.depth()) for circuit in isa]
    return {
        "label": arm.label,
        "observable": arm.observable,
        "shots": arm.shots,
        "repetitions": arm.repetitions,
        "source_depth": int(arm.circuit.depth()),
        "source_operation_counts": _operation_counts(arm.circuit),
        "transpiled_depth_max": max(depths),
        "transpiled_depths": depths,
        "transpiled_operation_counts": _operation_counts(isa[0]),
        "seed_transpiler": SEED_TRANSPILER,
        "estimated_qpu_seconds": arm.estimated_qpu_seconds,
    }


def _manifest(
    package: Mapping[str, Any],
    readiness: Mapping[str, Any],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema": f"scpn_{args.lane}_ibm_xy_observable_manifest_v1",
        "experiment_id": args.experiment_id,
        "parent_experiment_id": args.parent_experiment_id,
        "lane": args.lane,
        "package": dict(package),
        "readiness": dict(readiness),
    }


def _readiness_document(
    *,
    backend_name: str,
    args: argparse.Namespace,
    package_manifest: Mapping[str, Any],
    arm_summaries: Sequence[Mapping[str, Any]],
    max_depth: int,
    max_qpu_seconds: float,
) -> dict[str, Any]:
    estimated = sum(float(row["estimated_qpu_seconds"]) for row in arm_summaries)
    depth_ok = all(int(row["transpiled_depth_max"]) <= max_depth for row in arm_summaries)
    budget_ok = estimated <= max_qpu_seconds
    return {
        "schema": f"scpn_{args.lane}_ibm_xy_observable_readiness_v1",
        "timestamp_utc": _timestamp(),
        "backend": backend_name,
        "experiment_id": args.experiment_id,
        "parent_experiment_id": args.parent_experiment_id,
        "lane": args.lane,
        "hardware_submission": False,
        "status": "ready_for_submission" if depth_ok and budget_ok else "blocked",
        "observable_family": "direct_xy_pauli_correlators",
        "observables": list(args.observables),
        "n_rounds": args.n_rounds,
        "correction_angle": args.correction_angle,
        "base_gain": args.base_gain,
        "arms": list(arm_summaries),
        "max_depth": max_depth,
        "max_qpu_seconds": max_qpu_seconds,
        "estimated_qpu_seconds": estimated,
        "package_hash": hash_package_manifest(package_manifest),
        "reasons": _readiness_reasons(
            lane=args.lane,
            depth_ok=depth_ok,
            budget_ok=budget_ok,
        ),
    }


def _readiness_reasons(*, lane: str, depth_ok: bool, budget_ok: bool) -> list[str]:
    reasons: list[str] = []
    reasons.append(
        f"all {lane} direct-XY arms satisfy the transpiled depth ceiling"
        if depth_ok
        else f"at least one {lane} direct-XY arm exceeds the transpiled depth ceiling"
    )
    reasons.append(
        f"{lane} direct-XY estimate satisfies the approved QPU-second ceiling"
        if budget_ok
        else f"{lane} direct-XY estimate exceeds the approved QPU-second ceiling"
    )
    return reasons


def _analysis_summary(package: Mapping[str, Any]) -> dict[str, Any]:
    observables = package.get("observables")
    if not isinstance(observables, Sequence) or isinstance(observables, str):
        raise ValueError("S1b package must contain observable rows")
    deltas = [abs(float(row["feedback_minus_control"])) for row in observables]
    signed = [float(row["feedback_minus_control"]) for row in observables]
    return {
        "experiment_id": package["experiment_id"],
        "parent_experiment_id": package["parent_experiment_id"],
        "lane": package["lane"],
        "job_ids": list(package.get("job_ids", [])),
        "observable_family": package["observable_family"],
        "n_observables": len(deltas),
        "mean_abs_feedback_minus_control": sum(deltas) / len(deltas),
        "signed_feedback_minus_control": signed,
        "observables": list(observables),
        "claim_boundary": (
            f"{package['lane']} extends the S1 paper with direct XY-sector Pauli correlators. "
            "It does not by itself establish backend-general feedback control."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.submit and not args.confirm_budget:
        print("ERROR: --submit requires --confirm-budget", file=sys.stderr)
        return 2

    controller = _controller(args)
    package = build_s1_feedback_submission_package(
        controller,
        experiment_id=args.experiment_id,
        n_rounds=args.n_rounds,
        circuits=2 * len(args.observables),
        shots_per_circuit=args.shots,
        repetitions=args.repetitions,
        estimated_seconds_per_circuit=1.0,
    )
    backend = load_authenticated_backend(args.backend, args.instance, args.credentials_vault)
    arms = build_s1_xy_observable_arm_circuits(
        controller,
        observables=tuple(args.observables),
        n_rounds=args.n_rounds,
        shots=args.shots,
        repetitions=args.repetitions,
    )
    isa_by_key = {(arm.label, arm.observable): _transpile_arm(backend, arm) for arm in arms}
    arm_summaries = tuple(
        _arm_summary(arm, isa_by_key[(arm.label, arm.observable)]) for arm in arms
    )
    provisional_manifest = _manifest(package.to_dict(), {"arms": arm_summaries}, args=args)
    readiness = _readiness_document(
        backend_name=args.backend,
        args=args,
        package_manifest=provisional_manifest,
        arm_summaries=arm_summaries,
        max_depth=args.max_depth,
        max_qpu_seconds=args.max_qpu_seconds,
    )
    manifest = _manifest(package.to_dict(), readiness, args=args)
    readiness["package_hash"] = hash_package_manifest(manifest)
    timestamp = readiness["timestamp_utc"]
    readiness_path = (
        args.out_dir / f"{args.lane}_xy_observable_readiness_{args.backend}_{timestamp}.json"
    )
    readiness_sha = _write_json(readiness_path, readiness)
    print(f"readiness={readiness['status']}")
    print(f"readiness_json={readiness_path}")
    print(f"readiness_sha256={readiness_sha}")
    if readiness["status"] != "ready_for_submission":
        return 3
    if not args.submit:
        print("hardware_submission=false")
        print(f"Re-run with --submit --confirm-budget to submit {args.lane} direct-XY jobs.")
        return 0

    approval = HardwareApprovalRecord(
        approval_id=args.approval_id,
        approver=args.approver,
        package_hash=hash_package_manifest(manifest),
        max_qpu_seconds=args.max_qpu_seconds,
        allowed_provider="ibm_runtime",
        approved=True,
        notes=f"explicit command-line --submit --confirm-budget approval for {args.lane}",
    )
    arm_by_key = {(arm.label, arm.observable): arm for arm in arms}
    scheduler = ApprovalGatedFeedbackHardwareScheduler(
        provider="ibm_runtime",
        package_manifest=manifest,
        approval=approval,
        submitter=lambda command, _: run_ibm_sampler_arm(
            backend=backend,
            arm=arm_by_key[(command.label, command.payload["observable"])],
            isa_circuits=command.payload["isa_circuits"],
            timeout_s=float(command.payload["timeout_s"]),
        ),
    )
    results = []
    for arm in arms:
        command = build_s1_arm_command(
            arm,
            isa_circuits=isa_by_key[(arm.label, arm.observable)],
            timeout_s=args.timeout_s,
        )
        command.payload["observable"] = arm.observable
        results.append(scheduler.submit(command))

    raw_package = raw_count_package_from_xy_observable_results(
        experiment_id=args.experiment_id,
        n_qubits=SYSTEM_QUBITS,
        results=results,
    )
    raw_package.update(
        {
            "schema": f"scpn_{args.lane}_xy_observable_raw_counts_v1",
            "backend": args.backend,
            "parent_experiment_id": args.parent_experiment_id,
            "lane": args.lane,
            "n_rounds": args.n_rounds,
            "correction_angle": args.correction_angle,
            "base_gain": args.base_gain,
            "approval": {
                "approval_id": approval.approval_id,
                "approver": approval.approver,
                "package_hash": approval.package_hash,
                "max_qpu_seconds": approval.max_qpu_seconds,
                "allowed_provider": approval.allowed_provider,
            },
            "submission_records": [
                record.__dict__ | {"metadata": dict(record.metadata)}
                for record in scheduler.submissions
            ],
        }
    )
    raw_path = (
        args.out_dir / f"{args.lane}_xy_observable_raw_counts_{args.backend}_{timestamp}.json"
    )
    raw_sha = _write_json(raw_path, raw_package)
    analysis = _analysis_summary(raw_package)
    analysis_path = (
        args.out_dir / f"{args.lane}_xy_observable_analysis_{args.backend}_{timestamp}.json"
    )
    analysis_sha = _write_json(analysis_path, analysis)
    print("hardware_submission=true")
    print(f"job_ids={','.join(raw_package['job_ids'])}")
    print(f"raw_counts_json={raw_path}")
    print(f"raw_counts_sha256={raw_sha}")
    print(f"analysis_json={analysis_path}")
    print(f"analysis_sha256={analysis_sha}")
    print(f"mean_abs_feedback_minus_control={analysis['mean_abs_feedback_minus_control']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
