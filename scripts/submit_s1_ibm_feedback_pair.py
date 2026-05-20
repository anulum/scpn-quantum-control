#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S1 IBM feedback pair submission
"""Approval-gated S1 IBM feedback versus open-loop submission runner.

Default mode performs live backend loading, paired-arm circuit construction,
transpilation, budget accounting, and writes a no-submit readiness artefact.
QPU jobs are submitted only when both ``--submit`` and ``--confirm-budget`` are
provided.
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
    S1_FEEDBACK_ARM,
    S1FeedbackArmCircuit,
    build_s1_arm_command,
    build_s1_feedback_arm_circuits,
    raw_count_package_from_feedback_results,
    run_ibm_sampler_arm,
)
from scripts.analyse_s1_feedback_hardware import analyse_package  # noqa: E402
from scripts.prepare_s1_ibm_live_readiness import (  # noqa: E402
    DEFAULT_CREDENTIALS_VAULT,
    SEED_TRANSPILER,
    load_authenticated_backend,
)

EXPERIMENT_ID = "s1_dynamic_feedback_preregistration_2026-05-06"
TARGET_R = 0.72
N_ROUNDS = 3
SYSTEM_QUBITS = 3
DEFAULT_BACKEND = "ibm_kingston"
DEFAULT_SHOTS = 1024
DEFAULT_REPETITIONS = 12
DEFAULT_QPU_CEILING = 120.0
DEFAULT_MAX_DEPTH = 1200
DATA_DIR = REPO_ROOT / "data" / "s1_feedback_loop"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--instance")
    parser.add_argument("--credentials-vault", type=Path, default=DEFAULT_CREDENTIALS_VAULT)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--max-qpu-seconds", type=float, default=DEFAULT_QPU_CEILING)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-budget", action="store_true")
    parser.add_argument("--approval-id", default="s1_ibm_feedback_pair_2026-05-20")
    parser.add_argument("--approver", default="Miroslav Sotek")
    return parser.parse_args(argv)


def _controller() -> RealtimeSyncFeedbackController:
    return RealtimeSyncFeedbackController(
        np.array(
            [[0.0, 0.35, 0.20], [0.35, 0.0, 0.25], [0.20, 0.25, 0.0]],
            dtype=np.float64,
        ),
        np.array([0.1, 0.4, 0.7], dtype=np.float64),
    )


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _operation_counts(circuit: QuantumCircuit) -> dict[str, int]:
    return {str(name): int(count) for name, count in circuit.count_ops().items()}


def _transpile_arm(backend: Any, arm: S1FeedbackArmCircuit) -> list[QuantumCircuit]:
    isa = transpile(
        arm.circuit,
        backend=backend,
        optimization_level=1,
        seed_transpiler=SEED_TRANSPILER,
    )
    return [isa.copy(name=f"{arm.label}_{index:02d}") for index in range(arm.repetitions)]


def _arm_summary(arm: S1FeedbackArmCircuit, isa: Sequence[QuantumCircuit]) -> dict[str, Any]:
    depths = [int(circuit.depth()) for circuit in isa]
    first = isa[0]
    return {
        "label": arm.label,
        "shots": arm.shots,
        "repetitions": arm.repetitions,
        "source_depth": int(arm.circuit.depth()),
        "source_operation_counts": _operation_counts(arm.circuit),
        "transpiled_depth_max": max(depths),
        "transpiled_depths": depths,
        "transpiled_operation_counts": _operation_counts(first),
        "seed_transpiler": SEED_TRANSPILER,
        "estimated_qpu_seconds": arm.estimated_qpu_seconds,
    }


def _manifest(package: Mapping[str, Any], readiness: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "scpn_s1_ibm_feedback_pair_manifest_v1",
        "experiment_id": EXPERIMENT_ID,
        "package": dict(package),
        "readiness": dict(readiness),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return _sha256(path)


def _build_readiness_document(
    *,
    backend_name: str,
    package_manifest: Mapping[str, Any],
    arm_summaries: Sequence[Mapping[str, Any]],
    max_depth: int,
    max_qpu_seconds: float,
) -> dict[str, Any]:
    estimated = sum(float(row["estimated_qpu_seconds"]) for row in arm_summaries)
    depth_ok = all(int(row["transpiled_depth_max"]) <= max_depth for row in arm_summaries)
    budget_ok = estimated <= max_qpu_seconds
    return {
        "schema": "scpn_s1_ibm_feedback_pair_readiness_v1",
        "timestamp_utc": _timestamp(),
        "backend": backend_name,
        "experiment_id": EXPERIMENT_ID,
        "hardware_submission": False,
        "status": "ready_for_submission" if depth_ok and budget_ok else "blocked",
        "target_r": TARGET_R,
        "observable": "binary_phase_synchrony_from_final_counts",
        "arms": list(arm_summaries),
        "max_depth": max_depth,
        "max_qpu_seconds": max_qpu_seconds,
        "estimated_qpu_seconds": estimated,
        "package_hash": hash_package_manifest(package_manifest),
        "reasons": _readiness_reasons(depth_ok=depth_ok, budget_ok=budget_ok),
    }


def _readiness_reasons(*, depth_ok: bool, budget_ok: bool) -> list[str]:
    reasons: list[str] = []
    if depth_ok:
        reasons.append("both S1 arms satisfy the transpiled depth ceiling")
    else:
        reasons.append("at least one S1 arm exceeds the transpiled depth ceiling")
    if budget_ok:
        reasons.append("paired-arm estimate satisfies the approved QPU-second ceiling")
    else:
        reasons.append("paired-arm estimate exceeds the approved QPU-second ceiling")
    return reasons


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare or submit the paired S1 IBM feedback experiment."""
    args = _parse_args(argv)
    if args.submit and not args.confirm_budget:
        print("ERROR: --submit requires --confirm-budget", file=sys.stderr)
        return 2

    controller = _controller()
    package = build_s1_feedback_submission_package(
        controller,
        experiment_id=EXPERIMENT_ID,
        n_rounds=N_ROUNDS,
        circuits=2,
        shots_per_circuit=args.shots,
        repetitions=args.repetitions,
        estimated_seconds_per_circuit=1.0,
    )
    backend = load_authenticated_backend(args.backend, args.instance, args.credentials_vault)
    feedback_arm, control_arm = build_s1_feedback_arm_circuits(
        controller,
        n_rounds=N_ROUNDS,
        shots=args.shots,
        repetitions=args.repetitions,
    )
    feedback_isa = _transpile_arm(backend, feedback_arm)
    control_isa = _transpile_arm(backend, control_arm)
    arm_summaries = (
        _arm_summary(feedback_arm, feedback_isa),
        _arm_summary(control_arm, control_isa),
    )
    provisional_manifest = _manifest(package.to_dict(), {"arms": arm_summaries})
    readiness = _build_readiness_document(
        backend_name=args.backend,
        package_manifest=provisional_manifest,
        arm_summaries=arm_summaries,
        max_depth=args.max_depth,
        max_qpu_seconds=args.max_qpu_seconds,
    )
    manifest = _manifest(package.to_dict(), readiness)
    readiness["package_hash"] = hash_package_manifest(manifest)
    timestamp = readiness["timestamp_utc"]
    readiness_path = (
        args.out_dir / f"s1_ibm_feedback_pair_readiness_{args.backend}_{timestamp}.json"
    )
    readiness_sha = _write_json(readiness_path, readiness)
    print(f"readiness={readiness['status']}")
    print(f"readiness_json={readiness_path}")
    print(f"readiness_sha256={readiness_sha}")
    if readiness["status"] != "ready_for_submission":
        return 3
    if not args.submit:
        print("hardware_submission=false")
        print("Re-run with --submit --confirm-budget to submit the paired S1 jobs.")
        return 0

    approval = HardwareApprovalRecord(
        approval_id=args.approval_id,
        approver=args.approver,
        package_hash=hash_package_manifest(manifest),
        max_qpu_seconds=args.max_qpu_seconds,
        allowed_provider="ibm_runtime",
        approved=True,
        notes="explicit command-line --submit --confirm-budget approval",
    )
    scheduler = ApprovalGatedFeedbackHardwareScheduler(
        provider="ibm_runtime",
        package_manifest=manifest,
        approval=approval,
        submitter=lambda command, _: run_ibm_sampler_arm(
            backend=backend,
            arm=feedback_arm if command.label == S1_FEEDBACK_ARM else control_arm,
            isa_circuits=command.payload["isa_circuits"],
            timeout_s=float(command.payload["timeout_s"]),
        ),
    )
    feedback_result = scheduler.submit(
        build_s1_arm_command(feedback_arm, isa_circuits=feedback_isa, timeout_s=args.timeout_s)
    )
    control_result = scheduler.submit(
        build_s1_arm_command(control_arm, isa_circuits=control_isa, timeout_s=args.timeout_s)
    )
    raw_package = raw_count_package_from_feedback_results(
        experiment_id=EXPERIMENT_ID,
        target_r=TARGET_R,
        n_qubits=SYSTEM_QUBITS,
        feedback_result=feedback_result,
        control_result=control_result,
    )
    raw_package.update(
        {
            "schema": "scpn_s1_feedback_raw_counts_v1",
            "backend": args.backend,
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
    raw_path = args.out_dir / f"s1_feedback_raw_counts_{args.backend}_{timestamp}.json"
    raw_sha = _write_json(raw_path, raw_package)
    analysis = analyse_package(raw_package)
    analysis_path = args.out_dir / f"s1_feedback_analysis_summary_{args.backend}_{timestamp}.json"
    analysis_sha = _write_json(analysis_path, analysis)
    print("hardware_submission=true")
    print(f"job_ids={','.join(raw_package['job_ids'])}")
    print(f"raw_counts_json={raw_path}")
    print(f"raw_counts_sha256={raw_sha}")
    print(f"analysis_json={analysis_path}")
    print(f"analysis_sha256={analysis_sha}")
    print(f"decision={analysis['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
