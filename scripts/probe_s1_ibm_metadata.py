#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S1 IBM metadata probe
"""No-submit S1 IBM backend metadata capability probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scpn_quantum_control.control.realtime_feedback import RealtimeSyncFeedbackController
from scpn_quantum_control.hardware.feedback_capability_probe import (
    assess_feedback_backend_capability,
)
from scpn_quantum_control.hardware.feedback_provider_metadata import (
    snapshot_from_generic_metadata,
    snapshot_from_qiskit_backend,
)
from scpn_quantum_control.hardware.feedback_submission import (
    build_s1_feedback_submission_package,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "s1_feedback_loop"
DATE = "2026-05-06"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe S1 dynamic-circuit capability from IBM/Qiskit backend metadata "
            "without submitting jobs."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--metadata-json",
        type=Path,
        help="Offline provider-neutral metadata JSON exported from a backend.",
    )
    source.add_argument(
        "--backend",
        help="Backend name to load from an already-authenticated Qiskit Runtime account.",
    )
    parser.add_argument(
        "--instance",
        help="Optional Qiskit Runtime instance name. No credential string is accepted here.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory for capability decision JSON.",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package():
    controller = RealtimeSyncFeedbackController(
        np.array(
            [[0.0, 0.35, 0.20], [0.35, 0.0, 0.25], [0.20, 0.25, 0.0]],
            dtype=np.float64,
        ),
        np.array([0.1, 0.4, 0.7], dtype=np.float64),
    )
    return build_s1_feedback_submission_package(
        controller,
        experiment_id="s1_dynamic_feedback_preregistration_2026-05-06",
        n_rounds=3,
        circuits=2,
        shots_per_circuit=1024,
        repetitions=12,
        estimated_seconds_per_circuit=1.0,
    )


def load_snapshot_from_metadata_json(path: Path):
    """Load a capability snapshot from offline metadata JSON."""
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata JSON must contain an object")
    return snapshot_from_generic_metadata(metadata)


def load_snapshot_from_authenticated_backend(backend_name: str, instance: str | None):
    """Load a capability snapshot from an already-authenticated Qiskit Runtime account."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        raise RuntimeError("qiskit-ibm-runtime is required for --backend probing") from exc
    service_kwargs = {"instance": instance} if instance else {}
    service = QiskitRuntimeService(**service_kwargs)
    backend = service.backend(backend_name)
    return snapshot_from_qiskit_backend(backend, provider="ibm")


def build_decision_document(snapshot) -> dict[str, Any]:
    """Build the no-submit capability decision document."""
    package = _package()
    decision = assess_feedback_backend_capability(snapshot, package)
    return {
        "date": DATE,
        "script": "scripts/probe_s1_ibm_metadata.py",
        "submission_state": "metadata_probe_no_submission",
        "hardware_submission": False,
        "credential_argument_supported": False,
        "experiment_id": package.experiment_id,
        "package_budget": {
            "circuits": package.budget.circuits,
            "shots_per_circuit": package.budget.shots_per_circuit,
            "repetitions": package.budget.repetitions,
            "total_reserved_seconds": package.budget.total_reserved_seconds,
        },
        "capability_decision": decision.to_dict(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Probe IBM backend metadata for the S1 feedback readiness surface."""
    args = _parse_args(argv)
    if args.metadata_json is not None:
        snapshot = load_snapshot_from_metadata_json(args.metadata_json)
    else:
        snapshot = load_snapshot_from_authenticated_backend(args.backend, args.instance)
    document = build_decision_document(snapshot)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    backend_name = document["capability_decision"]["backend_name"]
    out_path = args.out_dir / f"s1_ibm_metadata_probe_{backend_name}_{DATE}.json"
    out_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote_json={out_path}")
    print(f"sha256_json={_sha256(out_path)}")
    print("hardware_submission=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
