#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
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
from scpn_quantum_control.hardware.provider_capability_discovery import (
    ProviderCapabilitySnapshot,
    build_openpulse_control_readiness,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "s1_feedback_loop"
DATE = "2026-05-06"
DEFAULT_CREDENTIALS_VAULT = Path("/media/anulum/724AA8E84AA8AA75/agentic-shared/CREDENTIALS.md")


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
        "--credentials-vault",
        type=Path,
        default=DEFAULT_CREDENTIALS_VAULT,
        help="Credential vault path used only for authenticated backend metadata lookup.",
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


def _parse_vault(path: Path) -> tuple[str | None, str | None]:
    if not path.exists():
        return None, None
    phase1_path = REPO_ROOT / "scripts" / "phase1_mini_bench_ibm_kingston.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("phase1_mini_bench_ibm_kingston", phase1_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {phase1_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    credential_value, vault_instance = module.parse_vault(path)
    return credential_value, vault_instance


def load_snapshot_from_authenticated_backend(
    backend_name: str,
    instance: str | None,
    credentials_vault: Path | None = None,
):
    """Load a capability snapshot from an already-authenticated Qiskit Runtime account."""
    backend = load_authenticated_backend(backend_name, instance, credentials_vault)
    return snapshot_from_qiskit_backend(backend, provider="ibm")


def load_authenticated_backend(
    backend_name: str,
    instance: str | None,
    credentials_vault: Path | None = None,
):
    """Load a Qiskit Runtime backend from saved auth or the local credentials vault."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        raise RuntimeError("qiskit-ibm-runtime is required for --backend probing") from exc
    credential_value, vault_instance = (
        _parse_vault(credentials_vault) if credentials_vault is not None else (None, None)
    )
    selected_instance = instance or vault_instance
    service_kwargs = {"channel": "ibm_cloud"} if credential_value else {}
    if credential_value:
        service_kwargs["token"] = credential_value
    if selected_instance:
        service_kwargs["instance"] = selected_instance
    service = QiskitRuntimeService(**service_kwargs)
    return service.backend(backend_name)


def build_decision_document(snapshot) -> dict[str, Any]:
    """Build the no-submit capability decision document."""
    package = _package()
    decision = assess_feedback_backend_capability(snapshot, package)
    openpulse_readiness = _openpulse_readiness_from_feedback_snapshot(snapshot)
    openpulse_payload = openpulse_readiness.to_dict()
    openpulse_status = "ready" if openpulse_readiness.ready else "blocked"
    return {
        "date": DATE,
        "script": "scripts/probe_s1_ibm_metadata.py",
        "submission_state": "metadata_probe_no_submission",
        "hardware_submission": False,
        "credential_string_argument_supported": False,
        "experiment_id": package.experiment_id,
        "package_budget": {
            "circuits": package.budget.circuits,
            "shots_per_circuit": package.budget.shots_per_circuit,
            "repetitions": package.budget.repetitions,
            "total_reserved_seconds": package.budget.total_reserved_seconds,
        },
        "openpulse_readiness_status": openpulse_status,
        "openpulse_blockers": list(openpulse_readiness.blockers),
        "openpulse_warnings": list(openpulse_readiness.warnings),
        "openpulse_readiness": openpulse_payload,
        "capability_decision": decision.to_dict(),
    }


def _openpulse_readiness_from_feedback_snapshot(snapshot: Any) -> Any:
    metadata = snapshot.metadata if isinstance(snapshot.metadata, Mapping) else {}
    provider_snapshot = ProviderCapabilitySnapshot(
        route_id="ibm_runtime::ibm",
        aggregator="ibm_runtime",
        provider=snapshot.provider,
        backend_id=snapshot.backend_name,
        target_name=snapshot.backend_name,
        n_qubits=snapshot.n_qubits,
        supported_ir_formats=("qiskit", "qiskit_qpy"),
        basis_gates=tuple(snapshot.basis_gates),
        native_features=tuple(snapshot.supported_features),
        online=True,
        simulator=bool(snapshot.simulator),
        no_submit=True,
        max_shots=snapshot.max_shots,
        max_circuits=snapshot.max_circuits,
        calibration_timestamp=None,
        metadata=metadata,
    )
    profile = metadata.get("openpulse_profile")
    if isinstance(profile, Mapping):
        n_control = profile.get("n_control_channels")
        if (isinstance(n_control, int) and n_control > 0) or bool(
            profile.get("supports_drive_channel_access")
        ):
            dt = 2.222e-10
        else:
            dt = 1.0
    else:
        dt = 1.0
    return build_openpulse_control_readiness(
        provider_snapshot,
        qubit=0,
        dt=dt,
        shots=1024,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Probe IBM backend metadata for the S1 feedback readiness surface."""
    args = _parse_args(argv)
    if args.metadata_json is not None:
        snapshot = load_snapshot_from_metadata_json(args.metadata_json)
    else:
        snapshot = load_snapshot_from_authenticated_backend(
            args.backend,
            args.instance,
            args.credentials_vault,
        )
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
