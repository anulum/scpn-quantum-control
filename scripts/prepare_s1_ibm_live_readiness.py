#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — prepare s1 IBM live readiness script
# scpn-quantum-control -- S1 IBM live readiness without submission
"""Prepare S1 IBM live-readiness artefacts without submitting QPU jobs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scpn_quantum_control.control.realtime_feedback import RealtimeSyncFeedbackController
from scpn_quantum_control.hardware.feedback_capability_probe import (
    assess_feedback_backend_capability,
)
from scpn_quantum_control.hardware.feedback_provider_metadata import snapshot_from_qiskit_backend
from scpn_quantum_control.hardware.feedback_submission import (
    FeedbackSubmissionPackage,
    build_s1_feedback_submission_package,
)
from scpn_quantum_control.hardware.provider_capability_discovery import (
    ProviderCapabilitySnapshot,
    build_openpulse_control_readiness,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-05-06"
CAPTURE_DATE = "2026-05-20"
DEFAULT_CREDENTIALS_VAULT = Path("~/.config/scpn-quantum-control/credentials.md").expanduser()
DATA_DIR = REPO_ROOT / "data" / "s1_feedback_loop"
DOCS_DIR = REPO_ROOT / "docs"
QPU_SECONDS_CEILING = 120.0
SEED_TRANSPILER = 20260520


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture live S1 IBM backend metadata and transpilation readiness without "
            "submitting QPU jobs."
        )
    )
    parser.add_argument("--backend", required=True, help="IBM backend name.")
    parser.add_argument("--instance", help="Optional Qiskit Runtime instance name.")
    parser.add_argument(
        "--credentials-vault",
        type=Path,
        default=DEFAULT_CREDENTIALS_VAULT,
        help="Credential vault path used only for authenticated backend metadata lookup.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory for generated JSON artefacts.",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=DOCS_DIR,
        help="Directory for generated Markdown readiness note.",
    )
    return parser.parse_args(argv)


def _controller() -> RealtimeSyncFeedbackController:
    return RealtimeSyncFeedbackController(
        np.array(
            [[0.0, 0.35, 0.20], [0.35, 0.0, 0.25], [0.20, 0.25, 0.0]],
            dtype=np.float64,
        ),
        np.array([0.1, 0.4, 0.7], dtype=np.float64),
    )


def _package() -> FeedbackSubmissionPackage:
    return build_s1_feedback_submission_package(
        _controller(),
        experiment_id="s1_dynamic_feedback_preregistration_2026-05-06",
        n_rounds=3,
        circuits=2,
        shots_per_circuit=1024,
        repetitions=12,
        estimated_seconds_per_circuit=1.0,
    )


def _parse_vault(path: Path) -> tuple[str | None, str | None]:
    if not path.exists():
        return None, None
    phase1_path = REPO_ROOT / "scripts" / "phase1_mini_bench_ibm_kingston.py"
    spec = importlib.util.spec_from_file_location("phase1_mini_bench_ibm_kingston", phase1_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {phase1_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    credential_value, vault_instance = module.parse_vault(path)
    return credential_value, vault_instance


def load_authenticated_backend(
    backend_name: str,
    instance: str | None,
    credentials_vault: Path | None = None,
) -> Any:
    """Load a Qiskit Runtime backend from saved auth or the local credentials vault."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        raise RuntimeError("qiskit-ibm-runtime is required for S1 live readiness") from exc
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _operation_counts(circuit: Any) -> dict[str, int]:
    return {str(name): int(count) for name, count in circuit.count_ops().items()}


def _backend_status(backend: Any) -> dict[str, Any]:
    status_method = getattr(backend, "status", None)
    if not callable(status_method):
        return {"available": None, "pending_jobs": None, "status_msg": "status unavailable"}
    status = status_method()
    return {
        "available": getattr(status, "operational", None),
        "pending_jobs": getattr(status, "pending_jobs", None),
        "status_msg": getattr(status, "status_msg", None),
    }


def _transpile_summary(backend: Any, n_rounds: int) -> dict[str, Any]:
    try:
        from qiskit import transpile
    except ImportError as exc:
        raise RuntimeError("qiskit is required for S1 live-readiness transpilation") from exc

    source = _controller().build_monitored_circuit(n_rounds)
    transpiled = transpile(
        source,
        backend=backend,
        optimization_level=1,
        seed_transpiler=SEED_TRANSPILER,
    )
    return {
        "source": {
            "n_qubits": source.num_qubits,
            "n_clbits": source.num_clbits,
            "depth": int(source.depth()),
            "operation_counts": _operation_counts(source),
        },
        "transpiled": {
            "n_qubits": transpiled.num_qubits,
            "n_clbits": transpiled.num_clbits,
            "depth": int(transpiled.depth()),
            "operation_counts": _operation_counts(transpiled),
            "seed_transpiler": SEED_TRANSPILER,
        },
        "submission_performed": False,
    }


def build_live_readiness_document(backend: Any) -> dict[str, Any]:
    """Build a no-submit S1 live-readiness document from an authenticated backend."""
    package = _package()
    snapshot = snapshot_from_qiskit_backend(backend, provider="ibm")
    decision = assess_feedback_backend_capability(snapshot, package)
    openpulse_readiness = _openpulse_readiness_from_feedback_snapshot(snapshot)
    openpulse_status = "ready" if openpulse_readiness.ready else "blocked"
    transpilation = _transpile_summary(backend, package.circuit.n_rounds)
    blockers: list[str] = []
    if decision.status != "ready":
        blockers.append("live backend capability decision is not ready")
    if not openpulse_readiness.ready:
        blockers.append("openpulse readiness is blocked")
    readiness_status = "blocked" if blockers else "ready_for_pair_runner"
    return {
        "date": CAPTURE_DATE,
        "preregistration_date": DATE,
        "script": "scripts/prepare_s1_ibm_live_readiness.py",
        "submission_state": "live_metadata_and_transpile_no_submission",
        "hardware_submission": False,
        "credential_string_argument_supported": False,
        "selected_backend": snapshot.backend_name,
        "backend_status": _backend_status(backend),
        "package_budget": {
            "circuits": package.budget.circuits,
            "shots_per_circuit": package.budget.shots_per_circuit,
            "repetitions": package.budget.repetitions,
            "total_reserved_seconds": package.budget.total_reserved_seconds,
            "qpu_seconds_ceiling": QPU_SECONDS_CEILING,
        },
        "capability_decision": decision.to_dict(),
        "openpulse_readiness_status": openpulse_status,
        "openpulse_blockers": list(openpulse_readiness.blockers),
        "openpulse_warnings": list(openpulse_readiness.warnings),
        "openpulse_readiness": openpulse_readiness.to_dict(),
        "transpilation": transpilation,
        "readiness_status": readiness_status,
        "blockers": blockers,
        "claim_boundary": (
            "This artefact captures live backend metadata and transpilation only. It "
            "does not submit IBM jobs and cannot support an S1 hardware-control claim."
        ),
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


def write_readiness_markdown(document: Mapping[str, Any], path: Path) -> None:
    """Write a concise Markdown readiness note."""
    capability = document["capability_decision"]
    transpiled = document["transpilation"]["transpiled"]
    budget = document["package_budget"]
    blockers = "\n".join(f"- {item}" for item in document["blockers"]) or "- none"
    openpulse_status = document.get("openpulse_readiness_status", "unknown")
    openpulse_blockers = document.get("openpulse_blockers", [])
    openpulse_blocker_text = (
        "\n".join(f"- {item}" for item in openpulse_blockers) if openpulse_blockers else "- none"
    )
    text = f"""<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- © Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- S1 IBM live readiness -->

# S1 IBM Live Readiness

Date: {document["date"]}

Preregistration date: {document["preregistration_date"]}

Backend: `{document["selected_backend"]}`

Submission state: `{document["submission_state"]}`

Hardware submission: `{str(document["hardware_submission"]).lower()}`

Capability status: `{capability["status"]}`

OpenPulse readiness status: `{openpulse_status}`

Readiness status: `{document["readiness_status"]}`

## Transpiled Dynamic-Circuit Payload

| Field | Value |
|---|---:|
| Qubits | {transpiled["n_qubits"]} |
| Classical bits | {transpiled["n_clbits"]} |
| Depth | {transpiled["depth"]} |
| Shots per circuit | {budget["shots_per_circuit"]} |
| Repetitions | {budget["repetitions"]} |
| QPU-second ceiling | {budget["qpu_seconds_ceiling"]} |
| Transpiler seed | {transpiled.get("seed_transpiler", "not recorded")} |

Operation counts:

```json
{json.dumps(transpiled["operation_counts"], indent=2, sort_keys=True)}
```

## Remaining Blockers

{blockers}

## OpenPulse Blockers

{openpulse_blocker_text}

## Claim Boundary

{document["claim_boundary"]}
"""
    path.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Capture the no-submit S1 IBM live-readiness artefacts."""
    args = _parse_args(argv)
    backend = load_authenticated_backend(args.backend, args.instance, args.credentials_vault)
    document = build_live_readiness_document(backend)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    backend_name = document["selected_backend"]
    json_path = args.data_dir / f"s1_ibm_live_readiness_{backend_name}_{CAPTURE_DATE}.json"
    md_path = args.docs_dir / f"s1_ibm_live_readiness_{backend_name}_{CAPTURE_DATE}.md"
    json_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    write_readiness_markdown(document, md_path)
    print(f"wrote_json={json_path}")
    print(f"sha256_json={_sha256(json_path)}")
    print(f"wrote_markdown={md_path}")
    print(f"sha256_markdown={_sha256(md_path)}")
    print("hardware_submission=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
