#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — March 2026 flagship retrieval receipts
"""Retrieve dated IBM receipts for the committed March 2026 job identifiers.

Companion to ``scripts/build_march_job_id_commitments.py``: for every public
label in the committed commitment file, the raw IBM job identifier is
resolved from the private mapping, each commitment digest is recomputed and
cross-checked against the private nonce sidecar (fail closed on any
mismatch), and the job metadata is then retrieved read-only from the IBM
Quantum service. The receipt records what IBM reports today — backend,
status, creation date, and usage seconds — bound to the public label and
commitment digest, never to the raw identifier.

Retrieval is free (no quantum seconds are consumed). Failures are recorded
per job with the exception class name only, because provider error messages
can embed the raw job identifier. A fail-closed leak check scans the
serialised payload for every raw identifier, nonce, and the API token
before anything is written.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_march_job_id_commitments import (  # noqa: E402
    COMMITMENT_SCHEMA,
    DEFAULT_NONCE_SIDECAR,
    DEFAULT_PRIVATE_MAPPING,
    NONCE_SIDECAR_SCHEMA,
    commitment_digest,
    label_by_raw,
    load_private_mapping,
    raw_job_id_entries,
)
from scripts.build_march_job_id_commitments import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_COMMITMENTS,
)

RECEIPTS_SCHEMA = "scpn.march_flagship_retrieval_receipts.v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "march_flagship_verifiability"


def load_commitments(path: Path) -> dict[str, Any]:
    """Load and validate the public commitment file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != COMMITMENT_SCHEMA:
        raise ValueError(f"{path} is not a {COMMITMENT_SCHEMA} document")
    if not isinstance(payload.get("commitments"), list) or not payload["commitments"]:
        raise ValueError(f"{path} carries no commitments")
    return payload


def load_nonces(path: Path) -> dict[str, str]:
    """Load the private nonce sidecar."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != NONCE_SIDECAR_SCHEMA or not isinstance(
        payload.get("nonces"), dict
    ):
        raise ValueError(f"nonce sidecar {path} is malformed")
    return dict(payload["nonces"])


def resolve_raw_ids(
    commitments: Mapping[str, Any],
    manifest: Mapping[str, Any],
    nonces: Mapping[str, str],
) -> dict[str, str]:
    """Map each committed public label to its raw id, verifying every digest."""
    labels = label_by_raw(raw_job_id_entries(manifest))
    raw_by_label = {label: raw for raw, label in labels.items()}
    resolved: dict[str, str] = {}
    for commitment in commitments["commitments"]:
        label = commitment["public_label"]
        raw = raw_by_label.get(label)
        if raw is None:
            raise ValueError(f"committed label {label} is absent from the private mapping")
        nonce = nonces.get(raw)
        if nonce is None:
            raise ValueError(f"committed label {label} has no nonce in the private sidecar")
        if commitment_digest(raw, nonce) != commitment["commitment_sha256"]:
            raise ValueError(f"commitment digest mismatch for {label}: refusing to retrieve")
        resolved[label] = raw
    return resolved


def load_runtime_service(instance: str | None, credentials_vault: Path | None) -> Any:
    """Authenticate a read-only Qiskit Runtime service from the credentials vault."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:  # pragma: no cover - exercised via injected stubs
        raise RuntimeError("qiskit-ibm-runtime is required for receipt retrieval") from exc
    from scripts.prepare_s1_ibm_live_readiness import _parse_vault

    credential_value, vault_instance = (
        _parse_vault(credentials_vault) if credentials_vault is not None else (None, None)
    )
    selected_instance = instance or vault_instance
    service_kwargs: dict[str, Any] = {"channel": "ibm_cloud"} if credential_value else {}
    if credential_value:
        service_kwargs["token"] = credential_value
    if selected_instance:
        service_kwargs["instance"] = selected_instance
    return QiskitRuntimeService(**service_kwargs)


def _usage_seconds(job: Any) -> float | None:
    """Extract IBM-reported usage seconds from job metrics, if available."""
    try:
        metrics = job.metrics()
    except Exception:
        return None
    usage = metrics.get("usage", {}) if isinstance(metrics, Mapping) else {}
    seconds = usage.get("seconds") if isinstance(usage, Mapping) else None
    if isinstance(seconds, (int, float)):
        return float(seconds)
    quantum_seconds = usage.get("quantum_seconds") if isinstance(usage, Mapping) else None
    if isinstance(quantum_seconds, (int, float)):
        return float(quantum_seconds)
    return None


def job_receipt(
    service: Any, public_label: str, commitment_sha256: str, raw_id: str
) -> dict[str, Any]:
    """Retrieve one job and return its public receipt (raw id never recorded)."""
    try:
        job = service.job(raw_id)
        backend = job.backend()
        creation = job.creation_date
        return {
            "public_label": public_label,
            "commitment_sha256": commitment_sha256,
            "retrieval": "ok",
            "backend": getattr(backend, "name", str(backend)),
            "status": str(job.status()),
            "creation_date_utc": creation.isoformat() if creation is not None else None,
            "usage_seconds": _usage_seconds(job),
        }
    except Exception as exc:
        return {
            "public_label": public_label,
            "commitment_sha256": commitment_sha256,
            "retrieval": "error",
            "error_type": type(exc).__name__,
        }


def build_receipts(
    service: Any,
    commitments: Mapping[str, Any],
    resolved: Mapping[str, str],
    commitments_sha256: str,
    retrieved_at_utc: str,
) -> dict[str, Any]:
    """Assemble the public receipts payload for every committed label."""
    receipts = [
        job_receipt(
            service,
            commitment["public_label"],
            commitment["commitment_sha256"],
            resolved[commitment["public_label"]],
        )
        for commitment in commitments["commitments"]
    ]
    receipts.sort(key=lambda r: r["public_label"])
    return {
        "schema": RECEIPTS_SCHEMA,
        "retrieved_at_utc": retrieved_at_utc,
        "commitments_schema": commitments["schema"],
        "commitments_sha256": commitments_sha256,
        "receipt_count": len(receipts),
        "ok_count": sum(1 for r in receipts if r["retrieval"] == "ok"),
        "receipts": receipts,
    }


def assert_no_private_leak(
    serialised: str,
    resolved: Mapping[str, str],
    nonces: Mapping[str, str],
    token: str | None,
) -> None:
    """Fail closed if a raw id, nonce, or the API token reaches the public text."""
    for label, raw in resolved.items():
        if raw in serialised:
            raise RuntimeError(f"raw job id for {label} leaked into the receipts payload")
    for nonce in nonces.values():
        if nonce in serialised:
            raise RuntimeError("a commitment nonce leaked into the receipts payload")
    if token and token in serialised:
        raise RuntimeError("the API token leaked into the receipts payload")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commitments", type=Path, default=DEFAULT_COMMITMENTS)
    parser.add_argument("--private-mapping", type=Path, default=DEFAULT_PRIVATE_MAPPING)
    parser.add_argument("--nonces", type=Path, default=DEFAULT_NONCE_SIDECAR)
    parser.add_argument("--credentials-vault", type=Path)
    parser.add_argument("--instance")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point."""
    args = _parse_args(argv)
    commitments = load_commitments(args.commitments)
    manifest = load_private_mapping(args.private_mapping)
    nonces = load_nonces(args.nonces)
    resolved = resolve_raw_ids(commitments, manifest, nonces)

    from scripts.prepare_s1_ibm_live_readiness import DEFAULT_CREDENTIALS_VAULT

    vault = args.credentials_vault or DEFAULT_CREDENTIALS_VAULT
    service = load_runtime_service(args.instance, vault)
    token = getattr(getattr(service, "_account", None), "token", None)

    retrieved_at = datetime.now(timezone.utc)
    payload = build_receipts(
        service,
        commitments,
        resolved,
        _sha256_file(args.commitments),
        retrieved_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    serialised = json.dumps(payload, indent=2)
    assert_no_private_leak(serialised, resolved, nonces, token)
    stamp = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir / f"march_retrieval_receipts_{stamp}.json"
    digest = _write_json(output, payload)
    print(f"receipts: {payload['ok_count']}/{payload['receipt_count']} retrieved ok")
    print(f"output: {output}")
    print(f"output sha256: {digest}")
    return 0 if payload["ok_count"] == payload["receipt_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
