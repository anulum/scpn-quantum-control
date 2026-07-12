#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — retrieve methods ansatz IBM validation script
# scpn-quantum-control -- Retrieve IBM ansatz-validation methods lane
"""Retrieve and reduce a completed IBM ansatz-validation methods job."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from submit_methods_ansatz_ibm_validation import _analyse_raw_rows

from scpn_quantum_control.hardware.runner import _extract_counts

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "rust_vqe_methods"
DEFAULT_CREDENTIALS_VAULT = Path("~/.config/scpn-quantum-control/credentials.md").expanduser()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve a completed IBM methods ansatz-validation job."
    )
    parser.add_argument("--submission-json", type=Path, required=True)
    parser.add_argument("--credentials-vault", type=Path, default=DEFAULT_CREDENTIALS_VAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args(argv)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return _sha256(path)


def _extract_status(job: Any) -> str:
    status = job.status()
    return str(getattr(status, "name", status))


def _entry_rows_from_submission(submission: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, circuit in enumerate(submission["transpiled_circuits"]):
        rows.append(
            {
                "index": index,
                "role": circuit["role"],
                "ansatz": circuit["ansatz"],
                "basis": circuit["basis"],
                "repetition": circuit["repetition"],
                "calibration_state": circuit["calibration_state"],
            }
        )
    return rows


def retrieve_completed_job(
    *,
    submission: Mapping[str, Any],
    credentials_vault: Path,
) -> tuple[str, list[dict[str, Any]]]:
    """Retrieve raw counts for a completed validation job."""
    token, instance = _parse_vault(credentials_vault)
    from qiskit_ibm_runtime import QiskitRuntimeService

    service_kwargs: dict[str, str] = {"channel": "ibm_cloud"}
    if token:
        service_kwargs["token"] = token
    if instance:
        service_kwargs["instance"] = instance
    service = QiskitRuntimeService(**service_kwargs)
    job_id = str(submission["job_ids"][0])
    job = service.job(job_id)
    status = _extract_status(job)
    if status not in {"DONE", "JobStatus.DONE"}:
        return status, []
    result = job.result()
    metadata_rows = _entry_rows_from_submission(submission)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(metadata_rows):
        pub_result = result[index]
        rows.append({**row, "counts": dict(_extract_counts(pub_result))})
    return status, rows


def main(argv: Sequence[str] | None = None) -> int:
    """Retrieve a completed methods ansatz-validation IBM job if available."""
    args = _parse_args(argv)
    submission_json = args.submission_json.resolve()
    submission = json.loads(submission_json.read_text(encoding="utf-8"))
    status, rows = retrieve_completed_job(
        submission=submission, credentials_vault=args.credentials_vault
    )
    print(f"job_status={status}")
    if not rows:
        print("raw_counts_available=false")
        return 2
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_payload = {
        "schema": "scpn_rust_vqe_methods_ansatz_ibm_raw_counts_v1",
        "experiment_id": submission["experiment_id"],
        "backend": submission["backend"],
        "timestamp_utc": timestamp,
        "job_ids": submission["job_ids"],
        "submission_json": str(submission_json.relative_to(REPO_ROOT)),
        "submission_sha256": _sha256(submission_json),
        "logical_to_physical_layout": submission["logical_to_physical_layout"],
        "shots": submission["shots"],
        "term_groups": submission["term_groups"],
        "circuits": rows,
    }
    raw_path = (
        args.out_dir / f"ansatz_ibm_validation_raw_counts_{submission['backend']}_{timestamp}.json"
    )
    raw_sha = _write_json(raw_path, raw_payload)
    analysis = _analyse_raw_rows(rows, submission["term_groups"])
    analysis["raw_counts_json"] = str(raw_path.relative_to(REPO_ROOT))
    analysis["raw_counts_sha256"] = raw_sha
    analysis_path = (
        args.out_dir / f"ansatz_ibm_validation_analysis_{submission['backend']}_{timestamp}.json"
    )
    analysis_sha = _write_json(analysis_path, analysis)
    print("raw_counts_available=true")
    print(f"raw_counts_json={raw_path}")
    print(f"raw_counts_sha256={raw_sha}")
    print(f"analysis_json={analysis_path}")
    print(f"analysis_sha256={analysis_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
