#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- hydrate async Phase 3 entanglement IBM jobs
"""Hydrate async Phase 3 entanglement IBM jobs into a reducer-ready artifact."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


def _load_script_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _phase1_module() -> Any:
    return _load_script_module(
        "phase1_mini_bench_ibm_kingston",
        REPO_ROOT / "scripts" / "phase1_mini_bench_ibm_kingston.py",
    )


def _extract_counts_symbol() -> Any:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from scpn_quantum_control.hardware.runner import _extract_counts

    return _extract_counts


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, Mapping):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            return [_json_safe(item) for item in value]
        return str(value)


def _elapsed_seconds(metrics: Mapping[str, Any]) -> float | None:
    timestamps = metrics.get("timestamps")
    if not isinstance(timestamps, Mapping):
        return None
    created = timestamps.get("created")
    finished = timestamps.get("finished")
    if not created or not finished:
        return None
    try:
        start = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
        stop = datetime.fromisoformat(str(finished).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (stop - start).total_seconds()


def _job_metadata(job: Any) -> dict[str, Any]:
    status = job.status()
    status_text = str(status).replace("JobStatus.", "")
    try:
        metrics = _json_safe(job.metrics())
    except Exception as exc:
        metrics = {"error": str(exc)}
    try:
        usage = _json_safe(job.usage_estimation)
    except Exception as exc:
        usage = {"error": str(exc)}
    creation_date = getattr(job, "creation_date", None)
    return {
        "job_id": str(job.job_id()),
        "status": status_text,
        "creation_date": str(creation_date) if creation_date is not None else None,
        "metrics": metrics,
        "usage_estimation": usage,
    }


def _metadata_from_pub_result(pub_result: Any) -> dict[str, Any]:
    metadata = getattr(pub_result, "metadata", None)
    if isinstance(metadata, Mapping):
        return _json_safe(dict(metadata))
    return {}


def _rows_from_result(
    *,
    metas: Sequence[Mapping[str, Any]],
    result: Sequence[Any],
    job_id: str,
    extract_counts: Any,
    metadata_note: str,
) -> list[dict[str, Any]]:
    if len(result) != len(metas):
        raise ValueError(
            f"result length {len(result)} does not match metadata length {len(metas)}"
        )
    rows: list[dict[str, Any]] = []
    for meta, pub_result in zip(metas, result):
        metadata = _metadata_from_pub_result(pub_result)
        metadata.setdefault("depth", None)
        metadata.setdefault("total_gates", None)
        metadata.setdefault("ecr_gates", None)
        metadata.setdefault("recovery_note", metadata_note)
        rows.append(
            {
                "meta": dict(meta),
                "counts": extract_counts(pub_result),
                "job_id": job_id,
                "metadata": metadata,
            }
        )
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    path.write_text(encoded, encoding="utf-8")
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def hydrate_artifact(path: Path) -> tuple[dict[str, Any], str]:
    """Fetch IBM jobs referenced by an async artifact and attach count rows."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    roles = payload.get("pending_job_roles")
    if not isinstance(roles, Mapping) or "main" not in roles or "readout" not in roles:
        raise ValueError("artifact must contain pending_job_roles.main and .readout")
    if payload.get("status") not in {"pending_jobs_submitted", "completed"}:
        raise ValueError(f"unexpected artifact status: {payload.get('status')}")
    metas_main = payload.get("metas_main")
    metas_readout = payload.get("metas_readout")
    if not isinstance(metas_main, list) or not isinstance(metas_readout, list):
        raise ValueError("artifact must contain metas_main and metas_readout lists")

    phase1 = _phase1_module()
    credential_value, instance = phase1.parse_vault(
        Path("/media/anulum/724AA8E84AA8AA75/agentic-shared/CREDENTIALS.md")
    )
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(
        channel="ibm_cloud",
        token=credential_value,
        instance=instance,
    )
    extract_counts = _extract_counts_symbol()
    jobs = {role: service.job(str(job_id)) for role, job_id in roles.items()}
    metadata = {role: _job_metadata(job) for role, job in jobs.items()}
    incomplete = {
        role: item["status"]
        for role, item in metadata.items()
        if str(item["status"]).replace("JobStatus.", "") != "DONE"
    }
    if incomplete:
        raise RuntimeError(f"jobs are not all DONE: {incomplete}")

    main_result = jobs["main"].result()
    readout_result = jobs["readout"].result()
    main_rows = _rows_from_result(
        metas=metas_main,
        result=main_result,
        job_id=str(roles["main"]),
        extract_counts=extract_counts,
        metadata_note="async artifact did not store transpiled-circuit metadata",
    )
    readout_rows = _rows_from_result(
        metas=metas_readout,
        result=readout_result,
        job_id=str(roles["readout"]),
        extract_counts=extract_counts,
        metadata_note="async artifact did not store transpiled-circuit metadata",
    )
    payload.update(
        {
            "status": "completed",
            "downloaded_utc": _timestamp(),
            "ibm_job_metadata": metadata,
            "wall_time_main_s": _elapsed_seconds(metadata["main"].get("metrics", {})),
            "wall_time_readout_s": _elapsed_seconds(metadata["readout"].get("metrics", {})),
            "circuits": main_rows + readout_rows,
            "recovery_note": (
                "counts were hydrated after async submission; per-circuit "
                "transpiled metadata is unavailable unless present in IBM pub metadata"
            ),
        }
    )
    sha = _write_json(path, payload)
    return payload, sha


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    return parser.parse_args()


def main() -> int:
    """Run the command-line entry point."""
    args = parse_args()
    artifact = args.artifact.resolve()
    payload, sha = hydrate_artifact(artifact)
    print(f"Hydrated: {artifact.relative_to(REPO_ROOT)}")
    print(f"SHA256: {sha}")
    print(f"Status: {payload['status']}")
    print(f"Circuits: {len(payload['circuits'])}")
    print(f"Jobs: {payload['pending_job_roles']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
