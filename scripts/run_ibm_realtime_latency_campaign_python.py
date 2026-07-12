#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — run IBM realtime latency campaign python script
"""Execute dedicated realtime-control latency campaign via Python IBM runner."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from prepare_s1_ibm_live_readiness import DEFAULT_CREDENTIALS_VAULT, load_authenticated_backend

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "realtime_control_latency"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="ibm_kingston")
    parser.add_argument("--instance")
    parser.add_argument("--credentials-vault", type=Path, default=DEFAULT_CREDENTIALS_VAULT)
    parser.add_argument("--matrix-json", type=Path)
    parser.add_argument("--timeout-s", type=float, default=2400.0)
    parser.add_argument("--poll-interval-s", type=float, default=2.0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=OUT_DIR / "ibm_runtime_realtime_python_latency_run_2026-05-22.json",
    )
    return parser.parse_args(argv)


def _latest_matrix_path() -> Path:
    candidates = sorted(OUT_DIR.glob("ibm_runtime_realtime_payload_matrix_*.json"))
    if not candidates:
        raise RuntimeError("no realtime payload matrix found")
    return candidates[-1]


def _terminal(status_text: str) -> bool:
    return status_text.upper() in {"DONE", "COMPLETED", "SUCCESS", "ERROR", "FAILED", "CANCELLED"}


def main(argv: Sequence[str] | None = None) -> int:
    """Run dedicated realtime matrix with Python orchestration and persist report."""
    args = _parse_args(argv)
    matrix_path = args.matrix_json or _latest_matrix_path()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("schema") != "scpn_ibm_runtime_realtime_payload_matrix_v1":
        raise RuntimeError(f"unexpected schema: {matrix.get('schema')!r}")

    backend = load_authenticated_backend(args.backend, args.instance, args.credentials_vault)
    service = backend.service
    api = service._active_api_client

    rows_out: list[dict[str, Any]] = []
    for row in matrix["rows"]:
        payload = row["payload"]
        params = payload["params"]
        backend_name = payload["backend"]

        submit_started = time.monotonic()
        response = api.program_run(
            program_id="sampler",
            backend_name=backend_name,
            params=params,
            image=None,
            log_level=None,
            session_id=None,
            job_tags=None,
            max_execution_time=None,
            start_session=False,
            session_time=None,
            private=False,
            calibration_id=None,
        )
        submit_finished = time.monotonic()
        job_id = str(response["id"])

        final_status = "UNKNOWN"
        while True:
            if (time.monotonic() - submit_started) > args.timeout_s:
                raise TimeoutError(f"job {job_id} timed out")
            state = api.job_get(job_id, exclude_params=True)
            status_text = str(state.get("state", {}).get("status", state.get("status", "UNKNOWN")))
            if _terminal(status_text):
                final_status = status_text
                break
            time.sleep(max(0.1, args.poll_interval_s))

        rows_out.append(
            {
                "lane": row["lane"],
                "scenario": row["scenario"],
                "job_id": job_id,
                "submit_overhead_seconds": submit_finished - submit_started,
                "submit_to_done_seconds": time.monotonic() - submit_started,
                "final_status": final_status,
            }
        )

    report = {
        "schema": "scpn_ibm_runtime_python_realtime_latency_run_v1",
        "backend": matrix["backend"],
        "runner": "python_ibm_runtime_program_run",
        "source_matrix": str(matrix_path),
        "rows": rows_out,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"matrix_json={matrix_path}")
    print(f"run_json={args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
