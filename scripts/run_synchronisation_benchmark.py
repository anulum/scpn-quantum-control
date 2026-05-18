#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- synchronisation benchmark runner
"""Run no-QPU standardised synchronisation benchmark rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from scpn_quantum_control.benchmark_harness.synchronisation_runner import (
    CHAIN_N8_BENCHMARK_ID,
    RING_N4_BENCHMARK_ID,
    run_kuramoto_chain_n8_decay_omega,
    run_kuramoto_ring_n4_linear_omega,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "synchronisation_benchmarks"
DOC_PATH = REPO_ROOT / "docs" / "synchronisation_benchmark_kuramoto_ring_n4.md"


def current_commit() -> str:
    """Return the current Git commit or unknown when unavailable."""

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def write_json(path: Path, payload: dict[str, Any]) -> str:
    """Write deterministic JSON and return its SHA-256 digest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8")
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def markdown(payload: dict[str, Any]) -> str:
    """Render a public benchmark result summary."""

    lines = [
        "# Kuramoto Ring n=4 Synchronisation Benchmark",
        "",
        "This no-QPU artefact records schema-compatible reference rows for the canonical four-node Kuramoto-XY ring benchmark.",
        "",
        f"Benchmark ID: `{payload['benchmark_id']}`",
        "",
        "| Backend | Observable | Value | Tolerance | Passed |",
        "|---|---|---:|---:|---|",
    ]
    for row in payload["rows"]:
        for obs in row["observables"]:
            lines.append(
                "| `{backend}` | `{name}` | `{value:.12g}` | `{tolerance:.3g}` | `{passed}` |".format(
                    backend=row["backend"],
                    name=obs["name"],
                    value=obs["value"],
                    tolerance=obs["tolerance"],
                    passed=obs["passed"],
                )
            )
    lines.extend(["", "## Claim boundary", "", str(payload["claim_boundary"])])
    return "\n".join(lines) + "\n"


def write_text(path: Path, text: str) -> str:
    """Write text and return its SHA-256 digest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    """Run the selected no-QPU synchronisation benchmark."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-id",
        default=RING_N4_BENCHMARK_ID,
        choices=(RING_N4_BENCHMARK_ID, CHAIN_N8_BENCHMARK_ID),
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DOC_PATH)
    args = parser.parse_args()

    command = f"scpn-bench sync-benchmark-run --benchmark-id {args.benchmark_id}"
    if args.benchmark_id == RING_N4_BENCHMARK_ID:
        payload = run_kuramoto_ring_n4_linear_omega(command=command, commit=current_commit())
        doc_path = args.doc_path
    else:
        payload = run_kuramoto_chain_n8_decay_omega(command=command, commit=current_commit())
        doc_path = (
            args.doc_path
            if args.doc_path != DOC_PATH
            else REPO_ROOT / "docs" / "synchronisation_benchmark_kuramoto_chain_n8.md"
        )
    json_path = args.out_dir / f"{args.benchmark_id}_reference_rows.json"
    json_digest = write_json(json_path, payload)
    doc_digest = write_text(doc_path, markdown(payload))
    print(f"wrote {json_path.relative_to(REPO_ROOT)} sha256={json_digest}")
    print(f"wrote {doc_path.relative_to(REPO_ROOT)} sha256={doc_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
