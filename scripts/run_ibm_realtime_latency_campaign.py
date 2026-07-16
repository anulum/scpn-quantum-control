#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — run IBM realtime latency campaign script
"""Execute dedicated realtime-control latency campaign via Rust IBM runner."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE1_VAULT_PARSER = REPO_ROOT / "scripts" / "phase1_mini_bench_ibm_kingston.py"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="ibm_kingston")
    parser.add_argument(
        "--runtime-url",
        default="https://quantum.cloud.ibm.com/api/v1",
    )
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--timeout-s", type=int, default=2400)
    parser.add_argument("--poll-interval-s", type=float, default=2.0)
    parser.add_argument(
        "--credentials-vault",
        type=Path,
        default=Path("~/.config/scpn-quantum-control/credentials.md").expanduser(),
    )
    return parser.parse_args(argv)


def _load_vault_credentials(path: Path) -> tuple[str, str]:
    spec = importlib.util.spec_from_file_location(
        "phase1_mini_bench_ibm_kingston", PHASE1_VAULT_PARSER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {PHASE1_VAULT_PARSER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    token, instance = module.parse_vault(path)
    if not token or not instance:
        raise RuntimeError("missing IBM token/instance in credentials vault")
    return token, instance


def _latest_matrix_path() -> Path:
    candidates = sorted(
        (REPO_ROOT / "data" / "realtime_control_latency").glob(
            "ibm_runtime_realtime_payload_matrix_*.json"
        )
    )
    if not candidates:
        raise RuntimeError("no realtime payload matrix found")
    return candidates[-1]


def main(argv: Sequence[str] | None = None) -> int:
    """Build and execute dedicated realtime IBM latency campaign."""
    args = _parse_args(argv)
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_ibm_runtime_realtime_payload_matrix.py"),
            "--backend",
            args.backend,
            "--shots",
            str(args.shots),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    matrix_path = _latest_matrix_path()
    output_path = (
        REPO_ROOT
        / "data"
        / "realtime_control_latency"
        / "ibm_runtime_realtime_rust_latency_run_2026-05-22.json"
    )
    token, instance = _load_vault_credentials(args.credentials_vault)
    env = os.environ.copy()
    env["IBM_API_KEY"] = token
    env["IBM_INSTANCE_CRN"] = instance
    subprocess.run(
        [
            "cargo",
            "run",
            "--manifest-path",
            str(REPO_ROOT / "scpn_quantum_engine" / "Cargo.toml"),
            "--features",
            "latency-runner",
            "--bin",
            "ibm_runtime_latency_runner",
            "--",
            "--payload-matrix",
            str(matrix_path),
            "--output",
            str(output_path),
            "--runtime-url",
            args.runtime_url,
            "--timeout-s",
            str(args.timeout_s),
            "--poll-interval-s",
            str(args.poll_interval_s),
        ],
        cwd=REPO_ROOT,
        check=True,
        env=env,
    )
    print(f"matrix_json={matrix_path}")
    print(f"run_json={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
