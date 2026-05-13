#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — IQM partial layout repeat runner
"""Run a collision-free partial IQM layout repeat under a strict credit budget."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_iqm_layout_pinned_dla_minimal as base
from qiskit import transpile

PUBLIC_DIR = base.PUBLIC_DIR
PRIVATE_DIR = base.PRIVATE_DIR

DEFAULT_CIRCUITS = (
    "iqm_dla_pinned_n4_d4_even",
    "iqm_dla_pinned_n4_d4_odd",
    "iqm_dla_pinned_n4_d6_even",
    "iqm_dla_pinned_n4_d6_odd",
    "iqm_readout_pinned_even_0011",
)


def parse_circuit_names(raw: str) -> tuple[str, ...]:
    """Parse a comma-separated non-empty circuit-name list."""
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not names:
        raise ValueError("at least one circuit name is required")
    if len(set(names)) != len(names):
        raise ValueError("circuit names must be unique")
    return names


def select_rows(
    *, layout: tuple[int, int, int, int], shots: int, circuit_names: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Select named rows from the canonical full-block plan."""
    plan = base.build_plan(layout=layout, shots=shots)
    by_name = {str(row["circuit_name"]): row for row in plan}
    missing = [name for name in circuit_names if name not in by_name]
    if missing:
        raise ValueError(f"unknown circuit names: {missing}")
    return [by_name[name] for name in circuit_names]


def run_partial(
    *,
    execute: bool,
    layout: tuple[int, int, int, int],
    shots: int,
    circuit_names: tuple[str, ...],
    fake_backend: str,
    quantum_computer: str,
    label: str,
) -> dict[str, Any]:
    """Dry-run or execute a named partial layout-repeat plan."""
    helper = base._load_helper()
    backend = base._resolve_backend(
        execute=execute, fake_backend=fake_backend, quantum_computer=quantum_computer
    )
    rows = select_rows(layout=layout, shots=shots, circuit_names=circuit_names)
    records: list[dict[str, Any]] = []
    started_all = time.time()
    for index, row in enumerate(rows, start=1):
        circuit = base._build_circuit(helper, row)
        isa = transpile(
            circuit, backend=backend, initial_layout=list(layout), optimization_level=1
        )
        record: dict[str, Any] = {
            **row,
            "backend_name": base._backend_name(backend),
            "transpiled_depth": isa.depth(),
            "transpiled_size": isa.size(),
            "transpiled_ops": {str(key): int(value) for key, value in isa.count_ops().items()},
            "status": "planned" if not execute else "submitted",
        }
        if execute:
            started = time.time()
            job = backend.run([isa], shots=shots)
            raw_job_id = base._job_id(job)
            job_hash = base.hashlib.sha256(raw_job_id.encode("utf-8")).hexdigest()
            result = job.result(timeout=900)
            raw_counts = base._counts(result)
            stats = base.analyse_counts(
                raw_counts,
                initial=str(row["meta"]["initial"]),
                n_qubits=int(row["meta"]["n_qubits"]),
            )
            record.update(
                {
                    **stats,
                    "job_id": raw_job_id,
                    "job_id_sha256": job_hash,
                    "wall_time_s": time.time() - started,
                    "status": "completed",
                }
            )
            print(
                f"completed {index}/{len(rows)} {row['circuit_name']} "
                f"leakage={record['parity_leakage']:.6f} "
                f"job_hash={record['job_id_sha256']}"
            )
        else:
            print(
                f"planned {index}/{len(rows)} {row['circuit_name']} "
                f"depth={record['transpiled_depth']} ops={record['transpiled_ops']}"
            )
        records.append(record)

    return {
        "schema": (
            "scpn_iqm_partial_layout_repeat_v1_private"
            if execute
            else "scpn_iqm_partial_layout_repeat_v1_plan"
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "iqm",
        "platform": "IQM Resonance" if execute else f"IQM fake {fake_backend}",
        "quantum_computer": quantum_computer if execute else None,
        "fake_backend": None if execute else fake_backend,
        "tier": "dla_parity_partial_layout_repeat",
        "execute": execute,
        "label": label,
        "requested_initial_layout": list(layout),
        "selected_circuits": list(circuit_names),
        "total_circuits": len(records),
        "total_shots": sum(int(row["shots"]) for row in records),
        "total_wall_time_s": time.time() - started_all,
        "records": records,
        "claim_boundary": (
            "Partial layout repeat is calibration-time diagnostic evidence only. "
            "It is not a replacement for a complete repeated layout block."
        ),
    }


def _write_outputs(payload: dict[str, Any], *, output_dir: Path) -> tuple[Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "executed" if payload["execute"] else "plan"
    layout_slug = "q" + "-".join(str(qubit) for qubit in payload["requested_initial_layout"])
    label = str(payload["label"]).replace("/", "_")
    public_path = output_dir / f"iqm_partial_layout_repeat_{label}_{layout_slug}_{suffix}.json"
    public_path.write_text(
        json.dumps(base._public_copy(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    private_path = None
    if payload["execute"]:
        private_path = (
            PRIVATE_DIR / f"iqm_partial_layout_repeat_{label}_{layout_slug}_private.json"
        )
        private_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return public_path, private_path


def main() -> int:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="submit real IQM jobs")
    parser.add_argument("--layout", default="2,7,12,13")
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--circuits", default=",".join(DEFAULT_CIRCUITS))
    parser.add_argument("--fake-backend", default="garnet")
    parser.add_argument("--quantum-computer", default="garnet")
    parser.add_argument(
        "--label",
        default=datetime.now(timezone.utc).strftime("calibration_repeat_%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument("--output-dir", type=Path, default=PUBLIC_DIR)
    args = parser.parse_args()

    payload = run_partial(
        execute=args.execute,
        layout=base.parse_layout(args.layout),
        shots=args.shots,
        circuit_names=parse_circuit_names(args.circuits),
        fake_backend=args.fake_backend,
        quantum_computer=args.quantum_computer,
        label=args.label,
    )
    public_path, private_path = _write_outputs(payload, output_dir=args.output_dir)
    print(f"wrote_public={public_path}")
    if private_path is not None:
        print(f"wrote_private={private_path}")
    print(f"total_circuits={payload['total_circuits']}")
    print(f"total_shots={payload['total_shots']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
