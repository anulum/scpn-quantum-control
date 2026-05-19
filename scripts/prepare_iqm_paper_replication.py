#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — IQM paper-replication readiness
"""Prepare a no-submit IQM replication package for the IBM paper runs.

The generated artefacts are readiness manifests only. The script builds the
paper-critical Phase 1 DLA/parity circuits and the SCPN/FIM negative-control
subset, transpiles them against an IQM fake backend when available, and records
the smallest real-hardware ladder to spend credits safely. It never reads the
IQM token, creates a remote IQM provider, or submits a QPU job.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from phase1_mini_bench_ibm_kingston import (  # noqa: E402
    T_STEP,
    build_readout_baseline_circuit,
    build_xy_trotter_circuit,
)
from prepare_fim_ibm_circuits import (  # noqa: E402
    build_fim_trotter_circuit,
    build_readout_circuit,
)

from scpn_quantum_control.hardware.iqm_backend import (  # noqa: E402
    IQMBackendConfig,
    IQMQuantumBackend,
)

DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "iqm_paper_replication"
FIM_PROTOCOL_PATH = (
    REPO_ROOT / "data" / "scpn_fim_hamiltonian" / "fim_ibm_candidate_protocol_2026-05-05.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bell_smoke_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    circuit.name = "iqm_smoke_bell_2q"
    return circuit


def _add_row(
    rows: list[dict[str, Any]],
    *,
    tier: str,
    priority: int,
    source_claim: str,
    meta: dict[str, Any],
    circuit: QuantumCircuit,
    shots: int,
) -> None:
    rows.append(
        {
            "tier": tier,
            "priority": priority,
            "source_claim": source_claim,
            "circuit_name": circuit.name,
            "n_qubits": circuit.num_qubits,
            "shots": int(shots),
            "raw_depth": circuit.depth(),
            "raw_size": circuit.size(),
            "raw_ops": json.dumps(
                {name: int(count) for name, count in circuit.count_ops().items()},
                sort_keys=True,
            ),
            "meta": meta,
            "_circuit": circuit,
        }
    )


def _build_raw_rows(smoke_shots: int, replication_shots: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    _add_row(
        rows,
        tier="smoke_account_probe",
        priority=0,
        source_claim="IQM account/backend/count-format smoke; not a paper claim.",
        meta={"experiment": "iqm_smoke_bell", "n_qubits": 2},
        circuit=_bell_smoke_circuit(),
        shots=smoke_shots,
    )

    for depth in (4, 6, 10):
        for sector_name, initial in {"even": "0011", "odd": "0001"}.items():
            circuit = build_xy_trotter_circuit(4, initial, depth, T_STEP)
            circuit.name = f"iqm_dla_min_n4_d{depth}_{sector_name}"
            _add_row(
                rows,
                tier="dla_parity_minimal",
                priority=1,
                source_claim="Phase 1 DLA/parity IBM paper minimal cross-provider replication.",
                meta={
                    "experiment": "A_dla_parity_n4",
                    "n_qubits": 4,
                    "depth": depth,
                    "sector": sector_name,
                    "initial": initial,
                    "t_step": T_STEP,
                    "paper_source": "phase1_dla_parity",
                },
                circuit=circuit,
                shots=replication_shots,
            )

    for depth in (2, 4, 6, 8, 10, 14, 20, 30):
        for sector_name, initial in {"even": "0011", "odd": "0001"}.items():
            circuit = build_xy_trotter_circuit(4, initial, depth, T_STEP)
            circuit.name = f"iqm_dla_core_n4_d{depth}_{sector_name}"
            _add_row(
                rows,
                tier="dla_parity_paper_core",
                priority=2,
                source_claim="Phase 1 DLA/parity IBM paper one-replicate depth grid.",
                meta={
                    "experiment": "A_dla_parity_n4",
                    "n_qubits": 4,
                    "depth": depth,
                    "sector": sector_name,
                    "initial": initial,
                    "t_step": T_STEP,
                    "paper_source": "phase1_dla_parity",
                },
                circuit=circuit,
                shots=replication_shots,
            )

    fim_rows = _selected_fim_protocol_rows()
    for index, fim_row in enumerate(fim_rows):
        circuit = build_fim_trotter_circuit(
            str(fim_row["initial_bitstring"]),
            int(fim_row["depth"]),
            float(fim_row["lambda_fim"]),
        )
        circuit.name = (
            f"iqm_fim_min_{index:02d}_l{fim_row['lambda_fim']}_"
            f"d{fim_row['depth']}_{fim_row['initial_bitstring']}"
        )
        _add_row(
            rows,
            tier="fim_negative_control_minimal",
            priority=3,
            source_claim="SCPN/FIM IBM negative-control paper minimal cross-provider check.",
            meta={**fim_row, "paper_source": "scpn_fim_hamiltonian"},
            circuit=circuit,
            shots=replication_shots,
        )

    for index in range(16):
        initial = format(index, "04b")
        circuit = build_readout_circuit(initial)
        circuit.name = f"iqm_readout_full_basis_{initial}"
        _add_row(
            rows,
            tier="readout_full_basis_optional",
            priority=4,
            source_claim="Optional n=4 full-basis readout calibration for publishable IQM correction.",
            meta={
                "experiment": "fim_readout_full_basis",
                "n_qubits": 4,
                "initial": initial,
                "paper_source": "scpn_fim_hamiltonian",
            },
            circuit=circuit,
            shots=replication_shots,
        )

    readout = build_readout_baseline_circuit(4, "0011")
    readout.name = "iqm_dla_readout_even_0011"
    _add_row(
        rows,
        tier="dla_readout_baseline_optional",
        priority=5,
        source_claim="Optional Phase 1 DLA readout baseline for same initial state as even sector.",
        meta={"experiment": "C_readout_baseline", "n_qubits": 4, "initial": "0011"},
        circuit=readout,
        shots=replication_shots,
    )

    return rows


def _selected_fim_protocol_rows() -> list[dict[str, Any]]:
    protocol = json.loads(FIM_PROTOCOL_PATH.read_text(encoding="utf-8"))
    selected: list[dict[str, Any]] = []
    for row in protocol["rows"]:
        if row["protocol_arm"] != "fim_sector_survival_pilot":
            continue
        if float(row["lambda_fim"]) not in {0.0, 4.0}:
            continue
        if int(row["depth"]) not in {2, 4}:
            continue
        if str(row["initial_bitstring"]) not in {"0000", "0011", "1111"}:
            continue
        selected.append(dict(row))
    return selected


def _transpile_rows(
    rows: list[dict[str, Any]],
    *,
    fake_backend: str,
    iqm_python: Path | None = None,
) -> tuple[str, str, list[dict[str, Any]]]:
    if iqm_python is not None:
        return _transpile_rows_subprocess(rows, fake_backend=fake_backend, iqm_python=iqm_python)

    adapter = IQMQuantumBackend()
    config = IQMBackendConfig(mode="fake", fake_backend=fake_backend)
    status = "iqm_fake_transpile_passed"
    blocked_reason = ""
    public_rows: list[dict[str, Any]] = []

    for row in rows:
        circuit = row.pop("_circuit")
        public_row = dict(row)
        try:
            transpiled = adapter.transpile_circuit(circuit, config)
            public_row.update(
                {
                    "iqm_fake_backend": fake_backend,
                    "iqm_fake_status": "passed",
                    "iqm_transpiled_depth": transpiled.depth(),
                    "iqm_transpiled_size": transpiled.size(),
                    "iqm_transpiled_ops": json.dumps(
                        {name: int(count) for name, count in transpiled.count_ops().items()},
                        sort_keys=True,
                    ),
                }
            )
        except ImportError as exc:
            status = "blocked_missing_iqm_dependency_or_fake_backend"
            blocked_reason = str(exc)
            public_row.update(
                {
                    "iqm_fake_backend": fake_backend,
                    "iqm_fake_status": "blocked",
                    "iqm_transpiled_depth": None,
                    "iqm_transpiled_size": None,
                    "iqm_transpiled_ops": "{}",
                }
            )
        public_rows.append(public_row)

    return status, blocked_reason, public_rows


def _transpile_rows_subprocess(
    rows: list[dict[str, Any]],
    *,
    fake_backend: str,
    iqm_python: Path,
) -> tuple[str, str, list[dict[str, Any]]]:
    public_rows = []
    payload_rows = []
    for row in rows:
        row.pop("_circuit")
        public_rows.append(dict(row))
        payload_rows.append({"circuit_name": row["circuit_name"], "meta": row["meta"]})

    helper = REPO_ROOT / "scripts" / "iqm_fake_transpile_payload.py"
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", encoding="utf-8", delete=False
    ) as handle:
        json.dump({"circuits": payload_rows}, handle)
        payload_path = Path(handle.name)
    try:
        completed = subprocess.run(
            [
                str(iqm_python),
                str(helper),
                "--input",
                str(payload_path),
                "--fake-backend",
                fake_backend,
            ],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
    finally:
        payload_path.unlink(missing_ok=True)

    if completed.returncode != 0:
        reason = (completed.stderr or completed.stdout or "IQM subprocess failed").strip()
        for row in public_rows:
            row.update(
                {
                    "iqm_fake_backend": fake_backend,
                    "iqm_fake_status": "blocked",
                    "iqm_transpiled_depth": None,
                    "iqm_transpiled_size": None,
                    "iqm_transpiled_ops": "{}",
                }
            )
        return "blocked_missing_iqm_dependency_or_fake_backend", reason, public_rows

    decoded = json.loads(completed.stdout)
    by_name = {row["circuit_name"]: row for row in decoded["rows"]}
    for row in public_rows:
        row.update(by_name[row["circuit_name"]])
    return "iqm_fake_transpile_passed", "", public_rows


def _summarise_tiers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for tier in sorted(
        {row["tier"] for row in rows},
        key=lambda t: min(r["priority"] for r in rows if r["tier"] == t),
    ):
        tier_rows = [row for row in rows if row["tier"] == tier]
        depths = [
            int(row["iqm_transpiled_depth"])
            for row in tier_rows
            if row.get("iqm_transpiled_depth") is not None
        ]
        summaries.append(
            {
                "tier": tier,
                "priority": min(int(row["priority"]) for row in tier_rows),
                "circuits": len(tier_rows),
                "shots": sum(int(row["shots"]) for row in tier_rows),
                "max_n_qubits": max(int(row["n_qubits"]) for row in tier_rows),
                "max_iqm_transpiled_depth": max(depths) if depths else None,
                "status": "passed"
                if all(row["iqm_fake_status"] == "passed" for row in tier_rows)
                else "blocked",
            }
        )
    return summaries


def generate(
    *,
    fake_backend: str = "garnet",
    smoke_shots: int = 128,
    replication_shots: int = 256,
    iqm_python: Path | None = None,
) -> dict[str, Any]:
    """Build the no-submit IQM paper-replication manifest."""
    raw_rows = _build_raw_rows(smoke_shots, replication_shots)
    status, blocked_reason, rows = _transpile_rows(
        raw_rows,
        fake_backend=fake_backend,
        iqm_python=iqm_python,
    )
    tier_summaries = _summarise_tiers(rows)
    return {
        "schema": "scpn_iqm_paper_replication_readiness_v1",
        "date": DATE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python scripts/prepare_iqm_paper_replication.py",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "provider": "iqm",
        "iqm_url": "https://resonance.iqm.tech/",
        "fake_backend": fake_backend,
        "submission_status": "not_submitted",
        "real_qpu_spend_authorised": False,
        "remote_provider_constructed": False,
        "iqm_python": str(iqm_python) if iqm_python else "",
        "transpile_status": status,
        "blocked_reason": blocked_reason,
        "total_circuits": len(rows),
        "total_shots": sum(int(row["shots"]) for row in rows),
        "tier_summaries": tier_summaries,
        "recommended_first_real_run": {
            "tier": "smoke_account_probe",
            "circuits": 1,
            "shots": smoke_shots,
            "stop_rule": (
                "Submit no further IQM jobs until the dashboard/resource calculator "
                "confirms actual credits consumed by this smoke job."
            ),
        },
        "credit_policy": (
            "Thirty IQM credits are treated as a micro-replication budget. "
            "Run smoke first, then DLA minimal, then DLA paper core, then FIM "
            "minimal, and spend on full-basis readout only if measured credit "
            "burn is acceptable."
        ),
        "claim_boundary": (
            "This readiness artefact does not contact IQM, does not spend credits, "
            "and does not create cross-provider hardware evidence. It prepares "
            "the exact paper-critical circuit families for later approved IQM runs."
        ),
        "rows": rows,
    }


def _write_markdown(path: Path, manifest: dict[str, Any], json_path: Path, csv_path: Path) -> None:
    def display_path(target: Path) -> str:
        try:
            return str(target.relative_to(REPO_ROOT))
        except ValueError:
            return str(target)

    lines = [
        "# IQM Paper Replication Readiness",
        "",
        "No IQM service was contacted. No QPU job was submitted. No credits were spent.",
        "",
        f"- JSON: `{display_path(json_path)}`",
        f"- CSV: `{display_path(csv_path)}`",
        f"- Provider: `{manifest['provider']}`",
        f"- Fake backend: `{manifest['fake_backend']}`",
        f"- Transpile status: `{manifest['transpile_status']}`",
        f"- Total circuits across all prepared tiers: `{manifest['total_circuits']}`",
        f"- Total planned shots across all prepared tiers: `{manifest['total_shots']}`",
        "",
        "## Tier Plan",
        "",
        "| Priority | Tier | Circuits | Shots | Status | Max fake depth |",
        "|---:|---|---:|---:|---|---:|",
    ]
    for row in manifest["tier_summaries"]:
        lines.append(
            f"| {row['priority']} | `{row['tier']}` | {row['circuits']} | "
            f"{row['shots']} | `{row['status']}` | {row['max_iqm_transpiled_depth']} |"
        )
    lines.extend(
        [
            "",
            "## Spend Rule",
            "",
            manifest["credit_policy"],
            "",
            "First real run:",
            "",
            f"- Tier: `{manifest['recommended_first_real_run']['tier']}`",
            f"- Circuits: `{manifest['recommended_first_real_run']['circuits']}`",
            f"- Shots: `{manifest['recommended_first_real_run']['shots']}`",
            f"- Stop rule: {manifest['recommended_first_real_run']['stop_rule']}",
            "",
            "## Claim Boundary",
            "",
            manifest["claim_boundary"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Generate the staged IQM paper-replication manifest and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake-backend", default="garnet")
    parser.add_argument("--smoke-shots", type=int, default=128)
    parser.add_argument("--replication-shots", type=int, default=256)
    parser.add_argument(
        "--iqm-python",
        type=Path,
        default=None,
        help="Optional isolated Python executable with iqm-client[qiskit] installed.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = generate(
        fake_backend=args.fake_backend,
        smoke_shots=args.smoke_shots,
        replication_shots=args.replication_shots,
        iqm_python=args.iqm_python,
    )
    stem = f"iqm_paper_replication_readiness_{DATE}"
    json_path = args.output_dir / f"{stem}.json"
    csv_path = args.output_dir / f"{stem}.csv"
    md_path = args.output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_csv(csv_path, list(manifest["rows"]))
    _write_markdown(md_path, manifest, json_path, csv_path)
    print(f"wrote_json={json_path}")
    print(f"wrote_csv={csv_path}")
    print(f"wrote_md={md_path}")
    print(f"sha256_json={_sha256(json_path)}")
    print(f"sha256_csv={_sha256(csv_path)}")
    print(f"transpile_status={manifest['transpile_status']}")
    print(f"total_circuits={manifest['total_circuits']}")
    print(f"total_shots={manifest['total_shots']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
