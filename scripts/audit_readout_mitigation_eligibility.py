#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — audit readout mitigation eligibility script
# scpn-quantum-control -- readout mitigation eligibility audit
"""Generate dataset-level readout-mitigation eligibility markers.

The audit is offline and consumes committed raw-count JSON files only. It does
not request new calibration circuits and does not submit hardware jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-05-06"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data"
    / "readout_mitigation_eligibility"
    / f"readout_mitigation_eligibility_{DATE}.json"
)
PROMOTED_DATASETS = (
    REPO_ROOT / "data" / "phase1_dla_parity" / "phase1_bench_2026-04-10T183728Z.json",
    REPO_ROOT / "data" / "phase2_dla_parity" / "phase2_reduced_ag_2026-05-05T121357Z.json",
    REPO_ROOT
    / "data"
    / "phase2_popcount_control"
    / "phase2_popcount_control_2026-05-05T135318Z.json",
    REPO_ROOT / "data" / "phase2_scaling_bc" / "phase2_scaling_bc_2026-05-05T124722Z.json",
    REPO_ROOT
    / "data"
    / "scpn_fim_hamiltonian"
    / "fim_ibm_pilot_raw_counts_2026-05-05_ibm-run-4c0bd60c3fc2c532.json",
    REPO_ROOT
    / "data"
    / "scpn_fim_hamiltonian"
    / "fim_ibm_repeated_followup_raw_counts_2026-05-05_ibm-run-cf4835290f607387.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_rows = payload.get("circuits") or payload.get("metadata_rows") or []
    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("meta")
    if isinstance(meta, dict):
        return meta
    return row


def _is_readout_row(meta: dict[str, Any]) -> bool:
    values = (
        meta.get("experiment"),
        meta.get("protocol_arm"),
        meta.get("block"),
        meta.get("circuit_name"),
    )
    return any("readout" in str(value).lower() for value in values if value is not None)


def _initial_bitstring(meta: dict[str, Any]) -> str | None:
    value = meta.get("initial") or meta.get("initial_bitstring")
    if value is None:
        return None
    bitstring = str(value)
    return bitstring if set(bitstring) <= {"0", "1"} else None


def _eligibility_status(n_qubits: int, prepared: set[str]) -> str:
    if n_qubits > 8:
        return "not_applicable"
    required = 2**n_qubits
    if len(prepared) == required:
        return "full_basis_confusion_matrix_available"
    if prepared:
        return "partial_exact_state_baseline_only"
    return "missing_readout_calibration"


def audit_dataset(path: Path) -> dict[str, Any]:
    """Return one readout-mitigation eligibility marker."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = _rows(payload)
    metadata_rows = [_metadata(row) for row in rows]
    n_values = sorted(
        {
            int(meta["n_qubits"])
            for meta in metadata_rows
            if isinstance(meta.get("n_qubits"), int) and int(meta["n_qubits"]) <= 8
        }
    )
    readout_rows = [meta for meta in metadata_rows if _is_readout_row(meta)]
    per_n = [_per_n_marker(n_qubits, readout_rows) for n_qubits in n_values]
    status = _dataset_status(per_n)
    prepared = sorted(
        {bitstring for marker in per_n for bitstring in marker["prepared_readout_bitstrings"]}
    )
    missing = sorted(
        {bitstring for marker in per_n for bitstring in marker["missing_readout_bitstrings"]}
    )
    required = sum(int(marker["required_full_basis_count"]) for marker in per_n) or None
    return {
        "dataset": _display_path(path),
        "sha256": _sha256(path),
        "n_qubits": n_values[0] if len(n_values) == 1 else None,
        "n_qubit_values": n_values,
        "n_rows": len(rows),
        "readout_rows": len(readout_rows),
        "prepared_readout_bitstrings": prepared,
        "prepared_readout_count": sum(int(marker["prepared_readout_count"]) for marker in per_n),
        "required_full_basis_count": required,
        "missing_readout_bitstrings": missing,
        "per_n_markers": per_n,
        "eligibility_status": status,
        "allowed_mitigation": _allowed_mitigation(status),
        "qpu_spend_required_for_full_basis": status
        in {"partial_exact_state_baseline_only", "missing_readout_calibration"},
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _per_n_marker(n_qubits: int, readout_rows: list[dict[str, Any]]) -> dict[str, Any]:
    prepared = set()
    for meta in readout_rows:
        bitstring = _initial_bitstring(meta)
        if bitstring is None:
            continue
        row_n = meta.get("n_qubits")
        if row_n == n_qubits or len(bitstring) == n_qubits:
            prepared.add(bitstring)
    required = 2**n_qubits
    missing = [
        format(index, f"0{n_qubits}b")
        for index in range(required)
        if format(index, f"0{n_qubits}b") not in prepared
    ]
    status = _eligibility_status(n_qubits, prepared)
    return {
        "n_qubits": n_qubits,
        "prepared_readout_bitstrings": sorted(prepared),
        "prepared_readout_count": len(prepared),
        "required_full_basis_count": required,
        "missing_readout_bitstrings": missing,
        "eligibility_status": status,
    }


def _dataset_status(per_n: list[dict[str, Any]]) -> str:
    if not per_n:
        return "not_applicable"
    statuses = {str(marker["eligibility_status"]) for marker in per_n}
    if statuses == {"full_basis_confusion_matrix_available"}:
        return "full_basis_confusion_matrix_available"
    if "partial_exact_state_baseline_only" in statuses:
        return "partial_exact_state_baseline_only"
    if "full_basis_confusion_matrix_available" in statuses:
        return "partial_exact_state_baseline_only"
    if "missing_readout_calibration" in statuses:
        return "missing_readout_calibration"
    return "not_applicable"


def _allowed_mitigation(status: str) -> str:
    if status == "full_basis_confusion_matrix_available":
        return "full_confusion_matrix_inversion_or_exact_state_correction"
    if status == "partial_exact_state_baseline_only":
        return "exact_state_or_parity_readout_correction_only"
    if status == "missing_readout_calibration":
        return "raw_counts_only_unless_new_calibration_is_approved"
    return "not_applicable"


def build_summary(paths: tuple[Path, ...]) -> dict[str, Any]:
    """Build the complete eligibility marker payload."""

    markers = [audit_dataset(path) for path in paths if path.exists()]
    return {
        "date": DATE,
        "schema": "scpn_readout_mitigation_eligibility_v1",
        "command": "python scripts/audit_readout_mitigation_eligibility.py",
        "claim_boundary": (
            "Markers classify committed n<=8 raw-count datasets by whether a "
            "complete computational-basis readout calibration is present. "
            "They do not request or imply new QPU calibration circuits."
        ),
        "markers": markers,
    }


def main() -> int:
    """Run the readout-mitigation eligibility audit CLI."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ns = parser.parse_args()
    ns.output.parent.mkdir(parents=True, exist_ok=True)
    summary = build_summary(PROMOTED_DATASETS)
    ns.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote={ns.output}")
    print(f"sha256={_sha256(ns.output)}")
    for marker in summary["markers"]:
        print(
            f"{marker['dataset']}: {marker['eligibility_status']} "
            f"({marker['prepared_readout_count']}/"
            f"{marker['required_full_basis_count']} basis states)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
