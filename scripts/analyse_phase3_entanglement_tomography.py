#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Phase 3 entanglement/tomography analysis
"""Analyse approved Phase 3 entanglement/tomography raw-count artefacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TODAY = date(2026, 5, 20).isoformat()
DEFAULT_REFERENCE_CSV = (
    REPO_ROOT
    / "data"
    / "phase3_entanglement_tomography"
    / "entanglement_observable_rows_2026-05-07.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "phase3_entanglement_tomography"
DEFAULT_DOCS_DIR = REPO_ROOT / "docs"
N_QUBITS = 4


def _bitstrings(width: int) -> list[str]:
    return [format(index, f"0{width}b") for index in range(2**width)]


def pauli_expectation_from_counts(counts: Mapping[str, int], pauli_label: str) -> float:
    """Estimate a Pauli expectation value from basis-rotated bitstring counts."""

    total = sum(int(value) for value in counts.values())
    if total <= 0:
        raise ValueError("counts must contain at least one shot")
    active = [index for index, basis in enumerate(pauli_label) if basis != "I"]
    expectation = 0.0
    for bitstring, shots in counts.items():
        if len(bitstring.replace(" ", "")) < len(pauli_label):
            raise ValueError("bitstring width is smaller than pauli_label width")
        compact = bitstring.replace(" ", "")[-len(pauli_label) :]
        parity = sum(1 for index in active if compact[index] == "1")
        eigenvalue = -1.0 if parity % 2 else 1.0
        expectation += eigenvalue * int(shots)
    return float(expectation / total)


def estimate_single_qubit_readout_matrices(
    readout_circuits: Sequence[Mapping[str, Any]],
    *,
    width: int,
) -> list[dict[str, float]]:
    """Estimate independent single-qubit assignment matrices from readout shots.

    Each returned matrix stores probabilities as ``prepared_X_observed_Y``.
    The calibration can be partial at the bitstring level, but each qubit must
    have at least one prepared-0 and one prepared-1 calibration marginal.
    """

    prepared_counts = [
        {
            "0": {"0": 0, "1": 0},
            "1": {"0": 0, "1": 0},
        }
        for _ in range(width)
    ]
    for circuit in readout_circuits:
        meta = circuit.get("meta", {})
        if meta.get("block") != "readout":
            continue
        initial = str(meta["initial"]).replace(" ", "")[-width:][::-1]
        if len(initial) != width:
            raise ValueError("readout calibration width does not match analysis width")
        for observed, shots in circuit.get("counts", {}).items():
            compact = str(observed).replace(" ", "")[-width:]
            if len(compact) != width:
                raise ValueError("readout count width does not match analysis width")
            for qubit, prepared_bit in enumerate(initial):
                observed_bit = compact[qubit]
                prepared_counts[qubit][prepared_bit][observed_bit] += int(shots)

    matrices: list[dict[str, float]] = []
    for qubit, counts in enumerate(prepared_counts):
        totals = {prepared: sum(observed.values()) for prepared, observed in counts.items()}
        if totals["0"] <= 0 or totals["1"] <= 0:
            raise ValueError(
                f"readout calibration is missing both prepared states for qubit {qubit}"
            )
        matrix = {
            "prepared_0_observed_0": counts["0"]["0"] / totals["0"],
            "prepared_0_observed_1": counts["0"]["1"] / totals["0"],
            "prepared_1_observed_0": counts["1"]["0"] / totals["1"],
            "prepared_1_observed_1": counts["1"]["1"] / totals["1"],
        }
        determinant = (
            matrix["prepared_0_observed_0"] * matrix["prepared_1_observed_1"]
            - matrix["prepared_1_observed_0"] * matrix["prepared_0_observed_1"]
        )
        if abs(determinant) < 1e-9:
            raise ValueError(f"readout assignment matrix is singular for qubit {qubit}")
        matrices.append(matrix)
    return matrices


def _single_qubit_assignment_matrix(matrix: Mapping[str, float]) -> np.ndarray:
    return np.array(
        [
            [float(matrix["prepared_0_observed_0"]), float(matrix["prepared_1_observed_0"])],
            [float(matrix["prepared_0_observed_1"]), float(matrix["prepared_1_observed_1"])],
        ],
        dtype=float,
    )


def _tensor_assignment_matrix(matrices: Sequence[Mapping[str, float]]) -> np.ndarray:
    assignment = np.array([[1.0]], dtype=float)
    for matrix in matrices:
        assignment = np.kron(assignment, _single_qubit_assignment_matrix(matrix))
    return assignment


def estimate_correlated_readout_matrix(
    readout_circuits: Sequence[Mapping[str, Any]],
    *,
    width: int,
) -> list[list[float]]:
    """Estimate the full correlated assignment matrix from all basis states."""

    bitstrings = _bitstrings(width)
    bitstring_index = {bitstring: index for index, bitstring in enumerate(bitstrings)}
    columns: dict[str, np.ndarray] = {}
    for circuit in readout_circuits:
        meta = circuit.get("meta", {})
        if meta.get("block") != "readout":
            continue
        prepared = str(meta["initial"]).replace(" ", "")[-width:][::-1]
        if prepared not in bitstring_index:
            raise ValueError("readout calibration prepared state has invalid width")
        total = sum(int(value) for value in circuit.get("counts", {}).values())
        if total <= 0:
            raise ValueError(f"readout calibration for {prepared} contains no shots")
        column = np.zeros(len(bitstrings), dtype=float)
        for observed, shots in circuit.get("counts", {}).items():
            compact = str(observed).replace(" ", "")[-width:]
            column[bitstring_index[compact]] += int(shots) / total
        columns[prepared] = column
    missing = sorted(set(bitstrings).difference(columns))
    if missing:
        raise ValueError(f"full readout calibration is missing states: {missing}")
    assignment = np.column_stack([columns[bitstring] for bitstring in bitstrings])
    if abs(float(np.linalg.det(assignment))) < 1e-12:
        raise ValueError("full readout assignment matrix is singular")
    return assignment.tolist()


def build_readout_mitigation_model(
    readout_circuits: Sequence[Mapping[str, Any]],
    *,
    width: int,
) -> dict[str, Any]:
    """Build the strongest readout mitigation model supported by calibration data."""

    prepared = {
        str(circuit.get("meta", {}).get("initial", "")).replace(" ", "")[-width:]
        for circuit in readout_circuits
        if circuit.get("meta", {}).get("block") == "readout"
    }
    full_states = {format(index, f"0{width}b") for index in range(2**width)}
    if prepared == full_states:
        return {
            "method": "full_correlated_readout_inverse",
            "source": "full computational-basis readout calibration from the same live artefact",
            "n_calibration_circuits": len(readout_circuits),
            "assignment_matrix": estimate_correlated_readout_matrix(readout_circuits, width=width),
            "claim_boundary": (
                "full correlated readout inversion only; not a ZNE/PEC result and "
                "not a correction for basis-rotation or coherent gate errors"
            ),
        }
    return {
        "method": "tensor_product_single_qubit_inverse",
        "source": "readout calibration marginals from the same live artefact",
        "n_calibration_circuits": len(readout_circuits),
        "single_qubit_assignment_matrices": estimate_single_qubit_readout_matrices(
            readout_circuits,
            width=width,
        ),
        "claim_boundary": (
            "independent single-qubit readout inversion only; not a full "
            "16-state correlated readout calibration and not a ZNE/PEC result"
        ),
    }


def mitigated_pauli_expectation(
    counts: Mapping[str, int],
    pauli_label: str,
    model: Mapping[str, Any],
) -> float:
    """Estimate a Pauli expectation using the selected readout mitigation model."""

    if model["method"] == "full_correlated_readout_inverse":
        width = len(pauli_label)
        total = sum(int(value) for value in counts.values())
        if total <= 0:
            raise ValueError("counts must contain at least one shot")
        bitstrings = _bitstrings(width)
        bitstring_index = {bitstring: index for index, bitstring in enumerate(bitstrings)}
        observed = np.zeros(len(bitstrings), dtype=float)
        for bitstring, shots in counts.items():
            compact = str(bitstring).replace(" ", "")[-width:]
            observed[bitstring_index[compact]] += int(shots) / total
        mitigated = np.linalg.inv(np.array(model["assignment_matrix"], dtype=float)) @ observed
        active = [index for index, basis in enumerate(pauli_label) if basis != "I"]
        expectation = 0.0
        for bitstring, probability in zip(bitstrings, mitigated):
            parity = sum(1 for index in active if bitstring[index] == "1")
            expectation += (-1.0 if parity % 2 else 1.0) * float(probability)
        return float(expectation)
    return readout_mitigated_pauli_expectation(
        counts,
        pauli_label,
        model["single_qubit_assignment_matrices"],
    )


def readout_mitigated_pauli_expectation(
    counts: Mapping[str, int],
    pauli_label: str,
    readout_matrices: Sequence[Mapping[str, float]],
) -> float:
    """Estimate a Pauli expectation after tensor-product readout inversion."""

    width = len(pauli_label)
    if len(readout_matrices) != width:
        raise ValueError("readout matrix count must match pauli_label width")
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        raise ValueError("counts must contain at least one shot")
    bitstrings = _bitstrings(width)
    observed = np.zeros(len(bitstrings), dtype=float)
    bitstring_index = {bitstring: index for index, bitstring in enumerate(bitstrings)}
    for bitstring, shots in counts.items():
        compact = str(bitstring).replace(" ", "")[-width:]
        observed[bitstring_index[compact]] += int(shots) / total
    mitigated = np.linalg.inv(_tensor_assignment_matrix(readout_matrices)) @ observed
    active = [index for index, basis in enumerate(pauli_label) if basis != "I"]
    expectation = 0.0
    for bitstring, probability in zip(bitstrings, mitigated):
        parity = sum(1 for index in active if bitstring[index] == "1")
        expectation += (-1.0 if parity % 2 else 1.0) * float(probability)
    return float(expectation)


def _load_reference_rows(path: Path) -> dict[tuple[str, str, str, int, str, str], dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: dict[tuple[str, str, str, int, str, str], dict[str, Any]] = {}
        for row in reader:
            lambda_key = str(row.get("lambda_fim") or "")
            key = (
                str(row["family"]),
                str(row["label"]),
                str(row["initial"]),
                int(row["depth"]),
                lambda_key,
                str(row["basis_setting"]),
            )
            rows[key] = dict(row)
    return rows


def _reference_key(meta: Mapping[str, Any]) -> tuple[str, str, str, int, str, str]:
    lambda_value = meta.get("lambda_fim")
    lambda_key = "" if lambda_value is None else str(float(lambda_value))
    return (
        str(meta["family"]),
        str(meta["label"]),
        str(meta["initial"]),
        int(meta["depth"]),
        lambda_key,
        str(meta["basis_setting"]),
    )


def _stderr(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(stdev(values) / (len(values) ** 0.5))


def analyse_counts_artifact(
    counts_path: Path,
    reference_csv: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reduce a completed live counts artefact into observable rows."""

    payload = json.loads(counts_path.read_text(encoding="utf-8"))
    references = _load_reference_rows(reference_csv)
    grouped: dict[tuple[str, str, str, int, str, str], list[float]] = defaultdict(list)
    grouped_mitigated: dict[tuple[str, str, str, int, str, str], list[float]] = defaultdict(list)
    analysis_jobs: set[str] = set()
    circuits = payload.get("circuits", [])
    readout_circuits = [
        circuit for circuit in circuits if circuit.get("meta", {}).get("block") == "readout"
    ]
    readout_model = build_readout_mitigation_model(readout_circuits, width=N_QUBITS)
    for circuit in circuits:
        meta = circuit.get("meta", {})
        if meta.get("block") != "main":
            continue
        key = _reference_key(meta)
        reference = references.get(key)
        if reference is None:
            raise ValueError(f"missing reference row for {key}")
        counts = circuit.get("counts", {})
        pauli_label = str(reference["pauli_label"])
        grouped[key].append(pauli_expectation_from_counts(counts, pauli_label))
        grouped_mitigated[key].append(
            mitigated_pauli_expectation(counts, pauli_label, readout_model)
        )
        if circuit.get("job_id"):
            analysis_jobs.add(str(circuit["job_id"]))

    rows: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        reference = references[key]
        exact = float(reference["exact_expectation"])
        measured = float(mean(values))
        mitigated_values = grouped_mitigated[key]
        mitigated = float(mean(mitigated_values))
        stderr = _stderr(values)
        mitigated_stderr = _stderr(mitigated_values)
        deviation = measured - exact
        mitigated_deviation = mitigated - exact
        rows.append(
            {
                "family": key[0],
                "label": key[1],
                "initial": key[2],
                "depth": key[3],
                "lambda_fim": key[4],
                "observable": reference["observable"],
                "pauli_label": reference["pauli_label"],
                "basis_setting": key[5],
                "n_repetitions": len(values),
                "mean_expectation": measured,
                "stderr_expectation": stderr,
                "readout_mitigated_mean_expectation": mitigated,
                "readout_mitigated_stderr_expectation": mitigated_stderr,
                "exact_expectation": exact,
                "deviation_from_exact": deviation,
                "absolute_deviation": abs(deviation),
                "readout_mitigated_deviation_from_exact": mitigated_deviation,
                "readout_mitigated_absolute_deviation": abs(mitigated_deviation),
                "readout_mitigation_delta": mitigated - measured,
                "z_score_vs_exact": deviation / stderr if stderr > 0.0 else None,
                "readout_mitigated_z_score_vs_exact": (
                    mitigated_deviation / mitigated_stderr if mitigated_stderr > 0.0 else None
                ),
                "half_chain_purity_reference": float(reference["half_chain_purity"]),
                "parity_survival_reference": float(reference["parity_survival_ideal"]),
            }
        )
    max_abs = max((row["absolute_deviation"] for row in rows), default=None)
    mean_abs = mean([row["absolute_deviation"] for row in rows]) if rows else None
    mitigated_abs_values = [row["readout_mitigated_absolute_deviation"] for row in rows]
    summary = {
        "schema": "scpn_phase3_entanglement_tomography_analysis_v1",
        "counts_artifact": _display_path(counts_path),
        "reference_csv": _display_path(reference_csv),
        "backend": payload.get("backend"),
        "status": payload.get("status"),
        "job_ids": [str(job_id) for job_id in payload.get("job_ids", [])],
        "analysis_job_ids": sorted(analysis_jobs),
        "n_observable_rows": len(rows),
        "max_absolute_deviation": max_abs,
        "mean_absolute_deviation": mean_abs,
        "readout_mitigated_max_absolute_deviation": (
            max(mitigated_abs_values) if mitigated_abs_values else None
        ),
        "readout_mitigated_mean_absolute_deviation": (
            mean(mitigated_abs_values) if mitigated_abs_values else None
        ),
        "readout_mitigation": readout_model,
        "claim_boundary": (
            "reduced-Pauli observable analysis only; no scalable tomography, "
            "quantum advantage, or backend-general claim"
        ),
    }
    return rows, summary


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _manifest(summary: Mapping[str, Any], *, json_path: Path, csv_path: Path) -> str:
    return "\n".join(
        [
            "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
            "<!-- Commercial license available -->",
            "<!-- © Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->",
            "<!-- © Code 2020-2026 Miroslav Sotek. All rights reserved. -->",
            "<!-- ORCID: 0009-0009-3560-0851 -->",
            "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
            "<!-- scpn-quantum-control -- entanglement/tomography analysis manifest -->",
            "",
            "# Phase 3 Entanglement/Tomography Analysis Manifest",
            "",
            f"Date: {TODAY}",
            "",
            "## Inputs",
            "",
            f"- Counts artefact: `{summary['counts_artifact']}`",
            f"- Reference CSV: `{summary['reference_csv']}`",
            f"- Backend: `{summary['backend']}`",
            f"- Job IDs: `{', '.join(summary['job_ids'])}`",
            "",
            "## Outputs",
            "",
            f"- JSON summary: `{_display_path(json_path)}`",
            f"- Observable rows: `{_display_path(csv_path)}`",
            f"- Observable rows SHA256: `{_sha256(csv_path)}`",
            "",
            "## Result Snapshot",
            "",
            f"- Observable rows: `{summary['n_observable_rows']}`",
            f"- Mean absolute deviation from exact reference: `{summary['mean_absolute_deviation']}`",
            f"- Maximum absolute deviation from exact reference: `{summary['max_absolute_deviation']}`",
            "- Readout-mitigated mean absolute deviation from exact reference: "
            f"`{summary['readout_mitigated_mean_absolute_deviation']}`",
            "- Readout-mitigated maximum absolute deviation from exact reference: "
            f"`{summary['readout_mitigated_max_absolute_deviation']}`",
            "",
            "## Readout Mitigation",
            "",
            f"- Method: `{summary['readout_mitigation']['method']}`",
            f"- Calibration circuits: `{summary['readout_mitigation']['n_calibration_circuits']}`",
            f"- Boundary: {summary['readout_mitigation']['claim_boundary']}",
            "",
            "## Boundary",
            "",
            str(summary["claim_boundary"]),
            "",
        ]
    )


def write_outputs(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    output_dir: Path,
    docs_dir: Path,
    result_tag: str = TODAY,
) -> tuple[Path, Path, Path]:
    """Write analysis JSON, CSV, and Markdown artefacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"entanglement_tomography_summary_{result_tag}.json"
    csv_path = output_dir / f"entanglement_tomography_rows_{result_tag}.csv"
    md_path = docs_dir / f"phase3_entanglement_tomography_manifest_{result_tag}.md"
    _write_csv(csv_path, rows)
    payload = dict(summary)
    payload["rows_sha256"] = _sha256(csv_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(
        _manifest(payload, json_path=json_path, csv_path=csv_path), encoding="utf-8"
    )
    return json_path, csv_path, md_path


def parse_args() -> argparse.Namespace:
    """Parse analysis command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("counts_artifact", type=Path)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument(
        "--result-tag",
        default=TODAY,
        help="filename tag for same-day backend/layout replication analyses",
    )
    return parser.parse_args()


def main() -> int:
    """Analyse an approved entanglement/tomography counts artefact."""

    args = parse_args()
    rows, summary = analyse_counts_artifact(args.counts_artifact, args.reference_csv)
    json_path, csv_path, md_path = write_outputs(
        rows,
        summary,
        output_dir=args.output_dir,
        docs_dir=args.docs_dir,
        result_tag=args.result_tag,
    )
    print(f"wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"wrote {csv_path.relative_to(REPO_ROOT)}")
    print(f"wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"n_observable_rows={summary['n_observable_rows']}")
    print(f"mean_absolute_deviation={summary['mean_absolute_deviation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
