#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — retrieve phase3 large system IBM script
# scpn-quantum-control -- retrieve Phase 3 larger-system IBM lane
"""Retrieve and reduce a completed Phase 3 larger-system IBM job."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
from analyse_phase3_entanglement_tomography import (
    build_readout_mitigation_model,
    mitigated_pauli_expectation,
    pauli_expectation_from_counts,
)
from qiskit.quantum_info import Pauli, Statevector
from submit_phase3_large_system_ibm import SourceSpec, build_source_circuit

from scpn_quantum_control.hardware.runner import _extract_counts
from scpn_quantum_control.mitigation.zne import zne_extrapolate

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "phase3_entanglement_tomography"
DOCS_DIR = REPO_ROOT / "docs" / "campaigns"
DEFAULT_CREDENTIALS_VAULT = Path("~/.config/scpn-quantum-control/credentials.md").expanduser()


@dataclass(frozen=True)
class AnalysisOutputs:
    """Paths written by the larger-system analysis writer."""

    raw_counts_json: Path
    reference_csv: Path
    scale_rows_csv: Path
    channel_summary_csv: Path
    analysis_json: Path
    manifest_md: Path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-json", type=Path, required=True)
    parser.add_argument("--credentials-vault", type=Path, default=DEFAULT_CREDENTIALS_VAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
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
    token, instance = module.parse_vault(path)
    return token, instance


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return _sha256(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)
    return _sha256(path)


def _stderr(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(stdev(values) / (len(values) ** 0.5))


def _bitstrings(width: int) -> list[str]:
    return [format(index, f"0{width}b") for index in range(2**width)]


def _pseudoinverse_readout_model(
    readout_circuits: Sequence[Mapping[str, Any]],
    *,
    width: int,
) -> dict[str, Any]:
    bitstrings = _bitstrings(width)
    bitstring_index = {bitstring: index for index, bitstring in enumerate(bitstrings)}
    columns: dict[str, np.ndarray] = {}
    for circuit in readout_circuits:
        meta = circuit.get("meta", {})
        if meta.get("block") != "readout":
            continue
        prepared = str(meta["initial"]).replace(" ", "")[-width:][::-1]
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
    return {
        "method": "full_correlated_readout_pseudoinverse",
        "source": "full computational-basis readout calibration with singular-matrix fallback",
        "n_calibration_circuits": len(readout_circuits),
        "assignment_matrix": assignment.tolist(),
        "condition_number": float(np.linalg.cond(assignment)),
        "rank": int(np.linalg.matrix_rank(assignment)),
        "claim_boundary": (
            "full correlated readout pseudo-inversion for a singular larger-width "
            "assignment matrix; this is a sensitivity replay, not a physical "
            "correction for gate, basis-rotation, crosstalk, or coherent errors"
        ),
    }


def _readout_model(
    readout_circuits: Sequence[Mapping[str, Any]],
    *,
    width: int,
) -> dict[str, Any]:
    try:
        return build_readout_mitigation_model(readout_circuits, width=width)
    except ValueError as exc:
        if "singular" not in str(exc):
            raise
        return _pseudoinverse_readout_model(readout_circuits, width=width)


def _mitigated_expectation(
    counts: Mapping[str, int],
    pauli_label: str,
    model: Mapping[str, Any],
) -> float:
    if model["method"] != "full_correlated_readout_pseudoinverse":
        return mitigated_pauli_expectation(counts, pauli_label, model)
    width = len(pauli_label)
    bitstrings = _bitstrings(width)
    bitstring_index = {bitstring: index for index, bitstring in enumerate(bitstrings)}
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        raise ValueError("counts must contain at least one shot")
    observed = np.zeros(len(bitstrings), dtype=float)
    for bitstring, shots in counts.items():
        compact = str(bitstring).replace(" ", "")[-width:]
        observed[bitstring_index[compact]] += int(shots) / total
    assignment = np.array(model["assignment_matrix"], dtype=float)
    mitigated = np.linalg.pinv(assignment) @ observed
    active = [index for index, basis in enumerate(pauli_label) if basis != "I"]
    expectation = 0.0
    for bitstring, probability in zip(bitstrings, mitigated):
        parity = sum(1 for index in active if bitstring[index] == "1")
        expectation += (-1.0 if parity % 2 else 1.0) * float(probability)
    return float(expectation)


def _job_status(job: Any) -> str:
    status = job.status()
    return str(getattr(status, "name", status)).replace("JobStatus.", "")


def _submission_metadata_rows(submission: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = submission.get("metadata_rows")
    if not isinstance(rows, list):
        raise ValueError("submission is missing metadata_rows")
    return [dict(row) for row in rows]


def _result_rows_from_submission(
    *,
    submission: Mapping[str, Any],
    result: Sequence[Any],
    job_id: str,
) -> list[dict[str, Any]]:
    metadata_rows = _submission_metadata_rows(submission)
    if len(result) != len(metadata_rows):
        raise ValueError(
            f"result length {len(result)} does not match metadata rows {len(metadata_rows)}"
        )
    rows: list[dict[str, Any]] = []
    for meta, pub_result in zip(metadata_rows, result, strict=True):
        rows.append(
            {"metadata": meta, "counts": dict(_extract_counts(pub_result)), "job_id": job_id}
        )
    return rows


def retrieve_completed_job(
    *,
    submission: Mapping[str, Any],
    credentials_vault: Path,
) -> tuple[str, list[dict[str, Any]]]:
    """Retrieve raw counts for a completed larger-system Phase 3 job."""

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
    status = _job_status(job)
    if status != "DONE":
        return status, []
    return status, _result_rows_from_submission(
        submission=submission,
        result=job.result(),
        job_id=job_id,
    )


def raw_payload_from_rows(
    *,
    submission: Mapping[str, Any],
    result_rows: Sequence[Mapping[str, Any]],
    submission_json: Path,
    submission_sha256: str,
    timestamp_utc: str,
) -> dict[str, Any]:
    """Build the repository raw-count artefact from retrieved IBM rows."""

    circuits: list[dict[str, Any]] = []
    for row in result_rows:
        meta = dict(row["metadata"])
        block = str(meta["block"])
        if block == "readout_calibration":
            meta = {
                "block": "readout",
                "initial": str(meta["calibration_state"]),
                "calibration_state": str(meta["calibration_state"]),
                "circuit_index": int(meta["circuit_index"]),
            }
        circuits.append(
            {
                "job_id": str(row.get("job_id", submission["job_ids"][0])),
                "meta": meta,
                "counts": {str(key): int(value) for key, value in row["counts"].items()},
            }
        )
    return {
        "schema": "scpn_phase3_large_system_raw_counts_v1",
        "experiment_id": submission["experiment_id"],
        "backend": submission["backend"],
        "timestamp_utc": timestamp_utc,
        "n_qubits": int(submission["n_qubits"]),
        "job_ids": [str(job_id) for job_id in submission["job_ids"]],
        "submission_json": _display_path(submission_json),
        "submission_sha256": submission_sha256,
        "physical_qubits": list(submission["physical_qubits"]),
        "shots": int(submission["shots"]),
        "claim_boundary": submission.get("claim_boundary"),
        "circuits": circuits,
    }


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


def _observable_name(pauli_label: str) -> str:
    active = [f"q{index}" for index, basis in enumerate(pauli_label) if basis != "I"]
    active_label = "".join(basis for basis in pauli_label if basis != "I")
    return f"{active_label}_{'_'.join(active)}"


def _exact_expectation(meta: Mapping[str, Any]) -> float:
    spec = SourceSpec(
        family=str(meta["family"]),
        label=str(meta["label"]),
        initial_bitstring=str(meta["initial"]),
        depth=int(meta["depth"]),
        lambda_fim=None if meta.get("lambda_fim") is None else float(meta["lambda_fim"]),
    )
    state = Statevector.from_instruction(build_source_circuit(spec))
    pauli = Pauli(str(meta["basis_setting"])[::-1])
    return float(state.expectation_value(pauli).real)


def reference_rows_for_submission(submission: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build exact reference rows for the submitted larger-system channels."""

    seen: set[tuple[str, str, str, int, str, str]] = set()
    rows: list[dict[str, Any]] = []
    for meta in _submission_metadata_rows(submission):
        if meta.get("block") != "main":
            continue
        key = _reference_key(meta)
        if key in seen:
            continue
        seen.add(key)
        pauli_label = str(meta["basis_setting"])
        rows.append(
            {
                "family": key[0],
                "label": key[1],
                "initial": key[2],
                "depth": key[3],
                "lambda_fim": key[4],
                "observable": _observable_name(pauli_label),
                "pauli_label": pauli_label,
                "basis_setting": key[5],
                "exact_expectation": _exact_expectation(meta),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row["family"]),
            str(row["label"]),
            str(row["initial"]),
            int(row["depth"]),
            str(row["lambda_fim"]),
            str(row["basis_setting"]),
        ),
    )


def _reference_lookup(
    reference_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str, int, str, str], Mapping[str, Any]]:
    return {
        (
            str(row["family"]),
            str(row["label"]),
            str(row["initial"]),
            int(row["depth"]),
            str(row.get("lambda_fim") or ""),
            str(row["basis_setting"]),
        ): row
        for row in reference_rows
    }


def analyse_large_system_payload(
    raw_payload: Mapping[str, Any],
    reference_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Reduce raw Phase 3 larger-system counts into scale and ZNE channel rows."""

    n_qubits = int(raw_payload["n_qubits"])
    references = _reference_lookup(reference_rows)
    circuits = list(raw_payload.get("circuits", []))
    readout_circuits = [
        circuit for circuit in circuits if circuit.get("meta", {}).get("block") == "readout"
    ]
    readout_model = _readout_model(readout_circuits, width=n_qubits)
    grouped: dict[tuple[str, str, str, int, str, str, int], list[float]] = defaultdict(list)
    grouped_mitigated: dict[tuple[str, str, str, int, str, str, int], list[float]] = defaultdict(
        list
    )
    analysis_jobs: set[str] = set()
    for circuit in circuits:
        meta = circuit.get("meta", {})
        if meta.get("block") != "main":
            continue
        reference_key = _reference_key(meta)
        reference = references.get(reference_key)
        if reference is None:
            raise ValueError(f"missing reference row for {reference_key}")
        scale = int(meta["zne_noise_scale"])
        key = (*reference_key, scale)
        counts = circuit["counts"]
        pauli_label = str(reference["pauli_label"])
        grouped[key].append(pauli_expectation_from_counts(counts, pauli_label))
        grouped_mitigated[key].append(_mitigated_expectation(counts, pauli_label, readout_model))
        if circuit.get("job_id"):
            analysis_jobs.add(str(circuit["job_id"]))

    scale_rows: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        reference = references[key[:6]]
        exact = float(reference["exact_expectation"])
        mitigated_values = grouped_mitigated[key]
        measured = float(mean(values))
        mitigated = float(mean(mitigated_values))
        scale_rows.append(
            {
                "family": key[0],
                "label": key[1],
                "initial": key[2],
                "depth": key[3],
                "lambda_fim": key[4],
                "observable": reference["observable"],
                "pauli_label": reference["pauli_label"],
                "basis_setting": key[5],
                "zne_noise_scale": key[6],
                "n_repetitions": len(values),
                "mean_expectation": measured,
                "stderr_expectation": _stderr(values),
                "readout_mitigated_mean_expectation": mitigated,
                "readout_mitigated_stderr_expectation": _stderr(mitigated_values),
                "exact_expectation": exact,
                "deviation_from_exact": measured - exact,
                "absolute_deviation": abs(measured - exact),
                "readout_mitigated_deviation_from_exact": mitigated - exact,
                "readout_mitigated_absolute_deviation": abs(mitigated - exact),
            }
        )

    by_channel: dict[tuple[str, str, str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scale_rows:
        by_channel[
            (
                row["family"],
                row["label"],
                row["initial"],
                int(row["depth"]),
                row["lambda_fim"],
                row["basis_setting"],
            )
        ].append(row)

    channel_rows: list[dict[str, Any]] = []
    for key, rows in sorted(by_channel.items()):
        ordered = sorted(rows, key=lambda row: int(row["zne_noise_scale"]))
        scales = [int(row["zne_noise_scale"]) for row in ordered]
        raw_values = [float(row["mean_expectation"]) for row in ordered]
        mitigated_values = [float(row["readout_mitigated_mean_expectation"]) for row in ordered]
        exact = float(ordered[0]["exact_expectation"])
        linear = zne_extrapolate(scales, raw_values, order=1)
        linear_mitigated = zne_extrapolate(scales, mitigated_values, order=1)
        quadratic = zne_extrapolate(scales, raw_values, order=2) if len(scales) >= 3 else None
        quadratic_mitigated = (
            zne_extrapolate(scales, mitigated_values, order=2) if len(scales) >= 3 else None
        )
        channel_rows.append(
            {
                "family": key[0],
                "label": key[1],
                "initial": key[2],
                "depth": key[3],
                "lambda_fim": key[4],
                "basis_setting": key[5],
                "pauli_label": ordered[0]["pauli_label"],
                "observable": ordered[0]["observable"],
                "noise_scales": ",".join(str(scale) for scale in scales),
                "scale_expectations": ",".join(f"{value:.12g}" for value in raw_values),
                "readout_mitigated_scale_expectations": ",".join(
                    f"{value:.12g}" for value in mitigated_values
                ),
                "exact_expectation": exact,
                "scale1_expectation": raw_values[0],
                "scale1_absolute_deviation": abs(raw_values[0] - exact),
                "linear_zne_expectation": float(linear.zero_noise_estimate),
                "linear_zne_fit_residual": float(linear.fit_residual),
                "linear_zne_absolute_deviation": abs(float(linear.zero_noise_estimate) - exact),
                "readout_mitigated_linear_zne_expectation": float(
                    linear_mitigated.zero_noise_estimate
                ),
                "readout_mitigated_linear_zne_fit_residual": float(linear_mitigated.fit_residual),
                "readout_mitigated_linear_zne_absolute_deviation": abs(
                    float(linear_mitigated.zero_noise_estimate) - exact
                ),
                "quadratic_zne_expectation": None
                if quadratic is None
                else float(quadratic.zero_noise_estimate),
                "quadratic_zne_fit_residual": None
                if quadratic is None
                else float(quadratic.fit_residual),
                "quadratic_zne_absolute_deviation": None
                if quadratic is None
                else abs(float(quadratic.zero_noise_estimate) - exact),
                "readout_mitigated_quadratic_zne_expectation": None
                if quadratic_mitigated is None
                else float(quadratic_mitigated.zero_noise_estimate),
                "readout_mitigated_quadratic_zne_fit_residual": None
                if quadratic_mitigated is None
                else float(quadratic_mitigated.fit_residual),
                "readout_mitigated_quadratic_zne_absolute_deviation": None
                if quadratic_mitigated is None
                else abs(float(quadratic_mitigated.zero_noise_estimate) - exact),
            }
        )

    scale1_deviations = [float(row["scale1_absolute_deviation"]) for row in channel_rows]
    linear_deviations = [float(row["linear_zne_absolute_deviation"]) for row in channel_rows]
    mitigated_linear_deviations = [
        float(row["readout_mitigated_linear_zne_absolute_deviation"]) for row in channel_rows
    ]
    summary = {
        "schema": "scpn_phase3_large_system_analysis_v1",
        "experiment_id": raw_payload["experiment_id"],
        "backend": raw_payload["backend"],
        "n_qubits": n_qubits,
        "job_ids": [str(job_id) for job_id in raw_payload["job_ids"]],
        "analysis_job_ids": sorted(analysis_jobs),
        "physical_qubits": list(raw_payload["physical_qubits"]),
        "shots": int(raw_payload["shots"]),
        "n_scale_rows": len(scale_rows),
        "n_channels": len(channel_rows),
        "noise_scales": sorted({int(row["zne_noise_scale"]) for row in scale_rows}),
        "scale1_mean_absolute_deviation": mean(scale1_deviations) if scale1_deviations else None,
        "scale1_max_absolute_deviation": max(scale1_deviations, default=None),
        "linear_zne_mean_absolute_deviation": mean(linear_deviations)
        if linear_deviations
        else None,
        "linear_zne_max_absolute_deviation": max(linear_deviations, default=None),
        "readout_mitigated_linear_zne_mean_absolute_deviation": mean(mitigated_linear_deviations)
        if mitigated_linear_deviations
        else None,
        "readout_mitigated_linear_zne_max_absolute_deviation": max(
            mitigated_linear_deviations,
            default=None,
        ),
        "readout_mitigation": readout_model,
        "claim_boundary": (
            f"n={n_qubits} Phase 3 larger-system reduced-Pauli ZNE stress analysis only; "
            "not full tomography, backend-general entanglement dynamics, quantum advantage, "
            "or a full causal mechanism claim."
        ),
    }
    return scale_rows, channel_rows, summary


def _manifest(
    summary: Mapping[str, Any],
    *,
    raw_path: Path,
    reference_path: Path,
    scale_path: Path,
    channel_path: Path,
    analysis_path: Path,
) -> str:
    return "\n".join(
        [
            "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
            "<!-- Commercial license available -->",
            "<!-- © Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->",
            "<!-- © Code 2020-2026 Miroslav Sotek. All rights reserved. -->",
            "<!-- ORCID: 0009-0009-3560-0851 -->",
            "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
            "<!-- scpn-quantum-control -- Phase 3 larger-system analysis manifest -->",
            "",
            "# Phase 3 Larger-System Analysis Manifest",
            "",
            f"- Backend: `{summary['backend']}`",
            f"- Width: `n={summary['n_qubits']}`",
            f"- Job IDs: `{', '.join(summary['job_ids'])}`",
            f"- Raw counts: `{_display_path(raw_path)}`",
            f"- Reference rows: `{_display_path(reference_path)}`",
            f"- Scale rows: `{_display_path(scale_path)}`",
            f"- Channel summary: `{_display_path(channel_path)}`",
            f"- Analysis JSON: `{_display_path(analysis_path)}`",
            f"- Channels: `{summary['n_channels']}`",
            f"- Scale rows: `{summary['n_scale_rows']}`",
            "- Scale-1 mean absolute deviation: "
            f"`{summary.get('scale1_mean_absolute_deviation')}`",
            "- Linear ZNE mean absolute deviation: "
            f"`{summary.get('linear_zne_mean_absolute_deviation')}`",
            "",
            "## Boundary",
            "",
            str(summary["claim_boundary"]),
            "",
        ]
    )


def write_analysis_outputs(
    *,
    raw_path: Path,
    reference_rows: Sequence[Mapping[str, Any]],
    scale_rows: Sequence[Mapping[str, Any]],
    channel_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    output_dir: Path,
    docs_dir: Path,
    result_tag: str,
) -> AnalysisOutputs:
    """Write reference, scale, channel, JSON, and Markdown analysis artefacts."""

    reference_path = output_dir / f"phase3_large_system_reference_rows_{result_tag}.csv"
    scale_path = output_dir / f"phase3_large_system_scale_rows_{result_tag}.csv"
    channel_path = output_dir / f"phase3_large_system_channel_summary_{result_tag}.csv"
    analysis_path = output_dir / f"phase3_large_system_analysis_{result_tag}.json"
    manifest_path = docs_dir / f"phase3_large_system_manifest_{result_tag}.md"
    reference_sha = _write_csv(reference_path, reference_rows)
    scale_sha = _write_csv(scale_path, scale_rows)
    channel_sha = _write_csv(channel_path, channel_rows)
    payload = dict(summary)
    payload.update(
        {
            "raw_counts_json": _display_path(raw_path),
            "raw_counts_sha256": _sha256(raw_path),
            "reference_rows_csv": _display_path(reference_path),
            "reference_rows_sha256": reference_sha,
            "scale_rows_csv": _display_path(scale_path),
            "scale_rows_sha256": scale_sha,
            "channel_summary_csv": _display_path(channel_path),
            "channel_summary_sha256": channel_sha,
        }
    )
    _write_json(analysis_path, payload)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        _manifest(
            payload,
            raw_path=raw_path,
            reference_path=reference_path,
            scale_path=scale_path,
            channel_path=channel_path,
            analysis_path=analysis_path,
        ),
        encoding="utf-8",
    )
    return AnalysisOutputs(
        raw_counts_json=raw_path,
        reference_csv=reference_path,
        scale_rows_csv=scale_path,
        channel_summary_csv=channel_path,
        analysis_json=analysis_path,
        manifest_md=manifest_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Retrieve and reduce one completed larger-system Phase 3 submission."""

    args = _parse_args(argv)
    submission_json = args.submission_json.resolve()
    submission = json.loads(submission_json.read_text(encoding="utf-8"))
    status, result_rows = retrieve_completed_job(
        submission=submission,
        credentials_vault=args.credentials_vault,
    )
    print(f"job_status={status}")
    if not result_rows:
        print("raw_counts_available=false")
        return 2
    timestamp = _timestamp()
    submission_sha = _sha256(submission_json)
    raw_payload = raw_payload_from_rows(
        submission=submission,
        result_rows=result_rows,
        submission_json=submission_json,
        submission_sha256=submission_sha,
        timestamp_utc=timestamp,
    )
    tag = f"{timestamp}_{submission['backend']}_n{submission['n_qubits']}"
    raw_path = args.out_dir / f"phase3_large_system_raw_counts_{tag}.json"
    raw_sha = _write_json(raw_path, raw_payload)
    reference_rows = reference_rows_for_submission(submission)
    scale_rows, channel_rows, summary = analyse_large_system_payload(raw_payload, reference_rows)
    outputs = write_analysis_outputs(
        raw_path=raw_path,
        reference_rows=reference_rows,
        scale_rows=scale_rows,
        channel_rows=channel_rows,
        summary=summary,
        output_dir=args.out_dir,
        docs_dir=args.docs_dir,
        result_tag=tag,
    )
    print("raw_counts_available=true")
    print(f"raw_counts_json={outputs.raw_counts_json}")
    print(f"raw_counts_sha256={raw_sha}")
    print(f"analysis_json={outputs.analysis_json}")
    print(f"manifest_md={outputs.manifest_md}")
    print(f"n_channels={summary['n_channels']}")
    print(f"linear_zne_mean_absolute_deviation={summary['linear_zne_mean_absolute_deviation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
