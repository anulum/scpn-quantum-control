#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — analyse phase3 entanglement ZNE script
# scpn-quantum-control -- Phase 3 ZNE reduced-Pauli analysis
"""Analyse Phase 3 reduced-Pauli ZNE raw-count artefacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from statistics import mean, stdev
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"
TODAY = date(2026, 5, 20).isoformat()
DEFAULT_COUNTS = (
    REPO_ROOT
    / "data"
    / "phase3_entanglement_tomography"
    / "entanglement_tomography_live_ibm_fez_2026-05-20T023600Z.json"
)
DEFAULT_REFERENCE_CSV = (
    REPO_ROOT
    / "data"
    / "phase3_entanglement_tomography"
    / "entanglement_observable_rows_2026-05-07.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "phase3_entanglement_tomography"
DEFAULT_DOCS_DIR = REPO_ROOT / "docs"
N_QUBITS = 4


def _load_script_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tomography_module() -> ModuleType:
    return _load_script_module(
        "analyse_phase3_entanglement_tomography",
        SCRIPTS_DIR / "analyse_phase3_entanglement_tomography.py",
    )


def _zne_module() -> ModuleType:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from scpn_quantum_control.mitigation import zne

    return zne


def _load_reference_rows(path: Path) -> dict[tuple[str, str, str, int, str, str], dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: dict[tuple[str, str, str, int, str, str], dict[str, Any]] = {}
        for row in reader:
            key = (
                str(row["family"]),
                str(row["label"]),
                str(row["initial"]),
                int(row["depth"]),
                str(row.get("lambda_fim") or ""),
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


def _fit(scales: Sequence[int], values: Sequence[float], *, order: int) -> tuple[float, float]:
    result = _zne_module().zne_extrapolate(list(scales), list(values), order=order)
    return float(result.zero_noise_estimate), float(result.fit_residual)


def analyse_zne_counts_artifact(
    counts_path: Path,
    reference_csv: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Reduce a completed ZNE counts artefact into scale and channel rows."""

    payload = json.loads(counts_path.read_text(encoding="utf-8"))
    references = _load_reference_rows(reference_csv)
    tomography = _tomography_module()
    circuits = payload.get("circuits", [])
    readout_circuits = [
        circuit for circuit in circuits if circuit.get("meta", {}).get("block") == "readout"
    ]
    readout_model = tomography.build_readout_mitigation_model(readout_circuits, width=N_QUBITS)

    grouped: dict[tuple[str, str, str, int, str, str, int], list[float]] = defaultdict(list)
    grouped_mitigated: dict[tuple[str, str, str, int, str, str, int], list[float]] = defaultdict(
        list
    )
    analysis_jobs: set[str] = set()
    for circuit in circuits:
        meta = circuit.get("meta", {})
        if meta.get("block") != "main":
            continue
        if "zne_noise_scale" not in meta:
            raise ValueError("ZNE main circuit is missing zne_noise_scale metadata")
        reference_key = _reference_key(meta)
        reference = references.get(reference_key)
        if reference is None:
            raise ValueError(f"missing reference row for {reference_key}")
        scale = int(meta["zne_noise_scale"])
        key = (*reference_key, scale)
        counts = circuit.get("counts", {})
        pauli_label = str(reference["pauli_label"])
        grouped[key].append(tomography.pauli_expectation_from_counts(counts, pauli_label))
        grouped_mitigated[key].append(
            tomography.mitigated_pauli_expectation(counts, pauli_label, readout_model)
        )
        if circuit.get("job_id"):
            analysis_jobs.add(str(circuit["job_id"]))

    scale_rows: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        reference = references[key[:6]]
        exact = float(reference["exact_expectation"])
        measured = float(mean(values))
        mitigated_values = grouped_mitigated[key]
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
        linear_raw, linear_raw_residual = _fit(scales, raw_values, order=1)
        linear_mitigated, linear_mitigated_residual = _fit(scales, mitigated_values, order=1)
        quadratic_raw = quadratic_raw_residual = None
        quadratic_mitigated = quadratic_mitigated_residual = None
        if len(scales) >= 3:
            quadratic_raw, quadratic_raw_residual = _fit(scales, raw_values, order=2)
            quadratic_mitigated, quadratic_mitigated_residual = _fit(
                scales,
                mitigated_values,
                order=2,
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
                "linear_zne_expectation": linear_raw,
                "linear_zne_fit_residual": linear_raw_residual,
                "linear_zne_absolute_deviation": abs(linear_raw - exact),
                "readout_mitigated_linear_zne_expectation": linear_mitigated,
                "readout_mitigated_linear_zne_fit_residual": linear_mitigated_residual,
                "readout_mitigated_linear_zne_absolute_deviation": abs(linear_mitigated - exact),
                "quadratic_zne_expectation": quadratic_raw,
                "quadratic_zne_fit_residual": quadratic_raw_residual,
                "quadratic_zne_absolute_deviation": (
                    abs(quadratic_raw - exact) if quadratic_raw is not None else None
                ),
                "readout_mitigated_quadratic_zne_expectation": quadratic_mitigated,
                "readout_mitigated_quadratic_zne_fit_residual": quadratic_mitigated_residual,
                "readout_mitigated_quadratic_zne_absolute_deviation": (
                    abs(quadratic_mitigated - exact) if quadratic_mitigated is not None else None
                ),
            }
        )

    scale1_deviations = [row["scale1_absolute_deviation"] for row in channel_rows]
    linear_deviations = [row["linear_zne_absolute_deviation"] for row in channel_rows]
    mitigated_linear_deviations = [
        row["readout_mitigated_linear_zne_absolute_deviation"] for row in channel_rows
    ]
    quadratic_deviations = [
        row["quadratic_zne_absolute_deviation"]
        for row in channel_rows
        if row["quadratic_zne_absolute_deviation"] is not None
    ]
    mitigated_quadratic_deviations = [
        row["readout_mitigated_quadratic_zne_absolute_deviation"]
        for row in channel_rows
        if row["readout_mitigated_quadratic_zne_absolute_deviation"] is not None
    ]
    summary = {
        "schema": "scpn_phase3_entanglement_zne_analysis_v1",
        "counts_artifact": _display_path(counts_path),
        "reference_csv": _display_path(reference_csv),
        "backend": payload.get("backend"),
        "status": payload.get("status"),
        "job_ids": [str(job_id) for job_id in payload.get("job_ids", [])],
        "analysis_job_ids": sorted(analysis_jobs),
        "n_scale_rows": len(scale_rows),
        "n_channels": len(channel_rows),
        "noise_scales": sorted({int(row["zne_noise_scale"]) for row in scale_rows}),
        "scale1_mean_absolute_deviation": (mean(scale1_deviations) if scale1_deviations else None),
        "scale1_max_absolute_deviation": max(scale1_deviations, default=None),
        "linear_zne_mean_absolute_deviation": (
            mean(linear_deviations) if linear_deviations else None
        ),
        "linear_zne_max_absolute_deviation": max(linear_deviations, default=None),
        "readout_mitigated_linear_zne_mean_absolute_deviation": (
            mean(mitigated_linear_deviations) if mitigated_linear_deviations else None
        ),
        "readout_mitigated_linear_zne_max_absolute_deviation": max(
            mitigated_linear_deviations,
            default=None,
        ),
        "quadratic_zne_mean_absolute_deviation": (
            mean(quadratic_deviations) if quadratic_deviations else None
        ),
        "quadratic_zne_max_absolute_deviation": max(quadratic_deviations, default=None),
        "readout_mitigated_quadratic_zne_mean_absolute_deviation": (
            mean(mitigated_quadratic_deviations) if mitigated_quadratic_deviations else None
        ),
        "readout_mitigated_quadratic_zne_max_absolute_deviation": max(
            mitigated_quadratic_deviations,
            default=None,
        ),
        "readout_mitigation": readout_model,
        "claim_boundary": (
            "small preregistered reduced-Pauli ZNE stress test only; not a "
            "backend-general, advantage, full-tomography, or full-causal mechanism claim"
        ),
    }
    return scale_rows, channel_rows, summary


def _manifest(
    summary: Mapping[str, Any],
    *,
    json_path: Path,
    scale_path: Path,
    channel_path: Path,
) -> str:
    return "\n".join(
        [
            "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
            "<!-- Commercial license available -->",
            "<!-- © Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->",
            "<!-- © Code 2020-2026 Miroslav Sotek. All rights reserved. -->",
            "<!-- ORCID: 0009-0009-3560-0851 -->",
            "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
            "<!-- scpn-quantum-control -- Phase 3 ZNE analysis manifest -->",
            "",
            "# Phase 3 Entanglement ZNE Analysis Manifest",
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
            f"- Scale rows: `{_display_path(scale_path)}`",
            f"- Scale rows SHA256: `{_sha256(scale_path)}`",
            f"- Channel summary: `{_display_path(channel_path)}`",
            f"- Channel summary SHA256: `{_sha256(channel_path)}`",
            "",
            "## Result Snapshot",
            "",
            f"- Scale rows: `{summary['n_scale_rows']}`",
            f"- Channels: `{summary['n_channels']}`",
            f"- Noise scales: `{summary.get('noise_scales', [])}`",
            "- Scale-1 mean absolute deviation: "
            f"`{summary.get('scale1_mean_absolute_deviation')}`",
            "- Linear ZNE mean absolute deviation: "
            f"`{summary.get('linear_zne_mean_absolute_deviation')}`",
            "- Readout-mitigated linear ZNE mean absolute deviation: "
            f"`{summary.get('readout_mitigated_linear_zne_mean_absolute_deviation')}`",
            "- Quadratic ZNE mean absolute deviation: "
            f"`{summary.get('quadratic_zne_mean_absolute_deviation')}`",
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
    scale_rows: Sequence[Mapping[str, Any]],
    channel_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    output_dir: Path,
    docs_dir: Path,
    result_tag: str = TODAY,
) -> tuple[Path, Path, Path, Path]:
    """Write ZNE JSON, scale CSV, channel CSV, and Markdown manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"entanglement_zne_summary_{result_tag}.json"
    scale_path = output_dir / f"entanglement_zne_scale_rows_{result_tag}.csv"
    channel_path = output_dir / f"entanglement_zne_channel_summary_{result_tag}.csv"
    md_path = docs_dir / f"phase3_entanglement_zne_manifest_{result_tag}.md"
    _write_csv(scale_path, scale_rows)
    _write_csv(channel_path, channel_rows)
    payload = dict(summary)
    payload["scale_rows_sha256"] = _sha256(scale_path)
    payload["channel_summary_sha256"] = _sha256(channel_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(
        _manifest(payload, json_path=json_path, scale_path=scale_path, channel_path=channel_path),
        encoding="utf-8",
    )
    return json_path, scale_path, channel_path, md_path


def parse_args() -> argparse.Namespace:
    """Parse analysis command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("counts_artifact", type=Path, nargs="?", default=DEFAULT_COUNTS)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument(
        "--result-tag",
        default=f"{TODAY}_ibm_fez_zne",
        help="filename tag for the ZNE reduction artefacts",
    )
    return parser.parse_args()


def main() -> int:
    """Analyse an approved Phase 3 ZNE raw-count artefact."""

    args = parse_args()
    scale_rows, channel_rows, summary = analyse_zne_counts_artifact(
        args.counts_artifact,
        args.reference_csv,
    )
    json_path, scale_path, channel_path, md_path = write_outputs(
        scale_rows,
        channel_rows,
        summary,
        output_dir=args.output_dir,
        docs_dir=args.docs_dir,
        result_tag=args.result_tag,
    )
    print(f"wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"wrote {scale_path.relative_to(REPO_ROOT)}")
    print(f"wrote {channel_path.relative_to(REPO_ROOT)}")
    print(f"wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"n_scale_rows={summary['n_scale_rows']}")
    print(f"n_channels={summary['n_channels']}")
    print(f"linear_zne_mean_absolute_deviation={summary['linear_zne_mean_absolute_deviation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
