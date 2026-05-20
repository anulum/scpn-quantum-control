#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
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
    analysis_jobs: set[str] = set()
    for circuit in payload.get("circuits", []):
        meta = circuit.get("meta", {})
        if meta.get("block") != "main":
            continue
        key = _reference_key(meta)
        reference = references.get(key)
        if reference is None:
            raise ValueError(f"missing reference row for {key}")
        grouped[key].append(
            pauli_expectation_from_counts(circuit.get("counts", {}), str(reference["pauli_label"]))
        )
        if circuit.get("job_id"):
            analysis_jobs.add(str(circuit["job_id"]))

    rows: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        reference = references[key]
        exact = float(reference["exact_expectation"])
        measured = float(mean(values))
        stderr = _stderr(values)
        deviation = measured - exact
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
                "exact_expectation": exact,
                "deviation_from_exact": deviation,
                "absolute_deviation": abs(deviation),
                "z_score_vs_exact": deviation / stderr if stderr > 0.0 else None,
                "half_chain_purity_reference": float(reference["half_chain_purity"]),
                "parity_survival_reference": float(reference["parity_survival_ideal"]),
            }
        )
    max_abs = max((row["absolute_deviation"] for row in rows), default=None)
    mean_abs = mean([row["absolute_deviation"] for row in rows]) if rows else None
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
            "<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->",
            "<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->",
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
) -> tuple[Path, Path, Path]:
    """Write analysis JSON, CSV, and Markdown artefacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"entanglement_tomography_summary_{TODAY}.json"
    csv_path = output_dir / f"entanglement_tomography_rows_{TODAY}.csv"
    md_path = docs_dir / f"phase3_entanglement_tomography_manifest_{TODAY}.md"
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
    )
    print(f"wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"wrote {csv_path.relative_to(REPO_ROOT)}")
    print(f"wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"n_observable_rows={summary['n_observable_rows']}")
    print(f"mean_absolute_deviation={summary['mean_absolute_deviation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
