#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Phase 3 GUESS DLA analysis
"""Analyse the Phase 3 folded-noise GUESS / symmetry-decay run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "phase3_guess_dla"
DEFAULT_INPUT = DATA_DIR / "phase3_guess_ibm_marrakesh_2026-05-06T234602Z.json"
TODAY = date(2026, 5, 7).isoformat()


@dataclass(frozen=True)
class WitnessRow:
    """Mean witness value for one state/depth/noise-scale cell."""

    sector: str
    initial: str
    depth: int
    noise_scale: int
    n_reps: int
    mean_leakage_raw: float
    std_leakage_raw: float
    mean_leakage_corrected: float | None
    mean_survival_raw: float
    mean_survival_corrected: float | None
    mean_state_retention: float

    def to_row(self) -> dict[str, object]:
        return {
            "sector": self.sector,
            "initial": self.initial,
            "depth": self.depth,
            "noise_scale": self.noise_scale,
            "n_reps": self.n_reps,
            "mean_leakage_raw": self.mean_leakage_raw,
            "std_leakage_raw": self.std_leakage_raw,
            "mean_leakage_corrected": self.mean_leakage_corrected,
            "mean_survival_raw": self.mean_survival_raw,
            "mean_survival_corrected": self.mean_survival_corrected,
            "mean_state_retention": self.mean_state_retention,
        }


@dataclass(frozen=True)
class FitRow:
    """Log-survival fit for one sector/depth series."""

    sector: str
    initial: str
    depth: int
    correction: str
    n_points: int
    alpha_per_noise_scale: float
    intercept_log_survival: float
    extrapolated_survival_scale0: float
    r_squared: float
    rmse_log_survival: float
    monotone_survival_decay: bool
    usable_for_guess_witness: bool

    def to_row(self) -> dict[str, object]:
        return {
            "sector": self.sector,
            "initial": self.initial,
            "depth": self.depth,
            "correction": self.correction,
            "n_points": self.n_points,
            "alpha_per_noise_scale": self.alpha_per_noise_scale,
            "intercept_log_survival": self.intercept_log_survival,
            "extrapolated_survival_scale0": self.extrapolated_survival_scale0,
            "r_squared": self.r_squared,
            "rmse_log_survival": self.rmse_log_survival,
            "monotone_survival_decay": self.monotone_survival_decay,
            "usable_for_guess_witness": self.usable_for_guess_witness,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parity(bitstring: str) -> int:
    return bitstring.replace(" ", "").count("1") % 2


def _leakage_from_counts(counts: Mapping[str, int], initial: str) -> float:
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("empty count dictionary")
    initial_parity = _parity(initial)
    opposite = sum(count for bits, count in counts.items() if _parity(bits) != initial_parity)
    return opposite / total


def _readout_flip_rates(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    rates: dict[str, float] = {}
    for row in rows:
        meta = row["meta"]
        if meta.get("block") != "readout":
            continue
        initial = str(meta["initial"])
        rates[initial] = _leakage_from_counts(row["counts"], initial)
    return rates


def _correct_leakage(observed: float, readout_flip: float | None) -> float | None:
    if readout_flip is None:
        return None
    denom = 1.0 - 2.0 * readout_flip
    if denom <= 0.0:
        return None
    return min(1.0, max(0.0, (observed - readout_flip) / denom))


def build_witness_rows(payload: Mapping[str, Any]) -> tuple[WitnessRow, ...]:
    """Aggregate raw and exact-state-readout-corrected witness rows."""
    circuits = [row for row in payload["circuits"] if row["meta"].get("block") == "main"]
    readout_rates = _readout_flip_rates(payload["circuits"])
    grouped: dict[tuple[str, str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in circuits:
        meta = row["meta"]
        key = (
            str(meta["sector"]),
            str(meta["initial"]),
            int(meta["depth"]),
            int(meta["noise_scale"]),
        )
        grouped[key].append(row)

    witness_rows: list[WitnessRow] = []
    for (sector, initial, depth, scale), rows in sorted(grouped.items()):
        leakages = [float(row["stats"]["parity_leakage"]) for row in rows]
        corrected = [_correct_leakage(value, readout_rates.get(initial)) for value in leakages]
        corrected_values = [value for value in corrected if value is not None]
        retentions = [float(row["stats"]["initial_state_retention"]) for row in rows]
        mean_corrected = float(mean(corrected_values)) if corrected_values else None
        witness_rows.append(
            WitnessRow(
                sector=sector,
                initial=initial,
                depth=depth,
                noise_scale=scale,
                n_reps=len(rows),
                mean_leakage_raw=float(mean(leakages)),
                std_leakage_raw=float(pstdev(leakages)) if len(leakages) > 1 else 0.0,
                mean_leakage_corrected=mean_corrected,
                mean_survival_raw=1.0 - float(mean(leakages)),
                mean_survival_corrected=None if mean_corrected is None else 1.0 - mean_corrected,
                mean_state_retention=float(mean(retentions)),
            )
        )
    return tuple(witness_rows)


def _linear_fit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float, float, float]:
    x_mean = mean(xs)
    y_mean = mean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    slope = (
        0.0 if denom == 0.0 else sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
    )
    intercept = y_mean - slope * x_mean
    preds = [slope * x + intercept for x in xs]
    ss_res = sum((y - pred) ** 2 for y, pred in zip(ys, preds))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    rmse = math.sqrt(ss_res / len(xs))
    return float(slope), float(intercept), float(r_squared), float(rmse)


def build_fit_rows(witness_rows: Sequence[WitnessRow]) -> tuple[FitRow, ...]:
    """Fit log survival versus explicit folded-noise scale."""
    grouped: dict[tuple[str, str, int], list[WitnessRow]] = defaultdict(list)
    for row in witness_rows:
        grouped[(row.sector, row.initial, row.depth)].append(row)

    fits: list[FitRow] = []
    for (sector, initial, depth), rows in sorted(grouped.items()):
        for correction in ("raw", "exact_state_readout_corrected"):
            scale_rows = sorted(rows, key=lambda item: item.noise_scale)
            survivals: list[float] = []
            for row in scale_rows:
                value = (
                    row.mean_survival_raw if correction == "raw" else row.mean_survival_corrected
                )
                if value is None:
                    survivals = []
                    break
                survivals.append(max(float(value), 1e-9))
            if not survivals:
                continue
            xs = [float(row.noise_scale) for row in scale_rows]
            ys = [math.log(value) for value in survivals]
            slope, intercept, r_squared, rmse = _linear_fit(xs, ys)
            monotone = all(a >= b for a, b in zip(survivals, survivals[1:]))
            usable = bool(monotone and r_squared >= 0.90 and rmse <= 0.08)
            fits.append(
                FitRow(
                    sector=sector,
                    initial=initial,
                    depth=depth,
                    correction=correction,
                    n_points=len(scale_rows),
                    alpha_per_noise_scale=max(0.0, -slope),
                    intercept_log_survival=intercept,
                    extrapolated_survival_scale0=min(1.0, max(0.0, math.exp(intercept))),
                    r_squared=r_squared,
                    rmse_log_survival=rmse,
                    monotone_survival_decay=monotone,
                    usable_for_guess_witness=usable,
                )
            )
    return tuple(fits)


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(payload: Mapping[str, Any], *, input_path: Path) -> dict[str, Any]:
    """Build the complete analysis payload."""
    witness_rows = build_witness_rows(payload)
    fit_rows = build_fit_rows(witness_rows)
    usable = [row for row in fit_rows if row.usable_for_guess_witness]
    raw_usable = [row for row in usable if row.correction == "raw"]
    corrected_usable = [row for row in usable if row.correction == "exact_state_readout_corrected"]
    return {
        "schema": "scpn_phase3_guess_dla_analysis_v1",
        "source_artifact": str(input_path.relative_to(REPO_ROOT)),
        "source_sha256": _sha256(input_path),
        "backend": payload["backend"],
        "job_ids": payload["job_ids"],
        "hardware_submission": True,
        "n_witness_rows": len(witness_rows),
        "n_fit_rows": len(fit_rows),
        "n_usable_raw_fits": len(raw_usable),
        "n_usable_corrected_fits": len(corrected_usable),
        "decision_flags": {
            "raw_has_any_usable_guess_witness": bool(raw_usable),
            "corrected_has_any_usable_guess_witness": bool(corrected_usable),
            "all_corrected_fits_usable": len(corrected_usable)
            == len([row for row in fit_rows if row.correction == "exact_state_readout_corrected"]),
        },
        "claim_boundary": {
            "supported": [
                "folded-noise parity-survival witness fit for sampled backend/layout/window",
                "exact-state parity-readout correction where matching readout states exist",
            ],
            "blocked": [
                "universal GUESS mitigation performance",
                "backend-general transfer",
                "quantum advantage",
                "full confusion-matrix readout mitigation",
            ],
        },
        "witness_rows": [row.to_row() for row in witness_rows],
        "fit_rows": [row.to_row() for row in fit_rows],
    }


def _manifest(
    summary: Mapping[str, Any], summary_path: Path, fit_path: Path, witness_path: Path
) -> str:
    return "\n".join(
        [
            "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
            "<!-- Commercial license available -->",
            "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
            "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
            "<!-- ORCID: 0009-0009-3560-0851 -->",
            "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
            "<!-- SCPN Quantum Control — Phase 3 GUESS DLA Manifest -->",
            "",
            "# Phase 3 GUESS DLA Manifest",
            "",
            f"Date: {TODAY}",
            "",
            "## Hardware jobs",
            "",
            f"- Backend: `{summary['backend']}`",
            f"- Job IDs: `{', '.join(summary['job_ids'])}`",
            f"- Source artefact: `{summary['source_artifact']}`",
            f"- Source SHA256: `{summary['source_sha256']}`",
            "",
            "## Decision flags",
            "",
            f"- Raw usable fits: `{summary['n_usable_raw_fits']}`",
            f"- Corrected usable fits: `{summary['n_usable_corrected_fits']}`",
            "",
            "## Artefacts",
            "",
            f"- Summary JSON: `{summary_path.relative_to(REPO_ROOT)}`",
            f"- Fit rows: `{fit_path.relative_to(REPO_ROOT)}`",
            f"- Witness rows: `{witness_path.relative_to(REPO_ROOT)}`",
            "",
            "## Reproduction",
            "",
            "```bash",
            "./.venv-linux/bin/python scripts/analyse_phase3_guess_dla.py",
            "```",
            "",
        ]
    )


def write_outputs(summary: Mapping[str, Any]) -> tuple[Path, Path, Path, Path]:
    """Write summary, fit rows, witness rows, and manifest."""
    summary_path = DATA_DIR / f"phase3_guess_summary_{TODAY}.json"
    fit_path = DATA_DIR / f"phase3_guess_fit_rows_{TODAY}.csv"
    witness_path = DATA_DIR / f"phase3_guess_extrapolation_rows_{TODAY}.csv"
    manifest_path = REPO_ROOT / "docs" / f"phase3_guess_dla_manifest_{TODAY}.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(fit_path, summary["fit_rows"])
    _write_csv(witness_path, summary["witness_rows"])
    manifest_path.write_text(
        _manifest(summary, summary_path, fit_path, witness_path),
        encoding="utf-8",
    )
    return summary_path, fit_path, witness_path, manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    summary = build_summary(payload, input_path=args.input)
    paths = write_outputs(summary)
    for path in paths:
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    print(f"usable raw fits: {summary['n_usable_raw_fits']}")
    print(f"usable corrected fits: {summary['n_usable_corrected_fits']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
