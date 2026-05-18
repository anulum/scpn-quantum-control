#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Phase 2 parity-readout mitigation cross-check
"""Cross-check Phase 2 parity leakage with readout-only parity correction.

The promoted datasets contain selected readout-only calibration states, not a
complete 2^n x 2^n confusion matrix. This script therefore performs the
strongest honest no-new-QPU correction available from the raw counts:
state-specific parity-confusion inversion for circuit rows whose initial state
has an exact readout-only calibration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from scipy.stats import combine_pvalues, ttest_ind

REPO_ROOT = Path(__file__).resolve().parents[1]
AG_INPUT = REPO_ROOT / "data" / "phase2_dla_parity" / "phase2_reduced_ag_2026-05-05T121357Z.json"
POPCOUNT_INPUT = (
    REPO_ROOT
    / "data"
    / "phase2_popcount_control"
    / "phase2_popcount_control_2026-05-05T135318Z.json"
)
OUT_PATH = (
    REPO_ROOT
    / "data"
    / "phase2_readout_mitigation"
    / "phase2_readout_mitigation_summary_2026-05-05.json"
)


@dataclass(frozen=True)
class CorrectedPair:
    """Raw and readout-corrected parity comparison for one dataset depth."""

    dataset: str
    comparison: str
    depth: int
    left_raw: float
    right_raw: float
    left_corrected: float
    right_corrected: float
    raw_relative: float
    corrected_relative: float
    corrected_welch_t: float
    corrected_welch_p: float
    n_left: int
    n_right: int


def _parity(bitstring: str) -> int:
    return bitstring.count("1") % 2


def _total(counts: dict[str, int]) -> int:
    return int(sum(counts.values()))


def _parity_leakage(counts: dict[str, int], initial: str) -> float:
    total = _total(counts)
    if total <= 0:
        raise ValueError("empty count dictionary")
    initial_parity = _parity(initial)
    leaked = sum(
        count for bitstring, count in counts.items() if _parity(bitstring) != initial_parity
    )
    return leaked / total


def _sem(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return stdev(values) / math.sqrt(len(values))


def _correct_parity_leakage(observed: float, readout_flip: float) -> float:
    denominator = 1.0 - 2.0 * readout_flip
    if denominator <= 0.0:
        raise ValueError(f"readout parity flip too large for inversion: {readout_flip}")
    return min(1.0, max(0.0, (observed - readout_flip) / denominator))


def _readout_map(payloads: list[dict[str, Any]]) -> dict[tuple[int, str], float]:
    calibrations: dict[tuple[int, str], float] = {}
    for payload in payloads:
        for row in payload["circuits"]:
            meta = row["meta"]
            is_readout = (
                meta.get("experiment") == "G_readout_baseline" or meta.get("block") == "readout"
            )
            if not is_readout:
                continue
            initial = str(meta["initial"])
            key = (int(meta["n_qubits"]), initial)
            calibrations[key] = _parity_leakage(row["counts"], initial)
    return calibrations


def _summarise_pair(
    dataset: str,
    comparison: str,
    depth: int,
    left_raw: list[float],
    right_raw: list[float],
    left_corrected: list[float],
    right_corrected: list[float],
) -> CorrectedPair:
    left_corr_mean = mean(left_corrected)
    right_corr_mean = mean(right_corrected)
    welch = ttest_ind(left_corrected, right_corrected, equal_var=False)
    return CorrectedPair(
        dataset=dataset,
        comparison=comparison,
        depth=depth,
        left_raw=float(mean(left_raw)),
        right_raw=float(mean(right_raw)),
        left_corrected=float(left_corr_mean),
        right_corrected=float(right_corr_mean),
        raw_relative=float((mean(left_raw) - mean(right_raw)) / mean(right_raw)),
        corrected_relative=float((left_corr_mean - right_corr_mean) / right_corr_mean),
        corrected_welch_t=float(welch.statistic),
        corrected_welch_p=float(welch.pvalue),
        n_left=len(left_corrected),
        n_right=len(right_corrected),
    )


def _ag_rows(
    payload: dict[str, Any], calibrations: dict[tuple[int, str], float]
) -> list[CorrectedPair]:
    buckets: dict[tuple[int, str], dict[str, list[float]]] = {}
    for row in payload["circuits"]:
        meta = row["meta"]
        if meta.get("experiment") != "A_dla_parity_n4_phase2":
            continue
        key = (int(meta["n_qubits"]), str(meta["initial"]))
        if key not in calibrations:
            continue
        depth = int(meta["depth"])
        sector = str(meta["sector"])
        observed = float(row["stats"]["parity_leakage"])
        corrected = _correct_parity_leakage(observed, calibrations[key])
        buckets.setdefault((depth, sector), {"raw": [], "corrected": []})
        buckets[(depth, sector)]["raw"].append(observed)
        buckets[(depth, sector)]["corrected"].append(corrected)

    pairs = []
    for depth in sorted({depth for depth, _ in buckets}):
        if (depth, "even") not in buckets or (depth, "odd") not in buckets:
            continue
        pairs.append(
            _summarise_pair(
                "phase2_ag_n4",
                "even_minus_odd",
                depth,
                buckets[(depth, "even")]["raw"],
                buckets[(depth, "odd")]["raw"],
                buckets[(depth, "even")]["corrected"],
                buckets[(depth, "odd")]["corrected"],
            )
        )
    return pairs


def _popcount_rows(
    payload: dict[str, Any], calibrations: dict[tuple[int, str], float]
) -> list[CorrectedPair]:
    comparisons = {
        "original_E0_minus_O0": ("E0_original_even", "O0_original_odd"),
        "within_even_E0_minus_E1": ("E0_original_even", "E1_even_swap"),
        "within_odd_O0_minus_O1": ("O0_original_odd", "O1_odd_swap"),
        "excitation_inversion_E0_minus_O3": ("E0_original_even", "O3_odd_high_excitation"),
    }
    buckets: dict[tuple[int, str], dict[str, list[float]]] = {}
    for row in payload["circuits"]:
        meta = row["meta"]
        if meta.get("block") != "parity_leakage":
            continue
        key = (int(meta["n_qubits"]), str(meta["initial"]))
        if key not in calibrations:
            continue
        depth = int(meta["depth"])
        label = str(meta["state_label"])
        observed = float(row["stats"]["parity_leakage"])
        corrected = _correct_parity_leakage(observed, calibrations[key])
        buckets.setdefault((depth, label), {"raw": [], "corrected": []})
        buckets[(depth, label)]["raw"].append(observed)
        buckets[(depth, label)]["corrected"].append(corrected)

    pairs = []
    for depth in sorted({depth for depth, _ in buckets}):
        for name, (left, right) in comparisons.items():
            pairs.append(
                _summarise_pair(
                    "phase2_popcount",
                    name,
                    depth,
                    buckets[(depth, left)]["raw"],
                    buckets[(depth, right)]["raw"],
                    buckets[(depth, left)]["corrected"],
                    buckets[(depth, right)]["corrected"],
                )
            )
    return pairs


def _fisher_by_group(pairs: list[CorrectedPair]) -> dict[str, dict[str, float | int]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for pair in pairs:
        grouped.setdefault((pair.dataset, pair.comparison), []).append(pair.corrected_welch_p)
    out: dict[str, dict[str, float | int]] = {}
    for (dataset, comparison), p_values in sorted(grouped.items()):
        chi2, p = combine_pvalues(p_values, method="fisher")
        out[f"{dataset}:{comparison}"] = {
            "chi2": float(chi2),
            "p": float(p),
            "n_depths_significant_at_0_05": sum(value < 0.05 for value in p_values),
            "n_depths_tested": len(p_values),
        }
    return out


def main() -> int:
    """Run the Phase 2 parity-readout mitigation cross-check CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-json", action="store_true")
    args = parser.parse_args()

    ag_payload = json.loads(AG_INPUT.read_text(encoding="utf-8"))
    popcount_payload = json.loads(POPCOUNT_INPUT.read_text(encoding="utf-8"))
    calibrations = _readout_map([ag_payload, popcount_payload])
    pairs = _ag_rows(ag_payload, calibrations) + _popcount_rows(popcount_payload, calibrations)
    calibrated_initials = sorted(f"n={n}:{initial}" for n, initial in calibrations)
    summary: dict[str, Any] = {
        "method": "state_specific_parity_confusion_inversion",
        "full_confusion_matrix_available": False,
        "full_confusion_matrix_note": (
            "Existing Phase 2 raw counts include selected readout-only calibration states, "
            "not all computational basis states; a literal 2^n x 2^n confusion-matrix "
            "inversion would require new calibration circuits."
        ),
        "calibrated_initials": calibrated_initials,
        "pairs": [asdict(pair) for pair in pairs],
        "fisher_by_group": _fisher_by_group(pairs),
    }

    if args.write_json:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Phase 2 readout-mitigation cross-check")
    print(f"  full confusion matrix available: {summary['full_confusion_matrix_available']}")
    print(f"  calibrated initial states: {len(calibrated_initials)}")
    for name, stats in summary["fisher_by_group"].items():
        print(
            f"  {name}: chi2={stats['chi2']:.6f}, p={stats['p']:.6e}, "
            f"sig={stats['n_depths_significant_at_0_05']}/{stats['n_depths_tested']}"
        )
    if args.write_json:
        print(f"  wrote: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
