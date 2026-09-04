# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — powered depth-ordering calibration-epoch design
"""Freeze the powered depth-ordering design from prior epoch evidence.

The completed window-variability study is used only to choose a future
confirmatory design. The new study uses none of its counts in the endpoint.
Power is evaluated conservatively with the maximum binomial variance for four
proportions per epoch and a one-sided noncentral-t approximation matching the
safeguarded HKSJ decision threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scipy.stats import nct, t

SCHEMA = "scpn.iqm-dla-depth-profile-powered-design.v1"
DESIGN_EVIDENCE_SCHEMA = "scpn.iqm-window-variability-epoch-sensitivity.v2"
CAMPAIGN = "iqm_dla_depth_profile_powered_epoch_prereg_2026-09-04"
DEPTHS = (8, 12)
EPOCHS = 12
REPETITIONS = 12
SHOTS_PER_REPETITION = 1024
SHOTS_PER_ARM_DEPTH = REPETITIONS * SHOTS_PER_REPETITION
READOUT_STATES = 4
READOUT_SHOTS = 2048
ALPHA = 0.05
TARGET_POWER = 0.90
REPO_ROOT = Path(__file__).resolve().parents[1]


def _portable(value: float) -> float:
    """Round derived floats to a cross-version stable representation."""
    return float(f"{value:.12g}")


def _load(path: Path) -> dict[str, Any]:
    """Load one JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _display_path(path: Path) -> str:
    """Return stable repository-relative provenance when the source is local."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _design_rows(payload: dict[str, Any]) -> tuple[list[float], list[float], list[str]]:
    """Extract prior epoch contrasts, shot variances, and calibration IDs."""
    epoch_level = payload.get("epoch_level_sensitivity")
    if not isinstance(epoch_level, dict):
        raise ValueError("window-variability artefact has no epoch_level_sensitivity object")
    raw_rows = epoch_level.get("per_epoch")
    if not isinstance(raw_rows, list) or len(raw_rows) < 2:
        raise ValueError("window-variability artefact has fewer than two calibration epochs")
    effects: list[float] = []
    variances: list[float] = []
    calibration_ids: list[str] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise ValueError("window-variability per_epoch row is not an object")
        depths = raw.get("depths")
        if not isinstance(depths, dict):
            raise ValueError("window-variability per_epoch row has no depths object")
        d8 = depths.get("8")
        d12 = depths.get("12")
        if not isinstance(d8, dict) or not isinstance(d12, dict):
            raise ValueError("window-variability epoch lacks depth 8 or depth 12")
        effects.append(
            float(d8["difference_even_minus_odd"]) - float(d12["difference_even_minus_odd"])
        )
        variances.append(float(d8["shot_noise_se"]) ** 2 + float(d12["shot_noise_se"]) ** 2)
        calibration_id = raw.get("calibration_set_id")
        if not isinstance(calibration_id, str) or not calibration_id:
            raise ValueError("window-variability epoch lacks calibration_set_id")
        calibration_ids.append(calibration_id)
    if len(calibration_ids) != len(set(calibration_ids)):
        raise ValueError("window-variability artefact repeats a calibration_set_id")
    return effects, variances, calibration_ids


def _dersimonian_laird(effects: list[float], variances: list[float]) -> tuple[float, float]:
    """Return the random-effects mean and DerSimonian-Laird tau."""
    if len(effects) != len(variances) or len(effects) < 2:
        raise ValueError("random-effects input cardinality is invalid")
    weights = [1.0 / variance for variance in variances]
    weight_sum = sum(weights)
    fixed_mean = sum(weight * effect for weight, effect in zip(weights, effects, strict=True))
    fixed_mean /= weight_sum
    q = sum(
        weight * (effect - fixed_mean) ** 2
        for weight, effect in zip(weights, effects, strict=True)
    )
    c = weight_sum - sum(weight**2 for weight in weights) / weight_sum
    tau_squared = max(0.0, (q - (len(effects) - 1)) / c)
    random_weights = [1.0 / (variance + tau_squared) for variance in variances]
    random_mean = sum(
        weight * effect for weight, effect in zip(random_weights, effects, strict=True)
    ) / sum(random_weights)
    return random_mean, math.sqrt(tau_squared)


def build_design(source: Path) -> dict[str, Any]:
    """Build the deterministic future-study design artefact."""
    source_payload = _load(source)
    if source_payload.get("schema") != DESIGN_EVIDENCE_SCHEMA:
        raise ValueError(
            "design evidence must use the descriptive calibration-epoch "
            f"schema {DESIGN_EVIDENCE_SCHEMA}"
        )
    effects, variances, calibration_ids = _design_rows(source_payload)
    design_effect, tau = _dersimonian_laird(effects, variances)
    conservative_within_epoch_variance = 1.0 / SHOTS_PER_ARM_DEPTH
    total_epoch_sd = math.sqrt(tau**2 + conservative_within_epoch_variance)
    degrees_of_freedom = EPOCHS - 1
    critical_t = float(t.ppf(1.0 - ALPHA, degrees_of_freedom))
    noncentrality = design_effect * math.sqrt(EPOCHS) / total_epoch_sd
    projected_power = float(nct.sf(critical_t, degrees_of_freedom, noncentrality))
    main_circuits_per_epoch = len(DEPTHS) * 2 * REPETITIONS
    main_shots_per_epoch = main_circuits_per_epoch * SHOTS_PER_REPETITION
    readout_shots_per_epoch = READOUT_STATES * READOUT_SHOTS
    return {
        "schema": SCHEMA,
        "campaign": CAMPAIGN,
        "source": {
            "schema": DESIGN_EVIDENCE_SCHEMA,
            "path": _display_path(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "role": "design_only_excluded_from_confirmatory_endpoint",
            "calibration_epochs": len(effects),
            "excluded_calibration_set_ids": sorted(calibration_ids),
            "observed_epoch_contrasts_delta8_minus_delta12": [
                _portable(effect) for effect in effects
            ],
            "random_effects_design_mean": _portable(design_effect),
            "dersimonian_laird_tau": _portable(tau),
        },
        "frozen_design": {
            "primary_contrast": "(delta_8 - delta_12) > 0",
            "alpha": ALPHA,
            "sidedness": "one-sided",
            "decision_method": "safeguarded_HKSJ_random_effects_t",
            "distinct_calibration_epochs": EPOCHS,
            "repetitions_per_state_depth_epoch": REPETITIONS,
            "shots_per_repetition": SHOTS_PER_REPETITION,
            "shots_per_arm_depth_epoch": SHOTS_PER_ARM_DEPTH,
            "conservative_within_epoch_variance": _portable(conservative_within_epoch_variance),
            "projected_total_epoch_sd": _portable(total_epoch_sd),
            "degrees_of_freedom": degrees_of_freedom,
            "critical_t": _portable(critical_t),
            "noncentrality": _portable(noncentrality),
            "projected_power": _portable(projected_power),
            "target_power": TARGET_POWER,
            "target_power_met": projected_power >= TARGET_POWER,
        },
        "matrix_and_budget": {
            "main_circuits_per_epoch": main_circuits_per_epoch,
            "readout_circuits_per_epoch": READOUT_STATES,
            "circuits_per_epoch": main_circuits_per_epoch + READOUT_STATES,
            "main_shots_per_epoch": main_shots_per_epoch,
            "readout_shots_per_epoch": readout_shots_per_epoch,
            "shots_per_epoch": main_shots_per_epoch + readout_shots_per_epoch,
            "total_shots": EPOCHS * (main_shots_per_epoch + readout_shots_per_epoch),
            "jobs_per_epoch": 2,
            "estimated_total_jobs": EPOCHS * 2,
            "estimated_credits": EPOCHS * 2,
            "credit_note": (
                "planning estimate from prior two-job calibration-epoch batching; "
                "dashboard is authoritative"
            ),
        },
        "admission_boundary": (
            "each paid block requires a calibration_set_id absent from the design evidence "
            "and all prior confirmatory epochs; repeated IDs are zero-submit skips"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Write the frozen powered depth-ordering design artefact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-evidence", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    source = Path(args.design_evidence)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    design = build_design(source)
    output.write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")
    frozen = design["frozen_design"]
    print(
        f"Powered depth-ordering design: {output} — "
        f"{frozen['distinct_calibration_epochs']} epochs, "
        f"power={frozen['projected_power']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
