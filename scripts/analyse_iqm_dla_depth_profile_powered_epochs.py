# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — frozen powered depth-ordering epoch analysis
"""Analyse the preregistered powered depth-ordering campaign.

The independent unit is a distinct IQM ``calibration_set_id``. The primary
endpoint is a one-sided safeguarded Hartung-Knapp-Sidik-Jonkman random-effects
test of the epoch contrast ``(delta_8 - delta_12) > 0``. The script refuses
partial, repeated-calibration, design-contaminated, or wrong-matrix inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from scipy.stats import t

if TYPE_CHECKING:
    from scripts import iqm_depth_ordering_protocol as protocol
elif __package__:
    from . import iqm_depth_ordering_protocol as protocol
else:
    import iqm_depth_ordering_protocol as protocol

SCHEMA = protocol.ANALYSIS_SCHEMA
CAMPAIGN = protocol.CAMPAIGN_ID
PRIMARY_LAYOUT = list(protocol.PRIMARY_LAYOUT)
DEPTHS = protocol.DEPTHS
SECTORS = protocol.SECTORS
REPETITIONS = protocol.REPETITIONS
MAIN_SHOTS = protocol.MAIN_SHOTS


def _object(path: Path) -> dict[str, Any]:
    """Load one JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _portable_json(value: Any) -> Any:
    """Round derived floats for stable cross-runtime JSON evidence."""
    if isinstance(value, float):
        return float(f"{value:.12g}")
    if isinstance(value, list):
        return [_portable_json(item) for item in value]
    if isinstance(value, dict):
        return {key: _portable_json(item) for key, item in value.items()}
    return value


def _parity(bitstring: str) -> int:
    """Return computational-basis parity after removing register spaces."""
    return bitstring.replace(" ", "").count("1") % 2


def _leak(counts: dict[str, Any], initial: str) -> tuple[int, int]:
    """Return parity-leaked and total shots for one count mapping."""
    if not all(
        isinstance(state, str)
        and state
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
        for state, value in counts.items()
    ):
        raise ValueError("count block has a malformed state or count")
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("empty count block")
    expected = _parity(initial)
    leaked = sum(value for state, value in counts.items() if _parity(state) != expected)
    return leaked, total


def _delta(raw_counts: dict[str, Any], depth: int) -> tuple[float, float, dict[str, Any]]:
    """Pool one epoch's repetitions and return delta plus binomial variance."""
    arms: dict[str, tuple[int, int]] = {}
    for sector, initial in SECTORS.items():
        leaked = total = 0
        for repetition in REPETITIONS:
            label = f"main_d{depth}_{sector}_rep{repetition}"
            block = raw_counts.get(label)
            if not isinstance(block, dict):
                raise ValueError(f"missing count block {label}")
            block_leaked, block_total = _leak(block, initial)
            if block_total != MAIN_SHOTS:
                raise ValueError(f"{label} has {block_total} shots, expected {MAIN_SHOTS}")
            leaked += block_leaked
            total += block_total
        arms[sector] = leaked, total
    leaked_even, shots_even = arms["even"]
    leaked_odd, shots_odd = arms["odd"]
    p_even = leaked_even / shots_even
    p_odd = leaked_odd / shots_odd
    variance = p_even * (1.0 - p_even) / shots_even + p_odd * (1.0 - p_odd) / shots_odd
    return (
        p_even - p_odd,
        variance,
        {
            "leak_even": p_even,
            "leak_odd": p_odd,
            "shots_even": shots_even,
            "shots_odd": shots_odd,
            "shot_noise_se": math.sqrt(variance),
        },
    )


def _random_effects(effects: list[float], variances: list[float]) -> dict[str, float]:
    """Return DL heterogeneity and safeguarded HKSJ mean inference."""
    if len(effects) != len(variances) or len(effects) < 2:
        raise ValueError("random-effects input cardinality is invalid")
    if any(variance <= 0.0 for variance in variances):
        raise ValueError("random-effects variances must be positive")
    fixed_weights = [1.0 / variance for variance in variances]
    fixed_weight_sum = sum(fixed_weights)
    fixed_mean = (
        sum(weight * effect for weight, effect in zip(fixed_weights, effects, strict=True))
        / fixed_weight_sum
    )
    q = sum(
        weight * (effect - fixed_mean) ** 2
        for weight, effect in zip(fixed_weights, effects, strict=True)
    )
    c = fixed_weight_sum - sum(weight**2 for weight in fixed_weights) / fixed_weight_sum
    tau_squared = max(0.0, (q - (len(effects) - 1)) / c)
    weights = [1.0 / (variance + tau_squared) for variance in variances]
    weight_sum = sum(weights)
    mean = sum(weight * effect for weight, effect in zip(weights, effects, strict=True))
    mean /= weight_sum
    hksj_scale = sum(
        weight * (effect - mean) ** 2 for weight, effect in zip(weights, effects, strict=True)
    ) / (len(effects) - 1)
    safeguarded_scale = max(1.0, hksj_scale)
    standard_error = math.sqrt(safeguarded_scale / weight_sum)
    statistic = mean / standard_error
    degrees_of_freedom = len(effects) - 1
    return {
        "mean": mean,
        "tau_dl": math.sqrt(tau_squared),
        "cochran_q": q,
        "degrees_of_freedom": degrees_of_freedom,
        "hksj_scale": hksj_scale,
        "safeguarded_scale": safeguarded_scale,
        "standard_error": standard_error,
        "t_statistic": statistic,
        "one_sided_p": float(t.sf(statistic, degrees_of_freedom)),
    }


def analyse(
    count_paths: list[Path], calibration_paths: list[Path], design_path: Path
) -> dict[str, Any]:
    """Validate the complete custody set and build the frozen result."""
    design = protocol.validate_frozen_design(design_path)
    required_epochs = protocol.EPOCHS
    if len(count_paths) != required_epochs or len(calibration_paths) != required_epochs:
        raise ValueError(
            f"powered depth ordering requires exactly {required_epochs} count/calibration pairs"
        )
    rows: list[dict[str, Any]] = []
    seen_epochs: set[int] = set()
    seen_calibrations: set[str] = set()
    for count_path, calibration_path in zip(count_paths, calibration_paths, strict=True):
        counts = _object(count_path)
        calibration = _object(calibration_path)
        epoch = counts.get("epoch")
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or not 1 <= epoch <= required_epochs
        ):
            raise ValueError(f"{count_path} has an invalid epoch")
        if epoch in seen_epochs:
            raise ValueError(f"duplicate powered depth-ordering epoch {epoch}")
        seen_epochs.add(epoch)
        calibration_date = calibration.get("date")
        if not isinstance(calibration_date, str) or not calibration_date:
            raise ValueError(f"{calibration_path} has no calibration date")
        calibration_id = protocol.validate_calibration_snapshot(
            calibration,
            calibration_path,
            expected_date=calibration_date,
        )
        if calibration_id in design.excluded_calibration_set_ids:
            raise ValueError(f"epoch {epoch} reuses design-evidence calibration {calibration_id}")
        if calibration_id in seen_calibrations:
            raise ValueError(f"calibration_set_id {calibration_id} is repeated")
        seen_calibrations.add(calibration_id)
        protocol.validate_retrieved_counts(
            counts,
            count_path,
            expected_epoch=epoch,
            calibration_set_id=calibration_id,
            calibration_date=calibration_date,
            design_sha256=design.sha256,
        )
        raw_counts = counts.get("counts")
        assert isinstance(raw_counts, dict)
        effects: dict[str, dict[str, Any]] = {}
        variances: dict[int, float] = {}
        deltas: dict[int, float] = {}
        for depth in DEPTHS:
            delta, variance, detail = _delta(raw_counts, depth)
            deltas[depth] = delta
            variances[depth] = variance
            effects[str(depth)] = {"difference_even_minus_odd": delta, **detail}
        contrast = deltas[8] - deltas[12]
        contrast_variance = variances[8] + variances[12]
        rows.append(
            {
                "epoch": epoch,
                "date": counts.get("date"),
                "calibration_set_id": calibration_id,
                "job_ids": counts.get("job_ids", []),
                "depths": effects,
                "contrast_delta8_minus_delta12": contrast,
                "contrast_shot_noise_se": math.sqrt(contrast_variance),
                "contrast_variance": contrast_variance,
            }
        )
    rows.sort(key=lambda row: row["epoch"])

    contrasts = [float(row["contrast_delta8_minus_delta12"]) for row in rows]
    contrast_variances = [float(row["contrast_variance"]) for row in rows]
    primary: dict[str, Any] = _random_effects(contrasts, contrast_variances)
    alpha = protocol.ALPHA
    primary.update(
        {
            "contrast": "(delta_8 - delta_12) > 0",
            "alpha": alpha,
            "epochs": required_epochs,
            "rejects_null": bool(primary["mean"] > 0.0 and primary["one_sided_p"] < alpha),
        }
    )
    per_depth = {
        str(depth): _random_effects(
            [float(row["depths"][str(depth)]["difference_even_minus_odd"]) for row in rows],
            [float(row["depths"][str(depth)]["shot_noise_se"]) ** 2 for row in rows],
        )
        for depth in DEPTHS
    }
    report = {
        "schema": SCHEMA,
        "campaign": CAMPAIGN,
        "status": "complete_frozen_primary",
        "design": {
            "path": design_path.as_posix(),
            "sha256": design.sha256,
        },
        "custody": {
            "count_sha256": {
                path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in count_paths
            },
            "calibration_sha256": {
                path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in calibration_paths
            },
        },
        "per_epoch": rows,
        "primary_depth_ordering": primary,
        "secondary_per_depth_random_effects": per_depth,
        "decision_boundary": (
            "window-variability counts and calibration epochs are design-only and excluded; "
            "no interim "
            "endpoint analysis or result-dependent epoch extension is permitted"
        ),
        "interpretation_boundary": (
            "device-noise depth ordering on IQM Garnet primary layout only; a distinct "
            "calibration_set_id is an operational epoch marker, not proof of complete "
            "physical independence"
        ),
    }
    return cast(dict[str, Any], _portable_json(report))


def main(argv: list[str] | None = None) -> int:
    """Write the complete frozen powered depth-ordering analysis artefact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch-counts", required=True, nargs="+")
    parser.add_argument("--calibrations", required=True, nargs="+")
    parser.add_argument("--design", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = analyse(
        [Path(path) for path in args.epoch_counts],
        [Path(path) for path in args.calibrations],
        Path(args.design),
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    primary = report["primary_depth_ordering"]
    print(
        f"Powered depth-ordering analysis: {output} — mean={primary['mean']:+.6f}, "
        f"p={primary['one_sided_p']:.6g}, rejects_null={primary['rejects_null']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
