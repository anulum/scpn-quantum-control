# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — committed data for the Studio mitigation and readout instrument
"""Build, write and check the data the Studio mitigation instrument shows.

The instrument keeps three things apart, and every value comes from a source
owner:

* **Readout correction**: stated calibration and measurement counts corrected
  by :func:`mitigate_counts_with_uncertainty`, which inverts the full
  measured-basis calibration without clipping. Each case keeps its raw counts,
  its classical-bit-to-qubit measurement map, its calibration matrix and shot
  accounting, and the signed corrected values with separate sampling and
  calibration standard errors. A calibration the source refuses keeps its raw
  counts and the source's reason, with no corrected values;
* **Zero-noise extrapolation**: counts at several noise scales corrected with
  one shared calibration and extrapolated by
  :func:`zne_extrapolate_with_readout_uncertainty`, with the raw expectation
  at every scale kept beside the corrected one;
* **Hardware pairs**: the measured raw-versus-corrected parity pairs of the
  committed readout-mitigation summary, through
  :func:`build_readout_mitigation_bundle` and the federation gate, with the
  summary's own method and caveat.

The calibration and measurement counts of the first two sections are stated
fixtures, not device measurements. The browser selects among these values and
computes none of them.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from ..mitigation.readout_matrix import (
    ReadoutConfusionMatrix,
    build_readout_confusion_matrix,
    computational_basis_labels,
    counts_to_probabilities,
)
from ..mitigation.readout_uncertainty import (
    ReadoutMitigationEstimate,
    mitigate_counts_with_uncertainty,
    zne_extrapolate_with_readout_uncertainty,
)
from .committed_data import differences, run_committed_data_cli, serialise
from .evidence_bundle import validate_bundle
from .readout_mitigation_bundle import (
    DEFAULT_READOUT_MITIGATION_ARTIFACT_PATH,
    build_readout_mitigation_bundle,
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
MITIGATION_INSTRUMENT_SCHEMA: Final[str] = "scpn_qc_studio_mitigation_instrument_v1"
MITIGATION_INSTRUMENT_ARTIFACT_ID: Final[str] = "studio-mitigation-instrument-20261031"
DEFAULT_MITIGATION_INSTRUMENT_JSON_PATH: Final[Path] = Path(
    "data/studio/mitigation_instrument_20261031.json"
)
MITIGATION_INSTRUMENT_CLAIM_BOUNDARY: Final[str] = (
    "Readout correction and zero-noise extrapolation of stated fixture counts, "
    "beside measured hardware parity pairs, kept separate; corrected values are "
    "signed estimates, not probabilities, and no fixture is a device measurement"
)
FIXTURE_CLAIM_BOUNDARY: Final[str] = (
    "Stated calibration and measurement counts, not device measurements; standard "
    "errors are first-order delta estimates at the empirical frequencies and exclude "
    "calibration drift, preparation bias and model error"
)
LABEL_ORDER_NOTE: Final[str] = (
    "Labels read classical bits from highest to lowest, so the rightmost character "
    "is classical bit 0"
)


@dataclass(frozen=True)
class QubitReadout:
    """Stated readout flip probabilities of one physical qubit.

    Parameters
    ----------
    qubit
        Physical qubit index.
    zero_to_one
        Probability that a prepared 0 reads as 1.
    one_to_zero
        Probability that a prepared 1 reads as 0.

    """

    qubit: int
    zero_to_one: float
    one_to_zero: float


@dataclass(frozen=True)
class ReadoutCase:
    """One stated readout-correction case.

    Parameters
    ----------
    case_id
        Stable identifier.
    title
        What the case demonstrates.
    measurement_map
        Physical qubit read into each classical bit, indexed by classical bit.
    device_qubits
        Physical qubits of the stated device; unmeasured ones stay listed.
    readout
        Flip probabilities of every measured qubit.
    calibration_shots
        Shots per prepared calibration state.
    counts
        Measurement counts in the classical-bit labels.

    """

    case_id: str
    title: str
    measurement_map: tuple[int, ...]
    device_qubits: int
    readout: tuple[QubitReadout, ...]
    calibration_shots: int
    counts: Mapping[str, int]


_FLIPS: Final[dict[int, QubitReadout]] = {
    0: QubitReadout(0, 0.02, 0.05),
    1: QubitReadout(1, 0.03, 0.06),
    2: QubitReadout(2, 0.04, 0.08),
}

READOUT_CASES: Final[tuple[ReadoutCase, ...]] = (
    ReadoutCase(
        "permuted-partial-map",
        "Classical bit 0 reads qubit 2 and bit 1 reads qubit 0; qubit 1 is not measured",
        (2, 0),
        3,
        (_FLIPS[0], _FLIPS[2]),
        10000,
        {"10": 3620, "11": 150, "00": 200, "01": 30},
    ),
    ReadoutCase(
        "identity-calibration",
        "A calibration that never flips leaves the observed frequencies unchanged",
        (0, 1),
        2,
        (QubitReadout(0, 0.0, 0.0), QubitReadout(1, 0.0, 0.0)),
        1000,
        {"00": 600, "11": 400},
    ),
    ReadoutCase(
        "negative-quasi-probabilities",
        "Correcting counts near one basis state gives negative signed values",
        (0, 1),
        2,
        (QubitReadout(0, 0.1, 0.1), QubitReadout(1, 0.1, 0.1)),
        10000,
        {"00": 960, "01": 20, "10": 20},
    ),
    ReadoutCase(
        "ill-conditioned",
        "Flip probabilities near one half amplify both errors through the inverse",
        (0, 1),
        2,
        (QubitReadout(0, 0.45, 0.45), QubitReadout(1, 0.45, 0.45)),
        10000,
        {"00": 600, "11": 400},
    ),
    ReadoutCase(
        "singular-calibration",
        "A calibration that reads every state at random cannot be inverted",
        (0, 1),
        2,
        (QubitReadout(0, 0.5, 0.5), QubitReadout(1, 0.5, 0.5)),
        10000,
        {"00": 600, "11": 400},
    ),
)
"""Readout cases; each repeats exactly from its stated counts."""

ZNE_NOISE_SCALES: Final[tuple[float, ...]] = (1.0, 2.0, 3.0)
"""Noise amplification factors of the extrapolation case."""
ZNE_MEASUREMENT_MAP: Final[tuple[int, ...]] = (0, 1)
"""Physical qubit read into each classical bit of the extrapolation case."""
ZNE_COUNTS: Final[tuple[Mapping[str, int], ...]] = (
    {"00": 1750, "11": 1650, "01": 310, "10": 290},
    {"00": 1560, "11": 1480, "01": 490, "10": 470},
    {"00": 1400, "11": 1320, "01": 650, "10": 630},
)
"""Stated counts at each noise scale."""
ZNE_OBSERVABLE: Final[str] = "Z0 Z1 parity"
"""Observable estimated at each scale."""
ZNE_ORDER: Final[int] = 1
"""Degree of the extrapolating polynomial."""


def _calibration_counts(
    measurement_map: Sequence[int],
    readout: Sequence[QubitReadout],
    shots: int,
) -> dict[str, dict[str, int]]:
    """Return calibration counts of every prepared classical label.

    Each prepared label is read with the stated flip probabilities of the qubit
    its classical bit maps to; the counts are exact for the stated shots.

    Parameters
    ----------
    measurement_map
        Physical qubit read into each classical bit.
    readout
        Flip probabilities of the measured qubits.
    shots
        Shots per prepared state.

    Returns
    -------
    dict
        Prepared label to observed counts.

    """
    by_qubit = {item.qubit: item for item in readout}
    width = len(measurement_map)
    labels = computational_basis_labels(width)
    calibration: dict[str, dict[str, int]] = {}
    for prepared in labels:
        column: dict[str, int] = {}
        for observed in labels:
            probability = 1.0
            for position in range(width):
                flips = by_qubit[measurement_map[width - 1 - position]]
                was, read = prepared[position], observed[position]
                flip = flips.zero_to_one if was == "0" else flips.one_to_zero
                probability *= flip if was != read else 1.0 - flip
            column[observed] = round(probability * shots)
        calibration[prepared] = column
    return calibration


def _matrix(calibration: ReadoutConfusionMatrix) -> list[list[float]]:
    """Return the calibration matrix rows as plain floats."""
    return [[float(value) for value in row] for row in calibration.matrix]


def _corrected(estimate: ReadoutMitigationEstimate) -> dict[str, object]:
    """Return signed values and their separate standard errors.

    A negative variance would make ``math.sqrt`` raise, so a defective
    covariance refuses the build instead of showing a made-up error.
    """
    sampling = [math.sqrt(row[i]) for i, row in enumerate(estimate.sampling_covariance)]
    calibration = [math.sqrt(row[i]) for i, row in enumerate(estimate.calibration_covariance)]
    values = list(estimate.quasiprobabilities)
    return {
        "quasiprobabilities": values,
        "sampling_standard_error": sampling,
        "calibration_standard_error": calibration,
        "total_standard_error": [
            math.hypot(left, right) for left, right in zip(sampling, calibration, strict=True)
        ],
        "sum": math.fsum(values),
        "negative_labels": [
            label for label, value in zip(estimate.labels, values, strict=True) if value < 0.0
        ],
    }


def _readout_case(case: ReadoutCase) -> dict[str, object]:
    """Return one case with its raw counts, calibration and correction or refusal."""
    calibration = build_readout_confusion_matrix(
        _calibration_counts(case.measurement_map, case.readout, case.calibration_shots),
        len(case.measurement_map),
    )
    labels = list(calibration.labels)
    raw = counts_to_probabilities(case.counts, calibration.labels)
    row: dict[str, object] = {
        "case_id": case.case_id,
        "title": case.title,
        "labels": labels,
        "measurement_map": [
            {"classical_bit": bit, "qubit": qubit}
            for bit, qubit in enumerate(case.measurement_map)
        ],
        "unmeasured_qubits": sorted(set(range(case.device_qubits)) - set(case.measurement_map)),
        "readout": [
            {"qubit": item.qubit, "zero_to_one": item.zero_to_one, "one_to_zero": item.one_to_zero}
            for item in case.readout
        ],
        "raw_counts": [int(case.counts.get(label, 0)) for label in labels],
        "raw_frequencies": [float(value) for value in raw],
        "shots": sum(case.counts.values()),
        "calibration": {
            "matrix": _matrix(calibration),
            "shots_by_prepared_state": [
                calibration.shots_by_prepared_state[label] for label in labels
            ],
            "condition_number": calibration.condition_number,
        },
    }
    try:
        estimate = mitigate_counts_with_uncertainty(case.counts, calibration)
    except ValueError as refusal:
        # A refused calibration's condition number is not reported: near a
        # singular matrix it is rounding, and differs by platform (finite or inf).
        cast(dict[str, object], row["calibration"])["condition_number"] = None
        row.update(status="refused", reason=str(refusal))
        return row
    row.update(status="corrected", corrected=_corrected(estimate))
    return row


def _parity_weights(labels: Sequence[str]) -> list[float]:
    """Return the Z0 Z1 parity of every label: +1 even, -1 odd."""
    return [1.0 if label.count("1") % 2 == 0 else -1.0 for label in labels]


def _zne() -> dict[str, object]:
    """Return the extrapolation case with raw and corrected values at every scale."""
    readout = (_FLIPS[0], _FLIPS[1])
    calibration = build_readout_confusion_matrix(
        _calibration_counts(ZNE_MEASUREMENT_MAP, readout, 10000), len(ZNE_MEASUREMENT_MAP)
    )
    weights = _parity_weights(calibration.labels)
    result = zne_extrapolate_with_readout_uncertainty(
        ZNE_NOISE_SCALES, ZNE_COUNTS, calibration, weights, order=ZNE_ORDER
    )
    covariance = result.observation_covariance
    points = []
    for index, (scale, counts, estimate) in enumerate(
        zip(ZNE_NOISE_SCALES, ZNE_COUNTS, result.readout_estimates, strict=True)
    ):
        raw = counts_to_probabilities(counts, calibration.labels)
        points.append(
            {
                "noise_scale": scale,
                "raw_counts": [int(counts.get(label, 0)) for label in calibration.labels],
                "shots": estimate.shots,
                "raw_expectation": float(np.dot(raw, weights)),
                "corrected_expectation": float(np.dot(estimate.quasiprobabilities, weights)),
                "corrected_standard_error": math.sqrt(covariance[index][index]),
            }
        )
    fit = result.extrapolation
    return {
        "claim_boundary": FIXTURE_CLAIM_BOUNDARY,
        "observable": ZNE_OBSERVABLE,
        "labels": list(calibration.labels),
        "observable_weights": weights,
        "measurement_map": [
            {"classical_bit": bit, "qubit": qubit} for bit, qubit in enumerate(ZNE_MEASUREMENT_MAP)
        ],
        "calibration": {
            "matrix": _matrix(calibration),
            "shots_by_prepared_state": [
                calibration.shots_by_prepared_state[label] for label in calibration.labels
            ],
            "condition_number": calibration.condition_number,
        },
        "points": points,
        "observation_covariance": [list(row) for row in covariance],
        "extrapolation": {
            "zero_noise_estimate": fit.zero_noise_estimate,
            "standard_error": fit.standard_error,
            "low": fit.low,
            "high": fit.high,
            "coverage": fit.coverage,
            "order": fit.order,
            "fit_residual": fit.fit_residual,
            "method": fit.method,
        },
    }


def _hardware(root: Path) -> dict[str, object]:
    """Return the validated hardware bundle and its measured pairs verbatim."""
    artifact_path = root / DEFAULT_READOUT_MITIGATION_ARTIFACT_PATH
    validated = validate_bundle(build_readout_mitigation_bundle(artifact_path=artifact_path))
    payload: dict[str, Any] = json.loads(artifact_path.read_text(encoding="utf-8"))
    fields = (
        "dataset",
        "comparison",
        "depth",
        "left_raw",
        "right_raw",
        "left_corrected",
        "right_corrected",
        "raw_relative",
        "corrected_relative",
        "n_left",
        "n_right",
    )
    return {
        "source": DEFAULT_READOUT_MITIGATION_ARTIFACT_PATH.as_posix(),
        "bundle_schema": validated.bundle.schema,
        "bundle_digest": validated.bundle.entity.digest,
        "admitted": validated.verdict.admitted,
        "method": payload["method"],
        "full_confusion_matrix_available": payload["full_confusion_matrix_available"],
        "caveat": payload["full_confusion_matrix_note"],
        "pairs": [{field: pair[field] for field in fields} for pair in payload["pairs"]],
    }


def build_mitigation_instrument_artifact(root: Path = REPO_ROOT) -> dict[str, object]:
    """Build the committed instrument data from its source owners.

    Parameters
    ----------
    root
        Repository root holding the committed readout-mitigation summary.

    Returns
    -------
    dict[str, object]
        JSON-ready payload with separate ``readout``, ``zne`` and ``hardware``
        sections.

    Raises
    ------
    ValueError
        If the readout-mitigation summary fails its shape check.

    """
    return {
        "schema": MITIGATION_INSTRUMENT_SCHEMA,
        "artifact_id": MITIGATION_INSTRUMENT_ARTIFACT_ID,
        "generated_by": "python -m scpn_quantum_control.studio.mitigation_instrument_artifact --write",
        "claim_boundary": MITIGATION_INSTRUMENT_CLAIM_BOUNDARY,
        "label_order": LABEL_ORDER_NOTE,
        "readout": {
            "claim_boundary": FIXTURE_CLAIM_BOUNDARY,
            "method": "full measured-basis inverse without clipping",
            "cases": [_readout_case(case) for case in READOUT_CASES],
        },
        "zne": _zne(),
        "hardware": _hardware(root),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print, write or check the committed instrument data.

    Parameters
    ----------
    argv
        Optional argument vector, defaulting to ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` on success, ``1`` when the committed file differs from a fresh
        build, ``2`` when it cannot be read.

    """
    return run_committed_data_cli(
        argv,
        build=build_mitigation_instrument_artifact,
        default_path=REPO_ROOT / DEFAULT_MITIGATION_INSTRUMENT_JSON_PATH,
        label="mitigation instrument data",
        description=__doc__,
    )


__all__ = [
    "DEFAULT_MITIGATION_INSTRUMENT_JSON_PATH",
    "FIXTURE_CLAIM_BOUNDARY",
    "LABEL_ORDER_NOTE",
    "MITIGATION_INSTRUMENT_ARTIFACT_ID",
    "MITIGATION_INSTRUMENT_CLAIM_BOUNDARY",
    "MITIGATION_INSTRUMENT_SCHEMA",
    "READOUT_CASES",
    "ZNE_COUNTS",
    "ZNE_NOISE_SCALES",
    "QubitReadout",
    "ReadoutCase",
    "build_mitigation_instrument_artifact",
    "differences",
    "main",
    "serialise",
]


if __name__ == "__main__":
    raise SystemExit(main())
