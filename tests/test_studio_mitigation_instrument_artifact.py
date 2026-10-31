# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — tests for the committed mitigation instrument data
"""The mitigation instrument data comes from its source owners and stays current."""

from __future__ import annotations

import json
import runpy
import shutil
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")

from scpn_quantum_control.studio import mitigation_instrument_artifact as artifact  # noqa: E402
from scpn_quantum_control.studio.readout_mitigation_bundle import (  # noqa: E402
    DEFAULT_READOUT_MITIGATION_ARTIFACT_PATH,
)


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    """Build the instrument data once from the real repository.

    Returns
    -------
    dict
        Freshly built payload.

    """
    return cast(dict[str, Any], artifact.build_mitigation_instrument_artifact())


def _cases(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index the readout cases by identifier."""
    return {case["case_id"]: case for case in payload["readout"]["cases"]}


def _bit_matrix(flips: dict[str, float]) -> np.ndarray:
    """Column-stochastic 2x2 readout matrix of one qubit, columns prepared 0 and 1."""
    return np.array(
        [
            [1.0 - flips["zero_to_one"], flips["one_to_zero"]],
            [flips["zero_to_one"], 1.0 - flips["one_to_zero"]],
        ]
    )


def _oracle_matrix(case: dict[str, Any]) -> np.ndarray:
    """Kronecker product of the per-qubit matrices in label order.

    The leftmost label character is the highest classical bit, so the product
    runs from the highest bit's qubit to bit 0's qubit.
    """
    by_qubit = {item["qubit"]: item for item in case["readout"]}
    matrix = np.ones((1, 1))
    for mapping in reversed(case["measurement_map"]):
        matrix = np.kron(matrix, _bit_matrix(by_qubit[mapping["qubit"]]))
    return matrix


def test_mitigation_inspector_01_permuted_partial_map_survives_correction(
    payload: dict[str, Any],
) -> None:
    """A permuted, partial map keeps its order, and the correction follows it."""
    case = _cases(payload)["permuted-partial-map"]
    assert case["measurement_map"] == [
        {"classical_bit": 0, "qubit": 2},
        {"classical_bit": 1, "qubit": 0},
    ]
    assert case["unmeasured_qubits"] == [1]
    assert case["labels"] == ["00", "01", "10", "11"]
    oracle = _oracle_matrix(case)
    np.testing.assert_allclose(case["calibration"]["matrix"], oracle, atol=1e-15)
    counts = np.array(case["raw_counts"], dtype=float)
    expected = np.linalg.solve(oracle, counts / counts.sum())
    np.testing.assert_allclose(case["corrected"]["quasiprobabilities"], expected, atol=1e-12)
    # Swapping the map gives a different, wrong correction: the map is used.
    swapped = dict(case, measurement_map=list(reversed(case["measurement_map"])))
    assert not np.allclose(
        np.linalg.solve(_oracle_matrix(swapped), counts / counts.sum()), expected
    )


def test_mitigation_inspector_02_identity_calibration_leaves_values(
    payload: dict[str, Any],
) -> None:
    """A calibration that never flips returns the raw frequencies exactly."""
    case = _cases(payload)["identity-calibration"]
    assert case["calibration"]["matrix"] == np.eye(4).tolist()
    assert case["corrected"]["quasiprobabilities"] == case["raw_frequencies"]
    assert case["corrected"]["calibration_standard_error"] == [0.0] * 4
    shots = case["shots"]
    for frequency, error in zip(
        case["raw_frequencies"], case["corrected"]["sampling_standard_error"], strict=True
    ):
        assert error == pytest.approx(np.sqrt(frequency * (1.0 - frequency) / shots), abs=1e-15)


def test_mitigation_inspector_03_negative_values_stay_signed(payload: dict[str, Any]) -> None:
    """Negative corrected values are kept and named, never clipped or renormalised."""
    case = _cases(payload)["negative-quasi-probabilities"]
    values = case["corrected"]["quasiprobabilities"]
    counts = np.array(case["raw_counts"], dtype=float)
    np.testing.assert_allclose(
        values, np.linalg.solve(_oracle_matrix(case), counts / counts.sum()), atol=1e-12
    )
    assert min(values) < 0.0 and max(values) > 1.0
    assert case["corrected"]["negative_labels"] == ["01", "10"]
    assert case["corrected"]["sum"] == pytest.approx(1.0, abs=1e-12)
    for item in payload["readout"]["cases"]:
        if item["status"] == "corrected":
            negative = [
                label
                for label, value in zip(
                    item["labels"], item["corrected"]["quasiprobabilities"], strict=True
                )
                if value < 0.0
            ]
            assert item["corrected"]["negative_labels"] == negative


def test_mitigation_inspector_04_ill_conditioned_keeps_its_diagnostics(
    payload: dict[str, Any],
) -> None:
    """Near-random readout amplifies errors visibly; a singular one is refused."""
    cases = _cases(payload)
    ill = cases["ill-conditioned"]
    # Each qubit's matrix has singular values 1 and 1 - 2p = 0.1.
    assert ill["calibration"]["condition_number"] == pytest.approx(100.0, rel=1e-12)
    corrected = ill["corrected"]
    assert all(
        error > 0.4 * abs(value)
        for value, error in zip(
            corrected["quasiprobabilities"], corrected["total_standard_error"], strict=True
        )
    )
    singular = cases["singular-calibration"]
    assert singular["status"] == "refused"
    assert "condition" in singular["reason"]
    assert "corrected" not in singular
    # Rounding near a singular matrix gives a platform-dependent figure, so none is shown.
    assert singular["calibration"]["condition_number"] is None
    assert sum(singular["raw_counts"]) == singular["shots"]


def test_every_case_keeps_its_raw_counts_and_shot_accounting(payload: dict[str, Any]) -> None:
    """Raw counts, frequencies and calibration shots are those stated by the case."""
    for case, stated in zip(payload["readout"]["cases"], artifact.READOUT_CASES, strict=True):
        assert case["raw_counts"] == [stated.counts.get(label, 0) for label in case["labels"]]
        assert case["shots"] == sum(stated.counts.values())
        assert case["raw_frequencies"] == pytest.approx(
            [count / case["shots"] for count in case["raw_counts"]], abs=1e-15
        )
        assert case["calibration"]["shots_by_prepared_state"] == [stated.calibration_shots] * 4
        if case["status"] == "corrected":
            corrected = case["corrected"]
            for sampling, calibration, total in zip(
                corrected["sampling_standard_error"],
                corrected["calibration_standard_error"],
                corrected["total_standard_error"],
                strict=True,
            ):
                assert total == pytest.approx(np.hypot(sampling, calibration), abs=1e-15)


def test_zne_matches_an_independent_generalised_least_squares(payload: dict[str, Any]) -> None:
    """Corrected points and the zero-noise interval follow from the stated counts."""
    zne = payload["zne"]
    labels = zne["labels"]
    weights = np.array(zne["observable_weights"])
    assert weights.tolist() == [1.0, -1.0, -1.0, 1.0]
    inverse = np.linalg.inv(np.array(zne["calibration"]["matrix"]))
    means = []
    for point, counts in zip(zne["points"], artifact.ZNE_COUNTS, strict=True):
        raw = np.array([counts.get(label, 0) for label in labels], dtype=float)
        assert point["raw_counts"] == raw.tolist()
        frequencies = raw / raw.sum()
        assert point["raw_expectation"] == pytest.approx(frequencies @ weights, abs=1e-15)
        corrected = (inverse @ frequencies) @ weights
        assert point["corrected_expectation"] == pytest.approx(corrected, abs=1e-12)
        means.append(corrected)
    covariance = np.array(zne["observation_covariance"])
    assert [point["corrected_standard_error"] for point in zne["points"]] == pytest.approx(
        np.sqrt(np.diag(covariance)), abs=1e-15
    )
    scales = np.array(artifact.ZNE_NOISE_SCALES)
    design = np.vander(scales, 2, increasing=True)
    precision = np.linalg.inv(covariance)
    coefficients_covariance = np.linalg.inv(design.T @ precision @ design)
    coefficients = coefficients_covariance @ design.T @ precision @ np.array(means)
    fit = zne["extrapolation"]
    assert fit["zero_noise_estimate"] == pytest.approx(coefficients[0], rel=1e-10)
    assert fit["standard_error"] == pytest.approx(
        np.sqrt(coefficients_covariance[0, 0]), rel=1e-10
    )
    assert fit["low"] < fit["zero_noise_estimate"] < fit["high"]
    assert fit["method"] == "gls-delta"


def test_hardware_pairs_are_the_summary_verbatim(payload: dict[str, Any]) -> None:
    """Measured pairs, method and caveat come unchanged from the committed summary."""
    hardware = payload["hardware"]
    summary = json.loads(
        (artifact.REPO_ROOT / DEFAULT_READOUT_MITIGATION_ARTIFACT_PATH).read_text(encoding="utf-8")
    )
    assert hardware["admitted"] is True
    assert hardware["full_confusion_matrix_available"] is False
    assert hardware["caveat"] == summary["full_confusion_matrix_note"]
    assert hardware["method"] == summary["method"]
    assert len(hardware["pairs"]) == len(summary["pairs"])
    for shown, source in zip(hardware["pairs"], summary["pairs"], strict=True):
        assert all(shown[key] == source[key] for key in shown)


def test_build_uses_the_given_root(tmp_path: Path, payload: dict[str, Any]) -> None:
    """A copied summary gives the same section; one without its caveat refuses.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory.
    payload
        Payload built from the real repository.

    """
    target = tmp_path / DEFAULT_READOUT_MITIGATION_ARTIFACT_PATH
    target.parent.mkdir(parents=True)
    shutil.copyfile(artifact.REPO_ROOT / DEFAULT_READOUT_MITIGATION_ARTIFACT_PATH, target)
    assert (
        artifact.build_mitigation_instrument_artifact(tmp_path)["hardware"] == payload["hardware"]
    )
    summary = json.loads(target.read_text(encoding="utf-8"))
    summary["full_confusion_matrix_note"] = ""
    target.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="caveat"):
        artifact.build_mitigation_instrument_artifact(tmp_path)


def test_main_writes_and_checks(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
) -> None:
    """The command line writes the payload and checks it as current.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory.
    capsys
        Captured output.
    monkeypatch
        Replaces the build with the module's payload.
    payload
        Payload built from the real repository.

    """
    monkeypatch.setattr(artifact, "build_mitigation_instrument_artifact", lambda: payload)
    target = tmp_path / "instrument.json"
    assert artifact.main(["--write", "--json-path", str(target)]) == 0
    assert artifact.main(["--check", "--json-path", str(target)]) == 0
    assert "mitigation instrument data: current" in capsys.readouterr().out


def test_committed_data_is_current(payload: dict[str, Any]) -> None:
    """The committed file equals a fresh build within the budget."""
    committed = json.loads(
        (artifact.REPO_ROOT / artifact.DEFAULT_MITIGATION_INSTRUMENT_JSON_PATH).read_text(
            encoding="utf-8"
        )
    )
    assert artifact.differences(committed, payload) == []


def test_module_runs_as_a_script(monkeypatch: pytest.MonkeyPatch) -> None:
    """``python -m ... --check`` exits with the check's status.

    Parameters
    ----------
    monkeypatch
        Sets the process arguments.

    """
    monkeypatch.setattr(sys, "argv", ["mitigation_instrument_artifact", "--check"])
    with pytest.raises(SystemExit) as exited:
        runpy.run_module(
            "scpn_quantum_control.studio.mitigation_instrument_artifact", run_name="__main__"
        )
    assert exited.value.code == 0
