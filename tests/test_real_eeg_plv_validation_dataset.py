# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Real EEG PLV Validation Dataset Builder
"""Tests for real EEG PLV measured-coupling helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_real_eeg_plv_validation_dataset.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "_build_real_eeg_plv_validation_dataset",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


eeg_module = _load_script_module()
aggregate_record_edges = eeg_module.aggregate_record_edges
bandpass_phase = eeg_module.bandpass_phase
plv_edges = eeg_module.plv_edges
record_condition = eeg_module.record_condition
record_to_path = eeg_module.record_to_path
record_to_url = eeg_module.record_to_url
records_for_subject_run = eeg_module.records_for_subject_run
segment_starts = eeg_module.segment_starts
validated_https_url = eeg_module._validated_https_url


def test_record_to_path_and_url_follow_physionet_layout(tmp_path):
    assert record_to_path("S012R01", raw_root=tmp_path) == tmp_path / "S012" / "S012R01.edf"
    assert (
        record_to_url("S012R01", dataset_base_url="https://physionet.org/files/eegmmidb/1.0.0/")
        == "https://physionet.org/files/eegmmidb/1.0.0/S012/S012R01.edf"
    )


def test_validated_https_url_rejects_non_https_sources():
    with pytest.raises(ValueError, match="Only absolute HTTPS"):
        validated_https_url("file:///tmp/S001R01.edf")

    with pytest.raises(ValueError, match="Only absolute HTTPS"):
        validated_https_url("http://physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf")


def test_records_for_subject_run_builds_full_eegmmidb_identifiers():
    assert records_for_subject_run([1, 9, 109], run=2) == [
        "S001R02",
        "S009R02",
        "S109R02",
    ]

    with pytest.raises(ValueError, match="subjects must be in"):
        records_for_subject_run([0], run=1)

    with pytest.raises(ValueError, match="run must be in"):
        records_for_subject_run([1], run=15)


def test_record_condition_tracks_baseline_and_mixed_runs():
    assert record_condition(["S001R01", "S002R01"]) == "baseline eyes open"
    assert record_condition(["S001R02", "S002R02"]) == "baseline eyes closed"
    assert record_condition(["S001R01", "S001R02"]) == "mixed EEGMMIDB runs"


def test_segment_starts_uses_full_overlapping_window_grid():
    starts = segment_starts(10, sfreq=2.0, window_s=2.0, step_s=1.0)

    assert starts == [0, 2, 4, 6]


def test_plv_edges_reports_near_locked_pair_with_uncertainty():
    sfreq = 100.0
    t = np.arange(0.0, 4.0, 1.0 / sfreq)
    phase = np.vstack([2.0 * np.pi * 10.0 * t, 2.0 * np.pi * 10.0 * t + 0.2])

    edges = plv_edges(phase, sfreq=sfreq, window_s=1.0, step_s=1.0)

    assert len(edges) == 1
    assert edges[0]["i"] == 1
    assert edges[0]["j"] == 2
    assert edges[0]["value"] > 0.999
    assert edges[0]["uncertainty"] < 1e-12
    assert edges[0]["n_segments"] == 4


def test_bandpass_phase_preserves_channel_and_sample_shape():
    sfreq = 100.0
    t = np.arange(0.0, 3.0, 1.0 / sfreq)
    data = np.vstack([np.sin(2.0 * np.pi * 10.0 * t), np.cos(2.0 * np.pi * 10.0 * t)])

    phase = bandpass_phase(data, sfreq=sfreq, low_hz=8.0, high_hz=13.0)

    assert phase.shape == data.shape
    assert np.isfinite(phase).all()


def test_aggregate_record_edges_uses_median_and_mad_uncertainty():
    rows = [
        {"i": 1, "j": 2, "value": 0.2, "uncertainty": 0.01, "n_segments": 10},
        {"i": 1, "j": 2, "value": 0.4, "uncertainty": 0.01, "n_segments": 10},
        {"i": 1, "j": 2, "value": 0.9, "uncertainty": 0.01, "n_segments": 10},
        {"i": 1, "j": 3, "value": 0.7, "uncertainty": 0.02, "n_segments": 8},
    ]

    aggregated = aggregate_record_edges(rows)

    by_edge = {(item["i"], item["j"]): item for item in aggregated}
    assert by_edge[(1, 2)]["value"] == 0.4
    assert by_edge[(1, 2)]["q25"] == 0.30000000000000004
    assert by_edge[(1, 2)]["q75"] == 0.65
    assert by_edge[(1, 2)]["n_records"] == 3
    assert by_edge[(1, 2)]["n_segments_total"] == 30
    assert by_edge[(1, 3)]["value"] == 0.7
    assert by_edge[(1, 3)]["uncertainty"] == 0.02
