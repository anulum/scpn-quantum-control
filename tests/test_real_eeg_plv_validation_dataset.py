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
bandpass_phase = eeg_module.bandpass_phase
plv_edges = eeg_module.plv_edges
segment_starts = eeg_module.segment_starts


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
