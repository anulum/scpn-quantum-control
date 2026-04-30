# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for EEG PLV Cohort Comparison
"""Tests for derived EEG PLV cohort comparison helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "compare_eeg_plv_cohorts.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("_compare_eeg_plv_cohorts", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


comparison_module = _load_script_module()
compare_payloads = comparison_module.compare_payloads
coupling_map = comparison_module.coupling_map


def _payload(*, condition: str, values: list[float]) -> dict:
    return {
        "schema_version": "scpn-quantum-control.measured-couplings.v1",
        "unit": "phase_locking_value",
        "normalisation": "same alpha-band PLV normalisation",
        "normalisation_locked": True,
        "source_dataset": {"condition": condition, "n_records": 2},
        "couplings": [
            {
                "i": 1,
                "j": 2,
                "value": values[0],
                "uncertainty": 0.01,
                "q25": values[0] - 0.01,
                "q75": values[0] + 0.01,
            },
            {
                "i": 1,
                "j": 3,
                "value": values[1],
                "uncertainty": 0.02,
                "q25": values[1] - 0.02,
                "q75": values[1] + 0.02,
            },
        ],
    }


def test_coupling_map_rejects_duplicate_edges():
    payload = _payload(condition="baseline eyes open", values=[0.2, 0.3])
    payload["couplings"][1]["j"] = 2

    with pytest.raises(ValueError, match="Duplicate coupling edge"):
        coupling_map(payload)


def test_compare_payloads_reports_descriptive_closed_minus_open_summary():
    payload = compare_payloads(
        open_payload=_payload(condition="baseline eyes open", values=[0.2, 0.4]),
        closed_payload=_payload(condition="baseline eyes closed", values=[0.3, 0.35]),
        open_path=Path("open.json"),
        closed_path=Path("closed.json"),
        command=["python", "script.py"],
    )

    assert payload["normalisation_locked"] is True
    assert payload["cohorts"]["baseline_eyes_open"]["condition"] == "baseline eyes open"
    assert payload["cohorts"]["baseline_eyes_closed"]["condition"] == "baseline eyes closed"
    assert payload["summary"]["edge_count"] == 2
    assert payload["summary"]["mean_delta_closed_minus_open"] == pytest.approx(0.025)
    assert payload["summary"]["mean_absolute_delta"] == pytest.approx(0.075)
    assert payload["summary"]["max_absolute_delta_edge"] == {
        "i": 1,
        "j": 2,
        "absolute_delta": pytest.approx(0.1),
        "delta_closed_minus_open": pytest.approx(0.1),
    }
    assert payload["edges"][0]["delta_closed_minus_open"] == pytest.approx(0.1)
    assert payload["edges"][1]["delta_closed_minus_open"] == pytest.approx(-0.05)


def test_compare_payloads_requires_matching_edge_sets():
    closed_payload = _payload(condition="baseline eyes closed", values=[0.3, 0.35])
    closed_payload["couplings"].pop()

    with pytest.raises(ValueError, match="same edge set"):
        compare_payloads(
            open_payload=_payload(condition="baseline eyes open", values=[0.2, 0.4]),
            closed_payload=closed_payload,
            open_path=Path("open.json"),
            closed_path=Path("closed.json"),
            command=["python", "script.py"],
        )
