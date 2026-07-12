# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — phase3 entanglement paper assets tests
# scpn-quantum-control -- Phase 3 entanglement/tomography paper asset tests
"""Tests for Phase 3 entanglement/tomography paper asset generation."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_module() -> ModuleType:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "generate_phase3_entanglement_paper_assets.py"
    )
    spec = importlib.util.spec_from_file_location(
        "generate_phase3_entanglement_paper_assets", script
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load entanglement paper asset generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture_rows() -> list[dict[str, str]]:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "data"
        / "phase3_entanglement_tomography"
        / "entanglement_tomography_rows_2026-05-20.csv"
    )
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_label_summary_preserves_promoted_result_order_and_metrics() -> None:
    module = _load_module()
    rows = _fixture_rows()

    summary = module.build_label_summary(rows)

    assert [row["label"] for row in summary] == [
        "dla_even_shallow",
        "dla_even_signal",
        "dla_odd_shallow",
        "dla_odd_signal",
        "fim_lambda0_reference",
        "fim_lambda4_feedback",
    ]
    by_label = {row["label"]: row for row in summary}
    assert by_label["dla_odd_shallow"]["mean_absolute_deviation"] == pytest.approx(
        0.2197078666908347
    )
    assert by_label["fim_lambda4_feedback"]["max_absolute_deviation"] == pytest.approx(
        0.27415458282391614
    )


def test_top_deviations_identify_dla_odd_signal_transverse_edge() -> None:
    module = _load_module()
    rows = _fixture_rows()

    top = module.build_top_deviations(rows, limit=2)

    assert [row["basis_setting"] for row in top] == ["XXII", "YYII"]
    assert all(row["label"] == "dla_odd_signal" for row in top)
    assert top[0]["absolute_deviation"] == pytest.approx(0.5560906424788263)


def test_backend_comparison_tracks_raw_and_readout_mitigated_deviation() -> None:
    module = _load_module()
    rows = [
        {
            "absolute_deviation": "0.20",
            "readout_mitigated_absolute_deviation": "0.10",
        },
        {
            "absolute_deviation": "0.40",
            "readout_mitigated_absolute_deviation": "0.30",
        },
    ]

    comparison = module.build_backend_comparison({"ibm_test": rows})

    assert comparison == [
        {
            "backend": "ibm_test",
            "n_observables": 2,
            "mean_absolute_deviation": pytest.approx(0.30),
            "max_absolute_deviation": pytest.approx(0.40),
            "readout_mitigated_mean_absolute_deviation": pytest.approx(0.20),
            "readout_mitigated_max_absolute_deviation": pytest.approx(0.30),
        }
    ]


def test_full_readout_amplification_summary_separates_transverse_edges() -> None:
    module = _load_module()
    rows = [
        {
            "family": "dla_parity",
            "label": "dla_odd_shallow",
            "basis_setting": "IIYY",
            "absolute_deviation": "0.47",
            "readout_mitigated_absolute_deviation": "0.99",
        },
        {
            "family": "dla_parity",
            "label": "dla_odd_shallow",
            "basis_setting": "IIZZ",
            "absolute_deviation": "0.11",
            "readout_mitigated_absolute_deviation": "0.40",
        },
        {
            "family": "fim_pair",
            "label": "fim_lambda4_feedback",
            "basis_setting": "IXXI",
            "absolute_deviation": "0.20",
            "readout_mitigated_absolute_deviation": "0.21",
        },
    ]

    summary = module.build_full_readout_amplification_summary(rows, limit=2)

    assert summary["top_rows"][0]["label"] == "dla_odd_shallow"
    assert summary["top_rows"][0]["basis_setting"] == "IIYY"
    assert summary["top_rows"][0]["channel_class"] == "transverse_edge"
    assert summary["top_rows"][0]["absolute_amplification"] == pytest.approx(0.52)
    by_class = {row["channel_class"]: row for row in summary["class_summary"]}
    assert by_class["transverse_edge"]["mean_absolute_amplification"] == pytest.approx(0.52)
    assert by_class["zz_edge"]["mean_absolute_amplification"] == pytest.approx(0.29)
