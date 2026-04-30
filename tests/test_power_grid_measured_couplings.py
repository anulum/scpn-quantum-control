# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Power-Grid Measured Coupling Builder
"""Tests for the IEEE 5-bus measured coupling artifact builder."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_power_grid_measured_couplings.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "_build_power_grid_measured_couplings", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


power_grid_module = _load_script_module()
build_payload = power_grid_module.build_payload
symmetrised_edge_value_and_uncertainty = power_grid_module.symmetrised_edge_value_and_uncertainty


def test_symmetrised_edge_uncertainty_tracks_nonzero_and_absent_lines():
    value, uncertainty = symmetrised_edge_value_and_uncertainty(0, 1)

    assert value == pytest.approx(0.0012533881486833107)
    assert uncertainty > 0.0

    absent_value, absent_uncertainty = symmetrised_edge_value_and_uncertainty(0, 3)
    assert absent_value == 0.0
    assert absent_uncertainty == 0.0


def test_build_payload_records_all_upper_triangle_edges_with_locked_normalisation():
    payload = build_payload(command=["python", "script.py"])

    assert payload["schema_version"] == "scpn-quantum-control.measured-couplings.v1"
    assert payload["system"] == "IEEE 5-bus power-grid swing-equation coupling matrix"
    assert payload["normalisation_locked"] is True
    assert payload["unit"] == "dimensionless_swing_equation_coupling"
    assert len(payload["couplings"]) == 10
    assert payload["source_dataset"]["raw_units"]["susceptance"] == "per_unit_on_100_MVA_base"
    assert payload["signal_processing"]["nodes"] == ["bus_1", "bus_2", "bus_3", "bus_4", "bus_5"]

    by_edge = {(item["i"], item["j"]): item for item in payload["couplings"]}
    assert by_edge[(1, 2)]["raw_susceptance_per_unit"] == 3.81
    assert by_edge[(1, 2)]["uncertainty_type"] == (
        "rounded_input_half_width_first_order_propagation"
    )
    assert by_edge[(1, 4)]["value"] == 0.0
    assert by_edge[(1, 4)]["uncertainty_type"] == "topological_absence_in_public_benchmark"
