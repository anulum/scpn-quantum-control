# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — entanglement tomography readiness tests
# scpn-quantum-control -- tests for entanglement/tomography readiness
"""Tests for the no-QPU entanglement/tomography readiness package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


def _load_module() -> ModuleType:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "generate_entanglement_tomography_readiness.py"
    )
    spec = importlib.util.spec_from_file_location(
        "generate_entanglement_tomography_readiness", script
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load entanglement/tomography readiness script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_observable_map_has_nine_reduced_pauli_settings() -> None:
    module = _load_module()

    observables = module.observable_map()
    settings = module.basis_settings(observables)

    assert len(observables) == 9
    assert len(settings) == 9
    assert "XX_q0q1" in observables
    assert "ZZ_q2q3" in observables


def test_exact_product_state_purity_and_pauli_expectations() -> None:
    module = _load_module()
    state = np.zeros(16, dtype=np.complex128)
    state[0] = 1.0

    assert module.half_chain_purity(state) == 1.0
    assert module.pauli_expectation(state, "ZZII") == 1.0
    assert module.pauli_expectation(state, "XXII") == 0.0


def test_readiness_rows_are_under_preregistered_circuit_ceiling() -> None:
    module = _load_module()

    rows, summary = module.build_rows()

    assert summary["schema"] == "scpn_phase3_entanglement_tomography_readiness_v1"
    assert summary["hardware_submission"] is False
    assert summary["qpu_minutes_spent"] == 0.0
    assert summary["ready_for_optional_hardware"] is True
    assert summary["readiness_decision"] == "ready_for_optional_hardware_preregistration"
    assert summary["n_basis_settings"] <= module.MAX_SETTINGS_PER_STATE_FAMILY
    assert summary["total_circuits"] <= module.MAX_DLA_FIM_CIRCUITS
    assert len(rows) == summary["n_observable_rows"]


def test_write_outputs_records_manifest_and_csv_hash(tmp_path: Path) -> None:
    module = _load_module()
    rows, summary = module.build_rows()

    json_path, csv_path, md_path = module.write_outputs(
        rows,
        summary,
        output_dir=tmp_path / "data",
        docs_dir=tmp_path / "docs",
    )

    assert json_path.exists()
    assert csv_path.exists()
    manifest = md_path.read_text(encoding="utf-8")
    assert "ready_for_optional_hardware_preregistration" in manifest
    assert "Hardware submission: `False`" in manifest
