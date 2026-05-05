# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- ansatz scaling tensor-network benchmark tests
"""Tests for the ansatz scaling and MPS truncation benchmark harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_ansatz_scaling_tn.py"


def _load_module() -> object:
    spec = importlib.util.spec_from_file_location("benchmark_ansatz_scaling_tn", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_discarded_weight_is_zero_when_bond_covers_spectrum() -> None:
    module = _load_module()
    singular_values = np.array([0.8, 0.6])

    assert module._discarded_weight(singular_values, 2) == 0.0  # type: ignore[attr-defined]


def test_discarded_weight_keeps_squared_tail() -> None:
    module = _load_module()
    singular_values = np.array([0.8, 0.5, 0.3])

    assert module._discarded_weight(singular_values, 1) == float(0.5**2 + 0.3**2)  # type: ignore[attr-defined]


def test_schmidt_values_for_product_state() -> None:
    module = _load_module()
    state = np.zeros(4, dtype=np.complex128)
    state[0] = 1.0

    values = module._schmidt_values(state, 2, 1)  # type: ignore[attr-defined]

    assert np.allclose(values, [1.0, 0.0])


def test_mps_rows_mark_large_n_skipped() -> None:
    module = _load_module()

    rows = module.mps_truncation_rows([4], [4], exact_max_qubits=3)  # type: ignore[attr-defined]

    assert rows == [
        {
            "n_qubits": 4,
            "status": "skipped",
            "reason": "above_exact_max_qubits",
            "exact_max_qubits": 3,
            "ground_energy": None,
            "max_bond": None,
            "worst_cut_discarded_weight": None,
            "max_midchain_entropy_bits": None,
        }
    ]
