# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — ansatz scaling tn benchmark tests
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

    rows = module.mps_truncation_rows([4], [4], exact_max_qubits=2, sparse_max_qubits=3)  # type: ignore[attr-defined]

    assert rows == [
        {
            "n_qubits": 4,
            "status": "skipped",
            "reason": "above_exact_max_qubits",
            "exact_max_qubits": 2,
            "sparse_max_qubits": 3,
            "solver": None,
            "eigen_residual_norm": None,
            "ground_energy": None,
            "max_bond": None,
            "worst_cut_discarded_weight": None,
            "max_midchain_entropy_bits": None,
        }
    ]


def test_mps_rows_use_sparse_solver_above_dense_limit() -> None:
    module = _load_module()

    rows = module.mps_truncation_rows([4], [4], exact_max_qubits=2, sparse_max_qubits=4)  # type: ignore[attr-defined]

    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["solver"] == "sparse_eigsh"
    assert float(rows[0]["eigen_residual_norm"]) < 1e-8


def test_reference_comparison_marks_missing_vqe_reference_skipped(tmp_path: Path) -> None:
    module = _load_module()
    tn_rows = [
        {
            "n_qubits": 6,
            "status": "ok",
            "solver": "dense_eigh",
            "ground_energy": -1.25,
            "eigen_residual_norm": 0.0,
            "max_bond": 8,
            "worst_cut_discarded_weight": 0.125,
        }
    ]

    rows = module.reference_comparison_rows([6], tn_rows, tmp_path / "missing.json")  # type: ignore[attr-defined]

    assert rows[0]["tn_retained_weight_lower_bound"] == 0.875
    assert rows[0]["vqe_status"] == "skipped"
    assert rows[0]["vqe_skip_reason"] == "no_committed_vqe_reference"


def test_reference_comparison_uses_best_committed_vqe_aggregate(tmp_path: Path) -> None:
    module = _load_module()
    summary_path = tmp_path / "vqe.json"
    summary_path.write_text(
        """
        {
          "aggregate": [
            {
              "ansatz": "two_local",
              "n_qubits": 4,
              "reps": 1,
              "n_seeds": 3,
              "best_energy": -5.0,
              "median_relative_error_pct": 7.0,
              "best_relative_error_pct": 2.0
            },
            {
              "ansatz": "knm_informed",
              "n_qubits": 4,
              "reps": 2,
              "n_seeds": 3,
              "best_energy": -6.0,
              "median_relative_error_pct": 0.3,
              "best_relative_error_pct": 0.1
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    tn_rows = [
        {
            "n_qubits": 4,
            "status": "ok",
            "solver": "dense_eigh",
            "ground_energy": -6.3,
            "eigen_residual_norm": 0.0,
            "max_bond": 4,
            "worst_cut_discarded_weight": 0.02,
        },
        {
            "n_qubits": 4,
            "status": "ok",
            "solver": "dense_eigh",
            "ground_energy": -6.3,
            "eigen_residual_norm": 0.0,
            "max_bond": 8,
            "worst_cut_discarded_weight": 0.01,
        },
    ]

    rows = module.reference_comparison_rows([4], tn_rows, summary_path)  # type: ignore[attr-defined]

    assert rows[0]["tn_max_bond"] == 8
    assert rows[0]["vqe_status"] == "ok"
    assert rows[0]["vqe_best_ansatz"] == "knm_informed"
    assert rows[0]["vqe_median_relative_error_pct"] == 0.3
