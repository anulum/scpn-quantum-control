# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the biological QEC CLI
"""Branch tests for the biological QEC command-line input loaders.

Covers the CSV loader branches and unsupported-suffix guards for the matrix and
array loaders and the rejection of a higher-dimensional z-error payload.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpn_quantum_control.qec.biological_cli import _load_array, _load_matrix, main


def test_load_matrix_reads_csv(tmp_path: Path) -> None:
    """A CSV coupling matrix is loaded."""
    path = tmp_path / "K.csv"
    path.write_text("0,1\n1,0\n", encoding="utf-8")
    matrix = _load_matrix(path)
    assert matrix.shape == (2, 2)


def test_load_matrix_rejects_unknown_suffix(tmp_path: Path) -> None:
    """An unsupported matrix file suffix is rejected."""
    path = tmp_path / "K.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="K input must be .npy, .csv, or .txt"):
        _load_matrix(path)


def test_load_array_reads_csv(tmp_path: Path) -> None:
    """A CSV error vector is loaded."""
    path = tmp_path / "z.csv"
    path.write_text("0,1,0\n", encoding="utf-8")
    vector = _load_array(path)
    assert vector.shape == (3,)


def test_load_array_rejects_unknown_suffix(tmp_path: Path) -> None:
    """An unsupported array file suffix is rejected."""
    path = tmp_path / "z.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="z_errors input must be .npy, .csv, or .txt"):
        _load_array(path)


def test_main_rejects_higher_dimensional_z_errors(tmp_path: Path) -> None:
    """A z-error payload that is neither a vector nor a matrix is rejected."""
    k_path = tmp_path / "K.npy"
    z_path = tmp_path / "z.npy"
    out_path = tmp_path / "out.json"
    K = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    np.save(k_path, K)
    np.save(z_path, np.zeros((2, 2, 2), dtype=np.int8))
    with pytest.raises(ValueError, match="z_errors input must be either a vector"):
        main(["--k", str(k_path), "--z-errors", str(z_path), "--output", str(out_path)])
