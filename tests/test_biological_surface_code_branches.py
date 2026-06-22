# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the biological surface code
"""Branch and guard tests for the biological surface code and its MWPM decoder.

Covers the acyclic logical-qubit estimate, the syndrome and correction vector
guards, the unreachable-defect-pair skip in matching, and the re-raise of an
unexpected native decoder error.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.qec.biological_surface_code import (
    BiologicalMWPMDecoder,
    BiologicalSurfaceCode,
)


def _triangle_k() -> NDArray[np.float64]:
    return np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=np.float64)


def _path_k() -> NDArray[np.float64]:
    return np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def _two_triangle_k() -> NDArray[np.float64]:
    k = np.zeros((6, 6), dtype=np.float64)
    for a, b in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]:
        k[a, b] = k[b, a] = 1.0
    return k


def test_estimate_logical_qubits_handles_acyclic_code() -> None:
    """An acyclic (cycle-free) graph yields an empty Z-stabiliser matrix."""
    code = BiologicalSurfaceCode(_path_k())
    assert code.Hz.size == 0
    assert code.estimate_logical_qubits() >= 0


def test_x_syndrome_rejects_wrong_length() -> None:
    """A Z-error vector of the wrong length is rejected."""
    code = BiologicalSurfaceCode(_triangle_k())
    with pytest.raises(ValueError, match="z_errors length must equal"):
        code.x_syndrome_from_z_errors(np.zeros(5, dtype=np.int8))


def test_apply_z_correction_rejects_wrong_error_length() -> None:
    """A residual computation with a mis-sized error vector is rejected."""
    code = BiologicalSurfaceCode(_triangle_k())
    with pytest.raises(ValueError, match="z_errors length must equal"):
        code.apply_z_correction(np.zeros(5, dtype=np.int8), np.zeros(3, dtype=np.int8))


def test_apply_z_correction_rejects_wrong_correction_length() -> None:
    """A residual computation with a mis-sized correction vector is rejected."""
    code = BiologicalSurfaceCode(_triangle_k())
    with pytest.raises(ValueError, match="correction length must equal"):
        code.apply_z_correction(np.zeros(3, dtype=np.int8), np.zeros(5, dtype=np.int8))


def test_apply_z_correction_rejects_non_binary_errors() -> None:
    """A non-binary error vector is rejected."""
    code = BiologicalSurfaceCode(_triangle_k())
    with pytest.raises(ValueError, match="z_errors must be binary"):
        code.apply_z_correction(np.array([2, 0, 0], dtype=np.int8), np.zeros(3, dtype=np.int8))


def test_decoder_skips_unreachable_defect_pairs() -> None:
    """Cross-component defect pairs without a connecting path are skipped in matching."""
    code = BiologicalSurfaceCode(_two_triangle_k())
    decoder = BiologicalMWPMDecoder(code)
    syndrome = np.array([1, 1, 0, 1, 1, 0], dtype=np.int8)
    correction = decoder.decode_z_errors(syndrome)
    assert correction.shape == (code.num_data,)


def test_decoder_reraises_unexpected_native_error() -> None:
    """An unexpected native-decoder error is propagated, not silently swallowed."""
    code = BiologicalSurfaceCode(_triangle_k())
    decoder = BiologicalMWPMDecoder(code)

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("unexpected native failure")

    decoder._rust_engine = cast(Any, SimpleNamespace(biological_decode_z_errors=_boom))
    syndrome = np.array([1, 1, 0], dtype=np.int8)
    with pytest.raises(ValueError, match="unexpected native failure"):
        decoder.decode_z_errors(syndrome)
