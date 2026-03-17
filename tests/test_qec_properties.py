# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Property-based tests for QEC surface code."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from scpn_quantum_control.qec.control_qec import ControlQEC


@given(d=st.integers(min_value=3, max_value=5))
@settings(max_examples=5, deadline=10000)
def test_zero_errors_zero_syndrome(d: int) -> None:
    """No errors must produce all-zero syndromes."""
    qec = ControlQEC(distance=d)
    N = qec.code.num_data
    err_x = np.zeros(N, dtype=np.int8)
    err_z = np.zeros(N, dtype=np.int8)
    syn_z, syn_x = qec.get_syndrome(err_x, err_z)
    assert np.all(syn_z == 0)
    assert np.all(syn_x == 0)


@given(d=st.integers(min_value=3, max_value=5))
@settings(max_examples=5, deadline=10000)
def test_correction_shape_matches_error(d: int) -> None:
    """Correction vector must have same shape as error vector."""
    qec = ControlQEC(distance=d)
    N = qec.code.num_data
    rng = np.random.default_rng(d)
    err_x = rng.binomial(1, 0.05, N).astype(np.int8)
    err_z = rng.binomial(1, 0.05, N).astype(np.int8)
    syn_z, syn_x = qec.get_syndrome(err_x, err_z)
    corr_x = qec.decoder.decode(syn_z, dual=False)
    corr_z = qec.decoder.decode(syn_x, dual=True)
    assert corr_x.shape == err_x.shape
    assert corr_z.shape == err_z.shape


@given(d=st.integers(min_value=3, max_value=5))
@settings(max_examples=5, deadline=10000)
def test_syndrome_shape_matches_distance(d: int) -> None:
    """Syndrome vectors must have d^2 entries."""
    qec = ControlQEC(distance=d)
    N = qec.code.num_data
    rng = np.random.default_rng(d)
    err_x = rng.binomial(1, 0.1, N).astype(np.int8)
    err_z = rng.binomial(1, 0.1, N).astype(np.int8)
    syn_z, syn_x = qec.get_syndrome(err_x, err_z)
    assert syn_z.shape == (d * d,)
    assert syn_x.shape == (d * d,)
