# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for P H1 Derivation
"""Tests for p_h1 derivation from BKT universals."""

from __future__ import annotations

import pytest

from scpn_quantum_control.analysis.p_h1_derivation import (
    P_H1_Derivation,
    derive_p_h1,
)


class TestDeriveP_H1:
    def test_returns_derivation(self):
        result = derive_p_h1()
        assert isinstance(result, P_H1_Derivation)

    def test_prediction_near_072(self):
        result = derive_p_h1()
        assert abs(result.p_h1_predicted - 0.72) < 0.01

    def test_relative_deviation_under_1_pct(self):
        result = derive_p_h1()
        assert result.relative_deviation_pct < 1.0

    def test_is_derivable(self):
        result = derive_p_h1()
        assert result.is_derivable

    def test_derivation_chain_length(self):
        result = derive_p_h1()
        assert len(result.derivation_chain) >= 5

    def test_hasenbusch_pinn_value(self):
        result = derive_p_h1()
        assert result.a_hp == pytest.approx(0.8983)

    def test_nk_sqrt_value(self):
        result = derive_p_h1()
        import numpy as np

        assert result.nk_sqrt == pytest.approx(np.sqrt(2 / np.pi), abs=1e-6)

    def test_gap3_final(self):
        """The definitive Gap 3 result."""
        result = derive_p_h1()
        print("\n  === GAP 3 DERIVATION ===")
        for step in result.derivation_chain:
            print(f"  {step}")
        print(f"\n  p_h1 predicted: {result.p_h1_predicted:.6f}")
        print(f"  p_h1 target:    {result.p_h1_target}")
        print(
            f"  Deviation:      {result.absolute_deviation:.4f} ({result.relative_deviation_pct:.1f}%)"
        )
        print(f"  Derivable:      {result.is_derivable}")
        assert result.is_derivable
