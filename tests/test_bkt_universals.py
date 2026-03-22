# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for BKT universal amplitude ratio check."""

from __future__ import annotations

from scpn_quantum_control.analysis.bkt_universals import (
    BKTUniversalsSummary,
    check_all_candidates,
)


class TestBKTUniversals:
    def test_returns_summary(self):
        result = check_all_candidates()
        assert isinstance(result, BKTUniversalsSummary)

    def test_at_least_5_candidates(self):
        result = check_all_candidates()
        assert len(result.candidates) >= 5

    def test_sorted_by_deviation(self):
        result = check_all_candidates()
        for i in range(len(result.candidates) - 1):
            assert result.candidates[i].deviation <= result.candidates[i + 1].deviation

    def test_best_is_first(self):
        result = check_all_candidates()
        assert result.best_value == result.candidates[0].value

    def test_best_within_10_percent(self):
        """At least one candidate should be within 10% of 0.72."""
        result = check_all_candidates()
        assert result.best_deviation < 0.072  # 10% of 0.72

    def test_hasenbusch_pinn_candidate(self):
        """A_HP × sqrt(2/π) should be among the best."""
        result = check_all_candidates()
        hp_candidates = [c for c in result.candidates if "A_HP" in c.expression]
        assert len(hp_candidates) > 0
        best_hp = min(hp_candidates, key=lambda c: c.deviation)
        assert best_hp.relative_deviation_pct < 5.0

    def test_gap3_report(self):
        """Record Gap 3 findings."""
        result = check_all_candidates()
        print("\n  Gap 3: BKT universal candidates for p_h1 = 0.72:")
        for c in result.candidates[:5]:
            print(
                f"    {c.expression}: {c.value:.6f} (Δ={c.deviation:.4f}, {c.relative_deviation_pct:.1f}%)"
            )
        print(f"  Best: {result.best_expression} = {result.best_value:.6f}")
        assert result.best_deviation < 0.1
