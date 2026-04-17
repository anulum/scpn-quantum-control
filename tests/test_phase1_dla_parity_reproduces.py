# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase 1 Hardware Reproducer
"""Reproducer: Phase 1 DLA parity campaign headline numbers.

Loads the raw `data/phase1_dla_parity/*.json` files that ship with the
repository and asserts that the statistics quoted in `CHANGELOG.md` and
`docs/preprint.md` for the v0.9.5 Phase 1 IBM Heron r2 campaign still
reproduce end-to-end. If any of these assertions break, the scientific
claim and the data have drifted from one another — either the data
changed, the analysis changed, or one of the publication numbers is
stale. Any of those needs action.

Published claims verified here:

- Peak relative asymmetry `+17.48%` at depth 6
- Mean relative asymmetry `+10.8%` for Trotter depths ≥ 4
- 7 of 8 depths individually significant at `p < 0.05` (Welch t-test)
- Fisher combined `chi² = 123.4` over `df = 16`
- Fisher combined `p ≪ 10^-16`
- All four Phase 1 data files present, ≥ 300 circuits total, 8 distinct
  n = 4 depth points

The test imports the analysis script via `importlib.util` (the
`scripts/` tree is not a Python package by design).
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSE_SCRIPT = REPO_ROOT / "scripts" / "analyse_phase1_dla_parity.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_analyse_phase1", ANALYSE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the @dataclass decorator can resolve
    # cls.__module__ back to this module (dataclasses walk sys.modules
    # to locate forward references in __annotations__).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def analyse():
    return _load_module()


@pytest.fixture(scope="module")
def summaries(analyse):
    circuits = analyse.load_phase1_circuits()
    by_depth = analyse.collect_n4_depth_points(circuits)
    return [analyse.summarise_depth(by_depth[d]) for d in sorted(by_depth.keys())]


class TestPhase1Dataset:
    """Raw-data presence and coverage checks."""

    def test_all_four_phase1_files_exist(self, analyse):
        missing = [p for p in analyse.PHASE1_FILES if not p.exists()]
        assert not missing, (
            f"Phase 1 dataset incomplete — missing files: {missing}. The "
            f"reproducer cannot run without the raw JSONs in "
            f"data/phase1_dla_parity/."
        )

    def test_circuit_count_at_least_claimed_floor(self, analyse):
        circuits = analyse.load_phase1_circuits()
        assert len(circuits) >= 300, (
            f"Expected ≥ 300 circuits across the Phase 1 sub-phases; "
            f"loaded {len(circuits)}. Check that every sub-phase JSON "
            f"still contains a non-empty 'circuits' block."
        )

    def test_eight_distinct_n4_depth_points(self, analyse):
        circuits = analyse.load_phase1_circuits()
        by_depth = analyse.collect_n4_depth_points(circuits)
        assert len(by_depth) == 8, (
            f"Phase 1 campaign was designed around 8 distinct Trotter "
            f"depths at n = 4; collected {len(by_depth)}."
        )


class TestPhase1Statistics:
    """Check the headline numbers quoted in CHANGELOG and preprint."""

    def test_peak_asymmetry_at_depth_6(self, summaries):
        depth_6 = next((s for s in summaries if s.depth == 6), None)
        assert depth_6 is not None, "Depth 6 missing from summaries"
        pct = 100.0 * depth_6.asymmetry_relative
        assert math.isclose(pct, 17.48, abs_tol=2.0), (
            f"Depth-6 relative asymmetry {pct:+.2f}% drifted from the "
            f"claim of +17.48% (CHANGELOG v0.9.5)."
        )

    def test_mean_asymmetry_depths_ge_4(self, summaries):
        deep = [s for s in summaries if s.depth >= 4]
        assert deep, "No depth ≥ 4 points present"
        mean_pct = 100.0 * sum(s.asymmetry_relative for s in deep) / len(deep)
        assert math.isclose(mean_pct, 10.8, abs_tol=2.0), (
            f"Mean relative asymmetry (depths ≥ 4) = {mean_pct:+.2f}%; "
            f"claim in CHANGELOG is +10.8%."
        )

    def test_seven_of_eight_depths_significant(self, summaries):
        significant = sum(1 for s in summaries if not math.isnan(s.welch_p) and s.welch_p < 0.05)
        assert significant >= 7, (
            f"Only {significant}/8 depths have Welch p < 0.05; the "
            f"CHANGELOG claims 7/8 are individually significant."
        )

    def test_fisher_combined_chi2(self, analyse, summaries):
        pvals = [s.welch_p for s in summaries if not math.isnan(s.welch_p)]
        chi2, combined_p = analyse.fisher_combined_pvalue(pvals)
        assert math.isclose(chi2, 123.4, abs_tol=10.0), (
            f"Fisher combined chi² = {chi2:.3f}; claim is 123.4."
        )
        assert 2 * len(pvals) == 16, f"Fisher df = {2 * len(pvals)}; claim is 16 (= 2 × 8 depths)."
        assert combined_p < 1e-16, f"Fisher combined p = {combined_p}; claim is p ≪ 10^-16."


class TestAnalysisScriptEndToEnd:
    """Full `main()` smoke run, no figures."""

    def test_main_callable(self, analyse, monkeypatch, tmp_path, capsys):
        # Avoid polluting the working tree with figure artefacts; the
        # test contract is only that main() completes and prints the
        # headline combined-p line, not that it produces PNGs.
        import matplotlib

        matplotlib.use("Agg")
        monkeypatch.setattr(
            "sys.argv", ["analyse_phase1_dla_parity.py", "--out-dir", str(tmp_path)]
        )
        rc = analyse.main()
        assert rc == 0
        out = capsys.readouterr().out
        assert "Fisher combined chi²" in out
        assert "Depth points with Welch p < 0.05" in out
