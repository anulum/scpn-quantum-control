# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the CHSH recomputation script
"""Tests for scripts/recompute_chsh_bell_test.py.

The exact-value tests pin the recomputed statistics against the committed
``bell_test_4q.json`` artifact — the same numbers now stated in the public
record after the 2026-07-16 dated amendment (7.54σ for the S=2.165 pair,
8.94σ for the S=2.188 pair). Synthetic-count tests cover the correlator
arithmetic and every fail-closed branch; the CLI is exercised end to end.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from scripts import recompute_chsh_bell_test as script


@pytest.fixture(scope="module")
def artifact() -> dict[str, object]:
    """Load the committed Bell-test artifact once per module."""
    payload = json.loads(script.DEFAULT_ARTIFACT.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


class TestPairCorrelator:
    def test_perfect_correlation(self) -> None:
        e_value, total = script.pair_correlator({"00": 50, "11": 50}, (0, 1))
        assert e_value == pytest.approx(1.0)
        assert total == 100

    def test_perfect_anticorrelation(self) -> None:
        e_value, total = script.pair_correlator({"01": 30, "10": 70}, (0, 1))
        assert e_value == pytest.approx(-1.0)
        assert total == 100

    def test_mixed_counts(self) -> None:
        e_value, total = script.pair_correlator({"00": 3, "01": 1, "10": 1, "11": 3}, (0, 1))
        assert e_value == pytest.approx(0.5)
        assert total == 8

    def test_little_endian_positions(self) -> None:
        # Qubit 2 and 3 live in the two LEFTMOST characters of a 4-bit string.
        e_value, _ = script.pair_correlator({"0011": 10}, (2, 3))
        assert e_value == pytest.approx(1.0)
        e_value, _ = script.pair_correlator({"0111": 10}, (2, 3))
        assert e_value == pytest.approx(-1.0)

    def test_bitstring_too_narrow_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="narrower than qubit pair"):
            script.pair_correlator({"01": 5}, (2, 3))

    def test_zero_shots_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="no shots recorded"):
            script.pair_correlator({"00": 0, "11": 0}, (0, 1))


class TestChshForPair:
    def test_wrong_setting_count_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="expected 4 analyser settings"):
            script.chsh_for_pair([{"counts": {"00": 1}}], (0, 1))

    def test_missing_counts_fails_closed(self) -> None:
        settings: list[dict[str, object]] = [
            {"counts": {"00": 1}},
            {"pub_index": 1},
            {"counts": {"00": 1}},
            {"counts": {"00": 1}},
        ]
        with pytest.raises(ValueError, match="lacks a 'counts' mapping"):
            script.chsh_for_pair(settings, (0, 1))

    def test_tsirelson_style_synthetic_counts(self) -> None:
        # E = ±1/sqrt(2) per setting reproduces S = 2*sqrt(2) exactly.
        plus = {"00": 8536, "01": 1464, "10": 1464, "11": 8536}
        minus = {"00": 1464, "01": 8536, "10": 8536, "11": 1464}
        stats = script.chsh_for_pair(
            [{"counts": plus}, {"counts": minus}, {"counts": plus}, {"counts": plus}],
            (0, 1),
        )
        assert stats.s_value == pytest.approx(4 * 0.7072, abs=1e-4)
        expected_sigma = math.sqrt(4 * (1 - 0.7072**2) / 20000)
        assert stats.sigma == pytest.approx(expected_sigma, abs=1e-6)

    def test_zero_sigma_significance_is_signed_infinity(self) -> None:
        above = script.PairStatistics(
            label="toy", settings_e=(1.0, -1.0, 1.0, 1.0), s_value=4.0, sigma=0.0
        )
        assert above.significance == math.inf
        below = script.PairStatistics(
            label="toy", settings_e=(-1.0, 1.0, -1.0, -1.0), s_value=-4.0, sigma=0.0
        )
        assert below.significance == -math.inf

    def test_zero_sigma_at_the_bound_is_zero(self) -> None:
        at_bound = script.PairStatistics(
            label="toy", settings_e=(1.0, 1.0, 1.0, 1.0), s_value=2.0, sigma=0.0
        )
        assert at_bound.significance == 0.0

    def test_minus_sign_sits_on_setting_one(self) -> None:
        same = {"00": 10, "11": 10}
        stats = script.chsh_for_pair([{"counts": same}] * 4, (0, 1))
        # E = +1 everywhere -> S = 1 - 1 + 1 + 1 = 2.
        assert stats.s_value == pytest.approx(2.0)
        assert stats.settings_e == (1.0, 1.0, 1.0, 1.0)


class TestCommittedArtifactValues:
    """Pin the corrected public numbers to the committed raw counts."""

    def test_pair_q0q1(self, artifact: dict[str, object]) -> None:
        pairs = script.recompute(artifact)
        stats = pairs[0]
        assert stats.label == "q0q1"
        assert stats.s_value == pytest.approx(2.1650, abs=5e-5)
        assert stats.sigma == pytest.approx(0.0219, abs=5e-5)
        assert stats.significance == pytest.approx(7.54, abs=5e-3)

    def test_pair_q2q3(self, artifact: dict[str, object]) -> None:
        pairs = script.recompute(artifact)
        stats = pairs[1]
        assert stats.label == "q2q3"
        assert stats.s_value == pytest.approx(2.1880, abs=5e-5)
        assert stats.sigma == pytest.approx(0.0210, abs=5e-5)
        assert stats.significance == pytest.approx(8.94, abs=5e-3)

    def test_only_the_higher_pair_clears_eight_sigma(self, artifact: dict[str, object]) -> None:
        low, high = script.recompute(artifact)
        assert low.significance < 8.0 < high.significance

    def test_setting_one_anomaly_is_real(self, artifact: dict[str, object]) -> None:
        for stats in script.recompute(artifact):
            others = [e for i, e in enumerate(stats.settings_e) if i != 1]
            assert stats.settings_e[1] < 0.35
            assert min(others) > 0.75

    def test_artifact_without_results_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="lacks a 'results' list"):
            script.recompute({"backend": "ibm_fez"})


class TestReportAndCli:
    def test_render_report_states_both_pairs_and_anomaly(
        self, artifact: dict[str, object]
    ) -> None:
        report = script.render_report(script.recompute(artifact))
        assert "pair q0q1" in report
        assert "7.54 sigma" in report
        assert "pair q2q3" in report
        assert "8.94 sigma" in report
        assert "anomalous" in report

    def test_main_prints_report(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert script.main([]) == 0
        out = capsys.readouterr().out
        assert "S = 2.1650" in out
        assert "S = 2.1880" in out

    def test_main_writes_json(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        out_path = tmp_path / "chsh.json"
        assert script.main(["--json", str(out_path)]) == 0
        capsys.readouterr()
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert set(payload) == {"q0q1", "q2q3"}
        assert payload["q0q1"]["s_value"] == pytest.approx(2.1650, abs=5e-5)
        assert payload["q2q3"]["significance"] == pytest.approx(8.94, abs=5e-3)
        assert len(payload["q0q1"]["settings_e"]) == 4

    def test_main_custom_artifact(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        same = {"0000": 10, "1111": 10}
        payload = {"results": [{"counts": same}] * 4}
        artifact_path = tmp_path / "toy.json"
        artifact_path.write_text(json.dumps(payload), encoding="utf-8")
        assert script.main(["--artifact", str(artifact_path)]) == 0
        assert "S = 2.0000" in capsys.readouterr().out
