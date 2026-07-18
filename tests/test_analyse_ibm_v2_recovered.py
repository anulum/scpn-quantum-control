# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the IBM v2 recovered-counts reproducer
"""Tests for scripts/analyse_ibm_v2_recovered.py.

Covers the survival observables (single-bitstring, odd-parity subspace, the
empty-shot guard), the full reproduction report including the committed-mean
absent branch and the DUAL PROTECTION claim-boundary observation, and the CLI
against both a synthetic pack and the committed recovered pack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import analyse_ibm_v2_recovered as script

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_PACK = REPO_ROOT / "data" / "ibm_hardware_v2_recovered_2026-07-18"


def _pub(target_prob: float, target: str = "0000", shots: int = 1000) -> dict[str, int]:
    hit = round(target_prob * shots)
    other = "1111" if target != "1111" else "0000"
    return {target: hit, other: shots - hit}


class TestObservables:
    def test_bitstring_survival(self) -> None:
        assert script._bitstring_survival({"0000": 850, "1111": 150}, "0000") == 0.85

    def test_bitstring_survival_empty(self) -> None:
        assert script._bitstring_survival({}, "0000") == 0.0

    def test_odd_parity_survival(self) -> None:
        # 0001 and 0111 are odd weight; 0000 and 0011 are even.
        counts = {"0001": 300, "0111": 100, "0000": 500, "0011": 100}
        assert script._odd_parity_survival(counts) == pytest.approx(0.4)

    def test_odd_parity_survival_empty(self) -> None:
        assert script._odd_parity_survival({}) == 0.0

    def test_per_pub_survival_bitstring(self) -> None:
        vals = script.per_pub_survival("C_fim", [_pub(0.9), _pub(0.92)])
        assert vals == pytest.approx([0.9, 0.92])

    def test_per_pub_survival_odd_parity(self) -> None:
        vals = script.per_pub_survival("A_odd", [{"0001": 900, "0000": 100}])
        assert vals == pytest.approx([0.9])


def _synthetic_pack() -> dict[str, Any]:
    return {
        "experiments": [
            {
                "experiment": "C_xy",
                "committed_aggregate_mean": 0.85,
                "per_pub_counts": [_pub(0.85)],
            },
            {
                "experiment": "C_fim",
                "committed_aggregate_mean": 0.92,
                "per_pub_counts": [_pub(0.92)],
            },
            {
                "experiment": "A_odd",
                "committed_aggregate_mean": None,
                "per_pub_counts": [{"0001": 900, "0000": 100}],
            },
        ]
    }


class TestReproduce:
    def test_reproduces_and_flags_dual_protection(self) -> None:
        report = script.reproduce(_synthetic_pack())
        assert report["n_experiments"] == 3
        # C_xy and C_fim reproduce exactly; A_odd has no committed mean to compare.
        assert report["n_reproduced_exactly"] == 2
        obs = report["dual_protection_observation"]
        assert obs["F_FIM"] > obs["F_XY"]
        assert obs["F_FIM_gt_F_XY"] is True
        assert "NOT evidence of coherence protection" in obs["claim_boundary"]

    def test_absent_committed_mean_is_not_reproduced(self) -> None:
        report = script.reproduce(_synthetic_pack())
        a_odd = next(r for r in report["rows"] if r["experiment"] == "A_odd")
        assert a_odd["abs_delta"] is None
        assert a_odd["reproduced"] is False

    def test_committed_pack_reproduces_eight_of_nine(self) -> None:
        pack = json.loads(
            (COMMITTED_PACK / "recovered_raw_counts.json").read_text(encoding="utf-8")
        )
        report = script.reproduce(pack)
        assert report["n_experiments"] == 9
        assert report["n_reproduced_exactly"] == 8  # A_odd differs by ~3.7%
        assert report["dual_protection_observation"]["F_FIM_gt_F_XY"] is True


class TestCli:
    def test_cli_on_committed_pack(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert script.main([]) == 0
        out = capsys.readouterr().out
        assert "F_FIM=0.9158 > F_XY=0.8484" in out
        assert "NOT coherence protection" in out

    def test_cli_writes_output(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        out_path = tmp_path / "analysis.json"
        assert script.main(["--output", str(out_path)]) == 0
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["n_reproduced_exactly"] == 8
        assert "wrote analysis" in capsys.readouterr().out
