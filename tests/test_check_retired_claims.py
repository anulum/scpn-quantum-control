# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for the retired public-claims guard
"""Tests for tools/check_retired_claims.py.

Covers the register loader's fail-closed branches, the surface walker's
inclusion/exclusion rules, the ±1-line retraction-context window, and the
CLI. The committed ``data/retired_claims.json`` register is itself pinned:
its patterns must catch every historical spelling of the retired claims and
must not fire on the corrected per-pair statements or on numeric
coincidences such as ``5.254019``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tools import check_retired_claims as guard

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_config(path: Path) -> Path:
    config = {
        "claims": [
            {
                "id": "toy-claim",
                "pattern": r"1000x faster",
                "retired": "2026-01-01",
                "reason": "toy",
                "allow_context": [r"\[retracted", "no longer claimed"],
            }
        ]
    }
    config_path = path / "retired_claims.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


class TestLoadClaims:
    def test_valid_register(self, tmp_path: Path) -> None:
        claims = guard.load_claims(_write_config(tmp_path))
        assert len(claims) == 1
        assert claims[0].claim_id == "toy-claim"
        assert claims[0].pattern.search("this is 1000x faster than that")
        assert claims[0].retired == "2026-01-01"

    def test_claims_not_a_list_fails_closed(self, tmp_path: Path) -> None:
        config_path = tmp_path / "bad.json"
        config_path.write_text(json.dumps({"claims": "nope"}), encoding="utf-8")
        with pytest.raises(ValueError, match="non-empty list"):
            guard.load_claims(config_path)

    def test_empty_claims_fails_closed(self, tmp_path: Path) -> None:
        config_path = tmp_path / "bad.json"
        config_path.write_text(json.dumps({"claims": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="non-empty list"):
            guard.load_claims(config_path)

    def test_non_object_entry_fails_closed(self, tmp_path: Path) -> None:
        config_path = tmp_path / "bad.json"
        config_path.write_text(json.dumps({"claims": ["nope"]}), encoding="utf-8")
        with pytest.raises(ValueError, match="must be objects"):
            guard.load_claims(config_path)

    def test_missing_key_fails_closed(self, tmp_path: Path) -> None:
        config_path = tmp_path / "bad.json"
        config_path.write_text(
            json.dumps({"claims": [{"id": "x", "pattern": "y"}]}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="missing key"):
            guard.load_claims(config_path)


class TestIterSurfaceFiles:
    def test_inclusion_and_exclusion(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("root", encoding="utf-8")
        (tmp_path / "setup.py").write_text("not a surface", encoding="utf-8")
        (tmp_path / "docs" / "sub").mkdir(parents=True)
        (tmp_path / "docs" / "page.md").write_text("docs", encoding="utf-8")
        (tmp_path / "docs" / "sub" / "deep.md").write_text("deep", encoding="utf-8")
        (tmp_path / "docs" / "figure.png").write_text("binary-ish", encoding="utf-8")
        (tmp_path / "docs" / "internal").mkdir()
        (tmp_path / "docs" / "internal" / "todo.md").write_text("local", encoding="utf-8")
        (tmp_path / "paper").mkdir()
        (tmp_path / "paper" / "main.tex").write_text("tex", encoding="utf-8")
        found = {p.relative_to(tmp_path).as_posix() for p in guard.iter_surface_files(tmp_path)}
        assert found == {
            "README.md",
            "docs/page.md",
            "docs/sub/deep.md",
            "paper/main.tex",
        }

    def test_missing_surface_dir_is_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("root", encoding="utf-8")
        found = list(guard.iter_surface_files(tmp_path))
        assert [p.name for p in found] == ["README.md"]

    def test_exempt_paths_are_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("root", encoding="utf-8")
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "preprint.md").write_text("published", encoding="utf-8")
        (tmp_path / "docs" / "live.md").write_text("live", encoding="utf-8")
        (tmp_path / "paper" / "sub").mkdir(parents=True)
        (tmp_path / "paper" / "sub" / "main.tex").write_text("tex", encoding="utf-8")
        found = {
            p.relative_to(tmp_path).as_posix()
            for p in guard.iter_surface_files(tmp_path, ("docs/preprint.md", "paper/"))
        }
        assert found == {"README.md", "docs/live.md"}

    def test_root_level_file_can_be_exempt(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("root", encoding="utf-8")
        (tmp_path / "NOTES.md").write_text("kept", encoding="utf-8")
        found = {p.name for p in guard.iter_surface_files(tmp_path, ("README.md",))}
        assert found == {"NOTES.md"}


class TestLoadExemptPaths:
    def test_absent_key_defaults_empty(self, tmp_path: Path) -> None:
        assert guard.load_exempt_paths(_write_config(tmp_path)) == ()

    def test_non_list_fails_closed(self, tmp_path: Path) -> None:
        config_path = tmp_path / "bad.json"
        config_path.write_text(
            json.dumps({"claims": [], "exempt_paths": "nope"}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="must be a list"):
            guard.load_exempt_paths(config_path)

    def test_committed_register_exempts_published_records(self) -> None:
        exempt = guard.load_exempt_paths(REPO_ROOT / "data" / "retired_claims.json")
        assert "docs/preprint.md" in exempt
        assert "paper/" in exempt


class TestContextWindow:
    @pytest.fixture()
    def claim(self) -> guard.RetiredClaim:
        return guard.RetiredClaim(
            claim_id="toy",
            pattern=re.compile("1000x"),
            retired="2026-01-01",
            reason="toy",
            allow_context=(re.compile(r"\[retracted"),),
        )

    def _scan(self, tmp_path: Path, text: str, claim: guard.RetiredClaim) -> list[guard.Finding]:
        target = tmp_path / "page.md"
        target.write_text(text, encoding="utf-8")
        return guard.scan_file(target, [claim], tmp_path)

    def test_bare_revival_is_a_finding(self, tmp_path: Path, claim: guard.RetiredClaim) -> None:
        findings = self._scan(tmp_path, "now 1000x faster again\n", claim)
        assert len(findings) == 1
        assert findings[0].claim_id == "toy"
        assert findings[0].line_number == 1
        assert findings[0].path.as_posix() == "page.md"

    def test_same_line_context_passes(self, tmp_path: Path, claim: guard.RetiredClaim) -> None:
        assert self._scan(tmp_path, "the 1000x figure [retracted 2026]\n", claim) == []

    def test_previous_line_context_passes(self, tmp_path: Path, claim: guard.RetiredClaim) -> None:
        text = "[retracted 2026: cold start]\nthe old 1000x figure\n"
        assert self._scan(tmp_path, text, claim) == []

    def test_next_line_context_passes(self, tmp_path: Path, claim: guard.RetiredClaim) -> None:
        text = "the old 1000x figure\n[retracted 2026: cold start]\n"
        assert self._scan(tmp_path, text, claim) == []

    def test_context_two_lines_away_does_not_pass(
        self, tmp_path: Path, claim: guard.RetiredClaim
    ) -> None:
        text = "the old 1000x figure\nfiller line\n[retracted 2026]\n"
        assert len(self._scan(tmp_path, text, claim)) == 1


class TestCommittedRegister:
    """Pin the committed register's patterns to the historical spellings."""

    @pytest.fixture(scope="class")
    def claims(self) -> dict[str, guard.RetiredClaim]:
        loaded = guard.load_claims(REPO_ROOT / "data" / "retired_claims.json")
        return {claim.claim_id: claim for claim in loaded}

    @pytest.mark.parametrize(
        "text",
        [
            "5401× Hamiltonian (n=4)",
            "`build_xy_hamiltonian_dense` (5 401× vs Qiskit)",
            "Rust-accelerated (5401x)",
        ],
    )
    def test_speedup_pattern_catches_spellings(
        self, claims: dict[str, guard.RetiredClaim], text: str
    ) -> None:
        assert claims["rust-5401x-speedup"].pattern.search(text)

    @pytest.mark.parametrize(
        "text",
        ["5.254019", "the 96.5× warmed baseline", "L=4: 269.5 µs"],
    )
    def test_speedup_pattern_ignores_numeric_coincidences(
        self, claims: dict[str, guard.RetiredClaim], text: str
    ) -> None:
        assert not claims["rust-5401x-speedup"].pattern.search(text)

    @pytest.mark.parametrize(
        "text",
        [
            "CHSH S = 2.165 (>8σ)",
            "CHSH S = 2.165 (> 8σ)",
            r"at ${>}8\sigma$ significance",
            r"at $>8\sigma$ significance.",
            "stated as >8 sigma for both",
        ],
    )
    def test_sigma_pattern_catches_spellings(
        self, claims: dict[str, guard.RetiredClaim], text: str
    ) -> None:
        assert claims["chsh-blanket-8sigma"].pattern.search(text)

    @pytest.mark.parametrize(
        "text",
        [
            "S = 2.188 ± 0.021 (8.9σ, pair q2–q3)",
            "attributes 7.54σ to the lower pair and 8.94σ to the higher",
            "7.5σ and 8.9σ violations",
        ],
    )
    def test_sigma_pattern_ignores_corrected_statements(
        self, claims: dict[str, guard.RetiredClaim], text: str
    ) -> None:
        assert not claims["chsh-blanket-8sigma"].pattern.search(text)

    @pytest.mark.parametrize(
        "text",
        [
            "| 17 | **DUAL PROTECTION on IBM hardware** | IBM v2 | F_FIM > F_XY |",
            "hardware-artefact dual protection on ibm_fez",
            'the IBM v2 "dual protection on ibm_fez" hardware result',
            "Dual Protection on IBM Heron r2 (F_FIM > F_XY)",
            "digital dual protection on ibm_kingston",
        ],
    )
    def test_dual_protection_pattern_catches_hardware_spellings(
        self, claims: dict[str, guard.RetiredClaim], text: str
    ) -> None:
        assert claims["fim-dual-protection-hardware"].pattern.search(text)

    @pytest.mark.parametrize(
        "text",
        [
            "| 11 | Dual protection: Lyapunov + spectral gap | NB40 | 5/6 confirmed |",
            "FIM enhances MBL (dual protection)",
            "the dual protection mechanism in the mean-field model",
            "dual protection of the simulated spectral gap",
        ],
    )
    def test_dual_protection_pattern_ignores_simulation_uses(
        self, claims: dict[str, guard.RetiredClaim], text: str
    ) -> None:
        assert not claims["fim-dual-protection-hardware"].pattern.search(text)


class TestCli:
    def _repo(self, tmp_path: Path, body: str) -> Path:
        root = tmp_path / "repo"
        (root / "data").mkdir(parents=True)
        _write_config(root / "data")
        (root / "data" / "retired_claims.json").rename(root / "data" / "retired_claims.json")
        (root / "README.md").write_text(body, encoding="utf-8")
        return root

    def test_clean_repo_exits_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = self._repo(tmp_path, "an honest page\n")
        assert guard.main(["--root", str(root)]) == 0
        assert "no findings" in capsys.readouterr().out

    def test_finding_exits_one_with_location(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = self._repo(tmp_path, "still 1000x faster\n")
        assert guard.main(["--root", str(root)]) == 1
        out = capsys.readouterr().out
        assert "README.md:1" in out
        assert "toy-claim" in out

    def test_config_override(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        root = tmp_path / "repo"
        root.mkdir()
        (root / "README.md").write_text("still 1000x faster\n", encoding="utf-8")
        config_path = _write_config(tmp_path)
        assert guard.main(["--root", str(root), "--config", str(config_path)]) == 1
        assert "toy-claim" in capsys.readouterr().out
