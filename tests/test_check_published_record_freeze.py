# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for the published-record freeze guard
"""Tests for tools/check_published_record_freeze.py.

Published records are read-only (owner ruling 2026-07-16). The guard must
fail on any modification, deletion, or addition inside a frozen record
tree, pass on an intact tree, and regenerate its manifest only via the
explicit ``--update`` path. The committed manifest is itself pinned
against the working tree.
"""

from __future__ import annotations

import json
import shutil
import subprocess  # nosec B404
from pathlib import Path

import pytest

from tools import check_published_record_freeze as freeze

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    (root / "paper" / "submissions" / "s1").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "paper" / "submissions" / "s1" / "main.tex").write_text(
        "published body\n", encoding="utf-8"
    )
    (root / "docs" / "preprint.md").write_text("published page\n", encoding="utf-8")
    return root


class TestManifestRoundTrip:
    def test_update_then_verify_passes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = _make_repo(tmp_path)
        assert freeze.main(["--root", str(root), "--update"]) == 0
        assert freeze.main(["--root", str(root)]) == 0
        assert "intact" in capsys.readouterr().out

    def test_manifest_content_is_sorted_and_hashed(self, tmp_path: Path) -> None:
        root = _make_repo(tmp_path)
        freeze.main(["--root", str(root), "--update"])
        payload = json.loads(
            (root / "data" / "published_record_freeze.json").read_text(encoding="utf-8")
        )
        assert list(payload["records"]) == sorted(payload["records"])
        assert set(payload["records"]) == {
            "docs/preprint.md",
            "paper/submissions/s1/main.tex",
        }
        digest = payload["records"]["docs/preprint.md"]
        assert len(digest) == 64 and int(digest, 16) >= 0

    def test_missing_docs_page_is_simply_not_frozen(self, tmp_path: Path) -> None:
        root = _make_repo(tmp_path)
        (root / "docs" / "preprint.md").unlink()
        records = freeze.build_manifest(root)
        assert set(records) == {"paper/submissions/s1/main.tex"}

    def test_missing_frozen_tree_yields_no_tree_entries(self, tmp_path: Path) -> None:
        root = tmp_path / "bare"
        (root / "docs").mkdir(parents=True)
        (root / "docs" / "preprint.md").write_text("page\n", encoding="utf-8")
        records = freeze.build_manifest(root)
        assert set(records) == {"docs/preprint.md"}

    def test_untracked_build_artifacts_are_not_frozen(self, tmp_path: Path) -> None:
        """Inside a git checkout only tracked files enter the freeze."""
        root = _make_repo(tmp_path)
        subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(root), "add", "paper", "docs"], check=True)
        (root / "paper" / "submissions" / "s1" / "build.pdf").write_text(
            "local artifact\n", encoding="utf-8"
        )
        records = freeze.build_manifest(root)
        assert "paper/submissions/s1/main.tex" in records
        assert "paper/submissions/s1/build.pdf" not in records

    def test_no_git_executable_falls_back_to_plain_walk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        root = _make_repo(tmp_path)
        records = freeze.build_manifest(root)
        assert "paper/submissions/s1/main.tex" in records


class TestDriftDetection:
    @pytest.fixture()
    def frozen_repo(self, tmp_path: Path) -> Path:
        root = _make_repo(tmp_path)
        assert freeze.main(["--root", str(root), "--update"]) == 0
        return root

    def test_modification_fails(
        self, frozen_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (frozen_repo / "docs" / "preprint.md").write_text("edited!\n", encoding="utf-8")
        assert freeze.main(["--root", str(frozen_repo)]) == 1
        out = capsys.readouterr().out
        assert "published record modified: docs/preprint.md" in out
        assert "read-only" in out

    def test_deletion_fails(self, frozen_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
        (frozen_repo / "paper" / "submissions" / "s1" / "main.tex").unlink()
        assert freeze.main(["--root", str(frozen_repo)]) == 1
        assert "missing published-record file" in capsys.readouterr().out

    def test_addition_inside_frozen_tree_fails(
        self, frozen_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (frozen_repo / "paper" / "submissions" / "s1" / "extra.tex").write_text(
            "new\n", encoding="utf-8"
        )
        assert freeze.main(["--root", str(frozen_repo)]) == 1
        assert "unpinned file inside a frozen record tree" in capsys.readouterr().out

    def test_manifest_override_path(self, frozen_repo: Path, tmp_path: Path) -> None:
        alt = tmp_path / "alt_manifest.json"
        assert freeze.main(["--root", str(frozen_repo), "--manifest", str(alt), "--update"]) == 0
        assert freeze.main(["--root", str(frozen_repo), "--manifest", str(alt)]) == 0


class TestManifestValidation:
    def test_empty_records_fails_closed(self, tmp_path: Path) -> None:
        manifest = tmp_path / "bad.json"
        manifest.write_text(json.dumps({"records": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="non-empty object"):
            freeze.load_manifest(manifest)

    def test_records_not_object_fails_closed(self, tmp_path: Path) -> None:
        manifest = tmp_path / "bad.json"
        manifest.write_text(json.dumps({"records": ["x"]}), encoding="utf-8")
        with pytest.raises(ValueError, match="non-empty object"):
            freeze.load_manifest(manifest)


class TestCommittedManifest:
    """The committed manifest must match the committed published sources."""

    def test_repo_freeze_holds(self) -> None:
        pinned = freeze.load_manifest(REPO_ROOT / "data" / "published_record_freeze.json")
        current = freeze.build_manifest(REPO_ROOT)
        assert freeze.compare(pinned, current) == []

    def test_repo_freeze_covers_the_published_pages(self) -> None:
        pinned = freeze.load_manifest(REPO_ROOT / "data" / "published_record_freeze.json")
        assert "docs/preprint.md" in pinned
        assert any(path.startswith("paper/submissions/") for path in pinned)
