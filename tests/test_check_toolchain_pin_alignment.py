# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — toolchain pin alignment tests
"""Refusal behaviour for quality-tool pins that drift between hook and CI."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tools.check_toolchain_pin_alignment import (
    HOOK_DISTRIBUTIONS,
    NON_PYTHON_HOOK_REPOS,
    PRE_COMMIT_CONFIG,
    AlignmentFinding,
    HookPin,
    check_toolchain_pin_alignment,
    load_hook_pins,
    load_requirement_pins,
    main,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

RUFF_REPO = "https://github.com/astral-sh/ruff-pre-commit"
MYPY_REPO = "https://github.com/pre-commit/mirrors-mypy"


def _config(tmp_path: Path, repos: list[dict[str, object]]) -> Path:
    path = tmp_path / PRE_COMMIT_CONFIG
    path.write_text(yaml.safe_dump({"repos": repos}), encoding="utf-8")
    return path


def _checkout(
    tmp_path: Path,
    *,
    repos: list[dict[str, object]],
    requirements: dict[str, str],
) -> Path:
    _config(tmp_path, repos)
    for name, body in requirements.items():
        (tmp_path / name).write_text(body, encoding="utf-8")
    return tmp_path


def _hook(repo: str, rev: str) -> dict[str, object]:
    return {"repo": repo, "rev": rev, "hooks": [{"id": "example"}]}


def test_repository_toolchain_pins_agree() -> None:
    """The committed hook revisions must match the pinned CI requirements."""
    assert check_toolchain_pin_alignment(REPOSITORY_ROOT) == ()


def test_every_remote_hook_repository_is_classified() -> None:
    """No remote hook may sit outside both classification tables."""
    pins = load_hook_pins(REPOSITORY_ROOT / PRE_COMMIT_CONFIG)
    classified = set(HOOK_DISTRIBUTIONS) | set(NON_PYTHON_HOOK_REPOS)

    assert pins
    assert {pin.repo for pin in pins} <= classified


def test_repository_hook_revisions_match_installed_distributions() -> None:
    """The mapped hook revisions must name the versions the requirements pin."""
    pins = {pin.repo: pin for pin in load_hook_pins(REPOSITORY_ROOT / PRE_COMMIT_CONFIG)}
    requirement_pins = load_requirement_pins(sorted(REPOSITORY_ROOT.glob("requirements*.txt")))

    for repo, distribution in HOOK_DISTRIBUTIONS.items():
        declared = set(requirement_pins[distribution].values())
        assert len(declared) == 1, distribution
        assert pins[repo].version == declared.pop()


def test_hook_lagging_the_requirement_generation_is_refused(tmp_path: Path) -> None:
    """A hook pinned to an older release than CI installs must be refused."""
    root = _checkout(
        tmp_path,
        repos=[_hook(RUFF_REPO, "v0.15.18")],
        requirements={"requirements-ci-py312-linux.txt": "ruff==0.16.4 \\\n"},
    )

    findings = check_toolchain_pin_alignment(root)

    assert [finding.kind for finding in findings] == ["hook_generation_mismatch"]
    assert "v0.15.18" in findings[0].detail
    assert "0.16.4" in findings[0].detail


def test_requirement_files_disagreeing_on_a_version_are_refused(tmp_path: Path) -> None:
    """A distribution pinned to two versions has no single generation to match."""
    root = _checkout(
        tmp_path,
        repos=[_hook(MYPY_REPO, "v2.3.0")],
        requirements={
            "requirements-ci-py312-linux.txt": "mypy==2.3.0\n",
            "requirements-ci-py313-linux.txt": "mypy==1.20.0\n",
        },
    )

    findings = check_toolchain_pin_alignment(root)

    assert [finding.kind for finding in findings] == ["requirement_version_split"]
    assert "requirements-ci-py312-linux.txt=2.3.0" in findings[0].detail
    assert "requirements-ci-py313-linux.txt=1.20.0" in findings[0].detail


def test_unclassified_hook_repository_is_refused(tmp_path: Path) -> None:
    """A newly added hook cannot enter the configuration unclassified."""
    root = _checkout(
        tmp_path,
        repos=[_hook("https://github.com/example/new-hook", "v1.0.0")],
        requirements={"requirements-ci-py312-linux.txt": "ruff==0.16.4\n"},
    )

    findings = check_toolchain_pin_alignment(root)

    assert [finding.kind for finding in findings] == ["unclassified_hook_repo"]
    assert "classify it before it can pass" in findings[0].detail


def test_mapped_hook_without_a_requirement_pin_is_refused(tmp_path: Path) -> None:
    """A mapped tool that CI never pins cannot be proven to agree."""
    root = _checkout(
        tmp_path,
        repos=[_hook(RUFF_REPO, "v0.16.4")],
        requirements={"requirements-ci-py312-linux.txt": "numpy==2.4.6\n"},
    )

    findings = check_toolchain_pin_alignment(root)

    assert [finding.kind for finding in findings] == ["hook_without_requirement_pin"]
    assert "ruff" in findings[0].detail


def test_non_python_hook_is_skipped_with_a_recorded_reason(tmp_path: Path) -> None:
    """A classified non-Python hook passes and keeps its written reason."""
    gitleaks = next(iter(NON_PYTHON_HOOK_REPOS))
    root = _checkout(
        tmp_path,
        repos=[_hook(gitleaks, "v8.21.2")],
        requirements={"requirements-ci-py312-linux.txt": "ruff==0.16.4\n"},
    )

    assert check_toolchain_pin_alignment(root) == ()
    assert NON_PYTHON_HOOK_REPOS[gitleaks].strip()


def test_local_hooks_carry_no_revision_and_are_skipped(tmp_path: Path) -> None:
    """In-repository hooks declare no revision and must not be compared."""
    root = _checkout(
        tmp_path,
        repos=[
            {"repo": "local", "hooks": [{"id": "check-secrets"}]},
            _hook(RUFF_REPO, "v0.16.4"),
        ],
        requirements={"requirements-ci-py312-linux.txt": "ruff==0.16.4\n"},
    )

    pins = load_hook_pins(root / PRE_COMMIT_CONFIG)

    assert [pin.repo for pin in pins] == [RUFF_REPO]
    assert check_toolchain_pin_alignment(root) == ()


def test_requirement_pin_names_are_normalised(tmp_path: Path) -> None:
    """A pin spelled with underscores or capitals must still be found."""
    path = tmp_path / "requirements-ci-py312-linux.txt"
    path.write_text(
        "Types_Defusedxml==0.7.0\n"
        "# comment\n"
        "    indented==1.0.0\n"
        "ruff==0.16.4 \\\n"
        "    --hash=sha256:deadbeef \\\n",
        encoding="utf-8",
    )

    pins = load_requirement_pins([path])

    assert pins["types-defusedxml"] == {path.name: "0.7.0"}
    assert pins["ruff"] == {path.name: "0.16.4"}
    assert "indented" not in pins


def test_hook_pin_version_strips_only_a_release_tag_marker() -> None:
    """A ``v`` prefix is a tag marker; a bare revision is used unchanged."""
    assert HookPin(repo=RUFF_REPO, rev="v0.16.4").version == "0.16.4"
    assert HookPin(repo=RUFF_REPO, rev="0.16.4").version == "0.16.4"


def test_alignment_finding_requires_an_explanation() -> None:
    """A finding without a reason is not a usable refusal."""
    with pytest.raises(ValueError, match="must explain the finding"):
        AlignmentFinding(kind="hook_generation_mismatch", detail="  ")


def test_malformed_configurations_are_refused(tmp_path: Path) -> None:
    """An unreadable configuration must raise, never report agreement."""
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / PRE_COMMIT_CONFIG).write_text("repos: []\n", encoding="utf-8")
    (empty / "requirements-ci-py312-linux.txt").write_text("ruff==0.16.4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="declares no hook repositories"):
        check_toolchain_pin_alignment(empty)

    scalar = tmp_path / "scalar"
    scalar.mkdir()
    (scalar / PRE_COMMIT_CONFIG).write_text("just a string\n", encoding="utf-8")
    with pytest.raises(ValueError, match="declares no hook repositories"):
        load_hook_pins(scalar / PRE_COMMIT_CONFIG)

    revless = tmp_path / "revless"
    revless.mkdir()
    _config(revless, [{"repo": RUFF_REPO, "hooks": [{"id": "ruff"}]}])
    with pytest.raises(ValueError, match="declares no rev"):
        load_hook_pins(revless / PRE_COMMIT_CONFIG)

    only_local = tmp_path / "only_local"
    only_local.mkdir()
    _config(only_local, [{"repo": "local", "hooks": [{"id": "check-secrets"}]}])
    with pytest.raises(ValueError, match="no remote hook repository"):
        load_hook_pins(only_local / PRE_COMMIT_CONFIG)


def test_missing_evidence_files_are_refused(tmp_path: Path) -> None:
    """A checkout without the configuration or requirements cannot be checked."""
    with pytest.raises(ValueError, match=f"missing {PRE_COMMIT_CONFIG}"):
        check_toolchain_pin_alignment(tmp_path)

    _config(tmp_path, [_hook(RUFF_REPO, "v0.16.4")])
    with pytest.raises(ValueError, match="no requirements"):
        check_toolchain_pin_alignment(tmp_path)


def test_cli_reports_agreement_for_the_repository(capsys: pytest.CaptureFixture[str]) -> None:
    """The committed checkout must exit clean with an explicit statement."""
    exit_code = main(["--source-root", str(REPOSITORY_ROOT)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "toolchain pin alignment: OK" in captured.out


def test_cli_reports_each_finding_and_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A drifting checkout exits refused and names the disagreement."""
    root = _checkout(
        tmp_path,
        repos=[_hook(RUFF_REPO, "v0.15.18")],
        requirements={"requirements-ci-py312-linux.txt": "ruff==0.16.4\n"},
    )

    exit_code = main(["--source-root", str(root)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "hook_generation_mismatch" in captured.err
    assert captured.out == ""


def test_cli_fails_closed_on_unreadable_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unusable checkout returns the evidence-error code, never a pass."""
    exit_code = main(["--source-root", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "toolchain pin evidence unavailable" in captured.err
