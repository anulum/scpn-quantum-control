# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — toolchain declaration alignment tests
"""Refusal behaviour for tool versions that disagree between declarations."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tools.check_toolchain_pin_alignment import (
    ACTION_TOOLS,
    HOOK_DISTRIBUTIONS,
    NON_PYTHON_HOOK_REPOS,
    PRE_COMMIT_CONFIG,
    PYPROJECT,
    AlignmentFinding,
    ToolDeclaration,
    check_toolchain_pin_alignment,
    collect_declarations,
    main,
    normalise,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

RUFF_REPO = "https://github.com/astral-sh/ruff-pre-commit"
MYPY_REPO = "https://github.com/pre-commit/mirrors-mypy"
PNPM_ACTION = next(iter(ACTION_TOOLS))


def _hook(repo: str, rev: str) -> dict[str, object]:
    return {"repo": repo, "rev": rev, "hooks": [{"id": "example"}]}


def _checkout(
    tmp_path: Path,
    *,
    repos: list[dict[str, object]] | None = None,
    requirements: dict[str, str] | None = None,
    pyproject: str | None = None,
    workflows: dict[str, str] | None = None,
    contributing: str | None = None,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = {"repos": repos if repos is not None else [_hook(RUFF_REPO, "v0.16.4")]}
    (tmp_path / PRE_COMMIT_CONFIG).write_text(yaml.safe_dump(config), encoding="utf-8")
    body = requirements if requirements is not None else {"requirements.txt": "ruff==0.16.4\n"}
    for name, text in body.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    if pyproject is not None:
        (tmp_path / PYPROJECT).write_text(pyproject, encoding="utf-8")
    if workflows:
        directory = tmp_path / ".github" / "workflows"
        directory.mkdir(parents=True, exist_ok=True)
        for name, text in workflows.items():
            (directory / name).write_text(text, encoding="utf-8")
    if contributing is not None:
        (tmp_path / "CONTRIBUTING.md").write_text(contributing, encoding="utf-8")
    return tmp_path


def _kinds(findings: tuple[AlignmentFinding, ...]) -> list[str]:
    return [finding.kind for finding in findings]


def test_repository_declarations_agree() -> None:
    """Every declaration this repository makes must name one version per tool."""
    assert check_toolchain_pin_alignment(REPOSITORY_ROOT) == ()


def test_repository_declares_each_tool_in_more_than_one_place() -> None:
    """The gate is only meaningful where a tool is declared more than once."""
    declarations = collect_declarations(REPOSITORY_ROOT)
    counts: dict[str, int] = {}
    for declaration in declarations:
        counts[declaration.tool] = counts.get(declaration.tool, 0) + 1

    assert {"ruff", "mypy", "pnpm"} <= set(counts)
    assert counts["ruff"] > 1
    assert counts["mypy"] > 1
    assert counts["pnpm"] > 1


def test_repository_covers_every_declaration_kind_it_uses() -> None:
    """Each collector must actually find the declarations it owns."""
    kinds = {declaration.kind for declaration in collect_declarations(REPOSITORY_ROOT)}

    assert kinds == {
        "pre_commit_rev",
        "requirement_pin",
        "pyproject_range",
        "workflow_action_input",
        "workflow_install",
        "documented_command",
    }


def test_every_remote_hook_repository_is_classified() -> None:
    """No remote hook may sit outside both classification tables."""
    document = yaml.safe_load((REPOSITORY_ROOT / PRE_COMMIT_CONFIG).read_text(encoding="utf-8"))
    remote = {
        str(entry["repo"]) for entry in document["repos"] if str(entry.get("repo")) != "local"
    }

    assert remote
    assert remote <= set(HOOK_DISTRIBUTIONS) | set(NON_PYTHON_HOOK_REPOS)


def test_hook_lagging_the_requirement_generation_is_refused(tmp_path: Path) -> None:
    """A hook pinned to an older release than CI installs must be refused."""
    root = _checkout(
        tmp_path,
        repos=[_hook(RUFF_REPO, "v0.15.18")],
        requirements={"requirements-ci-py312-linux.txt": "ruff==0.16.4 \\\n"},
    )

    findings = check_toolchain_pin_alignment(root)

    assert _kinds(findings) == ["version_disagreement"]
    assert "pre_commit_rev)=0.15.18" in findings[0].detail
    assert "requirement_pin)=0.16.4" in findings[0].detail


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

    assert _kinds(findings) == ["version_disagreement"]
    assert "requirements-ci-py313-linux.txt(requirement_pin)=1.20.0" in findings[0].detail


def test_workflow_action_input_must_agree_with_the_documented_command(tmp_path: Path) -> None:
    """A workflow input and a contributor command naming one tool must agree."""
    root = _checkout(
        tmp_path,
        workflows={
            "ci-studio.yml": (
                "jobs:\n"
                "  studio:\n"
                "    steps:\n"
                f"      - uses: {PNPM_ACTION}@abc123\n"
                "        with:\n"
                "          version: 11.9.0\n"
            )
        },
        contributing="Run `corepack prepare pnpm@11.8.0` before building.\n",
    )

    findings = check_toolchain_pin_alignment(root)

    assert _kinds(findings) == ["version_disagreement"]
    assert "workflow_action_input)=11.9.0" in findings[0].detail
    assert "documented_command)=11.8.0" in findings[0].detail


def test_two_workflows_pinning_one_tool_must_agree(tmp_path: Path) -> None:
    """A pinned install repeated across workflows is one declaration set."""
    root = _checkout(
        tmp_path,
        workflows={
            "ci-security.yml": "      - run: cargo install cargo-audit --locked --version 0.22.1\n",
            "sbom.yml": "      - run: cargo install cargo-audit --locked --version 0.21.0\n",
        },
    )

    findings = check_toolchain_pin_alignment(root)

    assert _kinds(findings) == ["version_disagreement"]
    assert "cargo-audit" in findings[0].detail


def test_pinned_version_outside_the_project_range_is_refused(tmp_path: Path) -> None:
    """A pin the project metadata forbids is a split, not a newer generation."""
    root = _checkout(
        tmp_path,
        repos=[_hook(RUFF_REPO, "v0.16.4")],
        requirements={"requirements.txt": "ruff==0.16.4\n"},
        pyproject='[project]\nname = "x"\nversion = "0"\ndependencies = ["ruff>=0.4,<0.16"]\n',
    )

    findings = check_toolchain_pin_alignment(root)

    assert _kinds(findings) == ["range_violation"]
    assert "does not satisfy" in findings[0].detail
    assert "ruff>=0.4,<0.16" in findings[0].detail


def test_pinned_version_inside_the_project_range_passes(tmp_path: Path) -> None:
    """A range the pin satisfies must not be reported."""
    root = _checkout(
        tmp_path,
        repos=[_hook(RUFF_REPO, "v0.16.4")],
        requirements={"requirements.txt": "ruff==0.16.4\n"},
        pyproject=(
            '[project]\nname = "x"\nversion = "0"\n'
            'optional-dependencies = {dev = ["ruff>=0.4,<1.0"]}\n'
        ),
    )

    assert check_toolchain_pin_alignment(root) == ()


def test_unparsable_pinned_version_is_refused(tmp_path: Path) -> None:
    """Declarations that agree on a non-version cannot be compared with a range."""
    root = _checkout(
        tmp_path,
        repos=[_hook(RUFF_REPO, "not-a-version")],
        requirements={"requirements.txt": "ruff==not-a-version\n"},
        pyproject='[project]\nname = "x"\nversion = "0"\ndependencies = ["ruff>=0.4"]\n',
    )

    findings = check_toolchain_pin_alignment(root)

    assert _kinds(findings) == ["unparsable_version"]
    assert "not-a-version" in findings[0].detail


def test_unclassified_hook_repository_is_refused(tmp_path: Path) -> None:
    """A newly added hook cannot enter the configuration unclassified."""
    root = _checkout(tmp_path, repos=[_hook("https://github.com/example/new-hook", "v1.0.0")])

    findings = check_toolchain_pin_alignment(root)

    assert _kinds(findings) == ["unclassified_hook_repo"]
    assert "classify it before it can pass" in findings[0].detail


def test_mapped_hook_without_a_requirement_pin_is_refused(tmp_path: Path) -> None:
    """A mapped tool that no requirement file pins cannot be proven to agree."""
    root = _checkout(
        tmp_path,
        repos=[_hook(RUFF_REPO, "v0.16.4")],
        requirements={"requirements.txt": "numpy==2.4.6\n"},
    )

    findings = check_toolchain_pin_alignment(root)

    assert _kinds(findings) == ["hook_without_requirement_pin"]
    assert "ruff" in findings[0].detail


def test_non_python_hook_is_skipped_with_a_recorded_reason(tmp_path: Path) -> None:
    """A classified non-Python hook passes and keeps its written reason."""
    gitleaks = next(iter(NON_PYTHON_HOOK_REPOS))
    root = _checkout(tmp_path, repos=[_hook(gitleaks, "v8.21.2")])

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
    )

    declarations = collect_declarations(root)

    assert [declaration.kind for declaration in declarations if declaration.tool == "ruff"] == [
        "pre_commit_rev",
        "requirement_pin",
    ]
    assert check_toolchain_pin_alignment(root) == ()


def test_requirement_pin_names_are_normalised(tmp_path: Path) -> None:
    """A pin spelled with underscores or capitals is still the same tool."""
    root = _checkout(
        tmp_path,
        repos=[_hook(MYPY_REPO, "v2.3.0")],
        requirements={
            "requirements.txt": (
                "# comment\n    indented==1.0.0\nMyPy==2.3.0 \\\n    --hash=sha256:deadbeef \\\n"
            )
        },
    )

    declarations = {declaration.tool for declaration in collect_declarations(root)}

    assert normalise("MyPy") == "mypy"
    assert "mypy" in declarations
    assert "indented" not in declarations
    assert check_toolchain_pin_alignment(root) == ()


def test_pre_commit_revision_strips_only_a_release_tag_marker(tmp_path: Path) -> None:
    """A ``v`` prefix is a tag marker; a bare revision is used unchanged."""
    prefixed = _checkout(tmp_path / "prefixed", repos=[_hook(RUFF_REPO, "v0.16.4")])
    bare = _checkout(tmp_path / "bare", repos=[_hook(RUFF_REPO, "0.16.4")])

    for root in (prefixed, bare):
        hook = next(
            declaration
            for declaration in collect_declarations(root)
            if declaration.kind == "pre_commit_rev"
        )
        assert hook.version == "0.16.4"


def test_declarations_describe_pins_and_ranges_differently() -> None:
    """A report must show a range as its text, not as an empty version."""
    pin = ToolDeclaration(
        tool="ruff", version="0.16.4", specifier="v0.16.4", source="x", kind="pre_commit_rev"
    )
    span = ToolDeclaration(
        tool="ruff", version="", specifier="ruff>=0.4,<1.0", source="y", kind="pyproject_range"
    )

    assert pin.is_pin is True
    assert pin.describe() == "x(pre_commit_rev)=0.16.4"
    assert span.is_pin is False
    assert span.describe() == "y(pyproject_range)=ruff>=0.4,<1.0"


def test_alignment_finding_requires_an_explanation() -> None:
    """A finding without a reason is not a usable refusal."""
    with pytest.raises(ValueError, match="must explain the finding"):
        AlignmentFinding(kind="version_disagreement", detail="  ")


def test_malformed_configurations_are_refused(tmp_path: Path) -> None:
    """An unreadable configuration must raise, never report agreement."""
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / PRE_COMMIT_CONFIG).write_text("repos: []\n", encoding="utf-8")
    (empty / "requirements.txt").write_text("ruff==0.16.4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="declares no hook repositories"):
        check_toolchain_pin_alignment(empty)

    scalar = tmp_path / "scalar"
    scalar.mkdir()
    (scalar / PRE_COMMIT_CONFIG).write_text("just a string\n", encoding="utf-8")
    (scalar / "requirements.txt").write_text("ruff==0.16.4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="declares no hook repositories"):
        check_toolchain_pin_alignment(scalar)

    revless = tmp_path / "revless"
    revless.mkdir()
    (revless / PRE_COMMIT_CONFIG).write_text(
        yaml.safe_dump({"repos": [{"repo": RUFF_REPO, "hooks": [{"id": "ruff"}]}]}),
        encoding="utf-8",
    )
    (revless / "requirements.txt").write_text("ruff==0.16.4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="declares no rev"):
        check_toolchain_pin_alignment(revless)

    only_local = tmp_path / "only_local"
    only_local.mkdir()
    (only_local / PRE_COMMIT_CONFIG).write_text(
        yaml.safe_dump({"repos": [{"repo": "local", "hooks": [{"id": "check-secrets"}]}]}),
        encoding="utf-8",
    )
    (only_local / "requirements.txt").write_text("ruff==0.16.4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no remote hook repository"):
        check_toolchain_pin_alignment(only_local)


def test_missing_evidence_files_are_refused(tmp_path: Path) -> None:
    """A checkout without the configuration or requirements cannot be checked."""
    with pytest.raises(ValueError, match=f"missing {PRE_COMMIT_CONFIG}"):
        check_toolchain_pin_alignment(tmp_path)

    (tmp_path / PRE_COMMIT_CONFIG).write_text(
        yaml.safe_dump({"repos": [_hook(RUFF_REPO, "v0.16.4")]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="no requirements"):
        check_toolchain_pin_alignment(tmp_path)


def test_unreadable_project_metadata_is_skipped_not_guessed(tmp_path: Path) -> None:
    """A dependency entry that is not a requirement must be ignored, not invented."""
    root = _checkout(
        tmp_path,
        repos=[_hook(RUFF_REPO, "v0.16.4")],
        requirements={"requirements.txt": "ruff==0.16.4\n"},
        pyproject=(
            '[project]\nname = "x"\nversion = "0"\n'
            'dependencies = ["ruff>=0.4,<1.0", "not a requirement!!"]\n'
        ),
    )

    kinds = {declaration.kind for declaration in collect_declarations(root)}

    assert "pyproject_range" in kinds
    assert check_toolchain_pin_alignment(root) == ()


def test_project_metadata_without_a_project_table_is_skipped(tmp_path: Path) -> None:
    """Metadata carrying no project table declares no ranges."""
    root = _checkout(
        tmp_path,
        repos=[_hook(RUFF_REPO, "v0.16.4")],
        requirements={"requirements.txt": "ruff==0.16.4\n"},
        pyproject='[build-system]\nrequires = ["hatchling"]\n',
    )

    assert not [
        declaration
        for declaration in collect_declarations(root)
        if declaration.kind == "pyproject_range"
    ]


def test_workflow_step_boundary_ends_an_action_input_scan(tmp_path: Path) -> None:
    """A ``version`` belonging to a later step must not be read as the action's.

    Both step shapes must end the scan: the next step may start with another
    ``uses:`` or with a plain ``name:``.
    """
    root = _checkout(
        tmp_path,
        workflows={
            "next-uses.yml": (
                "jobs:\n"
                "  build:\n"
                "    steps:\n"
                f"      - uses: {PNPM_ACTION}@abc123\n"
                "      - uses: actions/setup-node@def456\n"
                "        with:\n"
                "          version: 22\n"
            ),
            "next-step.yml": (
                "jobs:\n"
                "  build:\n"
                "    steps:\n"
                f"      - uses: {PNPM_ACTION}@abc123\n"
                "      - name: Install something else\n"
                "        with:\n"
                "          version: 9.9.9\n"
            ),
        },
    )

    assert not [
        declaration
        for declaration in collect_declarations(root)
        if declaration.kind == "workflow_action_input"
    ]


def test_cli_reports_agreement_for_the_repository(capsys: pytest.CaptureFixture[str]) -> None:
    """The committed checkout must exit clean with a counted statement."""
    exit_code = main(["--source-root", str(REPOSITORY_ROOT)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "toolchain declaration alignment: OK" in captured.out
    assert "declarations across" in captured.out


def test_cli_lists_every_declaration(capsys: pytest.CaptureFixture[str]) -> None:
    """The listing mode must show the evidence the verdict rests on."""
    exit_code = main(["--source-root", str(REPOSITORY_ROOT), "--list"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "ruff: .pre-commit-config.yaml(pre_commit_rev)=" in captured.out
    assert "pnpm: CONTRIBUTING.md(documented_command)=" in captured.out


def test_cli_reports_each_finding_and_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A drifting checkout exits refused and names the disagreement."""
    root = _checkout(
        tmp_path,
        repos=[_hook(RUFF_REPO, "v0.15.18")],
        requirements={"requirements.txt": "ruff==0.16.4\n"},
    )

    exit_code = main(["--source-root", str(root)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "version_disagreement" in captured.err
    assert captured.out == ""


def test_cli_fails_closed_on_unreadable_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unusable checkout returns the evidence-error code, never a pass."""
    exit_code = main(["--source-root", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "toolchain declaration evidence unavailable" in captured.err
