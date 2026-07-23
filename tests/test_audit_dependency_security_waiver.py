# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — constrained dependency advisory waiver tests
"""Tests for the fail-closed dependency advisory waiver audit."""

from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

from tools import audit_dependency_security_waiver as waiver
from tools import yaml_mapping_key_audit

REPO_ROOT = Path(__file__).resolve().parents[1]


def _distribution_record(
    name: str,
    *,
    version: str = "1.39.5",
    setuptools_requirement: str = "setuptools==81.0.0",
    extra_requirements: tuple[str, ...] = ("numpy",),
) -> waiver.DistributionRecord:
    requirements = (*extra_requirements, setuptools_requirement)
    return waiver.DistributionRecord(name=name, version=version, requirements=requirements)


def _valid_records() -> tuple[waiver.DistributionRecord, ...]:
    return tuple(_distribution_record(name) for name in waiver.BRAKET_DISTRIBUTIONS)


def _lock_text(
    *,
    version: str = waiver.SETUPTOOLS_VERSION,
    hashes: tuple[str, ...] = tuple(sorted(waiver.SETUPTOOLS_HASHES)),
    via: tuple[str, ...] = waiver.BRAKET_DISTRIBUTIONS,
) -> str:
    lines = [f"setuptools=={version} \\"]
    for index, digest in enumerate(hashes):
        suffix = " \\" if index < len(hashes) - 1 else ""
        lines.append(f"    --hash=sha256:{digest}{suffix}")
    lines.extend(("    # via", *(f"    #   {owner}" for owner in via), "wheel==0.46.3"))
    return "\n".join(lines) + "\n"


def _lock_mapping(**overrides: str) -> dict[str, str]:
    return {path: overrides.get(path, _lock_text()) for path in waiver.LOCK_PATHS}


def _workflow(
    *,
    gate_command: str = waiver.WAIVER_GATE_COMMAND,
    pip_audit_command: tuple[str, ...] = waiver.EXPECTED_PIP_AUDIT_COMMAND,
) -> str:
    return (
        "jobs:\n"
        "  security:\n"
        "    steps:\n"
        f"      - run: {gate_command}\n"
        f"      - run: {_shlex_join(pip_audit_command)}\n"
    )


def _shlex_join(parts: tuple[str, ...]) -> str:
    """Join the repository's shell-safe alphanumeric audit arguments."""
    return " ".join(parts)


def _documentation() -> str:
    return "\n".join(
        (
            waiver.WAIVER_DOC_HEADING,
            waiver.ADVISORY_ID,
            f"setuptools=={waiver.SETUPTOOLS_VERSION}",
            waiver.BUILD_BACKEND,
            waiver.DEPENDABOT_CONFIG_PATH,
            "Remove the waiver when both pin owners permit the fixed version.",
        )
    )


def _dependabot_config(*, rule: str = "      - dependency-name: setuptools\n") -> str:
    return (
        "version: 2\n"
        "updates:\n"
        "  - package-ecosystem: pip\n"
        '    directory: "/"\n'
        "    schedule:\n"
        "      interval: weekly\n"
        "    ignore:\n"
        f"{rule}"
    )


def _write_replay_repository(root: Path) -> None:
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "src").mkdir()
    (root / "scripts").mkdir()
    (root / "tools").mkdir()
    (root / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n',
        encoding="utf-8",
    )
    for path, text in _lock_mapping().items():
        (root / path).write_text(text, encoding="utf-8")
    (root / ".github" / "workflows" / "ci.yml").write_text(_workflow(), encoding="utf-8")
    (root / ".github" / "dependabot.yml").write_text(_dependabot_config(), encoding="utf-8")
    (root / "docs" / "test_infrastructure.md").write_text(_documentation(), encoding="utf-8")
    (root / "src" / "runtime.py").write_text("import pathlib\n", encoding="utf-8")


def test_live_repository_waiver_matches_installed_braket_metadata() -> None:
    """The checked-out CI boundary must match installed package metadata."""
    result = waiver.audit_repository(REPO_ROOT)

    assert result.passed
    assert result.errors == ()


def test_recorded_metadata_replay_exercises_the_complete_repository_contract(
    tmp_path: Path,
) -> None:
    """A complete recorded repository replay must pass without network access."""
    _write_replay_repository(tmp_path)

    result = waiver.audit_repository(tmp_path, _valid_records())

    assert result == waiver.WaiverAuditResult(errors=())


def test_main_reports_pass_for_a_complete_repository(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The operator CLI must report the advisory and removal condition."""
    _write_replay_repository(tmp_path)

    status = waiver.main(["--repo-root", str(tmp_path)])

    output = capsys.readouterr().out
    assert status == 0
    assert "dependency security waiver audit: PASS" in output
    assert waiver.ADVISORY_ID in output
    assert "setuptools<83.0.0" in output


def test_script_entrypoint_executes_the_repository_audit(tmp_path: Path) -> None:
    """Direct script execution must use the same audited public entrypoint."""
    script = REPO_ROOT / "tools" / "audit_dependency_security_waiver.py"
    _write_replay_repository(tmp_path)

    original_argv = sys.argv
    try:
        sys.argv = [str(script), "--repo-root", str(tmp_path)]
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = original_argv

    assert exc_info.value.code == 0


def test_semantic_helper_loader_falls_back_only_for_the_missing_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct scripts may fall back without masking a nested import defect."""

    def missing_tools_package(name: str) -> ModuleType:
        if name == "tools.yaml_mapping_key_audit":
            raise ModuleNotFoundError("missing tools package", name="tools")
        if name == "yaml_mapping_key_audit":
            return yaml_mapping_key_audit
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(importlib, "import_module", missing_tools_package)
    assert waiver._load_yaml_mapping_key_audit() is yaml_mapping_key_audit

    def missing_nested_dependency(name: str) -> ModuleType:
        raise ModuleNotFoundError("missing nested dependency", name="unexpected_dependency")

    monkeypatch.setattr(importlib, "import_module", missing_nested_dependency)
    with pytest.raises(ModuleNotFoundError, match="missing nested dependency"):
        waiver._load_yaml_mapping_key_audit()


def test_lock_parser_reads_exact_version_hashes_and_pin_owners() -> None:
    """The parser must preserve every security-relevant pip-compile field."""
    pin = waiver.parse_setuptools_lock_pin(_lock_text())

    assert pin.version == waiver.SETUPTOOLS_VERSION
    assert pin.hashes == waiver.SETUPTOOLS_HASHES
    assert pin.via == frozenset(waiver.BRAKET_DISTRIBUTIONS)


def test_lock_parser_rejects_missing_duplicate_and_malformed_pins() -> None:
    """Missing, duplicated, or non-SHA-256 stanzas must fail closed."""
    with pytest.raises(ValueError, match="found 0"):
        waiver.parse_setuptools_lock_pin("wheel==0.46.3\n")
    with pytest.raises(ValueError, match="found 2"):
        waiver.parse_setuptools_lock_pin(_lock_text() + _lock_text())
    with pytest.raises(ValueError, match="has no version"):
        waiver.parse_setuptools_lock_pin("setuptools==\n")
    malformed = _lock_text(hashes=("z" * 64,))
    with pytest.raises(ValueError, match="invalid sha256"):
        waiver.parse_setuptools_lock_pin(malformed)
    unhashed = waiver.parse_setuptools_lock_pin(
        "setuptools==81.0.0\n    # via\n    #   amazon-braket-schemas\n"
    )
    assert unhashed.hashes == frozenset()
    assert waiver.parse_setuptools_lock_pin("setuptools==81.0.0").via == frozenset()


def test_lock_audit_rejects_missing_version_hash_and_owner_drift() -> None:
    """Every Python CI lock must retain the same source-verified closure."""
    lock_texts = {
        waiver.LOCK_PATHS[0]: _lock_text(version="83.0.0"),
        waiver.LOCK_PATHS[1]: _lock_text(hashes=(next(iter(waiver.SETUPTOOLS_HASHES)),)),
        waiver.LOCK_PATHS[2]: _lock_text(via=(waiver.BRAKET_DISTRIBUTIONS[0],)),
    }
    errors = waiver.audit_lockfiles(lock_texts)

    assert any("pin is 83.0.0" in error for error in errors)
    assert any("distribution hashes drifted" in error for error in errors)
    assert any("pin owners drifted" in error for error in errors)
    assert waiver.audit_lockfiles({}) == tuple(
        f"missing CI lock: {path}" for path in waiver.LOCK_PATHS
    )
    malformed = _lock_mapping(**{waiver.LOCK_PATHS[0]: _lock_text(hashes=("z" * 64,))})
    assert "invalid sha256" in waiver.audit_lockfiles(malformed)[0]


def test_build_system_audit_rejects_setuptools_and_malformed_metadata() -> None:
    """The waiver is invalid if this project starts building with setuptools."""
    valid = '[build-system]\nrequires=["hatchling"]\nbuild-backend="hatchling.build"\n'
    assert waiver.audit_build_system(valid) == ()
    assert waiver.audit_build_system("[project]\nname='x'\n") == (
        "pyproject.toml has no [build-system] table",
    )

    invalid = (
        '[build-system]\nrequires=["setuptools>=81", "not a req @"]\n'
        'build-backend="setuptools.build_meta"\n'
    )
    errors = waiver.audit_build_system(invalid)
    assert f"project build backend must remain {waiver.BUILD_BACKEND}" in errors
    assert "project build requirements must not include setuptools" in errors
    assert any("invalid build requirement" in error for error in errors)
    assert waiver.audit_build_system('[build-system]\nrequires="hatchling"\n')[-1] == (
        "build-system.requires must be a string array"
    )


def test_distribution_audit_rejects_changed_or_invalid_braket_metadata() -> None:
    """Changed upstream metadata must invalidate the exception automatically."""
    assert waiver.audit_distribution_records(_valid_records()) == ()

    records = (
        _distribution_record(
            waiver.BRAKET_DISTRIBUTIONS[0],
            version="not-a-version",
            setuptools_requirement="setuptools>=83.0.0; python_version > '3.11'",
            extra_requirements=("bad requirement @",),
        ),
        waiver.DistributionRecord(
            name="unexpected-package",
            version="1.0.0",
            requirements=("numpy",),
        ),
    )
    errors = waiver.audit_distribution_records(records)

    assert "installed metadata must cover exactly both Braket pin owners" in errors
    assert any("invalid installed version" in error for error in errors)
    assert any("invalid Requires-Dist metadata" in error for error in errors)
    assert any("no longer hard-pins" in error for error in errors)
    assert any("found 0" in error for error in errors)


def test_dependabot_config_audit_requires_one_unconditional_ignore() -> None:
    """Dependabot must not retry an upstream-blocked setuptools update."""
    assert waiver.audit_dependabot_config(_dependabot_config()) == ()

    invalid_configs = (
        _dependabot_config(rule="      - dependency-name: wheel\n"),
        _dependabot_config(rule=""),
        _dependabot_config(
            rule=("      - dependency-name: setuptools\n        versions: ['>=83.0.0']\n")
        ),
        _dependabot_config(
            rule=("      - dependency-name: setuptools\n      - dependency-name: setuptools\n")
        ),
        _dependabot_config().replace("updates:\n", "updates:\nupdates:\n"),
        _dependabot_config().replace("dependency-name", '"dependency-\\u006eame"'),
        "",
        _dependabot_config().replace("package-ecosystem", "ecosystem"),
        _dependabot_config()
        + (
            "  - package-ecosystem: pip\n"
            '    directory: "/"\n'
            "    ignore:\n"
            "      - dependency-name: setuptools\n"
        ),
        _dependabot_config().replace("    ignore:\n      - dependency-name: setuptools\n", ""),
        "not: [valid\n",
    )
    for config in invalid_configs:
        errors = waiver.audit_dependabot_config(config)
        assert len(errors) == 1
        assert errors[0].startswith("Dependabot waiver configuration is invalid:") or (
            errors[0] == "Dependabot configuration contains an escaped mapping key"
        )


def test_dependabot_yaml_node_contracts_fail_closed() -> None:
    """Malformed composed node graphs must not bypass the waiver rule."""

    class Node:
        def __init__(self, kind: object = "scalar", value: object = "value") -> None:
            self.id = kind
            self.value = value

    with pytest.raises(ValueError, match="no string id"):
        waiver._yaml_kind(Node(kind=None))
    with pytest.raises(ValueError, match="expected a scalar"):
        waiver._yaml_scalar(Node(kind="sequence", value=[]))
    with pytest.raises(ValueError, match="no string value"):
        waiver._yaml_scalar(Node(value=1))
    with pytest.raises(ValueError, match="sequence has invalid children"):
        waiver._yaml_sequence(Node(kind="sequence", value=()))
    with pytest.raises(ValueError, match="expected a mapping"):
        waiver._yaml_mapping(Node(kind="scalar"))
    with pytest.raises(ValueError, match="mapping has invalid entries"):
        waiver._yaml_mapping(Node(kind="mapping", value=()))
    with pytest.raises(ValueError, match="mapping has an invalid entry"):
        waiver._yaml_mapping(Node(kind="mapping", value=[Node()]))


def test_installed_distribution_loader_reports_missing_packages() -> None:
    """Absent CI packages must produce an explicit policy error."""
    records, errors = waiver.installed_distribution_records(
        ("definitely-not-a-real-distribution-4184931",)
    )

    assert records == ()
    assert errors == (
        "required installed distribution is missing: definitely-not-a-real-distribution-4184931",
    )


def test_ci_workflow_audit_rejects_broader_or_unchecked_exceptions() -> None:
    """CI may neither omit the precheck nor ignore a second advisory."""
    assert waiver.audit_ci_workflow(_workflow()) == ()

    broader = (
        *waiver.EXPECTED_PIP_AUDIT_COMMAND,
        "--ignore-vuln",
        "PYSEC-2099-0001",
    )
    errors = waiver.audit_ci_workflow(_workflow(gate_command="true", pip_audit_command=broader))

    assert errors == (
        "CI must run the waiver audit exactly once",
        "CI pip-audit command must scan the full lock and ignore only PYSEC-2026-3447",
    )

    block_scalar = (
        "jobs:\n"
        "  security:\n"
        "    steps:\n"
        "      - run: |\n"
        "          python tools/audit_dependency_security_waiver.py\n"
        "      - run: >-\n"
        f"          {_shlex_join(waiver.EXPECTED_PIP_AUDIT_COMMAND)}\n"
    )
    assert waiver.audit_ci_workflow(block_scalar) == ()

    hidden_second_exception = (
        _workflow()
        + "      - run: |\n"
        + "          pip-audit -r requirements-ci-py312-linux.txt --ignore-vuln OTHER\n"
    )
    assert waiver.audit_ci_workflow(hidden_second_exception) == (
        "CI pip-audit command must scan the full lock and ignore only PYSEC-2026-3447",
    )


def test_ci_workflow_audit_rejects_nonblocking_execution_controls() -> None:
    """Conditions, error suppression, and shell overrides must fail closed."""
    audit_condition = _workflow().replace(
        f"      - run: {_shlex_join(waiver.EXPECTED_PIP_AUDIT_COMMAND)}\n",
        f"      - if: false\n        run: {_shlex_join(waiver.EXPECTED_PIP_AUDIT_COMMAND)}\n",
    )
    assert waiver.audit_ci_workflow(audit_condition) == (
        "jobs.security must not define execution control: if",
    )

    audit_nonblocking = _workflow().replace(
        f"      - run: {_shlex_join(waiver.EXPECTED_PIP_AUDIT_COMMAND)}\n",
        f"      - run: {_shlex_join(waiver.EXPECTED_PIP_AUDIT_COMMAND)}\n"
        "        continue-on-error: true\n",
    )
    assert waiver.audit_ci_workflow(audit_nonblocking) == (
        "jobs.security must not define execution control: continue-on-error",
    )

    job_controls = (
        _workflow()
        .replace(
            "jobs:\n",
            "nonblocking: &nonblocking {}\njobs:\n",
        )
        .replace(
            "  security:\n",
            "  security:\n"
            "    if: false\n"
            "    defaults:\n"
            "      run:\n"
            "        shell: bash {0} || true\n"
            "    <<: *nonblocking\n",
        )
    )
    assert waiver.audit_ci_workflow(job_controls) == (
        "jobs.security must not define execution control: if",
        "jobs.security must not define execution control: defaults",
        "jobs.security must not define execution control: shell",
        "jobs.security must not define execution control: <<",
    )

    workflow_defaults = "defaults:\n  run:\n    shell: bash {0} || true\n" + _workflow()
    assert waiver.audit_ci_workflow(workflow_defaults) == (
        "CI must not override run defaults at workflow scope",
    )

    quoted_workflow_defaults = (
        'nonblocking: &nonblocking {}\n"defaults" : *nonblocking\n' + _workflow()
    )
    assert waiver.audit_ci_workflow(quoted_workflow_defaults) == (
        "CI must not override run defaults at workflow scope",
    )

    quoted_condition = _workflow().replace(
        "  security:\n",
        "  security:\n    'if' : false\n",
    )
    assert waiver.audit_ci_workflow(quoted_condition) == (
        "jobs.security must not define execution control: if",
    )


def test_ci_workflow_audit_rejects_escaped_double_quoted_mapping_keys() -> None:
    """YAML escapes must not hide security controls from the raw audit."""
    pip_audit_command = _shlex_join(waiver.EXPECTED_PIP_AUDIT_COMMAND)
    expected = ("CI must not encode mapping keys with YAML escapes",)
    live_workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    pip_audit_step = f"      - run: {pip_audit_command}\n"
    assert live_workflow.count(pip_audit_step) == 1

    escaped_condition = live_workflow.replace(
        pip_audit_step,
        f'      - "\\u0069f": false\n        run: {pip_audit_command}\n',
    )
    escaped_nonblocking = live_workflow.replace(
        pip_audit_step,
        f'      - run: {pip_audit_command}\n        "continue-on-\\u0065rror": true\n',
    )
    escaped_defaults = '"d\\u0065faults":\n  run:\n    shell: bash {0} || true\n' + live_workflow

    assert waiver.audit_ci_workflow(escaped_condition) == expected
    assert waiver.audit_ci_workflow(escaped_nonblocking) == expected
    assert waiver.audit_ci_workflow(escaped_defaults) == expected
    for noncanonical_key in (
        '    ? "\\u0069f"\n    : false\n',
        '    !!str "\\x69f": false\n',
        '    &control "\\U00000069f": false\n',
        '    ? "i\\\n      \\u0066"\n    : false\n',
        '    ? "\\u0069f" # execution control\n    : false\n',
        '    ? # execution control\n      "\\u0069f"\n    : false\n',
        '    ? !!str\n      "\\u0069f"\n    : false\n',
        '    ? &control # execution control\n      "\\u0069f"\n    : false\n',
        '    nested:\n      - - "\\x69f": false\n',
    ):
        escaped_control = _workflow().replace("  security:\n", "  security:\n" + noncanonical_key)
        assert waiver.audit_ci_workflow(escaped_control) == expected

    escaped_value = _workflow().replace(
        f"      - run: {waiver.WAIVER_GATE_COMMAND}\n",
        f'      - name: "waiver \\u0061udit"\n        run: {waiver.WAIVER_GATE_COMMAND}\n',
    )
    assert waiver.audit_ci_workflow(escaped_value) == ()
    escaped_multiline_value = _workflow().replace(
        "    steps:\n",
        '    environment: ["ordinary\\\n      \\u0076alue"]\n    steps:\n',
    )
    assert waiver.audit_ci_workflow(escaped_multiline_value) == ()
    assert not yaml_mapping_key_audit.has_escaped_double_quoted_mapping_key("plain: value")

    flow_style_step = live_workflow.replace(
        pip_audit_step,
        f'      - {{"\\u0069f": false, run: {pip_audit_command}}}\n',
    )
    assert waiver.audit_ci_workflow(flow_style_step) == (
        f"CI pip-audit command must scan the full lock and ignore only {waiver.ADVISORY_ID}",
        *expected,
    )

    malformed = live_workflow.replace("jobs:\n", '"unterminated\\q\njobs:\n')
    assert "CI workflow must be valid YAML for semantic mapping-key audit" in (
        waiver.audit_ci_workflow(malformed)
    )


def test_semantic_mapping_key_audit_handles_aliases_and_empty_documents() -> None:
    """Aliases remain inspectable while empty documents and values stay valid."""
    assert not yaml_mapping_key_audit.has_escaped_double_quoted_mapping_key("")
    assert not yaml_mapping_key_audit.has_escaped_double_quoted_mapping_key(
        'label: "escaped \\u0076alue"\n'
    )
    alias_key = 'label: &control "\\u0069f"\n? *control\n: false\n'
    assert yaml_mapping_key_audit.has_escaped_double_quoted_mapping_key(alias_key)
    nested_mapping_key = '? {"\\u0069f": false}\n: value\n'
    assert yaml_mapping_key_audit.has_escaped_double_quoted_mapping_key(nested_mapping_key)
    recursive_alias = "root: &root [*root]\n"
    assert not yaml_mapping_key_audit.has_escaped_double_quoted_mapping_key(recursive_alias)


def test_semantic_mapping_key_audit_fails_closed_on_invalid_composer_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unavailable YAML and malformed compose graphs must fail closed."""

    class Composer:
        def __init__(self, root: object | None) -> None:
            self.root = root

        def compose(self, _source: str) -> object | None:
            return self.root

    class Mark:
        def __init__(self, index: object) -> None:
            self.index = index

    class Node:
        def __init__(
            self,
            *,
            kind: object = "scalar",
            value: object = "safe",
            style: object = None,
            start: object = 0,
            end: object = 0,
        ) -> None:
            self.id = kind
            self.value = value
            self.style = style
            self.start_mark = Mark(start)
            self.end_mark = Mark(end)

    def install(root: object) -> None:
        monkeypatch.setattr(
            yaml_mapping_key_audit,
            "import_module",
            lambda _name: cast(object, Composer(root)),
        )

    def missing_yaml(_name: str) -> object:
        raise ModuleNotFoundError("missing yaml")

    monkeypatch.setattr(
        yaml_mapping_key_audit,
        "import_module",
        missing_yaml,
    )
    with pytest.raises(ValueError, match="not valid composable YAML"):
        yaml_mapping_key_audit.has_escaped_double_quoted_mapping_key("safe: value\n")

    invalid_roots = (
        Node(kind=None),
        Node(kind="unknown"),
        Node(kind="sequence", value=None),
        Node(kind="mapping", value=None),
        Node(kind="mapping", value=[Node()]),
        Node(kind="mapping", value=[(Node(),)]),
    )
    for root in invalid_roots:
        install(root)
        with pytest.raises(ValueError, match="composed YAML"):
            yaml_mapping_key_audit.has_escaped_double_quoted_mapping_key("safe: value\n")

    for start, end in ((True, 1), (-1, 1), (1, 0), (0, 99)):
        key = Node(style='"', start=start, end=end)
        install(Node(kind="mapping", value=[(key, Node())]))
        with pytest.raises(ValueError, match="composed YAML"):
            yaml_mapping_key_audit.has_escaped_double_quoted_mapping_key('"safe": value\n')


def test_ci_workflow_audit_rejects_ambiguous_or_replaced_steps() -> None:
    """Canonical job placement and distinct run steps prevent YAML replacement."""
    missing_security_job = _workflow().replace("  security:\n", "  lint:\n")
    errors = waiver.audit_ci_workflow(missing_security_job)
    assert errors[-1] == "CI must define exactly one canonical jobs.security mapping"

    duplicate_jobs = _workflow() + "jobs:\n  security:\n    steps: []\n"
    assert waiver.audit_ci_workflow(duplicate_jobs)[-1] == (
        "CI must define exactly one canonical jobs.security mapping"
    )

    duplicate_pip_run_key = _workflow().replace(
        f"      - run: {_shlex_join(waiver.EXPECTED_PIP_AUDIT_COMMAND)}\n",
        f'      - run: {_shlex_join(waiver.EXPECTED_PIP_AUDIT_COMMAND)}\n        "run": true\n',
    )
    assert waiver.audit_ci_workflow(duplicate_pip_run_key) == (
        "pip-audit must own a standalone jobs.security run step",
    )

    duplicate_gate_run_key = _workflow().replace(
        f"      - run: {waiver.WAIVER_GATE_COMMAND}\n",
        f"      - run: {waiver.WAIVER_GATE_COMMAND}\n        run: true\n",
    )
    assert waiver.audit_ci_workflow(duplicate_gate_run_key) == (
        "waiver audit must own a standalone jobs.security run step",
    )

    missing_steps = _workflow().replace("    steps:\n", "    commands:\n")
    assert waiver.audit_ci_workflow(missing_steps) == (
        "jobs.security must define one canonical non-empty steps sequence",
    )

    duplicate_steps = _workflow().replace("    steps:\n", "    steps:\n    steps:\n")
    assert waiver.audit_ci_workflow(duplicate_steps) == (
        "jobs.security must define one canonical non-empty steps sequence",
    )

    empty_steps = _workflow().replace("      - run:", "        run:")
    assert waiver.audit_ci_workflow(empty_steps) == (
        "jobs.security must define one canonical non-empty steps sequence",
    )


def test_setuptools_import_audit_scans_all_project_python_roots(tmp_path: Path) -> None:
    """A project import of the waived build tool must invalidate the boundary."""
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "src" / "safe.py").write_text(
        '"import setuptools"\n# from setuptools import setup\nimport pathlib\n',
        encoding="utf-8",
    )
    (tmp_path / "scripts" / "unsafe.py").write_text(
        "from setuptools.command import build\n", encoding="utf-8"
    )
    (tmp_path / "tools" / "unsafe.py").write_text(
        "import pathlib, setuptools.command\n", encoding="utf-8"
    )
    (tmp_path / "tools" / "invalid.py").write_text("if True print('x')\n", encoding="utf-8")

    errors = waiver.audit_setuptools_imports(tmp_path)

    assert errors == (
        "project code imports setuptools: scripts/unsafe.py",
        "cannot audit Python imports in tools/invalid.py: invalid syntax (<unknown>, line 1)",
        "project code imports setuptools: tools/unsafe.py",
    )
    assert waiver.audit_setuptools_imports(tmp_path / "absent") == ()


def test_operator_documentation_audit_requires_every_stable_marker() -> None:
    """Operators must retain the threat model and deterministic removal rule."""
    assert waiver.audit_operator_documentation(_documentation()) == ()

    errors = waiver.audit_operator_documentation("temporary exception\n")

    assert len(errors) == 6
    assert errors[0].endswith(waiver.WAIVER_DOC_HEADING)
    assert errors[-1].endswith("Remove the waiver")


def test_repository_and_cli_report_missing_surfaces_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An incomplete checkout must return findings through the normal CLI path."""
    result = waiver.audit_repository(tmp_path, _valid_records())

    assert not result.passed
    assert any("cannot read pyproject.toml" in error for error in result.errors)
    assert any("cannot read CI workflow" in error for error in result.errors)
    assert any("cannot read Dependabot configuration" in error for error in result.errors)
    assert any("cannot read operator documentation" in error for error in result.errors)

    status = waiver.main(["--repo-root", str(tmp_path)])
    output = capsys.readouterr().out
    assert status == 1
    assert output.startswith("dependency security waiver audit: FAIL")
    assert "required installed distribution is missing" not in output
