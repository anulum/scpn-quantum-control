# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for release readiness audit helper
"""Tests for release readiness audit helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_tool_module(module_name: str, filename: str) -> ModuleType:
    tools_root = Path(__file__).resolve().parents[1] / "tools"
    module_path = tools_root / filename
    if str(tools_root) not in sys.path:
        sys.path.insert(0, str(tools_root))
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_audit_release_readiness = _load_tool_module(
    "audit_release_readiness_for_tests",
    "audit_release_readiness.py",
)
audit_release_readiness = _audit_release_readiness.audit_release_readiness
check_required_artifacts = _audit_release_readiness.check_required_artifacts
check_versions = _audit_release_readiness.check_versions
format_release_readiness = _audit_release_readiness.format_release_readiness
main = _audit_release_readiness.main


def _write_version_carriers(root: Path, version: str = "0.9.7") -> None:
    (root / "src" / "scpn_quantum_control").mkdir(parents=True)
    (root / "pyproject.toml").write_text(f'version = "{version}"\n', encoding="utf-8")
    (root / "src" / "scpn_quantum_control" / "__init__.py").write_text(
        f'__version__ = "{version}"\n',
        encoding="utf-8",
    )
    (root / "CITATION.cff").write_text(f'version: "{version}"\n', encoding="utf-8")
    (root / ".zenodo.json").write_text(f'{{"version": "{version}"}}\n', encoding="utf-8")


def _write_release_artifacts(root: Path) -> None:
    for rel_path in _audit_release_readiness.REQUIRED_RELEASE_ARTIFACTS:
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"exclusions": []}\n' if path.suffix == ".json" else "{}\n", encoding="utf-8"
        )


def _write_coverage_xml(root: Path) -> Path:
    coverage_xml = root / "coverage.xml"
    coverage_xml.write_text(
        """<?xml version="1.0" ?>
<coverage>
  <packages>
    <package name="scpn_quantum_control">
      <classes>
        <class filename="src/scpn_quantum_control/__init__.py" line-rate="1.0">
          <lines><line number="1" hits="1"/></lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
""",
        encoding="utf-8",
    )
    return coverage_xml


def _write_behavioural_test(root: Path) -> None:
    tests_root = root / "tests"
    tests_root.mkdir()
    (tests_root / "test_contract.py").write_text(
        "\n".join(
            [
                "import pytest",
                "",
                "def test_contract():",
                "    assert 1 == 1",
                "",
                "def test_error_contract():",
                "    with pytest.raises(ValueError):",
                "        raise ValueError('bad')",
            ]
        ),
        encoding="utf-8",
    )


def test_release_readiness_passes_when_all_component_gates_pass(tmp_path: Path):
    _write_version_carriers(tmp_path)
    _write_release_artifacts(tmp_path)
    coverage_xml = _write_coverage_xml(tmp_path)
    _write_behavioural_test(tmp_path)

    payload = audit_release_readiness(
        project_root=tmp_path,
        coverage_xml=coverage_xml,
        min_file_percent=95.0,
        min_aggregate_percent=95.0,
        justified_exclusions=None,
        fail_on_file_gap=False,
        min_assertion_density=0.5,
        min_raises_contract_density=0.5,
    )
    summary = format_release_readiness(payload)

    assert payload["ready_for_tag"] is True
    assert payload["blockers"] == []
    assert "ready_for_tag: True" in summary


def test_release_readiness_blocks_missing_coverage_report(tmp_path: Path):
    _write_version_carriers(tmp_path)
    _write_release_artifacts(tmp_path)
    _write_behavioural_test(tmp_path)

    payload = audit_release_readiness(
        project_root=tmp_path,
        coverage_xml=tmp_path / "missing.xml",
        min_file_percent=95.0,
        min_aggregate_percent=95.0,
        justified_exclusions=None,
        fail_on_file_gap=False,
        min_assertion_density=0.5,
        min_raises_contract_density=0.5,
    )

    assert payload["ready_for_tag"] is False
    assert any("coverage XML report missing" in blocker for blocker in payload["blockers"])


def test_release_readiness_reports_version_and_artifact_blockers(tmp_path: Path):
    _write_version_carriers(tmp_path, version="0.9.7")
    (tmp_path / "src" / "scpn_quantum_control" / "__init__.py").write_text(
        '__version__ = "0.9.8"\n',
        encoding="utf-8",
    )

    version_check = check_versions(tmp_path)
    artifact_check = check_required_artifacts(tmp_path)

    assert version_check.valid is False
    assert any("does not match" in blocker for blocker in version_check.blockers)
    assert artifact_check.valid is False
    assert artifact_check.blockers


def test_release_readiness_reports_file_gaps_without_blocking_aggregate_pass(tmp_path: Path):
    _write_version_carriers(tmp_path)
    _write_release_artifacts(tmp_path)
    (tmp_path / "src" / "scpn_quantum_control" / "low.py").write_text(
        "x = 1\n",
        encoding="utf-8",
    )
    coverage_xml = tmp_path / "coverage.xml"
    coverage_xml.write_text(
        """<?xml version="1.0" ?>
<coverage>
  <packages>
    <package name="scpn_quantum_control">
      <classes>
        <class filename="src/scpn_quantum_control/__init__.py" line-rate="1.0">
          <lines><line number="1" hits="1"/><line number="2" hits="1"/><line number="3" hits="1"/></lines>
        </class>
        <class filename="src/scpn_quantum_control/low.py" line-rate="0.5">
          <lines><line number="1" hits="1"/><line number="2" hits="0"/></lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
""",
        encoding="utf-8",
    )
    _write_behavioural_test(tmp_path)

    payload = audit_release_readiness(
        project_root=tmp_path,
        coverage_xml=coverage_xml,
        min_file_percent=95.0,
        min_aggregate_percent=80.0,
        justified_exclusions=None,
        fail_on_file_gap=False,
        min_assertion_density=0.5,
        min_raises_contract_density=0.5,
    )

    coverage_check = next(
        check for check in payload["checks"] if check["name"] == "coverage_gap_gate"
    )
    assert payload["ready_for_tag"] is True
    assert coverage_check["details"]["below_threshold_files"] == 1


def test_release_readiness_can_block_on_file_gaps(tmp_path: Path):
    _write_version_carriers(tmp_path)
    _write_release_artifacts(tmp_path)
    (tmp_path / "src" / "scpn_quantum_control" / "low.py").write_text(
        "x = 1\n",
        encoding="utf-8",
    )
    coverage_xml = tmp_path / "coverage.xml"
    coverage_xml.write_text(
        """<?xml version="1.0" ?>
<coverage>
  <packages>
    <package name="scpn_quantum_control">
      <classes>
        <class filename="src/scpn_quantum_control/__init__.py" line-rate="1.0">
          <lines><line number="1" hits="1"/><line number="2" hits="1"/><line number="3" hits="1"/></lines>
        </class>
        <class filename="src/scpn_quantum_control/low.py" line-rate="0.5">
          <lines><line number="1" hits="1"/><line number="2" hits="0"/></lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
""",
        encoding="utf-8",
    )
    _write_behavioural_test(tmp_path)

    payload = audit_release_readiness(
        project_root=tmp_path,
        coverage_xml=coverage_xml,
        min_file_percent=95.0,
        min_aggregate_percent=80.0,
        justified_exclusions=None,
        fail_on_file_gap=True,
        min_assertion_density=0.5,
        min_raises_contract_density=0.5,
    )

    assert payload["ready_for_tag"] is False
    assert any("low.py: below_threshold" in blocker for blocker in payload["blockers"])


def test_release_readiness_cli_exit_status(tmp_path: Path, capsys):
    _write_version_carriers(tmp_path)
    _write_release_artifacts(tmp_path)
    _write_behavioural_test(tmp_path)

    exit_code = main(
        [
            "--project-root",
            str(tmp_path),
            "--coverage-xml",
            str(tmp_path / "missing.xml"),
            "--min-assertion-density",
            "0.5",
            "--min-raises-contract-density",
            "0.5",
            "--fail-on-blocker",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "ready_for_tag: False" in output
