# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for coverage-gap audit helper
"""Tests for coverage-gap audit helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_tool_module(module_name: str, filename: str) -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "tools" / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_audit_coverage_gaps = _load_tool_module(
    "audit_coverage_gaps_for_tests",
    "audit_coverage_gaps.py",
)
audit_coverage_gaps = _audit_coverage_gaps.audit_coverage_gaps
audits_to_json = _audit_coverage_gaps.audits_to_json
format_audits = _audit_coverage_gaps.format_audits
load_justified_exclusions = _audit_coverage_gaps.load_justified_exclusions


def test_coverage_gap_audit_classifies_ok_low_and_missing_files(tmp_path: Path):
    project_root = tmp_path / "repo"
    source_root = project_root / "src" / "scpn_quantum_control"
    source_root.mkdir(parents=True)
    (source_root / "covered.py").write_text("x = 1\n", encoding="utf-8")
    (source_root / "low.py").write_text("x = 1\n", encoding="utf-8")
    (source_root / "missing.py").write_text("x = 1\n", encoding="utf-8")
    coverage_xml = project_root / "coverage.xml"
    coverage_xml.write_text(
        """<?xml version="1.0" ?>
<coverage>
  <packages>
    <package name="scpn_quantum_control">
      <classes>
        <class filename="src/scpn_quantum_control/covered.py" line-rate="1.0">
          <lines><line number="1" hits="1"/></lines>
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

    audits = audit_coverage_gaps(
        project_root=project_root,
        source_root=source_root,
        coverage_xml=coverage_xml,
        min_file_percent=95.0,
        justified_exclusions=None,
    )
    by_path = {Path(item.path).name: item for item in audits}

    assert by_path["covered.py"].status == "ok"
    assert by_path["covered.py"].missing_lines == 0
    assert by_path["low.py"].status == "below_threshold"
    assert by_path["low.py"].missing_lines == 1
    assert by_path["missing.py"].status == "missing_from_report"


def test_coverage_gap_outputs_are_deterministic(tmp_path: Path):
    project_root = tmp_path / "repo"
    source_root = project_root / "src" / "scpn_quantum_control"
    source_root.mkdir(parents=True)
    (source_root / "module.py").write_text("x = 1\n", encoding="utf-8")
    coverage_xml = project_root / "coverage.xml"
    coverage_xml.write_text(
        """<?xml version="1.0" ?>
<coverage>
  <packages>
    <package name="scpn_quantum_control">
      <classes>
        <class filename="src/scpn_quantum_control/module.py" line-rate="1.0">
          <lines><line number="1" hits="1"/></lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
""",
        encoding="utf-8",
    )

    audits = audit_coverage_gaps(
        project_root=project_root,
        source_root=source_root,
        coverage_xml=coverage_xml,
        min_file_percent=95.0,
        justified_exclusions=None,
    )

    assert '"status": "ok"' in audits_to_json(audits)
    assert "Coverage gap audit summary:" in format_audits(audits)


def test_coverage_gap_summary_warns_when_report_matches_no_source_files(tmp_path: Path):
    project_root = tmp_path / "repo"
    source_root = project_root / "src" / "scpn_quantum_control"
    source_root.mkdir(parents=True)
    (source_root / "module.py").write_text("x = 1\n", encoding="utf-8")

    audits = audit_coverage_gaps(
        project_root=project_root,
        source_root=source_root,
        coverage_xml=project_root / "missing-coverage.xml",
        min_file_percent=95.0,
        justified_exclusions=None,
    )

    summary = format_audits(audits)

    assert "coverage_report_warning" in summary
    assert "regenerate coverage.xml" in summary
    assert "missing_from_report: src/scpn_quantum_control/module.py" in summary


def test_coverage_gap_audit_accepts_justified_file_exclusions(tmp_path: Path):
    project_root = tmp_path / "repo"
    source_root = project_root / "src" / "scpn_quantum_control"
    source_root.mkdir(parents=True)
    (source_root / "low.py").write_text("x = 1\n", encoding="utf-8")
    (source_root / "missing.py").write_text("x = 1\n", encoding="utf-8")
    coverage_xml = project_root / "coverage.xml"
    coverage_xml.write_text(
        """<?xml version="1.0" ?>
<coverage>
  <packages>
    <package name="scpn_quantum_control">
      <classes>
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
    exclusions = {
        "src/scpn_quantum_control/low.py": "covered by hardware-only integration gate",
        "src/scpn_quantum_control/missing.py": "generated optional backend shim",
    }

    audits = audit_coverage_gaps(
        project_root=project_root,
        source_root=source_root,
        coverage_xml=coverage_xml,
        min_file_percent=95.0,
        justified_exclusions=exclusions,
    )
    by_path = {Path(item.path).name: item for item in audits}
    summary = format_audits(audits)

    assert by_path["low.py"].status == "justified_exclusion"
    assert by_path["missing.py"].status == "justified_exclusion"
    assert by_path["low.py"].is_gap is False
    assert "files_justified_exclusions: 2" in summary


def test_coverage_gap_audit_accepts_justified_glob_exclusions(tmp_path: Path):
    project_root = tmp_path / "repo"
    source_root = project_root / "src" / "scpn_quantum_control" / "paper0"
    source_root.mkdir(parents=True)
    (source_root / "generated_validation.py").write_text("x = 1\n", encoding="utf-8")

    audits = audit_coverage_gaps(
        project_root=project_root,
        source_root=project_root / "src" / "scpn_quantum_control",
        coverage_xml=project_root / "missing-coverage.xml",
        min_file_percent=95.0,
        justified_exclusions={
            "src/scpn_quantum_control/paper0/*_validation.py": (
                "generated source-validation module covered by register reconciliation"
            )
        },
    )

    assert audits[0].status == "justified_exclusion"
    assert audits[0].is_gap is False


def test_load_justified_exclusions_requires_reason(tmp_path: Path):
    path = tmp_path / "exclusions.json"
    path.write_text('{"exclusions": [{"path": "src/pkg.py", "reason": "  "}]}', encoding="utf-8")

    try:
        load_justified_exclusions(path)
    except ValueError as exc:
        assert "reason" in str(exc)
    else:
        raise AssertionError("Expected empty exclusion reason to fail")


def test_load_justified_exclusions_accepts_path_glob(tmp_path: Path):
    path = tmp_path / "exclusions.json"
    path.write_text(
        '{"exclusions": [{"path_glob": "src/pkg/generated_*.py", "reason": "generated"}]}',
        encoding="utf-8",
    )

    assert load_justified_exclusions(path) == {"src/pkg/generated_*.py": "generated"}
