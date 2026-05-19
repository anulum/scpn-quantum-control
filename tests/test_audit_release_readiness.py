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


def test_required_artifacts_lists_stable_core_release_inputs(tmp_path: Path):
    _write_version_carriers(tmp_path)
    _write_release_artifacts(tmp_path)

    artifact_check = check_required_artifacts(tmp_path)
    required = set(artifact_check.details["required"])

    assert artifact_check.valid is True
    assert "data/stable_core/backend_capability_matrix.json" in required
    assert "docs/stable_core_backend_capability_matrix.md" in required


def test_required_artifacts_blocks_missing_stable_core_release_inputs(tmp_path: Path):
    _write_version_carriers(tmp_path)
    _write_release_artifacts(tmp_path)
    stable_core_paths = (
        "data/stable_core/backend_capability_matrix.json",
        "docs/stable_core_backend_capability_matrix.md",
    )
    for rel_path in stable_core_paths:
        (tmp_path / rel_path).unlink()

    artifact_check = check_required_artifacts(tmp_path)

    assert artifact_check.valid is False
    for rel_path in stable_core_paths:
        assert f"missing required release artefact: {rel_path}" in artifact_check.blockers


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


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_hardware_evidence_packet(root: Path) -> Path:
    import json

    releases = root / "docs" / "internal" / "releases"
    releases.mkdir(parents=True)
    verifier = releases / "verify.json"
    verifier.write_text(
        json.dumps({"pack_count": 1, "packs": [{"id": "pack_a"}]}) + "\n",
        encoding="utf-8",
    )
    export = releases / "export.json"
    export.write_text(
        json.dumps({"exports": [{"id": "pack_a", "sha256": "abc", "bytes": 123}]}) + "\n",
        encoding="utf-8",
    )
    log = releases / "pack_a.log"
    log.write_text("reproduced pack_a\n", encoding="utf-8")
    packet = releases / "packet.json"
    packet.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hardware_evidence_cited": True,
                "verifier_summary_path": "docs/internal/releases/verify.json",
                "export_summary_path": "docs/internal/releases/export.json",
                "reproduction_logs": [
                    {
                        "pack_id": "pack_a",
                        "command": "python scripts/reproduce_pack_a.py",
                        "log_path": "docs/internal/releases/pack_a.log",
                        "sha256": _file_sha256(log),
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return packet


def test_release_readiness_accepts_hardware_evidence_packet(tmp_path: Path):
    _write_version_carriers(tmp_path)
    _write_release_artifacts(tmp_path)
    coverage_xml = _write_coverage_xml(tmp_path)
    _write_behavioural_test(tmp_path)
    packet = _write_hardware_evidence_packet(tmp_path)

    payload = audit_release_readiness(
        project_root=tmp_path,
        coverage_xml=coverage_xml,
        min_file_percent=95.0,
        min_aggregate_percent=95.0,
        justified_exclusions=None,
        fail_on_file_gap=False,
        min_assertion_density=0.5,
        min_raises_contract_density=0.5,
        hardware_result_pack_evidence=packet,
    )

    hardware_check = next(
        check for check in payload["checks"] if check["name"] == "hardware_result_pack_evidence"
    )
    assert payload["ready_for_tag"] is True
    assert hardware_check["details"]["cited_pack_ids"] == ["pack_a"]


def test_release_readiness_blocks_hardware_log_digest_mismatch(tmp_path: Path):
    import json

    _write_version_carriers(tmp_path)
    _write_release_artifacts(tmp_path)
    coverage_xml = _write_coverage_xml(tmp_path)
    _write_behavioural_test(tmp_path)
    packet = _write_hardware_evidence_packet(tmp_path)
    payload = json.loads(packet.read_text(encoding="utf-8"))
    payload["reproduction_logs"][0]["sha256"] = "0" * 64
    packet.write_text(json.dumps(payload), encoding="utf-8")

    result = audit_release_readiness(
        project_root=tmp_path,
        coverage_xml=coverage_xml,
        min_file_percent=95.0,
        min_aggregate_percent=95.0,
        justified_exclusions=None,
        fail_on_file_gap=False,
        min_assertion_density=0.5,
        min_raises_contract_density=0.5,
        hardware_result_pack_evidence=packet,
    )

    assert result["ready_for_tag"] is False
    assert any("reproduction log digest mismatch" in blocker for blocker in result["blockers"])


def test_release_readiness_requires_synchronisation_benchmark_artifacts() -> None:
    """Release artefact gate includes synchronisation benchmark surfaces."""

    required = set(_audit_release_readiness.REQUIRED_RELEASE_ARTIFACTS)

    assert "data/synchronisation_benchmarks/synchronisation_benchmark_registry.json" in required
    assert (
        "data/synchronisation_benchmarks/kuramoto_ring_n4_linear_omega_reference_rows.json"
        in required
    )
    assert (
        "data/synchronisation_benchmarks/kuramoto_chain_n8_decay_omega_reference_rows.json"
        in required
    )
    assert "docs/synchronisation_benchmark_suite.md" in required


def test_release_readiness_requires_knm_measured_candidate_artifacts() -> None:
    """Release artefact gate includes K_nm measured-candidate surfaces."""

    required = set(_audit_release_readiness.REQUIRED_RELEASE_ARTIFACTS)

    assert "data/knm_physical_validation/eeg_alpha_plv_knm_comparison.json" in required
    assert "data/knm_physical_validation/power_grid_ieee5bus_knm_comparison.json" in required
    assert "data/knm_physical_validation/power_grid_ieee14bus_knm_comparison.json" in required
    assert "docs/paper0_knm_measured_coupling_evidence_checklist.md" in required
