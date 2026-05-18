# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for behavioural test audit helper
"""Tests for behavioural test audit helper."""

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


_audit_test_behaviour = _load_tool_module(
    "audit_test_behaviour_for_tests",
    "audit_test_behaviour.py",
)
audit_test_module = _audit_test_behaviour.audit_test_module
audit_test_tree = _audit_test_behaviour.audit_test_tree
audits_to_json = _audit_test_behaviour.audits_to_json
evaluate_quality_gate = _audit_test_behaviour.evaluate_quality_gate
format_audits = _audit_test_behaviour.format_audits
main = _audit_test_behaviour.main


def test_behaviour_audit_counts_functions_classes_and_contracts(tmp_path: Path):
    path = tmp_path / "test_contracts.py"
    path.write_text(
        "\n".join(
            [
                "import pytest",
                "",
                "def assert_positive(value):",
                "    assert value > 0",
                "",
                "@pytest.mark.parametrize('value', [1, 2])",
                "def test_assertion_helper_contract(value):",
                "    assert_positive(value)",
                "",
                "class TestParser:",
                "    def test_exception_contract(self):",
                "        with pytest.raises(ValueError):",
                "            raise ValueError('bad')",
                "",
                "def test_smoke_only():",
                "    object()",
            ]
        ),
        encoding="utf-8",
    )

    audit = audit_test_module(path)

    assert audit.test_count == 3
    assert audit.parametrised_count == 1
    assert audit.assertion_count == 1
    assert audit.raises_contract_count == 1
    assert audit.smoke_only_tests == ("test_smoke_only",)


def test_behaviour_tree_json_and_summary_are_deterministic(tmp_path: Path):
    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    (tests_root / "test_one.py").write_text(
        "def test_contract():\n    assert 1 == 1\n",
        encoding="utf-8",
    )
    (tests_root / "helper.py").write_text(
        "def test_ignored():\n    object()\n",
        encoding="utf-8",
    )

    audits = audit_test_tree(tests_root)
    json_text = audits_to_json(audits)
    gate = evaluate_quality_gate(audits, min_assertion_density=1.0)
    summary = format_audits(audits, quality_gate=gate)

    assert len(audits) == 1
    assert '"path"' in json_text
    assert "Behavioural test audit summary:" in summary
    assert "modules_with_smoke_only_tests: 0" in summary
    assert "quality_gate_valid: True" in summary
    assert gate.valid is True


def test_behaviour_quality_gate_blocks_low_contract_density(tmp_path: Path):
    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    (tests_root / "test_mixed.py").write_text(
        "\n".join(
            [
                "import pytest",
                "",
                "def test_assertion():",
                "    assert 1 == 1",
                "",
                "def test_exception():",
                "    with pytest.raises(ValueError):",
                "        raise ValueError('bad')",
            ]
        ),
        encoding="utf-8",
    )

    gate = evaluate_quality_gate(
        audit_test_tree(tests_root),
        min_assertion_density=1.0,
        min_raises_contract_density=1.0,
    )

    assert gate.valid is False
    assert gate.assertion_density == 0.5
    assert gate.raises_contract_density == 0.5
    assert any("assertion density" in blocker for blocker in gate.blockers)
    assert any("raises-contract density" in blocker for blocker in gate.blockers)


def test_behaviour_cli_fail_on_smoke_only_uses_exit_status(tmp_path: Path, capsys):
    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    (tests_root / "test_smoke.py").write_text(
        "def test_smoke_only():\n    object()\n",
        encoding="utf-8",
    )

    exit_code = main(["--tests-root", str(tests_root), "--fail-on-smoke-only"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "test_smoke_only" in output


def test_behaviour_cli_quality_gate_json_and_exit_status(tmp_path: Path, capsys):
    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    (tests_root / "test_contract.py").write_text(
        "def test_contract():\n    assert 1 == 1\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--tests-root",
            str(tests_root),
            "--json",
            "--min-assertion-density",
            "2.0",
            "--fail-on-quality-gate",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 1
    assert '"quality_gate"' in output
    assert '"valid": false' in output
