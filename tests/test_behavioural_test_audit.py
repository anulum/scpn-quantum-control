# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — behavioural test audit tests
"""Tests for the behavioural-test audit helper."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_AUDIT_TOOL = ROOT / "tools" / "audit_test_behaviour.py"
_SPEC = importlib.util.spec_from_file_location("audit_test_behaviour", _AUDIT_TOOL)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

audit_test_module = _MODULE.audit_test_module
audit_test_tree = _MODULE.audit_test_tree
audits_to_json = _MODULE.audits_to_json
format_audits = _MODULE.format_audits
main = _MODULE.main


def test_behaviour_audit_counts_assertions_raises_and_parametrisation(tmp_path: Path) -> None:
    module = tmp_path / "test_sample.py"
    module.write_text(
        "\n".join(
            [
                "import pytest",
                "",
                "@pytest.mark.parametrize('value', [1, 2])",
                "def test_contract(value):",
                "    assert value > 0",
                "",
                "def test_exception_contract():",
                "    with pytest.raises(ValueError):",
                "        raise ValueError('expected')",
            ]
        ),
        encoding="utf-8",
    )

    audit = audit_test_module(module)

    assert audit.test_count == 2
    assert audit.assertion_count == 1
    assert audit.raises_contract_count == 1
    assert audit.parametrised_count == 1
    assert audit.smoke_only_tests == ()


def test_behaviour_audit_counts_assertion_helper_calls(tmp_path: Path) -> None:
    module = tmp_path / "test_helpers.py"
    module.write_text(
        "\n".join(
            [
                "import numpy as np",
                "",
                "def test_numpy_contract():",
                "    np.testing.assert_allclose([1.0], [1.0])",
                "",
                "def test_local_helper_contract():",
                "    assert_state_matches({'ok': True})",
            ]
        ),
        encoding="utf-8",
    )

    audit = audit_test_module(module)

    assert audit.test_count == 2
    assert audit.assertion_count == 2
    assert audit.smoke_only_tests == ()


def test_behaviour_audit_counts_test_class_methods(tmp_path: Path) -> None:
    module = tmp_path / "test_classes.py"
    module.write_text(
        "\n".join(
            [
                "class TestContracts:",
                "    def test_method_contract(self):",
                "        assert 2 + 2 == 4",
                "",
                "class Helper:",
                "    def test_not_pytest_class(self):",
                "        pass",
            ]
        ),
        encoding="utf-8",
    )

    audit = audit_test_module(module)

    assert audit.test_count == 1
    assert audit.assertion_count == 1
    assert audit.smoke_only_tests == ()


def test_behaviour_audit_flags_smoke_only_tests(tmp_path: Path) -> None:
    module = tmp_path / "test_smoke.py"
    module.write_text(
        "\n".join(
            [
                "def test_import_smoke():",
                "    import math",
                "    math.sqrt(4)",
            ]
        ),
        encoding="utf-8",
    )

    audit = audit_test_module(module)

    assert audit.smoke_only_tests == ("test_import_smoke",)


def test_behaviour_audit_json_is_machine_readable(tmp_path: Path) -> None:
    module = tmp_path / "test_json.py"
    module.write_text("def test_value():\n    assert 1 == 1\n", encoding="utf-8")

    decoded = json.loads(audits_to_json((audit_test_module(module),)))

    assert decoded[0]["test_count"] == 1
    assert decoded[0]["assertion_count"] == 1
    assert decoded[0]["smoke_only_tests"] == []


def test_behaviour_audit_tree_and_summary_cover_multiple_modules(tmp_path: Path) -> None:
    (tmp_path / "test_a.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")
    (tmp_path / "test_b.py").write_text("def test_b():\n    pass\n", encoding="utf-8")
    (tmp_path / "helper.py").write_text("def test_not_collected():\n    pass\n", encoding="utf-8")

    audits = audit_test_tree(tmp_path)
    summary = format_audits(audits)

    assert len(audits) == 2
    assert "modules: 2" in summary
    assert "tests: 2" in summary
    assert "test_b" in summary


def test_behaviour_audit_cli_json_and_smoke_gate(tmp_path: Path, capsys: object) -> None:
    (tmp_path / "test_a.py").write_text("def test_a():\n    pass\n", encoding="utf-8")

    assert main(["--tests-root", str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["smoke_only_tests"] == ["test_a"]
    assert main(["--tests-root", str(tmp_path), "--fail-on-smoke-only"]) == 1
