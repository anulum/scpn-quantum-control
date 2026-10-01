# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Fresh-process package import budget
"""Measure ordinary package imports without preloaded test-runner state."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import TypedDict, cast

import pytest

ROOT = Path(__file__).resolve().parents[1]


class ImportCase(TypedDict):
    """One real package or leaf and its measured first-party module ceiling."""

    module: str
    maximum_first_party_modules: int
    observed_modules: list[str]


class ImportBudget(TypedDict):
    """Versioned import-count contract with its exact namespace prefix."""

    schema_version: int
    module_prefix: str
    cases: list[ImportCase]


class ImportObservation(TypedDict):
    """Actual loaded modules and evidence of genuine package initialization."""

    first_party: list[str]
    optional: list[str]
    version: str
    root_file: str
    lean_shell: bool


BUDGET = cast(
    ImportBudget,
    json.loads(
        (ROOT / "data/split_preparation/import_cost_budget.json").read_text(encoding="utf-8")
    ),
)


@pytest.mark.parametrize("case", BUDGET["cases"], ids=[c["module"] for c in BUDGET["cases"]])
def test_normal_import_respects_the_measured_module_budget(case: ImportCase) -> None:
    """Execute the real root and leaf imports in an otherwise fresh interpreter.

    Parameters
    ----------
    case
        Ordinary import target and its independently measured module ceiling.

    """
    assert BUDGET["schema_version"] == 1
    assert BUDGET["module_prefix"] == "scpn_quantum_control"
    source = """
import importlib, json, sys
importlib.import_module(sys.argv[1])
prefix = sys.argv[2]
root = sys.modules[prefix]
print(json.dumps({
    'first_party': sorted(n for n in sys.modules if n == prefix or n.startswith(prefix + '.')),
    'optional': sorted(n for n in ('jax', 'torch', 'tensorflow', 'mitiq') if n in sys.modules),
    'version': root.__version__,
    'root_file': root.__file__,
    'lean_shell': getattr(root, '__lean_shell__', False),
}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / "oscillatools/src"), environment.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", source, case["module"], BUDGET["module_prefix"]],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = cast(ImportObservation, json.loads(result.stdout))
    assert case["module"] in observed["first_party"]
    assert 0 < len(observed["first_party"]) <= case["maximum_first_party_modules"]
    assert observed["optional"] == []
    assert observed["version"]
    assert Path(observed["root_file"]).resolve() == ROOT / "src/scpn_quantum_control/__init__.py"
    assert not observed["lean_shell"]


def test_public_version_survives_source_import_without_distribution_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reload the genuine package with source and standard-library paths only.

    Parameters
    ----------
    monkeypatch
        Scoped import search path for the real uninstalled source configuration.

    """
    package = importlib.import_module("scpn_quantum_control")
    original = package.__version__
    try:
        with monkeypatch.context() as patch:
            patch.setattr(sys, "path", [str(ROOT / "src"), sysconfig.get_path("stdlib")])
            assert importlib.reload(package).__version__ == "0.0.0+local"
            assert package.__file__ == str(ROOT / "src/scpn_quantum_control/__init__.py")
    finally:
        importlib.reload(package)
    assert package.__version__ == original
