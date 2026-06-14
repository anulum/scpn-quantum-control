# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the lean phase-module loader
"""Tests for tools/lean_phase_import.py.

The loader exists so pure-NumPy phase tooling can import a single leaf module
without paying the multi-second cost of the full ``scpn_quantum_control`` and
``scpn_quantum_control.phase`` package init surfaces. The in-process tests pin
the public contract; the subprocess tests prove, in a fresh interpreter, that
the heavy init bodies and their optional dependencies are not executed.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOLS_DIR = _REPO_ROOT / "tools"


def _load_tool_module(module_name: str, filename: str) -> ModuleType:
    module_path = _TOOLS_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_lean = _load_tool_module("lean_phase_import_for_tests", "lean_phase_import.py")
load_phase_module = _lean.load_phase_module


def test_load_returns_module_with_benchmark_entry_points() -> None:
    module = load_phase_module("qnode_affinity_benchmark")

    assert callable(module.run_phase_qnode_affinity_benchmark)
    assert callable(module.classify_affinity_evidence)
    assert module.__name__ == "scpn_quantum_control.phase.qnode_affinity_benchmark"


def test_loaded_benchmark_runs_and_classifies_non_isolated() -> None:
    module = load_phase_module("qnode_affinity_benchmark")

    result = module.run_phase_qnode_affinity_benchmark(
        repetitions=3,
        warmups=1,
        reserved_cpus=(),
        host_load_before=(5.0, 5.0, 5.0),
        host_load_after=(5.0, 5.0, 5.0),
    )

    assert result.evidence_label == "functional_non_isolated"
    assert not result.production_benchmark
    assert result.raw_timing_rows


def test_load_is_idempotent() -> None:
    first = load_phase_module("qnode_affinity_benchmark")
    second = load_phase_module("qnode_affinity_benchmark")

    assert first is second


def test_in_package_relative_import_resolves_to_real_source() -> None:
    # The benchmark leaf does ``from .qnode_circuit import ...``; loading it must
    # bind the sibling against the real source tree, not leave it unresolved.
    load_phase_module("qnode_affinity_benchmark")

    circuit = sys.modules["scpn_quantum_control.phase.qnode_circuit"]
    assert circuit.__file__ is not None
    assert circuit.__file__.endswith("phase/qnode_circuit.py")


@pytest.mark.parametrize("bad", ["a.b", "a/b", "..", "", "qnode affinity", "qnode-affinity"])
def test_non_identifier_submodule_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="bare identifier"):
        load_phase_module(bad)


def test_missing_submodule_raises_module_not_found() -> None:
    with pytest.raises(ModuleNotFoundError, match="no phase module"):
        load_phase_module("module_that_does_not_exist")


def test_lean_path_skips_heavy_init_in_fresh_interpreter(tmp_path: Path) -> None:
    # A fresh interpreter proves the behaviour the loader exists for: the leaf
    # loads, the package root is a lean shell whose __init__ never ran, and the
    # optional heavy dependency mitiq is never imported.
    probe = tmp_path / "probe.py"
    probe.write_text(
        textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(_TOOLS_DIR)!r})
            import lean_phase_import as lean

            module = lean.load_phase_module("qnode_affinity_benchmark")
            root = sys.modules["scpn_quantum_control"]
            assert getattr(root, "__lean_shell__", False) is True, "root init ran"
            assert "mitiq" not in sys.modules, "heavy optional dep imported"
            assert "scpn_quantum_control.mitigation" not in sys.modules, "mitigation imported"

            result = module.run_phase_qnode_affinity_benchmark(
                repetitions=2,
                warmups=1,
                reserved_cpus=(),
                host_load_before=(5.0, 5.0, 5.0),
                host_load_after=(5.0, 5.0, 5.0),
            )
            assert result.evidence_label == "functional_non_isolated"
            print("LEAN_OK")
            """
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(probe)],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert "LEAN_OK" in completed.stdout


def test_benchmark_cli_runs_end_to_end(tmp_path: Path) -> None:
    output = tmp_path / "affinity.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(_TOOLS_DIR / "run_phase_qnode_affinity_benchmark.py"),
            "--repetitions",
            "2",
            "--warmups",
            "1",
            "--output",
            str(output),
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["evidence_label"] == "functional_non_isolated"
    assert payload["production_benchmark"] is False
