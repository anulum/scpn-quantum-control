# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — quantum-sensing quality-gate tests
"""Lock the quantum-sensing owner into preflight and CI."""

from pathlib import Path

from tools import preflight
from tools import quantum_sensing_quality_gates as quality_gates
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and isolated complete NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-quantum-sensing-quality"][5:]
        == quality_gates.QUANTUM_SENSING_TYPING_RATCHET
    )
    docs = gates["ruff D quantum-sensing quality ratchet"]
    assert "D,D413,D417,D420" in docs
    assert "--preview" in docs
    assert "lint.explicit-preview-rules = true" in docs
    assert docs[-len(quality_gates.QUANTUM_SENSING_DOCSTRING_RATCHET) :] == (
        quality_gates.QUANTUM_SENSING_DOCSTRING_RATCHET
    )


def test_coverage_gate_executes_export_and_is_exact() -> None:
    """Require real export execution and exact source branch coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["quantum-sensing focused coverage"]
    report = gates["quantum-sensing exact coverage threshold"]
    assert "--branch" in run
    assert run[-len(quality_gates.QUANTUM_SENSING_COVERAGE_COHORT) :] == (
        quality_gates.QUANTUM_SENSING_COVERAGE_COHORT
    )
    assert "tests/test_export_s11_quantum_sensing_readiness.py" in run
    assert quality_gates.QUANTUM_SENSING_COVERAGE_DATA_FILE.startswith("/tmp/")
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.QUANTUM_SENSING_COVERAGE_INCLUDE}" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper-defined static and coverage commands verbatim in preflight."""
    assert dict(preflight.QUANTUM_SENSING_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(QUANTUM_SENSING_COVERAGE_GATES)" in source


def test_ci_runs_and_aggregates_quantum_sensing_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  quantum-sensing-quality:")
    end = workflow.index("\n\n  quantum-sync-oracle-quality:", start)
    block = workflow[start:end]
    assert all(path in block for path in quality_gates.QUANTUM_SENSING_TYPING_RATCHET)
    assert all(path in block for path in quality_gates.QUANTUM_SENSING_DOCSTRING_RATCHET)
    assert all(path in block for path in quality_gates.QUANTUM_SENSING_COVERAGE_COHORT)
    assert quality_gates.QUANTUM_SENSING_COVERAGE_INCLUDE in block
    assert "quantum-sensing-quality" in workflow[workflow.index("  ci-gate:") :]


def test_native_spectrum_keeps_rustdoc_parity_and_global_rust_ci() -> None:
    """Keep the native kernel documented, parity-tested, and Rust-owned."""
    rust = Path(quality_gates.NV_MAGNETOMETRY_RUSTDOC_SOURCE).read_text(encoding="utf-8")
    parity = Path(quality_gates.NV_MAGNETOMETRY_TEST).read_text(encoding="utf-8")
    workflow = read_ci_workflow_source()
    assert "//! Lorentzian CW-ODMR photoluminescence spectrum." in rust
    assert "/// Normalised ODMR spectrum" in rust
    assert "test_odmr_spectrum_rust_parity" in parity
    assert "Run Rust engine tests" in workflow
    assert "cargo test --locked" in workflow
