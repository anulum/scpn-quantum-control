# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — DLA protected-memory quality-gate tests
"""Lock the cohesive Python/Rust protected-memory owner into preflight and CI."""

from pathlib import Path

from tools import dla_protected_memory_quality_gates as quality_gates
from tools import preflight


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete isolated NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-dla-protected-memory-quality"][5:]
        == quality_gates.DLA_PROTECTED_MEMORY_TYPING_RATCHET
    )
    docs = gates["ruff D DLA-protected-memory quality ratchet"]
    assert "D,D413,D417,D420" in docs
    assert "lint.explicit-preview-rules = true" in docs
    assert all(path in docs for path in quality_gates.DLA_PROTECTED_MEMORY_DOCSTRING_RATCHET)


def test_coverage_gate_is_connected_and_exact() -> None:
    """Require all real owner suites and exact source-only branch coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["DLA-protected-memory focused coverage"]
    report = gates["DLA-protected-memory exact coverage threshold"]
    assert run[-len(quality_gates.DLA_PROTECTED_MEMORY_TESTS) :] == (
        quality_gates.DLA_PROTECTED_MEMORY_TESTS
    )
    assert "--branch" in run
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.DLA_PROTECTED_MEMORY_COVERAGE_INCLUDE}" in report


def test_polyglot_gate_pins_native_and_ffi_evidence() -> None:
    """Require filtered Rust unit tests and installed-extension validation."""
    gates = dict(quality_gates.build_polyglot_gates("/cargo", "/python"))
    rust = gates["Rust DLA-protected-memory parity tests"]
    ffi = gates["DLA-protected-memory Rust FFI validation"]
    assert quality_gates.DLA_PROTECTED_MEMORY_RUST_SOURCE in (
        quality_gates.DLA_PROTECTED_MEMORY_POLYGLOT_EVIDENCE
    )
    assert rust[-1] == "test_memory_"
    assert quality_gates.DLA_PROTECTED_MEMORY_RUST_FFI_TEST in ffi
    assert ffi[-1] == "dla_protected_memory_metrics"


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep every helper-defined command verbatim in preflight."""
    assert dict(preflight.DLA_PROTECTED_MEMORY_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    assert dict(preflight.DLA_PROTECTED_MEMORY_POLYGLOT_GATES) == dict(
        quality_gates.build_polyglot_gates(preflight._CARGO, preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text()
    assert "gates.extend(DLA_PROTECTED_MEMORY_COVERAGE_GATES)" in source
    assert "gates.extend(DLA_PROTECTED_MEMORY_POLYGLOT_GATES)" in source


def test_ci_runs_and_aggregates_protected_memory_gate() -> None:
    """Keep the Rust-building CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text()
    start = workflow.index("  dla-protected-memory-quality:")
    end = workflow.index("\n\n  differentiable-sparse-derivatives-quality:", start)
    block = workflow[start:end]
    assert all(path in block for path in quality_gates.DLA_PROTECTED_MEMORY_TYPING_RATCHET)
    assert all(path in block for path in quality_gates.DLA_PROTECTED_MEMORY_TESTS)
    assert quality_gates.DLA_PROTECTED_MEMORY_COVERAGE_INCLUDE in block
    assert "test_memory_" in block
    assert "dla_protected_memory_metrics" in block
    assert "dla-protected-memory-quality" in workflow[workflow.index("  ci-gate:") :]
