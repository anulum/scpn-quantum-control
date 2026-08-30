# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — psi-field lattice quality-gate tests
"""Lock Python and Rust lattice gates into preflight and CI."""

from pathlib import Path

from tools import preflight
from tools import psi_lattice_quality_gates as quality_gates


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete connected docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert gates["mypy-strict-psi-lattice-quality"][5:] == (
        quality_gates.PSI_LATTICE_TYPING_RATCHET
    )
    ruff = gates["ruff D psi-lattice quality ratchet"]
    assert (
        ruff[-len(quality_gates.PSI_LATTICE_DOCSTRING_RATCHET) :]
        == quality_gates.PSI_LATTICE_DOCSTRING_RATCHET
    )
    assert "--isolated" in ruff and "D,D413,D417,D420" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require real Python execution and exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["psi-lattice focused coverage"]
    report = gates["psi-lattice exact coverage threshold"]
    assert "--branch" in run
    assert run[-len(quality_gates.PSI_LATTICE_COVERAGE_COHORT) :] == (
        quality_gates.PSI_LATTICE_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert "--include=*/psi_field/lattice.py,*/psi_field/observables.py" in report


def test_polyglot_gate_targets_rust_lattice_owner() -> None:
    """Keep the Rust hot path and FFI contract explicit."""
    gates = dict(quality_gates.build_polyglot_gates("/cargo"))
    command = gates["Rust psi-lattice parity tests"]
    assert command == [
        "/cargo",
        "test",
        "--manifest-path",
        "scpn_quantum_engine/Cargo.toml",
        "--lib",
        "gauge_lattice",
    ]
    assert quality_gates.PSI_LATTICE_POLYGLOT_EVIDENCE == [
        "scpn_quantum_engine/src/gauge_lattice.rs",
        "tests/test_rust_ffi_validation.py",
        "src/scpn_quantum_engine.pyi",
    ]


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.PSI_LATTICE_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    assert dict(preflight.PSI_LATTICE_POLYGLOT_GATES) == dict(
        quality_gates.build_polyglot_gates(preflight._CARGO)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(PSI_LATTICE_COVERAGE_GATES)" in source
    assert "gates.extend(PSI_LATTICE_POLYGLOT_GATES)" in source


def test_ci_runs_and_aggregates_psi_lattice_gate() -> None:
    """Keep the polyglot CI job and transitive aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("  psi-lattice-quality:")
    end = workflow.index("\n\n  differentiable-sparse-derivatives-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.PSI_LATTICE_DOCSTRING_RATCHET:
        assert path in block
    for path in quality_gates.PSI_LATTICE_COVERAGE_COHORT:
        assert path in block
    for path in quality_gates.PSI_LATTICE_SOURCES:
        assert path in block
    assert "--fail-under=100" in block
    assert "cargo test --manifest-path scpn_quantum_engine/Cargo.toml --lib gauge_lattice" in block
    downstream_start = workflow.index("  differentiable-sparse-derivatives-quality:")
    downstream_end = workflow.index("\n\n  tn-mps-baseline-design-quality:", downstream_start)
    downstream = workflow[downstream_start:downstream_end]
    assert "psi-lattice-quality" in downstream
    aggregate = workflow[workflow.index("  ci-gate:") :]
    assert "differentiable-sparse-derivatives-quality" in aggregate
