# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Project Configuration
"""Local CI preflight — mirrors every CI gate so failures are caught before push.

Gates (in order):
  1. ruff check      — lint errors
  2. ruff format     — formatting drift
  3. docs surface    — public docs/docstring surface regression gate
  4. test-quality    — forbid coverage-bucket pytest modules
  5. version-sync    — version string consistency across 5 carrier files
  6. rust-pyi        — Rust PyO3 exports match local typing contract
  7. mypy            — type errors
  8. mypy-strict-dp  — strict typing ratchet for differentiable programming
  9. pytest+coverage — tests + temporary coverage threshold (--cov-fail-under=70)
  10. bandit         — security scan

Usage:
  python tools/preflight.py                # all gates (default)
  python tools/preflight.py --no-tests     # skip pytest entirely (quick lint pass)
  python tools/preflight.py --no-coverage  # run tests without coverage threshold
"""

from __future__ import annotations

import subprocess  # noqa: S404
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_PY = sys.executable

_PYTEST_BASE = [
    _PY,
    "-m",
    "pytest",
    "tests/",
    "-x",
    "--tb=short",
    "-q",
    "--ignore=tests/test_hardware_runner.py",
    "--ignore=tests/test_dynamical_lie_algebra.py",  # DLA: 27 min/test, skip for pre-push
]

_PYTEST_COV = _PYTEST_BASE + [
    "--cov=scpn_quantum_control",
    "--cov-fail-under=70",  # temporary CI mirror while module-specific coverage is rebuilt
]

STATIC_GATES: list[tuple[str, list[str]]] = [
    ("ruff check", [_PY, "-m", "ruff", "check", "src/", "tests/"]),
    ("ruff format", [_PY, "-m", "ruff", "format", "--check", "src/", "tests/"]),
    (
        "documentation-surface",
        [
            _PY,
            "tools/audit_documentation_surface.py",
            "--allowlist",
            "tools/documentation_surface_allowlist.json",
            "--fail-on-findings",
        ],
    ),
    ("test-quality", [_PY, "tools/audit_test_quality.py"]),
    ("version-sync", [_PY, "scripts/check_version_consistency.py"]),
    ("rust-pyi", [_PY, "tools/check_rust_pyi_exports.py"]),
    ("mypy", [_PY, "-m", "mypy"]),
    (
        "mypy-strict-differentiable",
        [
            _PY,
            "-m",
            "mypy",
            "--strict",
            "src/scpn_quantum_control/differentiable.py",
            "src/scpn_quantum_control/differentiable_claim_ledger.py",
            "src/scpn_quantum_control/differentiable_api.py",
            "src/scpn_quantum_control/benchmarks/differentiable_programming.py",
            "src/scpn_quantum_control/differentiable_external_validation.py",
            "src/scpn_quantum_control/differentiable_framework_overlay.py",
            "src/scpn_quantum_control/differentiable_module_hardening_audit.py",
            "src/scpn_quantum_control/benchmarks/differentiable_hardening_gate.py",
            "src/scpn_quantum_control/benchmarks/differentiable_evidence.py",
            "src/scpn_quantum_control/phase/differentiable_readiness.py",
            "src/scpn_quantum_control/phase/differentiable_audit.py",
            "src/scpn_quantum_control/phase/gradient_support_matrix.py",
            "src/scpn_quantum_control/phase/provider_gradient.py",
            "src/scpn_quantum_control/phase/hardware_gradient_policy.py",
            "src/scpn_quantum_control/phase/provider_gradient_audit.py",
            "src/scpn_quantum_control/phase/hardware_gradient_publication.py",
            "src/scpn_quantum_control/phase/provider_hardware_gradient_audit.py",
            "src/scpn_quantum_control/phase/hardware_gradient_campaign.py",
            "src/scpn_quantum_control/phase/gradient_backend.py",
            "src/scpn_quantum_control/phase/gradient_tape.py",
            "src/scpn_quantum_control/phase/natural_gradient.py",
            "src/scpn_quantum_control/phase/gradient_descent.py",
            "src/scpn_quantum_control/phase/qnode_affinity_benchmark.py",
            "src/scpn_quantum_control/phase/qnode_tape.py",
            "src/scpn_quantum_control/phase/qnode_provider_transforms.py",
            "src/scpn_quantum_control/phase/qnode_transforms.py",
            "src/scpn_quantum_control/phase/qnode_vector_transforms.py",
            "src/scpn_quantum_control/phase/qnode_framework_parity.py",
            "src/scpn_quantum_control/phase/qnode_circuit.py",
            "src/scpn_quantum_control/phase/pennylane_bridge.py",
            "src/scpn_quantum_control/phase/jax_bridge.py",
            "src/scpn_quantum_control/phase/torch_bridge.py",
            "src/scpn_quantum_control/phase/tensorflow_bridge.py",
            "src/scpn_quantum_control/phase/qiskit_bridge.py",
            "src/scpn_quantum_control/phase/qnn_framework_bridge_matrix.py",
            "src/scpn_quantum_control/phase/transform_nesting.py",
            "src/scpn_quantum_control/benchmarks/differentiable_external_comparison.py",
            "src/scpn_quantum_control/phase/xy_compiler.py",
            "src/scpn_quantum_control/phase/pennylane_import.py",
            "src/scpn_quantum_control/phase/qnn_optimizer_benchmark.py",
            "src/scpn_quantum_control/phase/qnn_training.py",
            "src/scpn_quantum_control/phase/qnn_conformance.py",
            "src/scpn_quantum_control/phase/qnn_finite_shot.py",
            "src/scpn_quantum_control/phase/qnn_convergence.py",
            "src/scpn_quantum_control/phase/qnn_loss_landscape.py",
            "src/scpn_quantum_control/phase/qgnn.py",
            "src/scpn_quantum_control/phase/qnn_framework_agreement.py",
            "src/scpn_quantum_control/phase/model_training_evidence.py",
            "src/scpn_quantum_control/phase/domain_benchmark_datasets.py",
            "src/scpn_quantum_control/phase/objectives.py",
            "src/scpn_quantum_control/phase/objective_planner.py",
            "src/scpn_quantum_control/phase/objective_audit.py",
            "src/scpn_quantum_control/phase/optimizer_audit.py",
            "src/scpn_quantum_control/phase/param_shift.py",
            "src/scpn_quantum_control/phase/general_unitary.py",
            "src/scpn_quantum_control/phase/phase_vqe.py",
            "src/scpn_quantum_control/phase/structured_ansatz.py",
            "src/scpn_quantum_control/phase/xy_kuramoto.py",
            "src/scpn_quantum_control/phase/kuramoto_variants.py",
            "src/scpn_quantum_control/phase/adapt_vqe.py",
            "src/scpn_quantum_control/phase/trotter_error.py",
            "src/scpn_quantum_control/phase/ansatz_methodology.py",
            "src/scpn_quantum_control/phase/results.py",
            "src/scpn_quantum_control/phase/provider_hardware_safety_audit.py",
        ],
    ),
]

BANDIT_GATE: tuple[str, list[str]] = (
    "bandit",
    [_PY, "-m", "bandit", "-r", "src/", "-ll", "-q"],
)


def run_gate(name: str, cmd: list[str]) -> bool:
    """Run a named preflight command and print a compact result summary."""
    t0 = time.monotonic()
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)  # noqa: S603
    elapsed = time.monotonic() - t0
    if result.returncode == 0:
        print(f"  PASS  {name} ({elapsed:.1f}s)")
        return True
    print(f"  FAIL  {name} ({elapsed:.1f}s)")
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines()[-10:]:
            print(f"        {line}")
    if result.stderr.strip():
        for line in result.stderr.strip().splitlines()[-10:]:
            print(f"        {line}")
    return False


def main() -> int:
    """Run the configured preflight gate suite."""
    skip_tests = "--no-tests" in sys.argv
    no_coverage = "--no-coverage" in sys.argv

    gates: list[tuple[str, list[str]]] = list(STATIC_GATES)

    if not skip_tests:
        if no_coverage:
            gates.append(("pytest", _PYTEST_BASE))
        else:
            gates.append(("pytest + coverage", _PYTEST_COV))

    gates.append(BANDIT_GATE)

    print(f"preflight: {len(gates)} gates")
    print()

    t_start = time.monotonic()
    failed: list[str] = []

    for name, cmd in gates:
        if not run_gate(name, cmd):
            failed.append(name)
            break

    elapsed = time.monotonic() - t_start
    print()
    if failed:
        print(f"BLOCKED: {', '.join(failed)} ({elapsed:.1f}s)")
        return 1
    print(f"ALL CLEAR: ready to push ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
