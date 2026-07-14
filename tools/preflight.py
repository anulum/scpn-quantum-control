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
  4. diff-promo-lang  — differentiable promotion-language evidence gate
  5. diff-baselines  — differentiable competitive-baseline freshness gate
  6. diff-transform  — differentiable transform-algebra metamorphic gate
  7. ruff D ratchet  — NumPy-style docstring ratchet for differentiable hardening
  8. test-quality    — forbid coverage-bucket pytest modules
  9. module-size     — tracked oversized-code responsibility inventory
  10. module-size typing — strict typing for the inventory tool
  11. licence-readiness — canonical cross-language source headers and licence boundaries
  12. licence typing — strict typing for the licence-readiness audit
  13. test typing    — additive strict-mypy cohort for repository tests
  14. test typing tool — strict typing for the cohort-policy audit
  15. coverage policy — preserve line gate and require branch telemetry
  16. coverage policy tool — strict typing for the coverage-policy audit
  17. coverage debt — current 100% recovery register and priority drift
  18. coverage debt tool — strict typing for the debt-register audit
  19. external-validation — environment and evidence-bundle manifest drift
  20. external-validation tool — strict typing for the manifest gate
  21. rustfmt        — canonical formatting across the Rust engine crate
  22. version-sync   — version string consistency across 5 carrier files
  23. rust-pyi       — Rust PyO3 exports match local typing contract
  24. mypy           — type errors
  25. mypy-strict-dp — strict typing ratchet for differentiable programming
  26. pytest+coverage — tests + temporary coverage threshold (--cov-fail-under=70)
  27. bandit         — security scan

Usage:
  python tools/preflight.py                # all gates (default)
  python tools/preflight.py --no-tests     # skip pytest entirely (quick lint pass)
  python tools/preflight.py --no-coverage  # run tests without coverage threshold
"""

from __future__ import annotations

import subprocess  # nosec B404
import sys
import time
from collections.abc import Iterable
from os import X_OK, access, environ, pathsep
from pathlib import Path
from shutil import which

ROOT = Path(__file__).resolve().parent.parent
_PY = sys.executable
_CARGO = which("cargo") or "cargo"
_RUNTIME_SOURCE_ROOTS = (ROOT / "src", ROOT / "oscillatools" / "src")
_HELP_FLAGS = frozenset({"-h", "--help"})

DIFFERENTIABLE_DOCSTRING_RATCHET = [
    "src/scpn_quantum_control/differentiable_architecture_map.py",
    "src/scpn_quantum_control/differentiable_claim_ledger.py",
    "src/scpn_quantum_control/differentiable_claim_rendering.py",
    "src/scpn_quantum_control/differentiable_competitive_baselines.py",
    "src/scpn_quantum_control/differentiable_dependency_environment_map.py",
    "src/scpn_quantum_control/differentiable_baseline_scorecard.py",
    "src/scpn_quantum_control/differentiable_external_validation.py",
    "src/scpn_quantum_control/differentiable_module_hardening_audit.py",
    "src/scpn_quantum_control/differentiable_transform_algebra.py",
    "src/scpn_quantum_control/studio/evidence_bundle.py",
    "src/scpn_quantum_control/phase/tensorflow_maintenance.py",
    "src/scpn_quantum_control/benchmarks/differentiable_isolated_benchmark_plan.py",
    "src/scpn_quantum_control/benchmarks/differentiable_hardening_gate.py",
    "tests/test_differentiable_external_validation.py",
    "tests/test_differentiable_competitive_baselines.py",
    "tests/test_differentiable_module_hardening_audit.py",
    "tests/test_differentiable_transform_algebra.py",
    "tests/test_phase_tensorflow_maintenance.py",
    "tests/test_differentiable_hardening_gate.py",
]

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
    "--cov=src/scpn_quantum_control",
    "--cov-branch",
    "--cov-fail-under=70",  # temporary combined local smoke guard; CI separately gates lines
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
    (
        "differentiable-promotion-language",
        [_PY, "tools/check_differentiable_promotion_language.py"],
    ),
    (
        "differentiable-competitive-baselines",
        [_PY, "tools/check_differentiable_competitive_baselines.py"],
    ),
    (
        "differentiable-transform-algebra",
        [_PY, "tools/check_differentiable_transform_algebra.py"],
    ),
    (
        "ruff D differentiable module-hardening ratchet",
        [
            _PY,
            "-m",
            "ruff",
            "check",
            "--isolated",
            "--select",
            "D,D413",
            "--config",
            'lint.pydocstyle.convention = "numpy"',
            *DIFFERENTIABLE_DOCSTRING_RATCHET,
        ],
    ),
    ("test-quality", [_PY, "tools/audit_test_quality.py"]),
    ("module-size-policy", [_PY, "tools/audit_module_size_policy.py"]),
    (
        "mypy-strict-module-size-policy",
        [_PY, "-m", "mypy", "--strict", "tools/audit_module_size_policy.py"],
    ),
    ("licence-readiness", [_PY, "tools/audit_license_readiness.py"]),
    (
        "mypy-strict-licence-readiness",
        [_PY, "-m", "mypy", "--strict", "tools/audit_license_readiness.py"],
    ),
    ("test-typing-policy", [_PY, "tools/audit_test_typing_policy.py"]),
    (
        "mypy-strict-test-typing-policy",
        [_PY, "-m", "mypy", "--strict", "tools/audit_test_typing_policy.py"],
    ),
    (
        "coverage-policy",
        [_PY, "tools/audit_coverage_policy.py", "--validate-policy"],
    ),
    (
        "mypy-strict-coverage-policy",
        [_PY, "-m", "mypy", "--strict", "tools/audit_coverage_policy.py"],
    ),
    ("coverage-debt", [_PY, "tools/audit_coverage_debt.py"]),
    (
        "mypy-strict-coverage-debt",
        [_PY, "-m", "mypy", "--strict", "tools/audit_coverage_debt.py"],
    ),
    (
        "differentiable-external-validation",
        [_PY, "tools/check_differentiable_external_validation.py"],
    ),
    (
        "mypy-strict-differentiable-external-validation",
        [
            _PY,
            "-m",
            "mypy",
            "--strict",
            "tools/check_differentiable_external_validation.py",
        ],
    ),
    (
        "rustfmt",
        [
            _CARGO,
            "fmt",
            "--manifest-path",
            "scpn_quantum_engine/Cargo.toml",
            "--all",
            "--",
            "--check",
        ],
    ),
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
            "src/scpn_quantum_control/differentiable_architecture_map.py",
            "src/scpn_quantum_control/differentiable_competitive_baselines.py",
            "src/scpn_quantum_control/diff.py",
            "src/scpn/diff.py",
            "src/scpn/__init__.py",
            "src/scpn_quantum_control/differentiable_dependency_environment_map.py",
            "src/scpn_quantum_control/differentiable_baseline_scorecard.py",
            "src/scpn_quantum_control/differentiable_api.py",
            "src/scpn_quantum_control/benchmarks/differentiable_programming.py",
            "src/scpn_quantum_control/differentiable_external_validation.py",
            "src/scpn_quantum_control/differentiable_framework_overlay.py",
            "src/scpn_quantum_control/differentiable_module_hardening_audit.py",
            "src/scpn_quantum_control/differentiable_transform_algebra.py",
            "src/scpn_quantum_control/benchmarks/differentiable_isolated_benchmark_plan.py",
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
            "src/scpn_quantum_control/phase/qnode_circuit_builders.py",
            "src/scpn_quantum_control/phase/qnode_circuit.py",
            "src/scpn_quantum_control/phase/qnode_circuit_contracts.py",
            "src/scpn_quantum_control/phase/qnode_circuit_differentiation.py",
            "src/scpn_quantum_control/phase/qnode_circuit_execution.py",
            "src/scpn_quantum_control/phase/qnode_circuit_support.py",
            "src/scpn_quantum_control/phase/pennylane_bridge.py",
            "src/scpn_quantum_control/phase/pennylane_provider_plugin.py",
            "src/scpn_quantum_control/phase/jax_bridge.py",
            "src/scpn_quantum_control/phase/jax_bridge_contracts.py",
            "src/scpn_quantum_control/phase/jax_compatibility.py",
            "src/scpn_quantum_control/phase/jax_gradients.py",
            "src/scpn_quantum_control/phase/jax_maturity.py",
            "src/scpn_quantum_control/phase/jax_qnode_transforms.py",
            "src/scpn_quantum_control/phase/torch_bridge.py",
            "src/scpn_quantum_control/phase/torch_bridge_contracts.py",
            "src/scpn_quantum_control/phase/torch_compatibility.py",
            "src/scpn_quantum_control/phase/torch_gradients.py",
            "src/scpn_quantum_control/phase/torch_maturity.py",
            "src/scpn_quantum_control/phase/torch_qnode_transforms.py",
            "src/scpn_quantum_control/phase/tensorflow_bridge.py",
            "src/scpn_quantum_control/phase/tensorflow_bridge_contracts.py",
            "src/scpn_quantum_control/phase/tensorflow_compatibility.py",
            "src/scpn_quantum_control/phase/tensorflow_gradients.py",
            "src/scpn_quantum_control/phase/tensorflow_maintenance.py",
            "src/scpn_quantum_control/phase/qiskit_bridge.py",
            "src/scpn_quantum_control/phase/qiskit_bridge_contracts.py",
            "src/scpn_quantum_control/phase/qiskit_gradients.py",
            "src/scpn_quantum_control/phase/qiskit_runtime.py",
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
            "src/scpn_quantum_control/phase/backend_selector.py",
            "src/scpn_quantum_control/phase/ansatz_bench.py",
            "src/scpn_quantum_control/phase/trotter_upde.py",
            "src/scpn_quantum_control/phase/adiabatic_preparation.py",
            "src/scpn_quantum_control/phase/ancilla_lindblad.py",
            "src/scpn_quantum_control/phase/avqds.py",
            "src/scpn_quantum_control/phase/varqite.py",
            "src/scpn_quantum_control/phase/variational_metric.py",
            "src/scpn_quantum_control/phase/coupling_learning.py",
            "src/scpn_quantum_control/phase/contraction_optimiser.py",
            "src/scpn_quantum_control/phase/cross_domain_transfer.py",
            "src/scpn_quantum_control/phase/floquet_kuramoto.py",
        ],
    ),
]

BANDIT_GATE: tuple[str, list[str]] = (
    "bandit",
    [_PY, "-m", "bandit", "-r", "src/", "-ll", "-q"],
)


def _admit_gate_command(cmd: list[str]) -> list[str]:
    """Return a shell-free command with a verified executable path."""
    if not cmd:
        raise ValueError("gate command is empty")
    executable = Path(cmd[0])
    if not executable.is_absolute():
        raise ValueError(f"gate executable is not absolute: {cmd[0]}")
    try:
        exists = executable.exists()
    except (OSError, ValueError) as exc:
        raise ValueError(f"gate executable is not resolvable: {cmd[0]}") from exc
    if not exists:
        raise ValueError(f"gate executable is not resolvable: {cmd[0]}")
    if not executable.is_file():
        raise ValueError(f"gate executable is not a file: {executable}")
    if not access(executable, X_OK):
        raise ValueError(f"gate executable is not executable: {executable}")
    return [str(executable), *cmd[1:]]


def _deduplicated_path_entries(entries: Iterable[str]) -> list[str]:
    """Return path entries in first-seen order without empty duplicates."""
    seen: set[str] = set()
    deduplicated: list[str] = []
    for entry in entries:
        if not entry or entry in seen:
            continue
        seen.add(entry)
        deduplicated.append(entry)
    return deduplicated


def _gate_environment() -> dict[str, str]:
    """Return the subprocess environment for preflight gates.

    Tool scripts execute from ``tools/`` but import the repository packages.
    Prepending the source roots keeps local runtime checks aligned with the
    package layout and the explicit mypy path used for the install-free
    ``oscillatools`` sibling source tree.
    """
    env = dict(environ)
    source_roots = [str(path) for path in _RUNTIME_SOURCE_ROOTS if path.is_dir()]
    existing_pythonpath = env.get("PYTHONPATH", "")
    entries = _deduplicated_path_entries([*source_roots, *existing_pythonpath.split(pathsep)])
    if entries:
        env["PYTHONPATH"] = pathsep.join(entries)
    return env


def run_gate(name: str, cmd: list[str]) -> bool:
    """Run a named preflight command and print a compact result summary."""
    t0 = time.monotonic()
    try:
        admitted_cmd = _admit_gate_command(cmd)
    except ValueError as exc:
        elapsed = time.monotonic() - t0
        print(f"  FAIL  {name} ({elapsed:.1f}s)")
        print(f"        {exc}")
        return False
    result = subprocess.run(  # nosec B603
        admitted_cmd,
        cwd=ROOT,
        capture_output=True,
        env=_gate_environment(),
        text=True,
        shell=False,
    )
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


def _wants_help(args: Iterable[str]) -> bool:
    """Return whether the supplied CLI arguments request usage text."""
    return any(arg in _HELP_FLAGS for arg in args)


def main() -> int:
    """Run the configured preflight gate suite."""
    args = sys.argv[1:]
    if _wants_help(args):
        print((__doc__ or "").strip())
        return 0

    skip_tests = "--no-tests" in args
    no_coverage = "--no-coverage" in args

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


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
