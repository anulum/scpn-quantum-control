# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — CI documentation surface gate tests
# SCPN Quantum Control -- CI documentation surface gate contract
"""Static contract for the CI documentation-surface gate."""

from __future__ import annotations

from pathlib import Path

import pytest
import tomllib

from scpn_quantum_control import TraceADArray, TraceADScalar
from scpn_quantum_control.control import realtime_runtime


def test_ci_lint_job_gates_documentation_surface() -> None:
    """CI must fail if repository documentation-surface findings reappear."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "Audit documentation surface" in workflow
    assert "python tools/audit_documentation_surface.py" in workflow
    assert "--allowlist tools/documentation_surface_allowlist.json" in workflow
    assert "--fail-on-findings" in workflow
    assert "Audit differentiable promotion language" in workflow
    assert "python tools/check_differentiable_promotion_language.py" in workflow


def test_ci_lint_job_gates_generated_differentiable_support_matrix() -> None:
    """CI must reject generated-page, manifest, typing, or docstring drift."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "Audit generated differentiable support-matrix page" in workflow
    assert "python tools/differentiable_support_matrix_page.py --check" in workflow
    assert "Type-check differentiable support-matrix page gate" in workflow
    assert "mypy --strict --explicit-package-bases" in workflow
    assert "tools/differentiable_support_matrix_page.py" in workflow
    assert "tests/test_differentiable_support_matrix_page.py" in workflow


def test_ci_lint_job_gates_generated_differentiable_reviewer_evidence() -> None:
    """CI must reject reviewer-page, catalogue, manifest, and typing drift."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "Audit generated differentiable reviewer-evidence page" in workflow
    assert "python tools/differentiable_reviewer_evidence_page.py --check" in workflow
    assert "Type-check differentiable reviewer-evidence page gate" in workflow
    assert "tools/differentiable_reviewer_evidence_catalog.py" in workflow
    assert "tools/differentiable_reviewer_evidence_page.py" in workflow
    assert "tests/test_differentiable_reviewer_evidence_page.py" in workflow


def test_ci_lint_job_gates_realtime_runtime_quality_cohort() -> None:
    """CI must retain strict typing and NumPy docstrings for realtime runtime."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "Type-check realtime runtime quality cohort" in workflow
    assert "python -m mypy --strict --explicit-package-bases" in workflow
    assert "Ruff NumPy docstrings for realtime runtime quality cohort" in workflow
    assert "python -m ruff check --isolated --select D,D413" in workflow
    assert "src/scpn_quantum_control/control/realtime_runtime.py" in workflow
    assert "tests/test_realtime_runtime.py" in workflow
    assert "tests/test_realtime_runtime_branches.py" in workflow


def test_realtime_runtime_autodoc_matches_public_exports() -> None:
    """Autodoc must include every realtime export once and in canonical order."""
    autodoc = Path("docs/autodoc.md").read_text(encoding="utf-8")
    block_start = autodoc.index("::: scpn_quantum_control.control.realtime_runtime")
    block_end = autodoc.index("\n\n", block_start)
    members_line = next(
        line.strip()
        for line in autodoc[block_start:block_end].splitlines()
        if line.strip().startswith("members:")
    )
    members = [
        item.strip()
        for item in members_line.removeprefix("members: [").removesuffix("]").split(",")
    ]

    assert members == realtime_runtime.__all__


def test_ci_gates_whole_program_trace_value_quality_and_exact_coverage() -> None:
    """CI must retain trace-value typing, docs, and focused 100% coverage."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "Type-check whole-program trace-value quality cohort" in workflow
    assert "Ruff NumPy docstrings for whole-program trace-value quality cohort" in workflow
    assert "whole-program-trace-value-quality:" in workflow
    assert "Run whole-program trace-value focused coverage" in workflow
    assert "Enforce whole-program trace-value exact coverage" in workflow
    assert "--data-file=.coverage.whole-program-trace-values" in workflow
    assert "--include=*/whole_program_trace_values.py" in workflow
    assert "--fail-under=100" in workflow
    assert "needs['whole-program-trace-value-quality'].result" in workflow


def test_ci_gates_program_ad_array_indexing_exact_quality() -> None:
    """CI must retain array-indexing typing, docs, and exact branch coverage."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "Type-check Program-AD array-indexing quality cohort" in workflow
    assert "Ruff NumPy docstrings for Program-AD array-indexing quality cohort" in workflow
    assert "Run Program-AD array-indexing focused coverage" in workflow
    assert "Enforce Program-AD array-indexing exact coverage" in workflow
    assert "--data-file=.coverage.program-ad-array-indexing" in workflow
    assert "--include=*/program_ad_array_indexing.py" in workflow
    assert "tests/test_program_ad_array_indexing_quality.py" in workflow
    assert "--fail-under=100" in workflow
    assert "Type-check differentiable scalar-kernel quality cohort" in workflow
    assert "Ruff NumPy docstrings for differentiable scalar-kernel quality cohort" in workflow
    assert "Run differentiable scalar-kernel focused coverage" in workflow
    assert "Enforce differentiable scalar-kernel exact coverage" in workflow
    assert "--data-file=.coverage.scalar-kernels-quality" in workflow
    assert "--include=*/differentiable_scalar_kernels.py" in workflow
    assert "Type-check differentiable Rust/Python inventory quality cohort" in workflow
    assert (
        "Ruff NumPy docstrings for differentiable Rust/Python inventory quality cohort" in workflow
    )
    assert "Run differentiable Rust/Python inventory focused coverage" in workflow
    assert "Enforce differentiable Rust/Python inventory exact coverage" in workflow
    assert "--data-file=.coverage.rust-python-inventory-quality" in workflow
    assert "--include=*/differentiable_rust_python_inventory.py" in workflow
    assert "Type-check unified differentiable API quality cohort" in workflow
    assert "Ruff NumPy docstrings for unified differentiable API quality cohort" in workflow
    assert "Run unified differentiable API focused coverage" in workflow
    assert "Enforce unified differentiable API exact coverage" in workflow
    assert "--data-file=.coverage.differentiable-api-quality" in workflow
    assert "--include=*/differentiable_api.py" in workflow
    assert "Type-check differentiable Levenberg-Marquardt quality cohort" in workflow
    assert (
        "Ruff NumPy docstrings for differentiable Levenberg-Marquardt quality cohort" in workflow
    )
    assert "Run differentiable Levenberg-Marquardt focused coverage" in workflow
    assert "Enforce differentiable Levenberg-Marquardt exact coverage" in workflow
    assert "--data-file=.coverage.levenberg-marquardt-quality" in workflow
    assert "--include=*/differentiable_levenberg_marquardt.py" in workflow
    assert "Type-check differentiable natural-gradient quality cohort" in workflow
    assert "Ruff NumPy docstrings for differentiable natural-gradient quality cohort" in workflow
    assert "Run differentiable natural-gradient focused coverage" in workflow
    assert "Enforce differentiable natural-gradient exact coverage" in workflow
    assert "--data-file=.coverage.natural-gradient-quality" in workflow
    assert "--include=*/differentiable_natural_gradient.py" in workflow


def test_trace_value_autodoc_exposes_both_public_value_types() -> None:
    """Direct trace-value autodoc must retain both operator-intercepted types."""
    autodoc = Path("docs/autodoc.md").read_text(encoding="utf-8")
    block_start = autodoc.index("::: scpn_quantum_control.whole_program_trace_values")
    block_end = autodoc.index("\n\n", block_start)
    members_line = next(
        line.strip()
        for line in autodoc[block_start:block_end].splitlines()
        if line.strip().startswith("members:")
    )
    members = [
        item.strip()
        for item in members_line.removeprefix("members: [").removesuffix("]").split(",")
    ]

    assert members == [TraceADScalar.__name__, TraceADArray.__name__]


def test_ci_lint_job_gates_additive_test_typing_policy() -> None:
    """CI must execute the registered strict-test cohort and type its audit."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "Audit additive strict test-typing cohort" in workflow
    assert "python tools/audit_test_typing_policy.py" in workflow
    assert "Type-check test-typing policy audit" in workflow
    assert "mypy --strict tools/audit_test_typing_policy.py" in workflow


def test_ci_coverage_job_collects_branches_and_preserves_the_line_gate() -> None:
    """CI must measure branches while enforcing lines through the policy audit."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "--cov-branch" in workflow
    assert "--cov-fail-under=0" in workflow
    assert "python tools/audit_coverage_policy.py --coverage-xml coverage.xml" in workflow
    assert "python tools/audit_coverage_policy.py --validate-policy" in workflow
    assert "mypy --strict tools/audit_coverage_policy.py" in workflow
    assert "Upload coverage policy evidence" in workflow
    assert "coverage.xml" in workflow
    assert "coverage-gap-audit.json" in workflow
    assert "Audit coverage-debt register" in workflow
    assert "python tools/audit_coverage_debt.py" in workflow
    assert "mypy --strict tools/audit_coverage_debt.py" in workflow
    assert "--coverage-audit coverage-gap-audit.json" in workflow
    assert "--check-current" in workflow


def test_ci_mlir_leaf_job_enforces_exact_branch_coverage() -> None:
    """CI must gate the complete post-baseline MLIR leaf owner at 100%."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "mlir-leaf-quality:" in workflow
    assert "Run MLIR leaf focused coverage" in workflow
    assert "tests/test_mlir_toolchain_probe_hardening.py" in workflow
    assert "tests/test_phase_qnode_compiler_lowering.py" in workflow
    assert "--data-file=.coverage.mlir-leaf-quality" in workflow
    assert "--source=src/scpn_quantum_control/compiler" in workflow
    assert "Enforce MLIR leaf exact coverage" in workflow
    assert "*/mlir_enzyme_audit.py" in workflow
    assert "*/mlir_phase_qnode_runtime.py" in workflow
    assert "*/mlir_transform_plan_assembly.py" in workflow
    assert "*/mlir_workload_compilation.py" in workflow
    assert "--fail-under=100" in workflow
    assert "needs['mlir-leaf-quality'].result" in workflow


def test_ci_phase_qnode_affinity_job_enforces_exact_quality_and_coverage() -> None:
    """CI must keep affinity evidence typed, documented, and exactly covered."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "Type-check Phase-QNode affinity quality cohort" in workflow
    assert "Ruff NumPy docstrings for Phase-QNode affinity quality cohort" in workflow
    assert "phase-qnode-affinity-quality:" in workflow
    assert "Run Phase-QNode affinity focused coverage" in workflow
    assert "tests/test_phase_qnode_affinity_benchmark.py" in workflow
    assert "tests/test_lean_phase_import.py" in workflow
    assert "--data-file=.coverage.phase-qnode-affinity" in workflow
    assert "Enforce Phase-QNode affinity exact coverage" in workflow
    assert "--include=*/qnode_affinity_benchmark.py" in workflow
    assert "--fail-under=100" in workflow
    assert "needs['phase-qnode-affinity-quality'].result" in workflow


def test_ci_studio_program_ad_job_enforces_exact_polyglot_quality() -> None:
    """CI must bind the Studio replay across Python, Rust, WASM, and TypeScript."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "Type-check Studio Program-AD replay quality cohort" in workflow
    assert "Ruff NumPy docstrings for Studio Program-AD replay quality cohort" in workflow
    assert "studio-program-ad-quality:" in workflow
    assert "Build the current Rust engine wheel for artifact regeneration" in workflow
    assert "Run Studio Program-AD focused coverage" in workflow
    assert "--data-file=.coverage.studio-program-ad" in workflow
    assert "Enforce Studio Program-AD exact coverage" in workflow
    assert "--include=*/program_ad_replay_artifact.py" in workflow
    assert "Enforce Program-AD browser owner exact coverage (ST-12)" in workflow
    assert "--coverage.include=src/panel/programAd.ts" in workflow
    assert "--coverage.include=src/panel/ProgramADReplayCard.tsx" in workflow
    assert workflow.count("--coverage.thresholds.branches=100") >= 1
    assert "needs['studio-program-ad-quality'].result" in workflow


def test_ci_phase_qnode_vector_job_enforces_exact_branch_coverage() -> None:
    """CI must give the vector-transform owner an exact focused branch gate."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "phase-qnode-vector-quality:" in workflow
    assert "Run Phase-QNode vector focused coverage" in workflow
    assert "tests/test_phase_qnode_vector_transforms.py" in workflow
    assert "--data-file=.coverage.phase-qnode-vector" in workflow
    assert "Enforce Phase-QNode vector exact coverage" in workflow
    assert "--include=*/qnode_vector_transforms.py" in workflow
    assert "--fail-under=100" in workflow
    assert "needs['phase-qnode-vector-quality'].result" in workflow


def test_ci_phase_jax_qnode_gate_runs_after_the_real_cpu_overlay() -> None:
    """CI must execute exact JAX QNode coverage with the verified CPU overlay."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "Type-check Phase-QNode JAX quality cohort" in workflow
    assert "Ruff NumPy docstrings for Phase-QNode JAX quality cohort" in workflow
    assert "Run Phase-QNode JAX focused coverage" in workflow
    assert "tests/test_phase_jax_qnode_transforms.py" in workflow
    assert "tests/test_phase_jax_qnode_statevector_edges.py" in workflow
    assert "--data-file=.coverage.phase-jax-qnode" in workflow
    assert "--source=src/scpn_quantum_control/phase" in workflow
    assert "Enforce Phase-QNode JAX exact coverage" in workflow
    assert "--include=*/jax_qnode_transforms.py" in workflow
    assert "--fail-under=100" in workflow
    overlay_position = workflow.index("Build CPU-only differentiable framework overlay")
    runtime_probe_position = workflow.index("Verify real JAX runtime for Phase-QNode coverage")
    coverage_position = workflow.index("Run Phase-QNode JAX focused coverage")
    parity_job_position = workflow.index("  differentiable-parity:")
    next_job_position = workflow.index("\n  security:", parity_job_position)
    assert (
        parity_job_position
        < overlay_position
        < runtime_probe_position
        < coverage_position
        < next_job_position
    )
    assert "import jax; devices=jax.devices()" in workflow
    assert "primary={devices[0]}" in workflow
    assert "SCPN_FRAMEWORK_OVERLAY:$PYTHONPATH" in workflow
    assert "differentiable-parity" in workflow[workflow.index("  ci-gate:") :]


def test_coverage_sources_are_filesystem_paths_not_importable_packages() -> None:
    """Coverage discovery must not import and then unload NumPy/Qiskit state."""
    filesystem_target = "--cov=src/scpn_quantum_control"
    importable_target = "--cov=scpn_quantum_control"
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    test_docs = Path("docs/test_infrastructure.md").read_text(encoding="utf-8")
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    # The reproduction image does not copy the Makefile (make is not used inside
    # the container), so check it only when present; the copied surfaces still
    # enforce the filesystem-path coverage contract in every environment.
    surfaces = [workflow, test_docs]
    makefile_path = Path("Makefile")
    if makefile_path.is_file():
        surfaces.append(makefile_path.read_text(encoding="utf-8"))
    for surface in surfaces:
        assert filesystem_target in surface
        assert importable_target not in surface
    assert pyproject["tool"]["coverage"]["run"]["source"] == ["src/scpn_quantum_control"]


def test_ci_lint_gates_differentiable_external_validation_manifests() -> None:
    """CI lint must reject pinned-input drift before expensive test matrices."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "Audit differentiable external-validation manifests" in workflow
    assert "python tools/check_differentiable_external_validation.py" in workflow
    assert "Type-check differentiable external-validation manifest audit" in workflow
    assert "mypy --strict tools/check_differentiable_external_validation.py" in workflow


def test_docker_reproduction_image_builds_credential_free_git_index() -> None:
    """Docker policy audits need an index without copying host Git metadata."""
    dockerfile_path = Path("Dockerfile")
    dockerignore_path = Path(".dockerignore")
    if not dockerfile_path.is_file() or not dockerignore_path.is_file():
        pytest.skip(
            "host-side meta-test: Dockerfile and .dockerignore are not copied "
            "into the reproduction image, so this contract runs on the host and CI only"
        )

    dockerfile = dockerfile_path.read_text(encoding="utf-8")
    dockerignore = dockerignore_path.read_text(encoding="utf-8")

    assert "RUN git init -q" in dockerfile
    assert "&& git add -A" in dockerfile
    assert "&& chown sqc:sqc /app /app/.git" in dockerfile
    assert "COPY .gitignore .gitignore" in dockerfile
    assert "credential-free synthetic Git index" in dockerfile
    assert "COPY scpn_quantum_engine/Cargo.lock scpn_quantum_engine/Cargo.lock" in dockerfile
    assert "COPY .github/dependabot.yml .github/dependabot.yml" in dockerfile
    assert dockerignore.splitlines()[0] == ".git"
    assert "!.github/dependabot.yml" in dockerignore.splitlines()
    assert "**/target/" in dockerignore.splitlines()
    assert "**/__pycache__/" in dockerignore.splitlines()
    assert "BACKUP/" in dockerignore.splitlines()
    assert "scripts/**/results/" in dockerignore.splitlines()


def test_ci_gates_differentiable_strict_mypy_ratchet() -> None:
    """CI must enforce strict mypy on promoted differentiable modules."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "mypy --strict" in workflow
    assert "src/scpn_quantum_control/differentiable.py" in workflow
    assert "src/scpn_quantum_control/differentiable_claim_ledger.py" in workflow
    assert "src/scpn_quantum_control/differentiable_architecture_map.py" in workflow
    assert "src/scpn_quantum_control/differentiable_dependency_environment_map.py" in workflow
    assert "src/scpn_quantum_control/differentiable_baseline_scorecard.py" in workflow
    assert "src/scpn_quantum_control/differentiable_api.py" in workflow
    assert "src/scpn_quantum_control/benchmarks/differentiable_programming.py" in workflow
    assert "src/scpn_quantum_control/differentiable_external_validation.py" in workflow
    assert "src/scpn_quantum_control/differentiable_framework_overlay.py" in workflow
    assert "src/scpn_quantum_control/differentiable_module_hardening_audit.py" in workflow
    assert "src/scpn_quantum_control/benchmarks/differentiable_isolated_benchmark_plan.py" in (
        workflow
    )
    assert "src/scpn_quantum_control/benchmarks/differentiable_hardening_gate.py" in workflow
    assert "src/scpn_quantum_control/benchmarks/differentiable_evidence.py" in workflow
    assert "src/scpn_quantum_control/phase/differentiable_readiness.py" in workflow
    assert "src/scpn_quantum_control/phase/differentiable_audit.py" in workflow
    assert "src/scpn_quantum_control/phase/gradient_support_matrix.py" in workflow
    assert "src/scpn_quantum_control/phase/provider_gradient.py" in workflow
    assert "src/scpn_quantum_control/phase/hardware_gradient_policy.py" in workflow
    assert "src/scpn_quantum_control/phase/provider_gradient_audit.py" in workflow
    assert "src/scpn_quantum_control/phase/hardware_gradient_publication.py" in workflow
    assert "src/scpn_quantum_control/phase/provider_hardware_gradient_audit.py" in workflow
    assert "src/scpn_quantum_control/phase/hardware_gradient_campaign.py" in workflow
    assert "src/scpn_quantum_control/phase/gradient_backend.py" in workflow
    assert "src/scpn_quantum_control/phase/gradient_tape.py" in workflow
    assert "src/scpn_quantum_control/phase/natural_gradient.py" in workflow
    assert "src/scpn_quantum_control/phase/gradient_descent.py" in workflow
    assert "src/scpn_quantum_control/phase/qnode_affinity_benchmark.py" in workflow
    assert "src/scpn_quantum_control/phase/qnode_tape.py" in workflow
    assert "src/scpn_quantum_control/phase/qnode_provider_transforms.py" in workflow
    assert "src/scpn_quantum_control/phase/qnode_transforms.py" in workflow
    assert "src/scpn_quantum_control/phase/qnode_vector_transforms.py" in workflow
    assert "src/scpn_quantum_control/phase/qnode_framework_parity.py" in workflow
    assert "src/scpn_quantum_control/phase/qnode_circuit.py" in workflow
    assert "src/scpn_quantum_control/phase/pennylane_bridge.py" in workflow
    assert "src/scpn_quantum_control/phase/jax_bridge.py" in workflow
    assert "src/scpn_quantum_control/phase/torch_bridge.py" in workflow
    assert "src/scpn_quantum_control/phase/tensorflow_bridge.py" in workflow
    assert "src/scpn_quantum_control/phase/tensorflow_maintenance.py" in workflow
    assert "src/scpn_quantum_control/phase/qiskit_bridge.py" in workflow
    assert "src/scpn_quantum_control/phase/qnn_framework_bridge_matrix.py" in workflow
    assert "src/scpn_quantum_control/phase/transform_nesting.py" in workflow
    assert "src/scpn_quantum_control/benchmarks/differentiable_external_comparison.py" in workflow
    assert "src/scpn_quantum_control/phase/xy_compiler.py" in workflow


def test_ci_gates_differentiable_docstring_ratchet() -> None:
    """CI must enforce Ruff D on docstring-clean differentiable modules."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "Ruff docstring ratchet for differentiable module hardening" in workflow
    assert "ruff check --isolated --select D,D413" in workflow
    assert "--config 'lint.pydocstyle.convention = \"numpy\"'" in workflow
    assert "src/scpn_quantum_control/differentiable_claim_ledger.py" in workflow
    assert "src/scpn_quantum_control/differentiable_claim_rendering.py" in workflow
    assert "src/scpn_quantum_control/differentiable_competitive_baselines.py" in workflow
    assert "src/scpn_quantum_control/differentiable_external_validation.py" in workflow
    assert "src/scpn_quantum_control/differentiable_module_hardening_audit.py" in workflow
    assert "src/scpn_quantum_control/differentiable_transform_algebra.py" in workflow
    assert "src/scpn_quantum_control/studio/evidence_bundle.py" in workflow
    assert "src/scpn_quantum_control/phase/tensorflow_maintenance.py" in workflow
    assert "src/scpn_quantum_control/benchmarks/differentiable_isolated_benchmark_plan.py" in (
        workflow
    )
    assert "src/scpn_quantum_control/benchmarks/differentiable_hardening_gate.py" in (workflow)
    assert "tests/test_differentiable_external_validation.py" in workflow
    assert "tests/test_differentiable_competitive_baselines.py" in workflow
    assert "tests/test_differentiable_module_hardening_audit.py" in workflow
    assert "tests/test_differentiable_transform_algebra.py" in workflow
    assert "tests/test_phase_tensorflow_maintenance.py" in workflow
    assert "tests/test_differentiable_hardening_gate.py" in workflow
    assert "tools/differentiable_support_matrix_page.py" in workflow
    assert "tests/test_differentiable_support_matrix_page.py" in workflow
    assert "tools/differentiable_reviewer_evidence_catalog.py" in workflow
    assert "tools/differentiable_reviewer_evidence_page.py" in workflow
    assert "tests/test_differentiable_reviewer_evidence_page.py" in workflow


def test_rust_audit_installer_retries_transient_crates_io_transport_errors() -> None:
    """The Rust advisory gate must tolerate transient crates.io transport errors."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "Install cargo-audit" in workflow
    assert 'CARGO_HTTP_MULTIPLEXING: "false"' in workflow
    assert "for attempt in 1 2 3" in workflow
    assert "cargo install cargo-audit --locked --version 0.22.1" in workflow
    assert "Check Rust formatting" in workflow
    assert "cargo fmt --all -- --check" in workflow
    assert "cargo audit --deny warnings" in workflow
