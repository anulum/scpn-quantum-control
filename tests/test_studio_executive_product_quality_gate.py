# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Studio-executive quality-gate tests
"""Lock the Studio-executive product gate into preflight and CI."""

from tools import preflight
from tools import studio_executive_product_quality_gates as quality_gates
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and isolated NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-studio-executive-product-quality"][5:]
        == quality_gates.STUDIO_EXECUTIVE_PRODUCT_QUALITY_RATCHET
    )
    ruff = gates["ruff D studio-executive-product quality ratchet"]
    assert "--isolated" in ruff and "--preview" in ruff
    assert "D,D413,D417,D420" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact joint source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["studio-executive-product focused coverage"]
    report = gates["studio-executive-product exact coverage threshold"]
    assert "--branch" in run
    assert run[-len(quality_gates.STUDIO_EXECUTIVE_PRODUCT_COVERAGE_COHORT) :] == (
        quality_gates.STUDIO_EXECUTIVE_PRODUCT_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert (
        "--include=*/studio_executive_product.py,*/studio/manifest.py,*/studio/federation.py,*/studio/verbs.py,*/studio/executive_cli.py"
        in report
    )


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.STUDIO_EXECUTIVE_PRODUCT_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command


def test_ci_runs_and_aggregates_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  studio-executive-product-quality:")
    end = workflow.index("\n\n  decisive-advantage-quality:", start)
    block = workflow[start:end]
    assert all(path in block for path in quality_gates.STUDIO_EXECUTIVE_PRODUCT_QUALITY_RATCHET)
    assert all(path in block for path in quality_gates.STUDIO_EXECUTIVE_PRODUCT_COVERAGE_COHORT)
    assert "requirements-ci-studio-platform.txt" in block
    assert "--no-deps --require-hashes" in block
    assert (
        "--include=*/studio_executive_product.py,*/studio/manifest.py,*/studio/federation.py,*/studio/verbs.py,*/studio/executive_cli.py"
        in block
    )
    assert "studio-executive-product-quality" in workflow[workflow.index("  ci-gate:") :]


def test_compiler_trace_is_qualified_by_the_existing_studio_cohort() -> None:
    """Require native and browser trace ownership in the original coherent category."""
    from pathlib import Path

    gates = dict(quality_gates.build_program_authoring_quality_gates("/python"))
    strict = gates["studio-program-authoring-strict"]
    docs = gates["studio-program-authoring-native-docs"]
    run = gates["studio-program-authoring-native-coverage"]
    exact = gates["studio-program-authoring-native-exact"]
    for owner in [
        "src/scpn_quantum_control/studio/compiler_trace.py",
        "tests/test_studio_compiler_trace.py",
        "tools/studio_compiler_trace_browser.py",
        "tools/tests/test_studio_compiler_trace_browser.py",
    ]:
        assert owner in strict and owner in docs
    assert "tests/test_studio_compiler_trace.py" in run
    assert any("*/studio/compiler_trace.py" in field for field in exact)
    assert "--fail-under=100" in exact
    repo = Path(__file__).resolve().parents[1]
    workflow = (repo / ".github/workflows/ci-studio.yml").read_text()
    assert workflow.count("--scenario compiler_trace_inspector") == 1
    assert "tools/tests/test_studio_compiler_trace_browser.py" in workflow
    assert "--coverage.include='src/features/compiler/*.{ts,tsx}'" in workflow
    assert "build_program_authoring_quality_gates" in workflow


def test_backend_profiles_have_native_and_actual_browser_gates() -> None:
    """Require additive profile ownership in the existing Studio CI job."""
    from pathlib import Path

    gates = dict(quality_gates.build_backend_profiles_quality_gates("/python"))
    for owner in (
        "src/scpn_quantum_control/hardware/backend_profiles.py",
        "tools/export_backend_profiles.py",
        "tests/test_backend_profiles.py",
        "tools/studio_backend_profiles_browser.py",
        "tools/tests/test_studio_backend_profiles_browser.py",
    ):
        assert owner in gates["studio-backend-profiles-strict"]
        assert owner in gates["studio-backend-profiles-native-docs"]
    assert (
        "tests/test_provider_route_catalogue.py"
        in gates["studio-backend-profiles-native-coverage"]
    )
    assert "--branch" in gates["studio-backend-profiles-native-coverage"]
    assert "--fail-under=100" in gates["studio-backend-profiles-native-exact"]
    workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/ci-studio.yml"
    ).read_text()
    assert workflow.count("--scenario operator_backend_profiles") == 1
    assert "tools/tests/test_studio_backend_profiles_browser.py" in workflow
    assert "--coverage.include='src/features/operators/profiles/*.{ts,tsx}'" in workflow
    assert "build_backend_profiles_quality_gates" in workflow


def test_operator_policy_is_in_original_native_and_real_browser_cohorts() -> None:
    """Retain whole policy ownership, exact thresholds and one original public dispatcher."""
    from pathlib import Path

    gates = dict(quality_gates.build_operator_policy_quality_gates("/python"))
    for owner in (
        "src/scpn_quantum_control/hardware/operator_policy_contracts.py",
        "src/scpn_quantum_control/hardware/operator_policy.py",
        "src/scpn_quantum_control/studio_workspace/operator_policy.py",
        "tools/export_operator_policy_decisions.py",
        "tests/test_workspace_operator_policy.py",
        "tools/studio_operator_policy_browser.py",
        "tools/tests/test_studio_operator_policy_browser.py",
    ):
        assert owner in gates["studio-operator-policy-strict"]
        assert owner in gates["studio-operator-policy-native-docs"]
    assert "--branch" in gates["studio-operator-policy-native-coverage"]
    assert "--fail-under=100" in gates["studio-operator-policy-native-exact"]
    workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/ci-studio.yml"
    ).read_text()
    assert workflow.count("--scenario operator_policy_decisions") == 1
    assert "tools/tests/test_studio_operator_policy_browser.py" in workflow
    assert "--coverage.include='src/features/operators/policy/*.{ts,tsx}'" in workflow
    assert "build_operator_policy_quality_gates" in workflow
    assert workflow.count("tests/test_workspace_operator_policy.py") == 3


def test_operator_dossier_preserves_original_exact_native_and_browser_gates() -> None:
    """Review source stays in the existing native and genuine runtime coverage owners."""
    from pathlib import Path

    gates = dict(quality_gates.build_operator_dossier_quality_gates("/python"))
    owners = gates["studio-operator-dossier-strict"]
    for path in (
        "src/scpn_quantum_control/studio/executive_execute.py",
        "src/scpn_quantum_control/studio/operator_review_dossier.py",
        "src/scpn_quantum_control/studio/operator_review_script.py",
        "tools/export_operator_review_dossiers.py",
        "tests/test_operator_review_dossier.py",
        "tools/studio_operator_dossier_browser.py",
        "tools/tests/test_studio_operator_dossier_browser.py",
    ):
        assert path in owners
    for name, command in gates.items():
        assert command[0] == "/python"
        if name.endswith("exact"):
            assert "--fail-under=100" in command
        if name.endswith("native-coverage"):
            assert "--branch" in command and "--rcfile=/dev/null" in command
    workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/ci-studio.yml"
    ).read_text()
    assert workflow.count("--scenario operator_review_dossiers") == 1
    assert "tools/tests/test_studio_operator_dossier_browser.py" in workflow
    assert "build_operator_dossier_quality_gates" in workflow
    assert "src/features/operators/dossiers" in workflow
