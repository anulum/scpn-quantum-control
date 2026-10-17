# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Whole native method-comparison quality contract
"""Keep every native owner, direct regression and unchanged exact gate enrolled."""

from pathlib import Path

from tools.studio_method_comparison_quality_gates import build_method_comparison_quality_gates


def test_whole_native_cohort_keeps_strict_docs_all_original_tests_and_exact_coverage(
    tmp_path: Path,
) -> None:
    """The executable quality contract binds original producers to every dedicated owner."""
    gates = dict(build_method_comparison_quality_gates("python", output_root=tmp_path))
    strict = gates["studio-method-comparison-strict"]
    assert "--strict" in strict and "--explicit-package-bases" in strict
    for owner in [
        "method_comparison.py",
        "method_comparison_contracts.py",
        "reproducible_comparison.py",
        "method_comparison_artifact.py",
    ]:
        assert any(path.endswith("/" + owner) for path in strict)
    tests = gates["studio-method-comparison-native-coverage"]
    for owner in [
        "tests/test_method_comparison.py",
        "tests/test_method_comparison_contracts.py",
        "tests/test_reproducible_comparison.py",
        "tests/test_studio_method_comparison_artifact.py",
        "tools/tests/test_export_method_comparison.py",
    ]:
        assert owner in tests
    assert "--branch" in tests
    report = gates["studio-method-comparison-native-exact"]
    assert "--fail-under=100" in report
    assert "D,D413,D417,D420" in gates["studio-method-comparison-native-docs"]
    data = next(arg for arg in tests if arg.startswith("--data-file="))
    assert data in report
    assert str(tmp_path) in data
    assert not list(tmp_path.iterdir())


def test_analysis_aliases_have_original_native_strict_docs_and_exact_coverage(
    tmp_path: Path,
) -> None:
    """Qualify both public aliases in the existing producer cohort without weakening it.

    Parameters
    ----------
    tmp_path
        Caller-owned gate output directory; building commands performs no I/O.

    """
    gates = dict(build_method_comparison_quality_gates("python", output_root=tmp_path))
    for filename in ("method_comparison.py", "method_comparison_contracts.py"):
        path = "src/scpn_quantum_control/analysis/" + filename
        assert path in gates["studio-method-comparison-strict"]
        assert path in gates["studio-method-comparison-native-docs"]
        assert any(
            "*/analysis/" + filename in arg
            for arg in gates["studio-method-comparison-native-coverage"]
        )
        assert any(
            "*/analysis/" + filename in arg
            for arg in gates["studio-method-comparison-native-exact"]
        )
    assert (
        "tests/test_analysis_method_comparison.py"
        in gates["studio-method-comparison-native-coverage"]
    )
    assert "--fail-under=100" in gates["studio-method-comparison-native-exact"]
