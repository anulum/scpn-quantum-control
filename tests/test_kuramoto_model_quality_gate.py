# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Kuramoto model owning CI integration tests
"""Exercise real convention quality commands and original source gate wiring."""

from __future__ import annotations

import subprocess
import sys

from tools import kuramoto_model_quality_gates as gates
from tools import preflight
from tools import scientific_problem_quality_gates as scientific
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_kuramoto_model_static_commands_execute_the_actual_owned_cohort() -> None:
    """Strict types, native documentation and generated-source checks run real processes."""
    for name, command in gates.build_static_quality_gates(sys.executable):
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=60
        )
        assert completed.returncode == 0, f"{name}: {completed.stdout}{completed.stderr}"


def test_kuramoto_model_ci_and_preflight_require_exact_owners_and_coverage() -> None:
    """New conventions own one category while the existing shared-core gate stays complete."""
    source = read_ci_workflow_source()
    start = source.index("  kuramoto-model-quality:")
    end = source.index("\n  stable-core-product-quality:", start)
    block = source[start:end]
    for path in (*gates.KURAMOTO_MODEL_TYPING, *gates.KURAMOTO_MODEL_TESTS):
        assert path in block
    assert gates.KURAMOTO_MODEL_INCLUDE in block
    assert "--fail-under=100" in block and "--branch" in block
    assert "tools/build_kuramoto_conventions.py" in block and "--check" in block
    aggregate = source[source.index("  ci-gate:") :]
    assert "kuramoto-model-quality" in aggregate
    for name, command in gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    assert gates.build_coverage_gates(preflight._PY) == preflight.KURAMOTO_MODEL_COVERAGE_GATES
    assert "tests/test_kuramoto_model_conventions.py" in scientific.SCIENTIFIC_PROBLEM_TESTS
    shared_start = source.index("  scientific-problem-quality:")
    shared_end = source.index("\n  qpu-compute-types-quality:", shared_start)
    assert "tests/test_kuramoto_model_conventions.py" in source[shared_start:shared_end]
    assert "--fail-under=100" in source[shared_start:shared_end]
