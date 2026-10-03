# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — solver public example and owning quality integration
"""Run the real public example and require its exact scientific CI cohort."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from tools import kuramoto_solver_quality_gates as gates
from tools import preflight
from tools.ci_workflow_inventory import ci_workflow_paths, read_ci_workflow_source


def test_solver_public_example_runs_original_scientific_factory() -> None:
    """A real child uses the public binding, original solver and SciPy failures."""
    completed = subprocess.run(
        [sys.executable, "examples/kuramoto_solver_outcomes.py"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    payload = json.loads(completed.stdout)
    np.testing.assert_array_equal(payload["fixed_times"], np.arange(5) / 4)
    np.testing.assert_allclose(
        payload["fixed_phases"], (8 + 0.4 * np.arange(5) / 4)[:, None], atol=2e-15
    )
    assert payload["completed"] == "completed" and payload["event"] == "event"
    np.testing.assert_allclose(payload["event_times"], [[0.5]], atol=1e-14)
    np.testing.assert_allclose(payload["event_phases"], [[[8.2]]], atol=1e-14)
    assert payload["failure"] == "failed" and not payload["failure_success"]
    assert payload["failure_last_time"] < 2 and payload["failure_message"]
    assert payload["query_state_retained"] == [0.0]


def test_solver_static_commands_execute_real_types_and_native_docs() -> None:
    """The registered exact type/doc commands execute rather than assert strings alone."""
    for name, command in gates.build_static_quality_gates(sys.executable):
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=90
        )
        assert completed.returncode == 0, f"{name}: {completed.stdout}{completed.stderr}"


def test_solver_ci_and_preflight_keep_whole_owner_coverage() -> None:
    """The coherent scientific category retains every original owner and aggregate gate."""
    source = read_ci_workflow_source()
    start = source.index("  kuramoto-solver-quality:")
    end = source.index("\n  stable-core-product-quality:", start)
    block = source[start:end]
    for path in (*gates.KURAMOTO_SOLVER_TYPING, *gates.KURAMOTO_SOLVER_TESTS):
        assert path in block
    assert gates.KURAMOTO_SOLVER_INCLUDE in block
    assert "--fail-under=100" in block and "--branch" in block
    for name, command in gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    assert gates.build_coverage_gates(preflight._PY) == preflight.KURAMOTO_SOLVER_COVERAGE_GATES
    assert "kuramoto-solver-quality" in source[source.index("  ci-gate:") :]
    coordinator = ci_workflow_paths()[0].read_text()
    assert "core-science" in coordinator[coordinator.index("  ci-gate:") :]
