# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for the Kuramoto competitive-benchmark shared types
"""Tests for :mod:`kuramoto_competitive_types`.

The value types are the contract both the harness and its external adapters
exchange, so the tests pin the deterministic problem construction (shapes,
symmetry, reproducibility, and every rejected bound) and the serialisable row.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.benchmarks import kuramoto_competitive_types as t


def test_build_default_problem_shapes_and_properties() -> None:
    problem = t.build_default_problem(n_oscillators=8, t_max=1.0, dt=0.05, seed=3)
    assert problem.coupling.shape == (8, 8)
    assert problem.omega.shape == (8,)
    assert problem.theta0.shape == (8,)
    assert problem.n_oscillators == 8
    assert problem.n_steps == 20
    # symmetric, zero diagonal, non-negative
    assert np.allclose(problem.coupling, problem.coupling.T)
    assert np.allclose(np.diag(problem.coupling), 0.0)
    assert np.all(problem.coupling >= 0.0)


def test_build_default_problem_is_deterministic() -> None:
    a = t.build_default_problem(seed=11)
    b = t.build_default_problem(seed=11)
    assert np.array_equal(a.coupling, b.coupling)
    assert np.array_equal(a.omega, b.omega)
    assert np.array_equal(a.theta0, b.theta0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_oscillators": 1}, "n_oscillators must be >= 2"),
        ({"t_max": 0.0}, "t_max must be positive"),
        ({"dt": 0.0}, "dt must be positive"),
        ({"t_max": 0.1, "dt": 0.2}, "must not exceed"),
    ],
)
def test_build_default_problem_rejects_bad_bounds(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        t.build_default_problem(**kwargs)


def test_competitor_row_to_dict_round_trips_fields() -> None:
    row = t.CompetitorRow(
        method="x",
        backend="b",
        family="ours",
        language="python",
        available=True,
        version="1.2.3",
        r_final=0.4,
        r_error_vs_reference=0.01,
        elapsed_ms=3.0,
        install_command=None,
        unavailable_reason=None,
    )
    payload = row.to_dict()
    assert payload["method"] == "x"
    assert payload["language"] == "python"
    assert payload["version"] == "1.2.3"
    assert json.dumps(payload)  # serialisable
