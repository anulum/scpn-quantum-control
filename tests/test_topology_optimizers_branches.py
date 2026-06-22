# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the topology-control optimisers
"""Guard and branch tests for the projected topology-control optimisers.

Covers the trace final-objective guard, the SPSA and SciPy optimiser config
guards, the SPSA zero-step trace branch and the SciPy no-callback and
diverging-final trace branches.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.topology_control import (
    CouplingTopologyObjective,
    NetworkCycleBackend,
    ProjectedScipyOptimizer,
    ProjectedSPSAOptimizer,
    TopologyConstraintLedger,
)
from scpn_quantum_control.topology_control.optimizers import TopologyOptimisationTrace

_K0 = np.array(
    [
        [0.0, 0.3, 0.0, 0.3],
        [0.3, 0.0, 0.3, 0.0],
        [0.0, 0.3, 0.0, 0.3],
        [0.3, 0.0, 0.3, 0.0],
    ],
    dtype=np.float64,
)


def _objective() -> CouplingTopologyObjective:
    return CouplingTopologyObjective(
        ph_backend=NetworkCycleBackend(threshold=0.2),
        ledger=TopologyConstraintLedger(algebraic_connectivity_min=0.01),
        source_matrix=_K0,
        source_distance_weight=0.25,
        allow_approximate_ph_backend=True,
    )


def test_trace_final_objective_requires_steps() -> None:
    """A trace with no steps has no final objective."""
    trace = TopologyOptimisationTrace(
        initial_matrix=_K0.copy(), final_matrix=_K0.copy(), steps=[], seed=0
    )
    with pytest.raises(ValueError, match="optimisation trace has no steps"):
        _ = trace.final_objective


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_steps": -1}, "max_steps must be non-negative"),
        ({"step_size": 0.0}, "step_size must be positive"),
        ({"perturbation": 0.0}, "perturbation must be positive"),
    ],
)
def test_spsa_config_guards(kwargs: dict[str, Any], match: str) -> None:
    """The SPSA optimiser rejects each out-of-range parameter."""
    with pytest.raises(ValueError, match=match):
        ProjectedSPSAOptimizer(**kwargs)


def test_spsa_zero_steps_emits_single_trace_step() -> None:
    """With zero steps the SPSA optimiser still records one trace entry."""
    trace = ProjectedSPSAOptimizer(max_steps=0).optimise(_K0, _objective())
    assert len(trace.steps) == 1
    assert trace.final_objective is trace.steps[-1].objective


def test_scipy_optimiser_rejects_non_positive_maxiter() -> None:
    """A non-positive SciPy iteration budget is rejected."""
    with pytest.raises(ValueError, match="maxiter must be positive"):
        ProjectedScipyOptimizer(maxiter=0)


def test_scipy_optimiser_records_final_when_no_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the solver never calls back, the final iterate is recorded."""

    def _no_callback(score: Any, x0: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(x=np.asarray(x0, dtype=np.float64))

    monkeypatch.setattr("scipy.optimize.minimize", _no_callback)
    trace = ProjectedScipyOptimizer(maxiter=5).optimise(_K0, _objective())
    assert len(trace.steps) == 1


def test_scipy_optimiser_appends_diverging_final(monkeypatch: pytest.MonkeyPatch) -> None:
    """A final iterate differing from the last callback step is appended."""

    def _diverging(score: Any, x0: Any, *, callback: Any = None, **_kwargs: Any) -> Any:
        if callback is not None:
            callback(np.asarray(x0, dtype=np.float64))
        return SimpleNamespace(x=np.zeros_like(np.asarray(x0, dtype=np.float64)))

    monkeypatch.setattr("scipy.optimize.minimize", _diverging)
    trace = ProjectedScipyOptimizer(maxiter=5).optimise(_K0, _objective())
    assert len(trace.steps) >= 2
