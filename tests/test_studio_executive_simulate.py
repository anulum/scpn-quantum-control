# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio executive simulate handler tests
"""Tests for the bounded XY-Kuramoto ``simulate`` handler."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from scpn_quantum_control.studio.executive import (
        ActionRegistry,
        ExecutiveRequest,
        preview_action,
        resolve_verb_contract,
        run_action,
    )
    from scpn_quantum_control.studio.executive_simulate import (
        SIMULATE_VERB,
        SimulateActionHandler,
        _as_float,
        _as_positive_int,
        _normalise_simulate,
        _safe_slug,
    )
else:
    pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")
    pytest.importorskip("qiskit", reason="qiskit not installed")
    from scpn_quantum_control.studio.executive import (
        ActionRegistry,
        ExecutiveRequest,
        preview_action,
        resolve_verb_contract,
        run_action,
    )
    from scpn_quantum_control.studio.executive_simulate import (
        SIMULATE_VERB,
        SimulateActionHandler,
        _as_float,
        _as_positive_int,
        _normalise_simulate,
        _safe_slug,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]

_NETWORK: dict[str, Any] = {
    "K_nm": [[0.0, 0.4, 0.1], [0.4, 0.0, 0.3], [0.1, 0.3, 0.0]],
    "omega": [-0.1, 0.05, 0.05],
    "t_max": 0.2,
    "dt": 0.1,
    "trotter_per_step": 1,
    "trotter_order": 1,
}


def _registry() -> ActionRegistry:
    registry = ActionRegistry()
    registry.register(SimulateActionHandler())
    return registry


def _request(*, backend: str | None = None, **overrides: Any) -> ExecutiveRequest:
    parameters = dict(_NETWORK)
    parameters.update(overrides)
    return ExecutiveRequest(
        verb=SIMULATE_VERB, action_id="simulate-3node", parameters=parameters, backend=backend
    )


# --------------------------------------------------------------------------- #
# end-to-end
# --------------------------------------------------------------------------- #
def test_simulate_evolves_and_summarises_trajectory() -> None:
    """Evolve the real public solver and expose a bounded trajectory summary."""
    record = run_action(_request(), registry=_registry())
    assert record.result.status == "succeeded", record.result.error
    outputs = record.result.outputs
    assert outputs["n_nodes"] == 3
    assert outputs["evolution_schema"] == "studio.quantum-evolution.v1"
    assert outputs["n_points"] >= 2
    assert 0.0 <= outputs["order_parameter_initial"] <= 1.0
    assert 0.0 <= outputs["order_parameter_final"] <= 1.0
    assert outputs["order_parameter_max"] >= outputs["order_parameter_mean"]
    assert outputs["order_parameter_delta"] == pytest.approx(
        outputs["order_parameter_final"] - outputs["order_parameter_initial"]
    )
    assert record.script is not None


def test_simulate_plan_defaults_backend_read_only() -> None:
    """Expose the read-only plan with its default Python backend."""
    plan = preview_action(_request(), registry=_registry())
    assert plan.backend == "python"
    assert plan.requires_approval is False
    assert len(plan.steps) == 5


@pytest.mark.parametrize("backend", ["rust", "qiskit"])
def test_simulate_accepts_declared_backends(backend: str) -> None:
    """Accept every backend declared by the simulate verb contract."""
    plan = preview_action(_request(backend=backend), registry=_registry())
    assert plan.backend == backend


def test_simulate_rejects_undeclared_backend() -> None:
    """Reject a backend absent from the public simulate contract."""
    handler = SimulateActionHandler()
    contract = resolve_verb_contract(SIMULATE_VERB)
    with pytest.raises(ValueError, match="is not declared for the simulate verb"):
        handler.plan(_request(backend="abacus"), contract)


def test_generated_simulate_script_embeds_summary_and_compiles() -> None:
    """Generate a valid script carrying the sealed trajectory summary."""
    record = run_action(_request(), registry=_registry())
    assert record.script is not None
    source = record.script.source
    compile(source, record.script.filename, "exec")
    assert repr(record.result.outputs["order_parameter_final"]) in source
    assert "QuantumKuramotoSolver" in source
    assert "solver.run(T_MAX, DT, TROTTER_PER_STEP)" in source


def test_generated_simulate_script_executes_real_solver(
    tmp_path: Path,
) -> None:
    """Run the generated script against the real solver outside coverage tracing."""
    record = run_action(_request(), registry=_registry())
    assert record.script is not None
    script_path = tmp_path / record.script.filename
    script_path.write_text(record.script.source, encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "order_parameter_final=" in completed.stdout
    assert "order_parameter_mean=" in completed.stdout
    assert completed.stdout.rstrip().endswith("verified")


def test_simulate_trotter_order_two_is_accepted() -> None:
    """Evolve the supported second-order Trotter route."""
    record = run_action(_request(trotter_order=2), registry=_registry())
    assert record.result.outputs["trotter_order"] == 2
    assert record.result.status == "succeeded"


# --------------------------------------------------------------------------- #
# execute + generate_script against a stubbed solver
#
# The real solver path (above) proves fidelity but drives Qiskit's statevector
# evolution. Qiskit 2.5.0 raises ``QiskitError('Input is neither Instruction,
# Clifford or AnnotatedOperation.')`` from ``Statevector.evolve`` whenever a
# ``sys.settrace``/``sys.monitoring`` trace function is active (i.e. under
# ``coverage``), an upstream defect that also breaks committed solver tests
# (e.g. ``test_xy_kuramoto``). This stub isolates the handler's output assembly
# and script generation from Qiskit so those lines stay measurable under
# coverage without depending on the buggy traced-Qiskit path.
# --------------------------------------------------------------------------- #
class _StubTrajectory:
    def __init__(self, order_parameter: list[float]) -> None:
        self.R = np.asarray(order_parameter, dtype=np.float64)
        self.times = np.linspace(0.0, 1.0, len(order_parameter), dtype=np.float64)


class _StubSolver:
    def __init__(self, n: int, k_nm: Any, omega: Any, *, trotter_order: int) -> None:
        self.n = n
        self.trotter_order = trotter_order

    def run(self, t_max: float, dt: float, trotter_per_step: int) -> _StubTrajectory:
        return _StubTrajectory([0.1, 0.2, 0.4])


def test_simulate_execute_and_script_with_stubbed_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise output assembly under coverage despite traced-Qiskit incompatibility."""
    monkeypatch.setattr(
        "scpn_quantum_control.studio.executive_simulate.QuantumKuramotoSolver",
        _StubSolver,
    )
    record = run_action(_request(), registry=_registry())
    assert record.result.status == "succeeded", record.result.error
    outputs = record.result.outputs
    assert outputs["n_points"] == 3
    assert outputs["order_parameter_initial"] == pytest.approx(0.1)
    assert outputs["order_parameter_final"] == pytest.approx(0.4)
    assert outputs["order_parameter_max"] == pytest.approx(0.4)
    assert outputs["order_parameter_mean"] == pytest.approx(0.7 / 3.0)
    assert outputs["order_parameter_delta"] == pytest.approx(0.3)
    assert record.script is not None
    compile(record.script.source, record.script.filename, "exec")
    assert repr(outputs["order_parameter_final"]) in record.script.source


# --------------------------------------------------------------------------- #
# _as_float
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [True, "1", None, float("inf"), float("nan")])
def test_as_float_rejects_bad(bad: Any) -> None:
    """Reject booleans, non-numbers, and non-finite scalar inputs."""
    with pytest.raises(ValueError):
        _as_float("v", bad)


def test_as_float_accepts_numbers() -> None:
    """Normalise integer scalar input to a finite float."""
    assert _as_float("v", 2) == 2.0


# --------------------------------------------------------------------------- #
# _as_positive_int
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [True, "two", 0, 999])
def test_as_positive_int_rejects_bad(bad: Any) -> None:
    """Reject non-integer and out-of-range bounded-count inputs."""
    with pytest.raises(ValueError):
        _as_positive_int("v", bad, maximum=64)


def test_as_positive_int_accepts_bounded() -> None:
    """Accept an integer within the declared upper bound."""
    assert _as_positive_int("v", 3, maximum=64) == 3


# --------------------------------------------------------------------------- #
# _safe_slug
# --------------------------------------------------------------------------- #
def test_safe_slug_normal_and_empty() -> None:
    """Produce filesystem-safe action slugs and a non-empty fallback."""
    assert _safe_slug("simulate-3node.1") == "simulate_3node_1"
    assert _safe_slug("!!!") == "action"


# --------------------------------------------------------------------------- #
# _normalise_simulate validation branches
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "overrides",
    [
        {"K_nm": "matrix"},
        {"K_nm": [1.0, [0.0, 1.0]]},
        {"K_nm": [[0.0]]},
        {"K_nm": [[0.0] * 13 for _ in range(13)], "omega": [0.0] * 13},
        {"K_nm": [[0.0, 1.0], [1.0, 0.0, 0.0]], "omega": [0.0, 0.0]},
        {"K_nm": [[1.0, 0.0], [0.0, 0.0]], "omega": [0.0, 0.0]},
        {"K_nm": [[0.0, 1.0], [2.0, 0.0]], "omega": [0.0, 0.0]},
        {"omega": "not-a-list"},
        {"omega": [0.1, 0.2]},
        {"t_max": 0.0},
        {"t_max": -1.0},
        {"dt": 0.0},
        {"dt": -0.1},
        {"dt": 0.9},
        {"t_max": 1.0, "dt": 0.001},
        {"trotter_per_step": 0},
        {"trotter_per_step": 999},
        {"trotter_per_step": True},
        {"trotter_per_step": "two"},
        {"trotter_order": 3},
        {"trotter_order": True},
    ],
)
def test_normalise_simulate_rejects_invalid(overrides: dict[str, Any]) -> None:
    """Reject malformed or unbounded simulation requests."""
    parameters = dict(_NETWORK)
    parameters.update(overrides)
    with pytest.raises(ValueError):
        _normalise_simulate(parameters)


def test_normalise_simulate_accepts_bounded_network() -> None:
    """Normalise a bounded symmetric network and finite schedule."""
    simulate_spec = _normalise_simulate(_NETWORK)
    assert len(simulate_spec["K_nm"]) == 3
    assert simulate_spec["trotter_order"] == 1
    assert simulate_spec["t_max"] == 0.2
