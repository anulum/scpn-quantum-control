# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — source-owned control proposal instrument data
"""Check the control instrument data against independent oracles and its CLI.

Oracles: the tracking cost ``sum_t ||u_t v - r||^2`` evaluated directly, the
co-design constraint rows recomputed from the declared caps and budget, and a
fresh call of the partition integrator for the trace timestamps.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.control_stack_runtime_adapters import (
    CONTROL_STACK_RUNTIME_ADAPTER_CLAIM_BOUNDARY,
)
from scpn_quantum_control.cosimulation.quantum_classical import cosimulate
from scpn_quantum_control.studio import control_proposal_instrument_artifact as artifact

pytestmark = pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    """Build the instrument data once from its source owners."""
    return artifact.build_control_proposal_instrument()


def _tracking_cost(b_matrix: list[list[float]], target: list[float], action: int) -> float:
    """Return ``||u v - r||^2`` with ``v = B @ ones`` for one binary action."""
    actuation = np.asarray(b_matrix).sum(axis=1)
    residual = action * actuation - np.asarray(target)
    return float(residual @ residual)


def test_control_proposal_instrument_01_signed_and_rotated_targets_keep_the_core_objective(
    payload: dict[str, Any],
) -> None:
    """Each action cost equals the vector tracking cost; the target's sign moves the optimum."""
    cases = {case["case_id"]: case for case in payload["mpc_targets"]}
    for case in cases.values():
        expected = [_tracking_cost(case["b_matrix"], case["target"], u) for u in (0, 1)]
        assert case["action_costs"] == pytest.approx(expected, abs=1e-12)
        assert case["proposed_action"] == int(np.argmin(expected))
        assert case["applied"] is False
    assert cases["signed target"]["action_costs"] == pytest.approx([1.0, 4.0])
    assert cases["signed target"]["proposed_action"] == 0
    assert cases["sign-flipped target"]["proposed_action"] == 1
    assert cases["rotated target"]["action_costs"] == pytest.approx([2.0, 4.0])


def test_control_proposal_instrument_02_binding_and_violated_edges_follow_the_constraints(
    payload: dict[str, Any],
) -> None:
    """Highlighted edges are exactly the caps at or over their limit in some checked case."""
    codesign = payload["codesign"]
    caps = np.asarray(codesign["edge_caps"])
    scales = [np.ones(3), *(np.asarray(item) for item in codesign["disturbances"])]
    for proposal in codesign["proposals"]:
        if not proposal["constraint_values"]:
            assert proposal["binding_edges"] == proposal["violated_edges"] == []
            continue
        plan = np.asarray(proposal["plan"])
        rows = [
            np.concatenate([plan - caps * scale, [plan.sum() - codesign["total_budget"]]])
            for scale in scales
        ]
        np.testing.assert_allclose(proposal["constraint_values"], rows, atol=1e-12)
        worst = np.max(np.asarray(rows)[:, :3], axis=0)
        assert proposal["binding_edges"] == [i for i in range(3) if worst[i] >= -1e-9]
        assert proposal["violated_edges"] == [i for i in range(3) if worst[i] > 0.0]
    accepted, infeasible = codesign["proposals"][:2]
    assert accepted["binding_edges"] == [0] and accepted["violated_edges"] == []
    assert infeasible["violated_edges"] == [0]


def test_control_proposal_instrument_03_rejected_proposals_never_apply_or_adopt(
    payload: dict[str, Any],
) -> None:
    """Only the accepted plan is adopted; vetoed and out-of-domain proposals change nothing."""
    proposals = payload["codesign"]["proposals"]
    assert [item["status"] for item in proposals] == [
        "accepted",
        "infeasible",
        "out_of_domain",
        "vetoed",
    ]
    adopted = proposals[0]["plan"]
    assert proposals[0]["adopted_plan"] == adopted
    assert proposals[1]["adopted_plan"] == proposals[2]["adopted_plan"] == adopted
    assert proposals[2]["constraint_values"] == [] and proposals[2]["objective"] is None
    assert proposals[3]["plan"] is None and proposals[3]["decision"] is None
    assert "policy refused" in proposals[3]["reason"]
    assert all(item["applied"] is False for item in proposals)
    assert all(item["hardware_actuation"] is False for item in proposals)
    assert payload["claim_boundary"] == CONTROL_STACK_RUNTIME_ADAPTER_CLAIM_BOUNDARY
    assert "only with explicit runtime actuation evidence" in payload["applied_status_rule"]


def test_control_proposal_instrument_04_trace_timestamps_are_the_integrator_samples(
    payload: dict[str, Any],
) -> None:
    """Timestamps equal a fresh core run bit for bit; no uniform grid is regenerated."""
    trace = payload["trace"]
    coupling = np.array([[0.0, 0.6, 0.2], [0.6, 0.0, 0.4], [0.2, 0.4, 0.0]])
    fresh = cosimulate(
        coupling, np.array([1.0, 1.3, 0.8]), dt=0.3, n_steps=4, max_quantum_nodes=2, seed=3
    )
    assert trace["times"] == fresh.times.tolist()
    assert trace["global_order"] == fresh.global_order.tolist()
    assert trace["times"] == [step * 0.3 for step in range(5)]
    # The exact binary value of 3 * dt is kept, not a rounded display label.
    assert trace["times"][3] == 3 * 0.3 != 0.9
    assert trace["simulated"] is True and trace["observed_hardware_telemetry"] is None


def test_latency_verdicts_are_replayed_and_refuse_an_over_budget_claim(
    payload: dict[str, Any],
) -> None:
    """Replayed samples are labelled as such; the over-budget series names its blockers."""
    within, over = payload["latency"]
    assert within["passes"] is True and within["blockers"] == []
    assert over["passes"] is False and len(over["blockers"]) == 2
    assert max(over["round_latencies_s"]) == over["max_round_latency_s"] == 0.015
    assert {within["clock_source"], over["clock_source"]} == {
        "replayed_observed_round_latencies_s"
    }


def test_committed_data_and_public_cli_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write, check and refuse changed data through the public command line.

    Parameters
    ----------
    tmp_path
        Owned scratch output.
    monkeypatch
        Supplies the real command's argument vector.

    """
    assert artifact.main(["--check"]) == 0
    target = tmp_path / "control.json"
    assert artifact.main(["--write", "--json-path", str(target)]) == 0
    assert artifact.main(["--check", "--json-path", str(target)]) == 0
    target.write_text("{}")
    assert artifact.main(["--check", "--json-path", str(target)]) == 1
    monkeypatch.setattr(sys, "argv", ["control_proposal_instrument_artifact", "--check"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module(artifact.__name__, run_name="__main__")
    assert stopped.value.code == 0
