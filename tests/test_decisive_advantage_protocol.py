# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the decisive advantage-benchmark protocol
"""Multi-angle tests for benchmarks/decisive_advantage_protocol.py.

Dimensions: dataclass invariants and serialisation, submission-gate checks, the
full fail-closed decision matrix (advantage / crossover / classical-wins /
inconclusive), the accuracy and budget filters, and the default protocol.
"""

from __future__ import annotations

from typing import Any

import pytest

from scpn_quantum_control.benchmarks.decisive_advantage_protocol import (
    DecisionCriterion,
    DecisionOutcome,
    DecisiveAdvantageProtocol,
    SubmissionGate,
    default_decisive_advantage_protocol,
    evaluate_decision,
)

PROTOCOL = default_decisive_advantage_protocol()
PROTOCOL_ID = PROTOCOL.protocol.protocol_id
TARGET = PROTOCOL.criterion.target_size


def _row(
    baseline: str, wall_ms: float, reference_error: float, *, status: str = "ok"
) -> dict[str, Any]:
    """Build one schema-valid measured row for the decisive protocol."""
    return {
        "protocol_id": PROTOCOL_ID,
        "n_qubits": TARGET,
        "baseline": baseline,
        "status": status,
        "wall_time_ms": wall_ms,
        "memory_bytes": 4096,
        "metric_payload": {"reference_error": reference_error, "order_parameter_R": 0.5},
        "command": ["run", "bench"],
        "machine": "ml350",
        "dependencies": {"numpy": "2.1.0"},
        "git_commit": "deadbeef",
        "notes": [],
    }


def _rowset(
    *,
    classical_wall: float,
    exact_wall: float,
    qpu_wall: float | None,
    classical_error: float = 0.005,
    exact_error: float | None = None,
    qpu_error: float = 0.005,
) -> list[dict[str, Any]]:
    """Full required-baseline rowset (validates) plus an optional QPU row."""
    rows = [
        _row("classical_ode", classical_wall, classical_error),
        _row("mps_tensor_network", classical_wall, classical_error),
        _row("sparse_eigsh", classical_wall, classical_error),
        _row("dense_eigh", exact_wall, classical_error if exact_error is None else exact_error),
    ]
    if qpu_wall is not None:
        rows.append(_row("qpu_hardware", qpu_wall, qpu_error))
    return rows


class TestDecisionCriterion:
    def test_valid_criterion_serialises(self) -> None:
        criterion = DecisionCriterion(
            observable="order_parameter_R",
            target_size=12,
            accuracy_target=0.01,
            budget_wall_time_ms=1000.0,
            best_classical_baselines=("classical_ode",),
            exact_baselines=("dense_eigh",),
        )
        assert criterion.to_dict()["target_size"] == 12

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"observable": ""}, "observable"),
            ({"target_size": 0}, "target_size"),
            ({"accuracy_target": -1.0}, "accuracy_target"),
            ({"accuracy_target": float("inf")}, "accuracy_target"),
            ({"budget_wall_time_ms": 0.0}, "budget_wall_time_ms"),
            ({"best_classical_baselines": ()}, "best_classical_baselines"),
            ({"exact_baselines": ()}, "exact_baselines"),
        ],
    )
    def test_invalid_criterion_raises(self, kwargs: dict[str, Any], match: str) -> None:
        base = {
            "observable": "R",
            "target_size": 12,
            "accuracy_target": 0.01,
            "budget_wall_time_ms": 1000.0,
            "best_classical_baselines": ("classical_ode",),
            "exact_baselines": ("dense_eigh",),
        }
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            DecisionCriterion(**base)  # type: ignore[arg-type]


class TestSubmissionGate:
    def test_within_ceilings_passes(self) -> None:
        gate = SubmissionGate(max_circuit_depth=400, max_total_shots=8192)
        assert gate.check(400, 8192) == (True, ())

    def test_both_breaches_reported(self) -> None:
        gate = SubmissionGate(max_circuit_depth=400, max_total_shots=8192)
        passed, reasons = gate.check(401, 9000)
        assert passed is False
        assert len(reasons) == 2

    def test_serialisation(self) -> None:
        gate = SubmissionGate(max_circuit_depth=400, max_total_shots=8192)
        assert gate.to_dict() == {"max_circuit_depth": 400, "max_total_shots": 8192}

    @pytest.mark.parametrize(
        ("depth", "shots", "match"),
        [(0, 10, "max_circuit_depth"), (10, 0, "max_total_shots")],
    )
    def test_invalid_gate_raises(self, depth: int, shots: int, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            SubmissionGate(max_circuit_depth=depth, max_total_shots=shots)


class TestProtocolConstruction:
    def test_size_absent_from_protocol_raises(self) -> None:
        criterion = DecisionCriterion(
            observable="R",
            target_size=99,  # not in the default protocol's sizes
            accuracy_target=0.01,
            budget_wall_time_ms=1000.0,
            best_classical_baselines=("classical_ode",),
            exact_baselines=("dense_eigh",),
        )
        with pytest.raises(ValueError, match="exactly the decision size"):
            DecisiveAdvantageProtocol(
                protocol=PROTOCOL.protocol,
                criterion=criterion,
                gate=PROTOCOL.gate,
                qpu_time_estimate_s=1.0,
            )

    @pytest.mark.parametrize("estimate", [0.0, -1.0, float("nan")])
    def test_bad_estimate_raises(self, estimate: float) -> None:
        with pytest.raises(ValueError, match="qpu_time_estimate_s"):
            DecisiveAdvantageProtocol(
                protocol=PROTOCOL.protocol,
                criterion=PROTOCOL.criterion,
                gate=PROTOCOL.gate,
                qpu_time_estimate_s=estimate,
            )

    def test_serialisation_roundtrip_keys(self) -> None:
        payload = PROTOCOL.to_dict()
        assert set(payload) == {"protocol", "criterion", "gate", "qpu_time_estimate_s"}

    def test_validate_rows_delegates(self) -> None:
        rows = _rowset(classical_wall=100.0, exact_wall=90.0, qpu_wall=200.0)
        assert PROTOCOL.validate_rows(rows).valid is True


class TestDefaultProtocol:
    def test_identity_and_estimate(self) -> None:
        assert PROTOCOL.protocol.sizes == (12,)
        assert PROTOCOL.qpu_time_estimate_s > 0.0
        assert PROTOCOL.criterion.observable == "order_parameter_R"

    def test_required_baselines_cover_classical_and_exact(self) -> None:
        required = set(PROTOCOL.protocol.required_baselines)
        assert {"classical_ode", "mps_tensor_network", "sparse_eigsh", "dense_eigh"} <= required


class TestDecisionMatrix:
    def test_qpu_beats_best_classical_is_advantage(self) -> None:
        rows = _rowset(classical_wall=6000.0, exact_wall=6000.0, qpu_wall=100.0)
        assert evaluate_decision(PROTOCOL, rows).label == "qpu_decides_advantage"

    def test_classical_faster_but_qpu_beats_exact_is_crossover(self) -> None:
        rows = _rowset(classical_wall=100.0, exact_wall=5000.0, qpu_wall=200.0)
        assert evaluate_decision(PROTOCOL, rows).label == "exact_hilbert_space_crossover_only"

    def test_classical_fastest_is_classical_wins(self) -> None:
        rows = _rowset(classical_wall=100.0, exact_wall=90.0, qpu_wall=5000.0)
        assert evaluate_decision(PROTOCOL, rows).label == "classical_wins"

    def test_no_qpu_row_is_inconclusive(self) -> None:
        rows = _rowset(classical_wall=100.0, exact_wall=90.0, qpu_wall=None)
        assert evaluate_decision(PROTOCOL, rows).label == "inconclusive"

    def test_invalid_rows_are_inconclusive(self) -> None:
        rows = _rowset(classical_wall=100.0, exact_wall=90.0, qpu_wall=200.0)
        rows[0]["git_commit"] = ""  # break schema validity
        outcome = evaluate_decision(PROTOCOL, rows)
        assert outcome.label == "inconclusive"
        assert any("validation" in reason for reason in outcome.reasons)

    def test_accuracy_filter_excludes_inaccurate_qpu(self) -> None:
        # QPU is fast but its reference_error exceeds the accuracy target → excluded.
        rows = _rowset(classical_wall=6000.0, exact_wall=6000.0, qpu_wall=100.0, qpu_error=0.5)
        assert evaluate_decision(PROTOCOL, rows).label == "inconclusive"

    def test_only_exact_qualifies_is_crossover(self) -> None:
        # Best-classical rows present for validity but inaccurate → excluded; only
        # the exact row and the QPU row qualify.
        rows = _rowset(
            classical_wall=100.0,
            exact_wall=5000.0,
            qpu_wall=200.0,
            classical_error=0.5,
            exact_error=0.005,  # exact stays accurate so it qualifies
        )
        assert evaluate_decision(PROTOCOL, rows).label == "exact_hilbert_space_crossover_only"

    def test_no_qualifying_reference_is_inconclusive(self) -> None:
        # Every classical and exact row is inaccurate → only the QPU row qualifies.
        rows = _rowset(
            classical_wall=100.0,
            exact_wall=100.0,
            qpu_wall=200.0,
            classical_error=0.5,
        )
        assert evaluate_decision(PROTOCOL, rows).label == "inconclusive"

    def test_budget_filter_excludes_over_budget_qpu(self) -> None:
        over = PROTOCOL.criterion.budget_wall_time_ms + 1.0
        rows = _rowset(classical_wall=100.0, exact_wall=90.0, qpu_wall=over)
        assert evaluate_decision(PROTOCOL, rows).label == "inconclusive"

    def test_skipped_rows_do_not_count_as_timing(self) -> None:
        # A size-gated skip is a valid row but must not count as a fast classical
        # timing: the fastest classical here is skipped, so the QPU still wins.
        rows = _rowset(classical_wall=6000.0, exact_wall=6000.0, qpu_wall=100.0)
        rows[2]["status"] = "skipped"
        rows[2]["notes"] = ["size-gated skip"]
        rows[2]["wall_time_ms"] = 1.0
        assert evaluate_decision(PROTOCOL, rows).label == "qpu_decides_advantage"

    def test_row_without_reference_error_does_not_qualify(self) -> None:
        # An ok row lacking the accuracy metric cannot decide; here the QPU row
        # loses its reference_error, so no QPU row qualifies.
        rows = _rowset(classical_wall=6000.0, exact_wall=6000.0, qpu_wall=100.0)
        del rows[4]["metric_payload"]["reference_error"]
        assert evaluate_decision(PROTOCOL, rows).label == "inconclusive"


class TestDecisionOutcome:
    def test_outcome_serialises(self) -> None:
        outcome = DecisionOutcome("classical_wins", ("reason",))
        assert outcome.to_dict() == {"label": "classical_wins", "reasons": ["reason"]}
