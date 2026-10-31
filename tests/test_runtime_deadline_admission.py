# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — deadline, freshness and veto admission before any actuation
"""Late, stale or vetoed proposals never reach the apply port; cancellation ends the loop.

The oracle is a controlled virtual monotonic clock: each tick's proposal
time and telemetry age are set exactly, so the expected admission of every
tick is known in advance. Nothing here actuates hardware.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from scpn_quantum_control.control.realtime_runtime import (
    GATED_TIMING_CLAIM,
    ActuationProposal,
    OfflineApplyPort,
    RealtimeRuntimeConfig,
    VirtualRealtimeClock,
    run_gated_actuation_loop,
)

_CONFIG = RealtimeRuntimeConfig(sample_period_s=0.01, deadline_s=0.004, jitter_budget_s=0.001)


class _Script:
    """Propose on a script: per tick, compute time and telemetry age, both in seconds."""

    def __init__(self, clock: VirtualRealtimeClock, plan: list[tuple[float, float]]) -> None:
        """Store the clock and the per-tick ``(compute_s, telemetry_age_s)`` plan."""
        self.clock = clock
        self.plan = plan

    def __call__(self, index: int) -> ActuationProposal:
        """Spend the tick's compute time and return a proposal with the planned age."""
        compute, age = self.plan[index]
        sampled = self.clock.now() - age
        self.clock.advance(compute)
        return ActuationProposal((float(index),), sampled)


def _no_veto(proposal: ActuationProposal) -> str | None:
    """Admit everything."""
    return None


def test_runtime_deadline_admission_01_only_timely_fresh_unvetoed_proposals_are_applied() -> None:
    """On-time, late, stale and vetoed ticks are told apart; only on-time ones are applied."""
    clock = VirtualRealtimeClock()
    plan = [(0.001, 0.0), (0.006, 0.0), (0.001, 0.05), (0.001, 0.0), (0.002, 0.001)]

    def veto(proposal: ActuationProposal) -> str | None:
        """Veto tick 3 independently of timing."""
        return "operator interlock" if proposal.values == (3.0,) else None

    port = OfflineApplyPort()
    result = run_gated_actuation_loop(
        5,
        _Script(clock, plan),
        config=_CONFIG,
        max_telemetry_age_s=0.02,
        veto=veto,
        apply_port=port,
        clock=clock,
    )
    outcomes = [decision.outcome for decision in result.decisions]
    assert outcomes == ["applied", "late", "stale", "vetoed", "applied"]
    assert [proposal.values for proposal in port.received] == [(0.0,), (4.0,)]
    assert result.applied == 2 and result.completed and result.cancelled_at is None
    assert result.decisions[1].elapsed_s == pytest.approx(0.006)
    assert result.decisions[2].telemetry_age_s == pytest.approx(0.051)
    assert result.decisions[3].reason == "operator interlock"


def test_runtime_deadline_admission_01_a_slow_veto_makes_the_proposal_late() -> None:
    """The deadline is checked again after the veto, so a slow check cannot push it through."""
    clock = VirtualRealtimeClock()

    def slow_veto(proposal: ActuationProposal) -> str | None:
        """Admit, but take longer than the deadline."""
        clock.advance(0.005)
        return None

    port = OfflineApplyPort()
    result = run_gated_actuation_loop(
        1,
        _Script(clock, [(0.001, 0.0)]),
        config=_CONFIG,
        max_telemetry_age_s=0.02,
        veto=slow_veto,
        apply_port=port,
        clock=clock,
    )
    assert (
        result.decisions[0].outcome == "late" and "during the veto" in result.decisions[0].reason
    )
    assert port.received == []


def test_runtime_deadline_admission_02_cancellation_ends_the_owned_loop() -> None:
    """Cancellation set during tick 2 stops the loop before tick 3 proposes; no thread is left."""
    clock = VirtualRealtimeClock()
    cancel = threading.Event()
    proposed: list[int] = []

    def propose(index: int) -> ActuationProposal:
        """Record the tick and ask for cancellation at tick 2."""
        proposed.append(index)
        if index == 2:
            cancel.set()
        return ActuationProposal((float(index),), clock.now())

    threads_before = threading.active_count()
    port = OfflineApplyPort()
    result = run_gated_actuation_loop(
        10,
        propose,
        config=_CONFIG,
        max_telemetry_age_s=0.02,
        veto=_no_veto,
        apply_port=port,
        clock=clock,
        cancel=cancel,
    )
    assert proposed == [0, 1, 2]
    assert not result.completed and result.cancelled_at == 3
    assert result.applied == 3 == len(port.received)
    assert threading.active_count() == threads_before
    already = run_gated_actuation_loop(
        3, propose, config=_CONFIG, max_telemetry_age_s=0.02, veto=_no_veto, cancel=cancel
    )
    assert already.decisions == () and already.cancelled_at == 0 and proposed == [0, 1, 2]


def test_runtime_deadline_admission_03_recorded_timing_is_measured_software_evidence() -> None:
    """Timings come from the given clock and carry a claim that is not hard-real-time."""
    clock = VirtualRealtimeClock()
    result = run_gated_actuation_loop(
        2,
        _Script(clock, [(0.0015, 0.0), (0.0025, 0.0)]),
        config=_CONFIG,
        max_telemetry_age_s=0.02,
        veto=_no_veto,
        clock=clock,
    )
    assert [decision.elapsed_s for decision in result.decisions] == pytest.approx([0.0015, 0.0025])
    assert result.timing_claim == GATED_TIMING_CLAIM
    assert "not a hard-real-time guarantee" in result.timing_claim
    assert "physical safety certificate" in result.timing_claim
    # The wall-clock default also runs, offline, without actuating anything.
    wall = run_gated_actuation_loop(
        1,
        lambda index: ActuationProposal((0.0,), 0.0),
        config=RealtimeRuntimeConfig(sample_period_s=1.0, deadline_s=1.0, align_to_period=False),
        max_telemetry_age_s=1e9,
        veto=_no_veto,
    )
    assert wall.decisions[0].outcome == "applied"


def test_gated_loop_rejects_malformed_inputs() -> None:
    """Tick count, age limit and proposals are validated."""
    malformed: tuple[Any, ...] = (0, True, 1.5)
    for ticks in malformed:
        with pytest.raises(ValueError, match="n_ticks"):
            run_gated_actuation_loop(
                ticks,
                lambda index: ActuationProposal((0.0,), 0.0),
                config=_CONFIG,
                max_telemetry_age_s=1.0,
                veto=_no_veto,
            )
    with pytest.raises(ValueError, match="max_telemetry_age_s"):
        run_gated_actuation_loop(
            1,
            lambda index: ActuationProposal((0.0,), 0.0),
            config=_CONFIG,
            max_telemetry_age_s=0.0,
            veto=_no_veto,
        )
    with pytest.raises(ValueError, match="non-empty and finite"):
        ActuationProposal((), 0.0)
    with pytest.raises(ValueError, match="non-empty and finite"):
        ActuationProposal((float("nan"),), 0.0)
    with pytest.raises(ValueError, match="telemetry_time_s"):
        ActuationProposal((0.0,), float("inf"))
