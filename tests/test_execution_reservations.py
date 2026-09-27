# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — execution reservation lifecycle tests
"""Exercise live reservation ownership, contention and release on public scopes."""

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from threading import Event
from time import monotonic

import pytest

from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.execution_memory import ExecutionBuffer, ExecutionMemoryPlan
from scpn_quantum_control.execution_reservations import (
    ExecutionCancelledError,
    active_reserved_bytes,
    reserve_execution_memory,
)


def _plan(size: int) -> ExecutionMemoryPlan:
    return ExecutionMemoryPlan((ExecutionBuffer("state", "forward", (size,), "uint8"),))


def test_concurrent_scope_cannot_reuse_reserved_allowance() -> None:
    """Real parallel callers contend for one shared byte allowance."""
    baseline = active_reserved_bytes()
    with reserve_execution_memory(_plan(80), max_gib=128 / 1024**3):
        assert active_reserved_bytes() == baseline + 80
        with ThreadPoolExecutor(max_workers=1) as pool:

            def competing_call() -> None:
                with reserve_execution_memory(_plan(80), max_gib=128 / 1024**3):
                    raise AssertionError("overcommitted reservation admitted")

            with pytest.raises(DenseAllocationError, match="concurrent"):
                pool.submit(competing_call).result(timeout=5)
        assert active_reserved_bytes() == baseline + 80
    assert active_reserved_bytes() == baseline


def test_growing_plan_is_charged_atomically_and_refusal_preserves_prior_charge() -> None:
    """A retained-tape update cannot overwrite its charge when total admission fails."""
    baseline = active_reserved_bytes()
    with reserve_execution_memory(_plan(32), max_gib=128 / 1024**3) as reservation:
        reservation.resize(_plan(80))
        assert active_reserved_bytes() == baseline + 80
        with pytest.raises(DenseAllocationError, match="concurrent"):
            reservation.resize(_plan(129))
        assert active_reserved_bytes() == baseline + 80
        reservation.resize(_plan(16))
        assert active_reserved_bytes() == baseline + 16
    assert active_reserved_bytes() == baseline


def test_exception_releases_charge_and_closed_scope_rejects_reuse() -> None:
    """An interrupted computation releases its reservation without masking its error."""
    baseline = active_reserved_bytes()
    with (
        pytest.raises(KeyboardInterrupt),
        reserve_execution_memory(_plan(32), max_gib=128 / 1024**3) as reservation,
    ):
        raise KeyboardInterrupt
    assert active_reserved_bytes() == baseline
    with pytest.raises(RuntimeError, match="closed"):
        reservation.checkpoint()
    with pytest.raises(RuntimeError, match="closed"):
        reservation.resize(_plan(16))


def test_deadline_and_cancel_refuse_without_retaining_charge() -> None:
    """Expired and cancelled scopes cannot enter or outlive disposal."""
    baseline = active_reserved_bytes()
    cancelled = Event()
    cancelled.set()
    with (
        pytest.raises(ExecutionCancelledError),
        reserve_execution_memory(_plan(32), cancelled=cancelled),
    ):
        raise AssertionError("cancelled scope entered")
    with (
        pytest.raises(TimeoutError),
        reserve_execution_memory(_plan(32), deadline_monotonic=monotonic() - 1),
    ):
        raise AssertionError("expired scope entered")
    cancelled.clear()
    with (
        pytest.raises(ExecutionCancelledError),
        reserve_execution_memory(_plan(32), cancelled=cancelled) as reservation,
    ):
        cancelled.set()
        reservation.checkpoint()
    assert active_reserved_bytes() == baseline


def test_other_thread_cannot_operate_an_owned_reservation() -> None:
    """A reservation belongs to its originating caller, not any holder of its object."""
    with reserve_execution_memory(_plan(32)) as reservation:
        with ThreadPoolExecutor(max_workers=1) as pool, pytest.raises(RuntimeError, match="owner"):
            pool.submit(reservation.checkpoint).result(timeout=5)
        reservation.checkpoint()


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), True])
def test_malformed_deadline_refuses(deadline: float) -> None:
    """Nonfinite and boolean deadline metadata never authorize a live scope."""
    with (
        pytest.raises(ValueError, match="deadline"),
        reserve_execution_memory(_plan(32), deadline_monotonic=deadline),
    ):
        raise AssertionError("invalid deadline admitted")


def test_fork_resets_child_ledger_and_rejects_inherited_owner() -> None:
    """A real child cannot reuse parent ownership or release the parent's charge."""
    context = get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    baseline = active_reserved_bytes()
    with reserve_execution_memory(_plan(80), max_gib=128 / 1024**3) as reservation:

        def child() -> None:
            receiver.close()
            with pytest.raises(RuntimeError, match="owner"):
                reservation.checkpoint()
            with pytest.raises(RuntimeError, match="owner"):
                reservation.resize(_plan(16))
            assert active_reserved_bytes() == 0
            with reserve_execution_memory(_plan(32), max_gib=128 / 1024**3):
                assert active_reserved_bytes() == 32
            assert active_reserved_bytes() == 0
            sender.send("child scope released")
            sender.close()

        process = context.Process(target=child)
        process.start()
        sender.close()
        try:
            assert receiver.poll(10), "fork child did not report lifecycle completion"
            assert receiver.recv() == "child scope released"
            process.join(timeout=10)
            assert process.exitcode == 0
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            receiver.close()
        reservation.checkpoint()
        assert active_reserved_bytes() == baseline + 80
    assert active_reserved_bytes() == baseline


def test_nested_scopes_cannot_override_parent_cancellation() -> None:
    """Independent child policy still observes an active ancestor's cancellation."""
    baseline = active_reserved_bytes()
    parent_cancelled = Event()
    child_cancelled = Event()
    with reserve_execution_memory(_plan(16), cancelled=parent_cancelled):
        with (
            reserve_execution_memory(_plan(16), cancelled=child_cancelled) as child,
            reserve_execution_memory(_plan(16)) as grandchild,
        ):
            parent_cancelled.set()
            with pytest.raises(ExecutionCancelledError):
                grandchild.checkpoint()
            with pytest.raises(ExecutionCancelledError):
                child.resize(_plan(32))
            assert active_reserved_bytes() == baseline + 48
        parent_cancelled.clear()
        with reserve_execution_memory(_plan(16)) as sibling:
            sibling.checkpoint()
    assert active_reserved_bytes() == baseline


def test_nested_exception_restores_parent_context_without_child_policy_leak() -> None:
    """An interrupted child releases its charge and cannot contaminate later siblings."""
    baseline = active_reserved_bytes()
    cancelled = Event()
    with reserve_execution_memory(_plan(16)) as parent:
        with (
            pytest.raises(ExecutionCancelledError),
            reserve_execution_memory(_plan(16), cancelled=cancelled) as child,
        ):
            cancelled.set()
            child.checkpoint()
        parent.checkpoint()
        with reserve_execution_memory(_plan(16)) as sibling:
            sibling.checkpoint()
        assert active_reserved_bytes() == baseline + 16
    with reserve_execution_memory(_plan(16)) as next_scope:
        next_scope.checkpoint()
    assert active_reserved_bytes() == baseline


def test_nested_scope_cannot_extend_parent_deadline() -> None:
    """A child's later deadline cannot authorize work after its parent expires."""
    baseline = active_reserved_bytes()
    parent_deadline = monotonic() + 2
    with (
        reserve_execution_memory(_plan(16), deadline_monotonic=parent_deadline),
        reserve_execution_memory(_plan(16), deadline_monotonic=parent_deadline + 60) as child,
    ):
        Event().wait(max(0, parent_deadline - monotonic()))
        with pytest.raises(TimeoutError):
            child.checkpoint()
    assert active_reserved_bytes() == baseline


def test_handoff_retains_charge_in_parent_without_unreserved_gap() -> None:
    """A child transfers declared live bytes and its tighter cap atomically."""
    baseline = active_reserved_bytes()
    with reserve_execution_memory(_plan(16), max_gib=256 / 1024**3) as parent:
        with reserve_execution_memory(_plan(32), max_gib=128 / 1024**3) as child:
            with pytest.raises(ValueError, match="drops"):
                child.handoff(parent, _plan(47))
            assert active_reserved_bytes() == baseline + 48
            with pytest.raises(DenseAllocationError, match="handoff"):
                child.handoff(parent, _plan(129))
            assert active_reserved_bytes() == baseline + 48
            child.handoff(parent, _plan(48))
            assert active_reserved_bytes() == baseline + 48
            assert parent.decision.budget_bytes == 128
            with reserve_execution_memory(_plan(16)) as sibling:
                sibling.checkpoint()
                assert active_reserved_bytes() == baseline + 64
            with pytest.raises(RuntimeError, match="closed"):
                child.checkpoint()
        assert active_reserved_bytes() == baseline + 48
        parent.resize(_plan(64))
        with pytest.raises(DenseAllocationError, match="concurrent"):
            parent.resize(_plan(129))
    assert active_reserved_bytes() == baseline


def test_handoff_rejects_nonancestor_destination() -> None:
    """An unrelated or descendant owner cannot acquire a scope's declaration."""
    with (
        reserve_execution_memory(_plan(16)) as parent,
        reserve_execution_memory(_plan(16)) as child,
    ):
        with pytest.raises(ValueError, match="ancestor"):
            parent.handoff(child, _plan(32))
        with pytest.raises(ValueError, match="ancestor"):
            child.handoff(child, _plan(32))


def test_handoff_respects_another_live_scopes_tighter_allowance() -> None:
    """Atomic handoff keeps a peer's cap and leaves both charges intact on refusal."""
    baseline = active_reserved_bytes()
    with (
        reserve_execution_memory(_plan(16), max_gib=128 / 1024**3) as parent,
        reserve_execution_memory(_plan(16), max_gib=64 / 1024**3),
        reserve_execution_memory(_plan(16), max_gib=128 / 1024**3) as child,
    ):
        with pytest.raises(DenseAllocationError, match="handoff"):
            child.handoff(parent, _plan(49))
        assert active_reserved_bytes() == baseline + 48
        child.handoff(parent, _plan(48))
        assert active_reserved_bytes() == baseline + 64
    assert active_reserved_bytes() == baseline


def test_handoff_cannot_close_an_owner_with_active_descendants() -> None:
    """Closing an ancestor would invalidate live children, so handoff refuses first."""
    baseline = active_reserved_bytes()
    with (
        reserve_execution_memory(_plan(16)) as parent,
        reserve_execution_memory(_plan(16)) as middle,
        reserve_execution_memory(_plan(16)) as child,
    ):
        with pytest.raises(ValueError, match="current"):
            middle.handoff(parent, _plan(32))
        child.checkpoint()
        assert active_reserved_bytes() == baseline + 48
    assert active_reserved_bytes() == baseline


def test_resize_refreshes_controller_usage_and_preserves_charge_on_refusal(tmp_path: Path) -> None:
    """Controller usage changes constrain a live owner's next allocation."""
    (tmp_path / "memory.max").write_text("1000")
    usage = tmp_path / "memory.current"
    usage.write_text("0")
    baseline = active_reserved_bytes()
    with reserve_execution_memory(_plan(32), cgroup_root=tmp_path, max_gib=300 / 1024**3) as scope:
        usage.write_text("900")
        with pytest.raises(DenseAllocationError, match="concurrent"):
            scope.resize(_plan(64))
        assert active_reserved_bytes() == baseline + 32
        assert scope.decision.bytes_required == 32
        assert scope.decision.budget_bytes == 300
        usage.write_text("700")
        scope.resize(_plan(64))
        assert scope.decision.bytes_required == 64
        assert scope.decision.budget_bytes == 90
        assert scope.decision.capacity.cgroup_available_bytes == 300
        usage.write_text("0")
        with pytest.raises(DenseAllocationError, match="concurrent"):
            scope.resize(_plan(91))
        assert active_reserved_bytes() == baseline + 64
    assert active_reserved_bytes() == baseline
    with reserve_execution_memory(_plan(91), cgroup_root=tmp_path, max_gib=300 / 1024**3):
        assert active_reserved_bytes() == baseline + 91
    assert active_reserved_bytes() == baseline


def test_resize_refresh_includes_peer_charges(tmp_path: Path) -> None:
    """Fresh headroom cannot be spent again while another caller holds its charge."""
    (tmp_path / "memory.max").write_text("1000")
    usage = tmp_path / "memory.current"
    usage.write_text("0")
    baseline = active_reserved_bytes()
    with reserve_execution_memory(_plan(16), cgroup_root=tmp_path, max_gib=300 / 1024**3) as scope:
        with reserve_execution_memory(_plan(80), max_gib=300 / 1024**3):
            usage.write_text("700")
            with pytest.raises(DenseAllocationError, match="concurrent"):
                scope.resize(_plan(16))
            assert active_reserved_bytes() == baseline + 96
            assert scope.decision.budget_bytes == 300
        scope.resize(_plan(16))
        assert scope.decision.budget_bytes == 90
        assert active_reserved_bytes() == baseline + 16
    assert active_reserved_bytes() == baseline


def test_handoff_refreshes_both_owners_controller_headroom(tmp_path: Path) -> None:
    """Transfer cannot retain bytes under either owner's stale controller allowance."""
    parent_root = tmp_path / "parent"
    child_root = tmp_path / "child"
    for root in (parent_root, child_root):
        root.mkdir()
        (root / "memory.max").write_text("1000")
        (root / "memory.current").write_text("0")
    baseline = active_reserved_bytes()
    with reserve_execution_memory(
        _plan(16), cgroup_root=parent_root, max_gib=300 / 1024**3
    ) as parent:
        with reserve_execution_memory(
            _plan(32), cgroup_root=child_root, max_gib=300 / 1024**3
        ) as child:
            for root in (parent_root, child_root):
                (root / "memory.current").write_text("900")
                with pytest.raises(DenseAllocationError, match="handoff"):
                    child.handoff(parent, _plan(48))
                assert active_reserved_bytes() == baseline + 48
                assert parent.decision.bytes_required == 16
                assert child.decision.bytes_required == 32
                (root / "memory.current").write_text("0")
            (child_root / "memory.current").write_text("700")
            child.handoff(parent, _plan(48))
            assert parent.decision.bytes_required == 48
            assert parent.decision.budget_bytes == 90
            assert active_reserved_bytes() == baseline + 48
        with pytest.raises(DenseAllocationError, match="concurrent"):
            parent.resize(_plan(91))
    assert active_reserved_bytes() == baseline
