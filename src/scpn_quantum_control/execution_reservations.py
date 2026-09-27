# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — owned local execution memory reservations
"""Serialize declared-memory charges across cooperating callers in one process.

These scopes do not reserve operating-system pages or coordinate independent
processes. Cancellation and deadlines are checked cooperatively; a native call
already executing is not forcibly terminated. Disposal releases the charge only
when the owning scope actually exits.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
from threading import Event, RLock, get_ident
from time import monotonic

from .dense_budget import DenseAllocationError, cgroup_headroom_bytes, host_available_memory_bytes
from .execution_memory import (
    ExecutionBuffer,
    ExecutionMemoryDecision,
    ExecutionMemoryPlan,
    MemoryCapacity,
    check_execution_memory,
    require_execution_memory,
)


class ExecutionCancelledError(RuntimeError):
    """An owner observed cancellation at a cooperative execution checkpoint."""


class _Ledger:
    def __init__(self) -> None:
        self.lock = RLock()
        self.entries: dict[object, tuple[int, int]] = {}


_ledger = _Ledger()
_current_reservation: ContextVar[ExecutionMemoryReservation | None] = ContextVar(
    "execution_memory_reservation", default=None
)


def _after_fork() -> None:
    global _ledger
    _ledger = _Ledger()
    _current_reservation.set(None)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def active_reserved_bytes() -> int:
    """Return currently charged bytes in this process, under the ledger lock."""
    with _ledger.lock:
        return sum(size for size, _ in _ledger.entries.values())


class ExecutionMemoryReservation:
    """Owned declared-memory charge created by ``reserve_execution_memory``.

    Parameters
    ----------
    decision
        Accepted live-capacity admission snapshot.
    deadline_monotonic
        Absolute monotonic deadline, or no deadline.
    cancelled
        Caller-owned cancellation event, or no cancellation signal.
    cgroup_root
        Controller snapshot root to reread at charge changes, or process discovery.

    """

    def __init__(
        self,
        decision: ExecutionMemoryDecision,
        deadline_monotonic: float | None,
        cancelled: Event | None,
        *,
        cgroup_root: Path | None = None,
    ) -> None:
        if deadline_monotonic is not None and (
            isinstance(deadline_monotonic, bool) or not math.isfinite(deadline_monotonic)
        ):
            raise ValueError("deadline must be finite monotonic seconds")
        if not decision.allowed:
            raise DenseAllocationError("refused admission cannot create a reservation")
        self.decision = decision
        self._cgroup_root = cgroup_root
        self._deadline = deadline_monotonic
        self._cancelled = cancelled
        self._pid = os.getpid()
        self._thread = get_ident()
        self._token = object()
        self._closed = False
        self._parent = _current_reservation.get()
        self.checkpoint()
        self._charge(decision.bytes_required)

    def checkpoint(self) -> None:
        """Refuse closed/foreign ownership, cancellation or an expired deadline.

        Raises
        ------
        RuntimeError
            Closed scope or a caller other than its owning process/thread.
        ExecutionCancelledError
            The owner's cancellation signal was observed.
        TimeoutError
            The absolute monotonic deadline has elapsed.

        """
        current: ExecutionMemoryReservation | None = self
        while current is not None:
            current._require_owner()
            if current._closed:
                raise RuntimeError("execution reservation is closed")
            if current._cancelled is not None and current._cancelled.is_set():
                raise ExecutionCancelledError("execution cancellation observed")
            if current._deadline is not None and monotonic() >= current._deadline:
                raise TimeoutError("execution deadline elapsed")
            current = current._parent

    def resize(self, plan: ExecutionMemoryPlan) -> None:
        """Refresh host/controller headroom and atomically replace the live charge.

        A successful update retains the tighter prior cap and refreshes the
        decision. Refusal leaves both the prior charge and decision unchanged.
        Caller-supplied device capacity remains its original observation.

        Parameters
        ----------
        plan
            Replacement declaration, including all buffers still live.

        Raises
        ------
        DenseAllocationError
            The replacement plus active peers exceeds the shared allowance.

        """
        self.checkpoint()
        self._charge(plan.bytes_required)

    def handoff(self, destination: ExecutionMemoryReservation, plan: ExecutionMemoryPlan) -> None:
        """Atomically retain this scope's charge in an active ancestor.

        Success closes the source and restores its active parent context.
        Both owners refresh host/controller headroom before the transfer. The
        destination retains the tightest prior and refreshed cap.

        Parameters
        ----------
        destination
            Live ancestor owned by the same process and thread.
        plan
            Complete destination plan including both existing declared charges.

        Raises
        ------
        ValueError
            Destination is not an ancestor or the plan drops declared live bytes.
        DenseAllocationError
            Combined declaration exceeds the tightest active allowance.

        """
        self.checkpoint()
        destination.checkpoint()
        ancestor = self._parent
        while ancestor is not None and ancestor is not destination:
            ancestor = ancestor._parent
        if ancestor is None:
            raise ValueError("handoff destination must be an active ancestor")
        if _current_reservation.get() is not self:
            raise ValueError("handoff source must be the current active scope")
        required = plan.bytes_required
        with _ledger.lock:
            source_bytes, source_cap = _ledger.entries[self._token]
            destination_bytes, destination_cap = _ledger.entries[destination._token]
            if required < source_bytes + destination_bytes:
                raise ValueError("handoff plan drops declared live bytes")
            others = [
                entry
                for token, entry in _ledger.entries.items()
                if token not in (self._token, destination._token)
            ]
            source_snapshot = self._refresh(required)
            destination_snapshot = destination._refresh(required)
            if not source_snapshot.allowed or not destination_snapshot.allowed:
                raise DenseAllocationError("handoff exceeds refreshed execution capacity")
            cap = min(
                source_cap,
                destination_cap,
                source_snapshot.budget_bytes,
                destination_snapshot.budget_bytes,
            )
            budget = min(cap, *(limit for _, limit in others)) if others else cap
            if required + sum(size for size, _ in others) > budget:
                raise DenseAllocationError("handoff exceeds shared execution memory")
            decision = replace(destination_snapshot, budget_bytes=cap)
            _ledger.entries[destination._token] = (required, cap)
            del _ledger.entries[self._token]
            destination.decision = decision
            self._closed = True
            _current_reservation.set(self._parent)

    def _require_owner(self) -> None:
        if os.getpid() != self._pid or get_ident() != self._thread:
            raise RuntimeError("execution reservation owner mismatch")

    def _refresh(self, size: int) -> ExecutionMemoryDecision:
        capacity = MemoryCapacity(
            host_available_memory_bytes(),
            cgroup_headroom_bytes(self._cgroup_root),
            self.decision.capacity.device_available_bytes,
        )
        plan = ExecutionMemoryPlan(
            (ExecutionBuffer("reservation", "intermediate", (size,), "uint8"),)
        )
        return check_execution_memory(
            plan,
            capacity,
            max_bytes=self.decision.budget_bytes,
            require_device=capacity.device_available_bytes is not None,
        )

    def _charge(self, size: int) -> None:
        with _ledger.lock:
            decision = self._refresh(size)
            if not decision.allowed:
                raise DenseAllocationError(
                    "concurrent execution memory exceeds refreshed capacity: "
                    + ", ".join(decision.blockers)
                )
            others = [
                entry for token, entry in _ledger.entries.items() if token is not self._token
            ]
            budget = (
                min(decision.budget_bytes, *(cap for _, cap in others))
                if others
                else decision.budget_bytes
            )
            total = size + sum(amount for amount, _ in others)
            if total > budget:
                raise DenseAllocationError("concurrent execution memory exceeds shared allowance")
            _ledger.entries[self._token] = (size, decision.budget_bytes)
            self.decision = decision

    def _close(self) -> None:
        self._require_owner()
        with _ledger.lock:
            if self._closed:
                return
            _ledger.entries.pop(self._token)
            self._closed = True


@contextmanager
def reserve_execution_memory(
    plan: ExecutionMemoryPlan,
    *,
    max_gib: float | None = None,
    cgroup_root: Path | None = None,
    device_available_bytes: int | None = None,
    require_device: bool = False,
    native_symbol: str | None = None,
    deadline_monotonic: float | None = None,
    cancelled: Event | None = None,
) -> Iterator[ExecutionMemoryReservation]:
    """Admit, charge and dispose a real local execution scope.

    Nested scopes check every active ancestor's lifecycle policy. A child cannot
    clear its parent's cancellation or extend its parent's deadline. Disposal
    restores the prior context even when the child exits with an exception.

    Parameters
    ----------
    plan
        All simultaneously live declared buffers for this execution.
    max_gib
        Requested cap constrained by the actual process capacity snapshot.
    cgroup_root
        Optional explicit controller root; default discovers process membership.
    device_available_bytes
        Actual observation from the device-owning caller, or unknown.
    require_device
        Refuse absent required device headroom.
    native_symbol
        Required native callable, or no native requirement.
    deadline_monotonic
        Absolute monotonic deadline checked before entry and at checkpoints.
    cancelled
        Cancellation event checked before entry and at checkpoints.

    Yields
    ------
    ExecutionMemoryReservation
        Process/thread-owned charge, released on success or any escaping exception.

    """
    decision = require_execution_memory(
        plan,
        max_gib=max_gib,
        cgroup_root=cgroup_root,
        device_available_bytes=device_available_bytes,
        require_device=require_device,
        native_symbol=native_symbol,
    )
    reservation = ExecutionMemoryReservation(
        decision, deadline_monotonic, cancelled, cgroup_root=cgroup_root
    )
    context_token = _current_reservation.set(reservation)
    try:
        yield reservation
    finally:
        try:
            reservation._close()
        finally:
            _current_reservation.reset(context_token)
