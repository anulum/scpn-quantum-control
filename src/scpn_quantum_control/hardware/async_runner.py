# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Async hardware runner
# Language policy: EXEMPT from the Rust-path rule. This module is an
# asyncio-based hardware-bridge / I/O adapter over the IBM Python
# client. The compute happens inside Qiskit and IBM's cloud; the role
# here is orchestration. See docs/language_policy.md §"Current-state
# audit" and feedback_rustify_all.md (memory) carve-out.
"""Concurrent IBM job submission via asyncio.

Closes audit item C13. The synchronous :class:`HardwareRunner` submits
one circuit batch at a time; Phase 2 campaigns fan out across multiple
IBM instances and would benefit from parallel submission. This module
provides :class:`AsyncHardwareRunner`, a thin async wrapper that keeps
the legacy sync path untouched while exposing:

* :meth:`submit_one_async` — wrap a single ``sampler.run(...)`` call in
  a coroutine that returns a job handle without blocking the event
  loop.
* :meth:`submit_batch_async` — fan out many sub-batches across
  (optionally) several :class:`HardwareRunner` instances using
  ``asyncio.gather``. Concurrency is bounded by a semaphore so that an
  unbounded-fan-out does not overwhelm IBM's rate limits.
* :meth:`wait_for_job_async` — poll a submitted job via
  ``asyncio.to_thread`` so ``job.result()`` does not block the loop.

The implementation is deliberately pure-async plus ``to_thread`` — no
``aiohttp`` or custom reactor. IBM's Python client is thread-safe
synchronously, so wrapping it with ``to_thread`` is the correct way to
concurrency without rewriting the library. Cancellation propagates via
the standard ``asyncio.CancelledError`` mechanics.

Tests exercise the class with a mock Sampler / Service, so CI does not
need an IBM token. Real hardware usage is the same API surface.

Usage
-----

.. code-block:: python

    import asyncio
    from scpn_quantum_control.hardware.async_runner import AsyncHardwareRunner

    async def main():
        runners = [HardwareRunner(...) for _ in range(3)]
        async_runner = AsyncHardwareRunner(runners, max_concurrent=3)
        results = await async_runner.submit_batch_async(
            circuits_per_instance=[c_a, c_b, c_c],
            shots=4096,
            name="dla_parity",
        )

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .runner import HardwareRunner, JobResult


def _get_logger(name: str) -> Any:
    try:
        from ..logging_setup import get_logger

        return get_logger(name)
    except Exception:
        return logging.getLogger(name)


_log = _get_logger(__name__)


@dataclass
class AsyncJobHandle:
    """Opaque handle returned by :meth:`AsyncHardwareRunner.submit_one_async`.

    Wraps the IBM job object together with the owning runner so callers
    can retrieve the result or cancel the job without needing to know
    which underlying :class:`HardwareRunner` submitted it.
    """

    job_id: str
    runner: HardwareRunner
    experiment: str
    submitted_at: float = field(default_factory=time.time)
    _job: Any = None


class AsyncHardwareRunner:
    """Concurrent driver over one or more :class:`HardwareRunner` instances.

    Parameters
    ----------
    runners:
        One or more already-connected :class:`HardwareRunner` instances.
        The constructor accepts a single runner for the common single-
        instance case.
    max_concurrent:
        Upper bound on the number of simultaneously in-flight
        submissions. Defaults to ``len(runners)``. Set to a lower number
        to leave headroom for other jobs on the same instance; set
        higher (up to the provider's quota) for Phase 2 fan-outs.
    """

    def __init__(
        self,
        runners: HardwareRunner | list[HardwareRunner],
        *,
        max_concurrent: int | None = None,
    ) -> None:
        # Accept either a single runner or any iterable of runners.
        # Duck-typing on ``__iter__`` rather than ``isinstance(...,
        # HardwareRunner)`` keeps tests and downstream extensions that
        # provide their own runner-like class from having to subclass.
        if hasattr(runners, "backend_name") and not hasattr(runners, "__iter__"):
            runners = [runners]  # type: ignore[list-item]
        elif isinstance(runners, (list, tuple)):
            runners = list(runners)
        else:
            # Single-object path when the object happens to be iterable
            # (unlikely for HardwareRunner, but keep the strict fallback
            # for correctness).
            runners = [runners] if not isinstance(runners, list) else runners
        if not runners:
            raise ValueError("AsyncHardwareRunner needs at least one HardwareRunner")
        self._runners: list[HardwareRunner] = list(runners)
        # Distinguish None (take the default) from 0 (explicit, invalid).
        self._max_concurrent: int = (
            max_concurrent if max_concurrent is not None else len(self._runners)
        )
        if self._max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self._rr_index = 0  # round-robin across runners

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _next_runner(self) -> HardwareRunner:
        """Round-robin pick of the next runner."""
        runner = self._runners[self._rr_index % len(self._runners)]
        self._rr_index += 1
        return runner

    def _submit_blocking(
        self,
        runner: HardwareRunner,
        circuits: list[Any],
        shots: int,
        name: str,
    ) -> AsyncJobHandle:
        """Run the blocking SamplerV2.run call. Called inside ``to_thread``."""
        from qiskit_ibm_runtime import SamplerV2 as Sampler

        isa = [runner.transpile(c) for c in circuits]

        sampler = Sampler(mode=runner._backend)  # noqa: SLF001
        sampler.options.default_shots = shots

        job = sampler.run(isa)
        job_id = job.job_id()
        _log.info(
            "async_job_submitted",
            job_id=job_id,
            backend=runner.backend_name,
            experiment=name,
            shots=shots,
            n_circuits=len(isa),
        )
        runner._log_job(job_id, name)  # noqa: SLF001
        return AsyncJobHandle(
            job_id=job_id,
            runner=runner,
            experiment=name,
            _job=job,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit_one_async(
        self,
        circuits: list[Any],
        *,
        shots: int = 4096,
        name: str = "async_experiment",
        runner: HardwareRunner | None = None,
    ) -> AsyncJobHandle:
        """Submit a single sub-batch and return its :class:`AsyncJobHandle`.

        The actual ``sampler.run(...)`` call happens inside
        ``asyncio.to_thread`` so the event loop stays responsive.
        """
        chosen = runner or self._next_runner()
        async with self._semaphore:
            return await asyncio.to_thread(
                self._submit_blocking,
                chosen,
                circuits,
                shots,
                name,
            )

    async def submit_batch_async(
        self,
        circuits_per_instance: list[list[Any]],
        *,
        shots: int = 4096,
        name: str = "async_batch",
    ) -> list[AsyncJobHandle]:
        """Fan out ``circuits_per_instance`` concurrently across runners.

        Each sub-list becomes one submission; sub-list *i* is dispatched
        to runner ``i % len(runners)`` unless constrained further by
        the ``max_concurrent`` semaphore.
        """
        coros = [
            self.submit_one_async(
                batch,
                shots=shots,
                name=f"{name}_{idx}",
            )
            for idx, batch in enumerate(circuits_per_instance)
        ]
        return list(await asyncio.gather(*coros))

    async def wait_for_job_async(
        self,
        handle: AsyncJobHandle,
        *,
        timeout_s: float = 600,
    ) -> list[JobResult]:
        """Await the completion of a previously submitted job.

        The blocking ``job.result(timeout=...)`` runs inside
        ``asyncio.to_thread`` so callers can ``gather`` on many
        in-flight jobs at once.
        """
        if handle._job is None:
            raise RuntimeError("AsyncJobHandle has no underlying job object")

        from datetime import datetime

        from .runner import _extract_counts

        def _collect() -> list[JobResult]:
            result = handle._job.result(timeout=timeout_s)
            wall = time.time() - handle.submitted_at
            out: list[JobResult] = []
            for i, pub_result in enumerate(result):
                counts = _extract_counts(pub_result)
                out.append(
                    JobResult(
                        job_id=handle.job_id,
                        backend_name=handle.runner.backend_name,
                        experiment_name=f"{handle.experiment}_{i}",
                        counts=counts,
                        wall_time_s=wall,
                        timestamp=datetime.now().isoformat(),
                        metadata={},
                    ),
                )
            return out

        return await asyncio.to_thread(_collect)

    async def wait_all_async(
        self,
        handles: list[AsyncJobHandle],
        *,
        timeout_s: float = 600,
    ) -> list[list[JobResult]]:
        """Gather results for all handles in parallel."""
        return list(
            await asyncio.gather(
                *[self.wait_for_job_async(h, timeout_s=timeout_s) for h in handles],
            ),
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    @property
    def n_runners(self) -> int:
        return len(self._runners)


__all__ = [
    "AsyncHardwareRunner",
    "AsyncJobHandle",
]
