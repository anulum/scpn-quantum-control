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
        runners: HardwareRunner | list[HardwareRunner] | None = None,
        *,
        max_concurrent: int | None = None,
        backend: str = "ibm_heron_r2",
        shots: int = 4096,
        mitigation: str = "GUESS",
        **kwargs,
    ) -> None:
        self.backend = backend
        self.default_shots = shots
        self.mitigation = mitigation
        self.runner_kwargs = kwargs

        # Accept either a single runner or any iterable of runners.
        if runners is not None:
            if hasattr(runners, "backend_name") and not hasattr(runners, "__iter__"):
                runners = [runners]  # type: ignore[list-item]
            elif isinstance(runners, (list, tuple)):
                runners = list(runners)
            else:
                runners = [runners] if not isinstance(runners, list) else runners
            self._runners: list[HardwareRunner] = list(runners)
            if not self._runners:
                raise ValueError("AsyncHardwareRunner requires at least one runner")
        else:
            self._runners = []

        self._max_concurrent: int = (
            max_concurrent
            if max_concurrent is not None
            else (len(self._runners) if self._runners else 10)
        )
        if self._max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self._rr_index = 0

    # round-robin across runners

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

    def submit_circuit_batch(self, ansatz, observable, **kwargs):
        """Submit ``StructuredAnsatz`` jobs without fabricating unfinished results.

        Real QPU submissions return real job identifiers immediately when
        queued. Observables are evaluated only when real counts are available.
        Local simulation and ZNE are opt-in because both change the scientific
        meaning and resource profile of a campaign.
        """

        class JobWrapper:
            def __init__(self, runner_obj, ansatz, observable, kwargs):
                self.runner_obj = runner_obj
                self.ansatz = ansatz
                self.observable = observable
                self.kwargs = kwargs
                self.job_id = None
                self.submitted_at = time.time()

            def _run_blocking(self):
                import os

                from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling
                from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
                from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

                zne_sync_order = None  # set on IBM path if explicit ZNE succeeds
                ibm_job_ids: list[str] = []
                zne_job_ids: list[str] = []
                counts = None
                status = "NOT_SUBMITTED"
                shots = self.kwargs.get("shots", self.runner_obj.default_shots)
                allow_local_simulation = bool(
                    self.kwargs.get(
                        "allow_local_simulation",
                        self.runner_obj.runner_kwargs.get("allow_local_simulation", False),
                    )
                )
                enable_zne = bool(
                    self.kwargs.get(
                        "enable_zne",
                        self.runner_obj.runner_kwargs.get("enable_zne", False),
                    )
                )
                qc = self.ansatz.build_circuit()
                if qc.num_clbits == 0:
                    qc.measure_all()

                token = os.environ.get("SCPN_IBM_TOKEN")
                crn = os.environ.get(
                    "SCPN_IBM_CRN",
                    "crn:v1:bluemix:public:quantum-computing:us-east:"
                    "a/78db885720334fd19191b33a839d0c35:"
                    "841cc36d-0afd-4f96-ada2-8c56e1c443a0::",
                )

                try:
                    if token:
                        service = QiskitRuntimeService(
                            channel="ibm_cloud", token=token, instance=crn
                        )
                        target = (
                            "ibm_fez"
                            if self.runner_obj.backend == "ibm_heron_r2"
                            else self.runner_obj.backend
                        )
                        try:
                            backend = service.backend(target)
                        except Exception:
                            backend = service.least_busy(simulator=False, operational=True)

                        # Basic error mitigation: high optimization + dynamical decoupling
                        pm = generate_preset_pass_manager(
                            optimization_level=3, backend=backend, seed_transpiler=42
                        )
                        isa_qc = pm.run(qc)

                        # Dynamical decoupling — Qiskit 2.4.0 verified pattern:
                        # ALAPScheduleAnalysis must run first (in PassManager) to
                        # annotate the DAG with timing before PadDynamicalDecoupling.
                        try:
                            from qiskit.circuit.library import XGate
                            from qiskit.transpiler import PassManager

                            durations = backend.target.durations()
                            dd_pm = PassManager(
                                [
                                    ALAPScheduleAnalysis(
                                        durations=durations,
                                        target=backend.target,
                                    ),
                                    PadDynamicalDecoupling(
                                        durations=durations,
                                        dd_sequence=[XGate(), XGate()],
                                        target=backend.target,
                                    ),
                                ]
                            )
                            isa_qc = dd_pm.run(isa_qc)
                        except Exception as dd_err:
                            print(f"DD skipped ({dd_err}); submitting without DD.", flush=True)

                        sampler = SamplerV2(mode=backend)
                        sampler.options.default_shots = min(shots, 4000)

                        print(
                            "IBM Runtime: Dispatching circuit with DD + opt_level=3"
                            f" to {backend.name}...",
                            flush=True,
                        )

                        if enable_zne:
                            # Mitiq ZNE must operate on scalar observables, not
                            # counts dictionaries. This path blocks until each
                            # scaled job returns and records every IBM job id.
                            try:
                                from mitiq import zne
                                from mitiq.zne.inference import RichardsonFactory
                                from mitiq.zne.scaling import fold_global

                                from scpn_quantum_control.analysis import SyncOrderParameter

                                def _zne_executor(circ):
                                    """Run scaled circuit and return sync_order for extrapolation."""
                                    _job = sampler.run([circ])
                                    zne_job_id = str(_job.job_id())
                                    ibm_job_ids.append(zne_job_id)
                                    zne_job_ids.append(zne_job_id)
                                    _res = _job.result()
                                    _counts = _res[0].data.meas.get_counts()
                                    return SyncOrderParameter()(counts=_counts)["sync_order"]

                                zne_sync_order = zne.execute_with_zne(
                                    isa_qc,
                                    _zne_executor,
                                    factory=RichardsonFactory([1, 2, 3]),
                                    scale_noise=fold_global,
                                )
                                print(
                                    f"ZNE complete: extrapolated sync_order={zne_sync_order:.4f}",
                                    flush=True,
                                )
                            except Exception as zne_err:
                                print(
                                    f"ZNE skipped ({zne_err}); running unmitigated.",
                                    flush=True,
                                )

                        # Scale=1 run — collects full counts for all observables
                        job = sampler.run([isa_qc])
                        self.job_id = str(job.job_id())
                        ibm_job_ids.append(self.job_id)
                        print(f"IBM Runtime: Job queued -> {self.job_id}", flush=True)
                        status = "QUEUED_ON_IBM"
                        try:
                            res = job.result(timeout=15)
                            counts = res[0].data.meas.get_counts()
                            status = "DONE"
                        except Exception:
                            print(f"Job {self.job_id} still queued on IBM.", flush=True)
                    elif allow_local_simulation:
                        from qiskit.primitives import StatevectorSampler

                        sampler = StatevectorSampler()
                        job = sampler.run([qc], shots=shots)
                        self.job_id = "local_simulated"
                        res = job.result()
                        counts = res[0].data.meas.get_counts()
                        status = "DONE_LOCAL_SIMULATION"
                    else:
                        self.job_id = None
                        status = "NO_IBM_TOKEN"

                except Exception as e:
                    if allow_local_simulation:
                        print(
                            f"IBM Submission Error: {e}. Using explicit local simulation.",
                            flush=True,
                        )
                        from qiskit.primitives import StatevectorSampler

                        sampler = StatevectorSampler()
                        job = sampler.run([qc], shots=shots)
                        self.job_id = "local_simulated"
                        res = job.result()
                        counts = res[0].data.meas.get_counts()
                        status = "DONE_LOCAL_SIMULATION"
                    else:
                        print(f"IBM Submission Error: {e}.", flush=True)
                        self.job_id = None
                        status = "IBM_SUBMISSION_ERROR"

                final_result = {}
                if counts is not None:
                    observables = (
                        self.observable if isinstance(self.observable, list) else [self.observable]
                    )
                    for ob in observables:
                        if callable(ob):
                            final_result.update(ob(counts=counts, **self.kwargs))

                # Overwrite sync_order with ZNE-extrapolated value if available
                if zne_sync_order is not None:
                    final_result["sync_order"] = zne_sync_order
                    final_result["zne_applied"] = True
                    final_result["zne_scale_factors"] = [1, 2, 3]
                    final_result["zne_factory"] = "RichardsonFactory"
                    final_result["zne_job_ids"] = zne_job_ids

                final_result["job_id"] = self.job_id
                final_result["job_ids"] = ibm_job_ids
                final_result["runtime"] = time.time() - self.submitted_at
                final_result["status"] = status
                if counts is None:
                    final_result["counts_available"] = False

                return final_result

            async def result(self):
                import asyncio

                return await asyncio.to_thread(self._run_blocking)

        return JobWrapper(self, ansatz, observable, kwargs)

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
