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

Submission happens **once per job**. The wrapper returned by
:meth:`AsyncHardwareRunner.submit_circuit_batch` holds a single in-flight
submission task, so sequential awaits, concurrent ``gather`` and a cancelled
awaiter all share one crossing of the provider boundary. A submission that
raises is recorded as ambiguous — the provider may already hold the work — and
is re-raised on every later await instead of being resubmitted; recovery is an
explicit caller decision, readable through ``submission_state`` and
``submission_error``.

The device and the shot count are reported, never substituted quietly. A named
backend that cannot be resolved raises :class:`BackendSubstitutionError` instead
of running elsewhere; passing ``allow_backend_substitution=True`` accepts a
different device and records the swap. Execution opt-ins require actual booleans,
not truthy strings or integers. Positive integer shot requests are passed unchanged
to the provider; this adapter does not invent a universal shot limit or reduce a
request to make it fit. Provider rejection is not permission to retry with fewer
shots. Queued and completed submission receipts return ``requested_backend``,
``backend_name``, ``backend_substituted``, ``requested_shots``,
``effective_shots`` and ``shots_capped``.

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
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

from ..dense_budget import DenseAllocationError
from .runner import HardwareRunner, JobResult, _require_local_statevector_simulator


def _require_shots(shots: object) -> int:
    """Accept positive integer counts without boolean or fractional coercion."""
    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots <= 0:
        raise ValueError("shots must be a positive integer")
    return int(shots)


class BackendSubstitutionError(RuntimeError):
    """A named backend was unavailable and substitution was not permitted.

    Raised instead of quietly running on a different device. It is a
    configuration error, not a provider fault, so it propagates to the caller
    rather than being folded into a submission-error status or a local-simulation
    fallback.
    """


def _get_logger(name: str) -> Any:
    try:
        from ..logging_setup import get_logger

        return get_logger(name)
    except Exception:
        return logging.getLogger(name)


_log = _get_logger(__name__)


def _guard_local_statevector_simulation(qc: Any, max_dense_gib: float | None) -> None:
    """Budget guard for explicit local simulation of real Qiskit circuits."""
    if not hasattr(qc, "num_qubits"):
        return
    _require_local_statevector_simulator(qc, max_dense_gib)


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
        **kwargs: Any,
    ) -> None:
        """Initialize a runner pool with bounded submission concurrency."""
        self.backend = backend
        self.default_shots = shots
        self.mitigation = mitigation
        self.runner_kwargs = kwargs

        # Accept either a single runner or any iterable of runners.
        if runners is not None:
            if hasattr(runners, "backend_name") and not hasattr(runners, "__iter__"):
                runners = [runners]
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

    def submit_circuit_batch(self, ansatz: Any, observable: Any, **kwargs: Any) -> Any:
        """Submit ``StructuredAnsatz`` jobs without fabricating unfinished results.

        Real QPU submissions return real job identifiers immediately when
        queued. Observables are evaluated only when real counts are available.
        Local simulation and ZNE are opt-in because both change the scientific
        meaning and resource profile of a campaign.

        Parameters
        ----------
        ansatz:
            Object providing ``build_circuit()``.
        observable:
            Callable or list of callables consuming measured counts.
        **kwargs:
            ``shots`` overrides the runner default with a positive integer.
            ``allow_backend_substitution``, ``allow_local_simulation`` and
            ``enable_zne`` require booleans and default to False. Per-call
            options override runner options. Admission occurs on first await,
            before circuit construction or SDK loading. Shot counts are never
            capped; provider-specific rejection propagates after dispatch.
            Failed requested ZNE propagates without an unmitigated replacement.
            Known IDs remain available through the wrapper's ``job_ids``.
            IBM receipts record transpiler settings and whether optional
            dynamical decoupling succeeded (or its skip reason).

        Returns
        -------
        Any
            Memory-only job wrapper with an async ``result()`` method.

        Raises
        ------
        ValueError
            On first await if shots or execution opt-ins are malformed.
        """

        class JobWrapper:
            """Awaitable wrapper around simulator, ZNE, or submitted QPU work."""

            def __init__(
                self,
                runner_obj: AsyncHardwareRunner,
                ansatz: Any,
                observable: Any,
                kwargs: dict[str, Any],
            ) -> None:
                self.runner_obj = runner_obj
                self.ansatz = ansatz
                self.observable = observable
                self.kwargs = kwargs
                self.job_id: str | None = None
                self.submitted_at = time.time()
                self._result: dict[str, Any] | None = None
                self._failure: BaseException | None = None
                self._task: asyncio.Task[dict[str, Any]] | None = None
                self._start_lock = asyncio.Lock()
                self._provider_job: Any = None
                self._provider_dispatch_started = False
                self._job_ids: list[str] = []

            def _run_blocking(self) -> dict[str, Any]:
                shots = _require_shots(self.kwargs.get("shots", self.runner_obj.default_shots))
                options = {}
                for name in ("allow_local_simulation", "allow_backend_substitution", "enable_zne"):
                    value = self.kwargs.get(name, self.runner_obj.runner_kwargs.get(name, False))
                    if not isinstance(value, bool):
                        raise ValueError(f"{name} must be a boolean")
                    options[name] = value
                allow_local_simulation = options["allow_local_simulation"]
                allow_backend_substitution = options["allow_backend_substitution"]
                enable_zne = options["enable_zne"]

                import os

                from qiskit.providers import JobTimeoutError
                from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling
                from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
                from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

                zne_sync_order = None  # set on IBM path if explicit ZNE succeeds
                ibm_job_ids = self._job_ids
                zne_job_ids: list[str] = []
                counts = None
                status = "NOT_SUBMITTED"
                execution_provenance: dict[str, Any] = {}
                max_dense_gib = self.kwargs.get(
                    "max_dense_gib",
                    self.runner_obj.runner_kwargs.get("max_dense_gib"),
                )
                qc = self.ansatz.build_circuit()
                if qc.num_clbits == 0:
                    qc.measure_all()

                token = os.environ.get("SCPN_IBM_TOKEN")
                crn = os.environ.get("SCPN_IBM_CRN") or os.environ.get("SCPN_IBM_INSTANCE")
                if not token and enable_zne and allow_local_simulation:
                    raise RuntimeError("requested ZNE is unavailable on the local simulation path")

                try:
                    if token:
                        service_kwargs = {"channel": "ibm_cloud", "token": token}
                        if crn:
                            service_kwargs["instance"] = crn
                        service = QiskitRuntimeService(**service_kwargs)
                        target = (
                            "ibm_fez"
                            if self.runner_obj.backend == "ibm_heron_r2"
                            else self.runner_obj.backend
                        )
                        try:
                            backend = service.backend(target)
                            execution_provenance["backend_substituted"] = False
                        except Exception as lookup_error:
                            if not allow_backend_substitution:
                                raise BackendSubstitutionError(
                                    f"requested backend {target!r} is unavailable "
                                    f"({lookup_error}); pass "
                                    "allow_backend_substitution=True to accept a "
                                    "different device, which changes the device the "
                                    "results describe"
                                ) from lookup_error
                            backend = service.least_busy(simulator=False, operational=True)
                            execution_provenance["backend_substituted"] = True
                            execution_provenance["backend_substitution_reason"] = str(lookup_error)
                        execution_provenance["requested_backend"] = target
                        execution_provenance["backend_name"] = getattr(backend, "name", None)

                        # Basic error mitigation: high optimization + dynamical decoupling
                        pm = generate_preset_pass_manager(
                            optimization_level=3, backend=backend, seed_transpiler=42
                        )
                        isa_qc = pm.run(qc)
                        execution_provenance["transpilation"] = {
                            "optimization_level": 3,
                            "seed_transpiler": 42,
                            "backend_name": execution_provenance["backend_name"],
                        }
                        execution_provenance["dd_applied"] = False
                        execution_provenance["zne_requested"] = enable_zne

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
                            execution_provenance["dd_applied"] = True
                        except Exception as dd_err:
                            execution_provenance["dd_skip_reason"] = str(dd_err)
                            print(f"DD skipped ({dd_err}); submitting without DD.", flush=True)

                        sampler = SamplerV2(mode=backend)
                        effective_shots = shots
                        sampler.options.default_shots = effective_shots
                        execution_provenance["requested_shots"] = shots
                        execution_provenance["effective_shots"] = effective_shots
                        execution_provenance["shots_capped"] = False

                        print(
                            "IBM Runtime: Dispatching circuit with opt_level=3"
                            f" (DD applied={execution_provenance['dd_applied']})"
                            f" to {backend.name}...",
                            flush=True,
                        )

                        if enable_zne:
                            # Mitiq ZNE must operate on scalar observables, not
                            # counts dictionaries. This path blocks until each
                            # scaled job returns and records every IBM job id.
                            from mitiq import zne
                            from mitiq.zne.inference import RichardsonFactory
                            from mitiq.zne.scaling import fold_global

                            from scpn_quantum_control.analysis import SyncOrderParameter

                            def _zne_executor(circ: Any) -> float:
                                """Run scaled circuit and return sync_order for extrapolation."""
                                self._provider_dispatch_started = True
                                _job = sampler.run([circ])
                                zne_job_id = str(_job.job_id())
                                self.job_id = zne_job_id
                                ibm_job_ids.append(zne_job_id)
                                zne_job_ids.append(zne_job_id)
                                _res = _job.result()
                                from .runner import _extract_counts

                                _counts = _extract_counts(_res[0])
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

                        # Scale=1 run — collects full counts for all observables
                        self._provider_dispatch_started = True
                        job = sampler.run([isa_qc])
                        self._provider_job = job
                        self.job_id = str(job.job_id())
                        ibm_job_ids.append(self.job_id)
                        print(f"IBM Runtime: Job queued -> {self.job_id}", flush=True)
                        status = "QUEUED_ON_IBM"
                        try:
                            res = job.result(timeout=15)
                            from .runner import _extract_counts

                            counts = _extract_counts(res[0])
                            status = "DONE"
                        except (TimeoutError, JobTimeoutError):
                            print(f"Job {self.job_id}: result not yet available.", flush=True)
                    elif allow_local_simulation:
                        from qiskit.primitives import StatevectorSampler

                        _guard_local_statevector_simulation(qc, max_dense_gib)
                        sampler = StatevectorSampler()
                        job = sampler.run([qc], shots=shots)
                        self.job_id = "local_simulated"
                        res = job.result()
                        counts = res[0].data.meas.get_counts()
                        status = "DONE_LOCAL_SIMULATION"
                    else:
                        self.job_id = None
                        status = "NO_IBM_TOKEN"

                except BackendSubstitutionError:
                    raise
                except Exception as e:
                    if (
                        isinstance(e, DenseAllocationError)
                        or self._provider_dispatch_started
                        or enable_zne
                    ):
                        raise
                    if allow_local_simulation:
                        print(
                            f"IBM Submission Error: {e}. Using explicit local simulation.",
                            flush=True,
                        )
                        from qiskit.primitives import StatevectorSampler

                        _guard_local_statevector_simulation(qc, max_dense_gib)
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

                if status == "DONE_LOCAL_SIMULATION":
                    execution_provenance = {
                        "requested_backend": self.runner_obj.backend,
                        "backend_name": "StatevectorSampler",
                        "backend_substituted": True,
                        "requested_shots": shots,
                        "effective_shots": shots,
                        "shots_capped": False,
                        "dd_applied": False,
                        "zne_requested": False,
                        "transpilation": None,
                    }
                final_result = self._evaluate_observables(counts)

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
                final_result.update(execution_provenance)
                final_result["counts_available"] = counts is not None

                return final_result

            def _evaluate_observables(self, counts: dict[str, int] | None) -> dict[str, Any]:
                """Evaluate the existing observable contract only after counts arrive."""
                result: dict[str, Any] = {}
                if counts is not None:
                    observables = (
                        self.observable if isinstance(self.observable, list) else [self.observable]
                    )
                    for observable in observables:
                        if callable(observable):
                            result.update(observable(counts=counts, **self.kwargs))
                return result

            def _retrieve_existing(self, previous: dict[str, Any]) -> dict[str, Any]:
                """Retrieve the original job without recompiling or resubmitting it."""
                from qiskit.providers import JobTimeoutError

                from .runner import _extract_counts

                try:
                    response = self._provider_job.result(timeout=15)
                except (TimeoutError, JobTimeoutError):
                    return dict(previous)
                counts = _extract_counts(response[0])
                result = self._evaluate_observables(counts)
                result.update(previous)
                result["status"] = "DONE"
                result["counts_available"] = True
                result["runtime"] = time.time() - self.submitted_at
                return result

            def _awaiting_provider_result(self) -> bool:
                """Whether the cached submission receipt still lacks provider counts."""
                return self._result is not None and self._result.get("status") == "QUEUED_ON_IBM"

            @property
            def job_ids(self) -> tuple[str, ...]:
                """Known provider job IDs, retained even if later mitigation fails.

                Returns
                -------
                tuple[str, ...]
                    Immutable snapshot in dispatch order. Missing IDs on an
                    ambiguous dispatch must be reconciled with the provider;
                    an empty tuple is not proof of zero resource consumption.

                """
                return tuple(self._job_ids)

            @property
            def submission_state(self) -> str:
                """The provider-boundary state of this wrapper.

                Returns
                -------
                str
                    ``"not_started"`` before any submission, ``"in_flight"``
                    while one is running, ``"awaiting_result"`` after a result
                    timeout, ``"completed"`` once a final result is
                    held, or ``"ambiguous"`` when the submission raised and it
                    is unknown whether the provider accepted the work.

                """
                if self._failure is not None:
                    return "ambiguous"
                if self._awaiting_provider_result():
                    return "awaiting_result"
                if self._result is not None:
                    return "completed"
                if self._task is not None:
                    return "in_flight"
                return "not_started"

            @property
            def submission_error(self) -> BaseException | None:
                """The recorded failure, or ``None`` if there is none.

                A recorded failure is deliberately not retried: the provider may
                already hold the work, so recovery is an explicit caller
                decision rather than an automatic resubmission.
                """
                return self._failure

            def _capture_completion(self, task: asyncio.Task[dict[str, Any]]) -> None:
                """Record completion even when no client remains to await it."""
                try:
                    self._result = deepcopy(task.result())
                except BaseException as exc:
                    self._failure = exc

            async def _shared_submission(self) -> asyncio.Task[dict[str, Any]]:
                """Return the one in-flight submission task, starting it once."""
                async with self._start_lock:
                    if self._task is None:
                        self._task = asyncio.ensure_future(asyncio.to_thread(self._run_blocking))
                        self._task.add_done_callback(self._capture_completion)
                    elif (
                        self._result is not None
                        and self._awaiting_provider_result()
                        and self._task.done()
                    ):
                        previous = self._result
                        if self._provider_job is None:
                            raise RuntimeError("pending submission has no original provider job")
                        self._result = None
                        self._task = asyncio.ensure_future(
                            asyncio.to_thread(self._retrieve_existing, previous)
                        )
                        self._task.add_done_callback(self._capture_completion)
                    elif self._task.get_loop() is not asyncio.get_running_loop():
                        raise RuntimeError("active submission belongs to another event loop")
                    return self._task

            async def result(self) -> dict[str, Any]:
                """Await this job's result, submitting at most once.

                Sequential and concurrent awaits share a single provider
                submission. A cancelled awaiter does not cancel the underlying
                submission, so a later await joins the same work instead of
                issuing a second one. Once the submission has failed, every
                later await re-raises the recorded failure rather than
                resubmitting.
                Completion is recorded independently of surviving awaiters;
                cancelling the last client does not leave a completed job
                labelled in_flight or an unobserved failure unrecorded.
                An active task must be awaited on its owning event loop;
                cross-loop retrieval is refused without marking provider work
                as failed. Completed cached outcomes may be read subsequently.
                After a result timeout, the next call polls the same retained
                provider job; concurrent calls share that retrieval task.
                The legacy QUEUED_ON_IBM receipt means counts are not available,
                not a verified provider queue status. Other provider/decoding
                errors propagate, retain the job identity, and never trigger
                local fallback after dispatch. Returned dictionaries are detached
                copies so caller edits cannot change retrieval state or provenance.
                This wrapper is memory-only; process-restart recovery requires
                archiving job IDs and using the provider's retrieval interface.

                Returns
                -------
                dict
                    The submission result payload.

                Raises
                ------
                BaseException
                    The recorded submission failure, re-raised unchanged on
                    every subsequent await.

                """
                if self._failure is not None:
                    raise self._failure
                if self._result is not None and not self._awaiting_provider_result():
                    return deepcopy(self._result)

                task = await self._shared_submission()
                try:
                    outcome = await asyncio.shield(task)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    self._failure = exc
                    raise
                return deepcopy(outcome)

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
        ``shots`` must be a positive integer and is forwarded unchanged;
        malformed counts raise ValueError before runner selection/transpilation.
        """
        shots = _require_shots(shots)
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
        Positive integer ``shots`` are validated before any sub-batch starts,
        including an empty batch; invalid counts raise ValueError.
        """
        shots = _require_shots(shots)
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
        """Maximum number of concurrent runner workers."""
        return self._max_concurrent

    @property
    def n_runners(self) -> int:
        """Number of runner workers in the local pool."""
        return len(self._runners)


__all__ = [
    "BackendSubstitutionError",
    "AsyncHardwareRunner",
    "AsyncJobHandle",
]
