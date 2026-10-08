# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — native provider job lifecycle
"""Bind durable attempts to original IBM jobs without replacing native execution."""

from __future__ import annotations

import io
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any, cast

from qiskit import QuantumCircuit, qpy
from qiskit.primitives.containers import SamplerPub

from .provider_job_journal import ProviderJobJournal, SubmissionUnknownError
from .runner import HardwareRunner, JobResult


def native_batch_payload(circuits: Sequence[QuantumCircuit]) -> bytes:
    """Encode an ordered batch using the installed native QPY codec.

    Parameters
    ----------
    circuits
        Actual compiled circuits crossing the original sampler boundary.

    Returns
    -------
    bytes
        Exact native QPY bytes; no circuit equivalence or target inference.

    """
    stream = io.BytesIO()
    qpy.dump(list(circuits), stream)
    return stream.getvalue()


def submit_durable_batch(
    runner: HardwareRunner,
    circuits: list[Any],
    shots: int,
    experiment: str,
    journal: ProviderJobJournal,
    attempt_id: str,
) -> Any:
    """Persist native identity before the original sampler dispatch.

    Parameters
    ----------
    runner
        Original connected hardware runner and exact selected backend.
    circuits
        Native source batch passed to the original transpiler.
    shots
        Admitted requested shots, forwarded unchanged.
    experiment
        Original caller label.
    journal
        Durable state owner.
    attempt_id
        Canonical UUID for this one effect.

    Returns
    -------
    Any
        Actual native provider job, already bound durably before returning.

    Raises
    ------
    SubmissionUnknownError
        If any prior dispatch may have taken effect.

    """
    from qiskit_ibm_runtime import SamplerV2

    if not 1 <= len(circuits) <= 256:
        raise ValueError("durable circuit batch must contain between one and256 circuits")
    compiled = [runner.transpile(circuit) for circuit in circuits]
    journal.prepare(
        attempt_id,
        target=runner.backend_name,
        payload=native_batch_payload(compiled),
        shots=shots,
        experiment=experiment,
        circuits=len(compiled),
    )
    sampler = SamplerV2(mode=runner._backend)
    sampler.options.default_shots = shots
    journal.begin_submit(attempt_id)
    job = sampler.run(compiled, shots=shots)
    journal.bind_job(attempt_id, job.job_id())
    return job


def retrieve_durable_job(
    runner: HardwareRunner,
    journal: ProviderJobJournal,
    attempt_id: str,
    provider_job_id: str | None = None,
) -> Any:
    """Retrieve and verify an existing native job; this function never submits.

    Parameters
    ----------
    runner
        Original connected runner with read-only provider retrieval capability.
    journal
        Reopened durable journal.
    attempt_id
        Existing attempt identity.
    provider_job_id
        Owner-identified existing handle for an uncertain dispatch. For a bound
        attempt it must equal the stored handle.

    Returns
    -------
    Any
        Original provider job whose native request and exact target match.

    Raises
    ------
    SubmissionUnknownError
        If no existing handle is known; no new submission is attempted.
    ValueError
        If the recovered native request, target, shots or handle differs.

    """
    row = journal.snapshot(attempt_id)
    stored_id = row["provider_job_id"]
    if stored_id is not None and provider_job_id is not None and stored_id != provider_job_id:
        raise ValueError("recovery handle differs from original provider job")
    job_id = stored_id if stored_id is not None else provider_job_id
    if not isinstance(job_id, str):
        raise SubmissionUnknownError("an existing provider handle must be reconciled explicitly")
    if runner.backend_name != row["target"]:
        raise ValueError("recovery runner target differs")
    job = runner.retrieve_job(job_id)
    if job.job_id() != job_id or job.backend().name != row["target"]:
        raise ValueError("provider returned a different handle or target")
    inputs = job.inputs
    pubs = inputs["pubs"]
    options = inputs.get("options")
    observed_default = options.get("default_shots") if options is not None else None
    if options is not None and observed_default != row["shots"]:
        raise ValueError("recovered shots differ")
    compiled = []
    for original_pub in pubs:
        pub = SamplerPub.coerce(original_pub, shots=cast(int | None, observed_default))
        if pub.parameter_values.num_parameters:
            raise ValueError("recovered parameter bindings differ from compiled request")
        if pub.shots != row["shots"]:
            raise ValueError("recovered per-publication shots differ")
        compiled.append(pub.circuit)
    if native_batch_payload(compiled) != row["payload"]:
        raise ValueError("recovered native payload differs")
    journal.bind_job(attempt_id, job_id)
    return job


def observe_durable_job(
    job: Any, journal: ProviderJobJournal, attempt_id: str
) -> dict[str, object]:
    """Observe the original provider status without submitting or claiming billing.

    Parameters
    ----------
    job
        Original or exactly recovered native handle.
    journal
        Durable state owner.
    attempt_id
        Bound attempt.

    Returns
    -------
    dict[str, object]
        Detached durable state after the native observation.

    """
    row = journal.snapshot(attempt_id)
    if job.job_id() != row["provider_job_id"]:
        raise ValueError("observation handle differs from original provider job")
    status = job.status()
    native_status = status if isinstance(status, str) else status.name
    state = {"DONE": "completed", "CANCELLED": "cancelled", "ERROR": "failed"}.get(
        native_status, "submitted"
    )
    # DONE confirms execution, but result decoding still owns completion evidence.
    if state == "completed":
        state = "submitted"
    journal.record(attempt_id, state, {"provider_status": native_status})
    return journal.snapshot(attempt_id)


def cancel_durable_job(
    job: Any, journal: ProviderJobJournal, attempt_id: str
) -> dict[str, object]:
    """Persist cancellation intent before the effect, then observe confirmation.

    Parameters
    ----------
    job
        Original native provider handle.
    journal
        Durable state owner.
    attempt_id
        Bound attempt whose terminal result wins a concurrent cancellation.

    Returns
    -------
    dict[str, object]
        Observed state; an accepted cancellation request alone is not terminal.

    """
    row = journal.snapshot(attempt_id)
    if row["state"] in {"completed", "cancelled", "failed"}:
        return row
    if job.job_id() != row["provider_job_id"]:
        raise ValueError("cancellation handle differs from original provider job")
    journal.record(attempt_id, "cancellation_requested", {})
    job.cancel()
    return observe_durable_job(job, journal, attempt_id)


def store_job_results(
    journal: ProviderJobJournal, attempt_id: str, results: list[JobResult]
) -> None:
    """Persist native decoded results as completion evidence of the bound job.

    Parameters
    ----------
    journal
        Original durable custody owner.
    attempt_id
        Identity whose stored provider handle must match every result.
    results
        Original decoded count records from the existing runner.

    Raises
    ------
    ValueError
        If identities, batch length or native sample conservation disagree.

    """
    row = journal.snapshot(attempt_id)
    if len(results) != row["circuits"]:
        journal.record(attempt_id, "submitted", {"partial_results": [asdict(r) for r in results]})
        raise ValueError("provider result batch is incomplete")
    for result in results:
        if result.job_id != row["provider_job_id"] or result.backend_name != row["target"]:
            raise ValueError("provider result identity differs")
        if result.counts is None or sum(result.counts.values()) != row["shots"]:
            journal.record(
                attempt_id, "submitted", {"partial_results": [asdict(r) for r in results]}
            )
            raise ValueError("provider sample total differs from the original request")
    journal.record(attempt_id, "completed", {"results": [asdict(r) for r in results]})


def retain_native_result(journal: ProviderJobJournal, attempt_id: str, result: Any) -> None:
    """Retain the original SDK result before decoding any publication or count.

    Parameters
    ----------
    journal
        Original durable request owner.
    attempt_id
        Bound attempt identity.
    result
        Actual native Runtime result, encoded by its installed SDK codec.
        Partial or malformed count views do not discard this evidence.

    """
    from qiskit_ibm_runtime import RuntimeEncoder

    encoded = json.dumps(result, cls=RuntimeEncoder, allow_nan=False)
    journal.record(attempt_id, "submitted", {"native_runtime_json": encoded})


def native_job_result(job: Any, timeout_s: object) -> Any:
    """Retrieve through the actual native job's supported timeout interface.

    Parameters
    ----------
    job
        Original cloud Runtime job or local SDK PrimitiveJob.
    timeout_s
        Finite nonnegative wait bound in seconds. Local SDK jobs use their
        original in_final_state contract, since result() accepts no timeout.

    Returns
    -------
    Any
        Original native result, without a second submission or synthetic output.

    Raises
    ------
    ValueError
        If the wait bound is invalid.
    qiskit.providers.JobTimeoutError
        If a local native job remains unfinished at the deadline.

    """
    from qiskit.primitives import PrimitiveJob
    from qiskit.providers import JobTimeoutError

    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, int | float)
        or not math.isfinite(timeout_s)
        or timeout_s < 0
    ):
        raise ValueError("timeout_s must be finite nonnegative seconds")
    if isinstance(job, PrimitiveJob):
        deadline = time.monotonic() + timeout_s
        while not job.in_final_state():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise JobTimeoutError("local native job did not finish within the wait bound")
            time.sleep(min(0.05, remaining))
        return job.result()
    return job.result(timeout=timeout_s)


def restore_job_results(snapshot: dict[str, object]) -> list[JobResult]:
    """Read detached original results from the committed terminal event.

    Parameters
    ----------
    snapshot
        Verified durable journal snapshot.

    Returns
    -------
    list[JobResult]
        Original result values without a provider or submission call.

    Raises
    ------
    ValueError
        If the terminal snapshot has no retained result evidence.

    """
    events = cast(list[dict[str, Any]], snapshot["events"])
    for event in reversed(events):
        if event["state"] == "completed":
            return [JobResult(**row) for row in event["observation"]["results"]]
    raise ValueError("completed attempt has no retained results")
