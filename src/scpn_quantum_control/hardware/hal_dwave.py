# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — D-Wave Leap adapter for the hardware HAL
"""Direct D-Wave Leap BQM adapter for the provider-neutral HAL."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from importlib import import_module
from math import isfinite
from typing import Any, Literal, cast

import numpy as np

from ._count_integrity import (
    strict_integer_value,
    strict_provider_job_id,
    strict_shot_conservation,
)
from .hal import (
    BackendProfile,
    QuantumJobRef,
    QuantumJobResult,
    QuantumWorkload,
    _resolve_stored_job,
    _validate_workload_for_profile,
)
from .provider_modalities import (
    AnnealingObservation,
    AnnealingSample,
    ModalitySemantics,
    NativeAnnealingRecord,
)
from .provider_semantics import modality_submission_semantics

DWAVE_BQM_SCHEMA = "scpn.dwave.bqm.v1"
DWAVE_EXECUTION_MODE = "dwave_leap_bqm"


def dwave_bqm_workload(
    *,
    linear: Mapping[str, float],
    quadratic: Mapping[tuple[str, str], float],
    workload_id: str,
    n_variables: int,
    reads: int,
    offset: float = 0.0,
    vartype: str = "BINARY",
    metadata: Mapping[str, object] | None = None,
    schema: str = DWAVE_BQM_SCHEMA,
    capture_semantics: bool = False,
    requested_target: str | None = None,
) -> QuantumWorkload:
    """Encode the original Ising/QUBO plan and optional native annealing companion.

    Parameters
    ----------
    linear
        Finite biases keyed by every original variable label.
    quadratic
        Finite pair biases whose distinct endpoints belong to the same model.
    workload_id
        Stable caller identity for the original model.
    n_variables
        Declared native variable count. The legacy HAL width field does not
        reinterpret annealing variables as gate-model qubits.
    reads
        Positive integral requested occurrence total.
    offset
        Finite native model energy offset.
    vartype
        Original BINARY or SPIN domain, canonicalized to its uppercase name.
    metadata
        Caller annotations separate from source and execution settings.
    schema
        Supported original BQM plan version.
    capture_semantics
        Add a versioned source digest, original encoded variable order and
        native domain without changing the existing JSON codec.
    requested_target
        Optional exact declared adapter selector, requiring native capture.
        A selector name does not attest a physical solver.

    Returns
    -------
    QuantumWorkload
        Existing canonical BQM JSON plan, whose variables retain the original
        builder's sorted order, and optional native companion.

    Raises
    ------
    ValueError
        If schema, variables, pair endpoints, finite biases, native domain,
        resource settings or target admission are invalid.

    """
    variables = sorted(str(variable) for variable in linear)
    payload: dict[str, object] = {
        "schema": schema,
        "vartype": _normalise_vartype(vartype),
        "variables": variables,
        "linear": {
            variable: _coerce_float(bias, field_name="linear bias")
            for variable, bias in linear.items()
        },
        "quadratic": [
            {
                "u": str(u),
                "v": str(v),
                "bias": _coerce_float(bias, field_name="quadratic bias"),
            }
            for (u, v), bias in sorted(
                quadratic.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
            )
        ],
        "offset": _coerce_float(offset, field_name="offset"),
    }
    _validate_bqm_payload(payload, n_variables)
    if not capture_semantics and requested_target is not None:
        raise ValueError("target pin requires native semantics capture")
    program = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    semantics = (
        ModalitySemantics(
            program_sha256=hashlib.sha256(program.encode()).hexdigest(),
            modality="annealing",
            native_axes=tuple(variables),
            vartype=_normalise_vartype(vartype),
            requested_target=requested_target,
        )
        if capture_semantics
        else None
    )
    return QuantumWorkload(
        workload_id=workload_id,
        ir_format="bqm",
        program=program,
        n_qubits=n_variables,
        shots=reads,
        metadata=dict(metadata or {}),
        semantics=semantics,
    )


class DWaveLeapHALAdapter:
    """Retain synchronous native annealing outputs separately from binary projections.

    Parameters
    ----------
    profile
        D-Wave route declaring supported IR and resource limits.
    sampler
        Optional explicitly supplied native-compatible sampler. Its truth
        value does not change admission or select a default route.
    sampler_factory
        Optional lazy native-compatible sampler constructor, retained after
        its first successful construction. Absence uses the optional SDK.
    bqm_factory
        Optional constructor receiving each current original admitted plan.
        Absence uses the optional dimod BQM constructor.
    solver
        Declared adapter selector; absent means default, malformed names refuse.
        The selector itself does not attest a native or physical solver identity.

    Raises
    ------
    ValueError
        If profile or explicit selector is invalid.

    """

    supports_provider_semantics = True

    def __init__(
        self,
        profile: BackendProfile,
        *,
        sampler: Any | None = None,
        sampler_factory: Callable[[], Any] | None = None,
        bqm_factory: Callable[[dict[str, object]], Any] | None = None,
        solver: str | None = None,
    ) -> None:
        if profile.backend_id != "dwave_leap":
            raise ValueError("DWaveLeapHALAdapter requires the dwave_leap profile")
        self.profile = profile
        self.backend_id = profile.backend_id
        self._sampler = sampler
        self._sampler_factory = sampler_factory
        self._bqm_factory = bqm_factory
        self._solver = strict_provider_job_id(
            solver if solver is not None else "default", field_name="D-Wave solver"
        )
        self._jobs: dict[str, QuantumJobRef] = {}
        self._results: dict[str, QuantumJobResult] = {}

    def submit(
        self, workload: QuantumWorkload, *, approval_id: str | None = None
    ) -> QuantumJobRef:
        """Admit original source, native variable domain and selector before sampling.

        Parameters
        ----------
        workload
            Original supported BQM plan and optional source-bound companion.
        approval_id
            Required caller authorization for synchronous submission.

        Returns
        -------
        QuantumJobRef
            Completed handle retaining the original model and requested reads.
            A captured result keeps actual returned columns, SPIN/BINARY values,
            energies and structured-record bytes separately from legacy counts.
            Native model construction belongs to the supplied or SDK constructor;
            no compiled digest or physical execution claim is fabricated.

        Raises
        ------
        PermissionError
            If caller approval is absent.
        ValueError
            If profile, source, native domain/order, selector, provider identity,
            record layout, sample values or exact occurrence totals disagree.
        TypeError
            If the selected sampler lacks callable sample retrieval.
        RuntimeError
            If the configured native builder requires an unavailable SDK.

        """
        if not approval_id:
            raise PermissionError("approval_id is required for D-Wave Leap submission")
        if workload.ir_format != "bqm":
            raise ValueError("D-Wave Leap direct adapter requires bqm workloads")
        _validate_workload_for_profile(self.profile, workload)
        payload = _decode_bqm_payload(workload)
        _validate_bqm_payload(payload, workload.n_qubits)
        submission = modality_submission_semantics(
            workload.semantics,
            program=workload.program,
            ir_format=workload.ir_format,
            native_axes=tuple(cast(Sequence[str], payload["variables"])),
            modality="annealing",
            vartype=_normalise_vartype(payload["vartype"]),
            target_name=self._solver,
            shots=workload.shots,
        )
        bqm = self._build_bqm(payload)
        sample_method = getattr(self._sampler_for(), "sample", None)
        if not callable(sample_method):
            raise TypeError("D-Wave sampler object does not provide sample()")
        sample_set = sample_method(bqm, num_reads=workload.shots, label=workload.workload_id)
        observation = None
        if submission is not None:
            assert isinstance(submission.request, ModalitySemantics)
            observation = _annealing_observation(sample_set, submission.request, workload.shots)
        counts = (
            dict(observation.counts)
            if observation is not None
            else _normalise_sample_counts(
                sample_set,
                cast(Sequence[str], payload["variables"]),
                vartype=_normalise_vartype(payload["vartype"]),
            )
        )
        observed_shots = strict_shot_conservation(counts, expected_shots=workload.shots)
        provider_job_id = _provider_job_id(sample_set)
        hal_job_id = _hal_job_id(self.backend_id, workload.workload_id, provider_job_id)
        job = QuantumJobRef(
            job_id=hal_job_id,
            backend_id=self.backend_id,
            workload_id=workload.workload_id,
            status="completed",
            submission=submission,
            metadata={
                "approval_id": approval_id,
                "provider_job_id": provider_job_id,
                "execution_mode": DWAVE_EXECUTION_MODE,
                "solver": self._solver,
                "ir_format": workload.ir_format,
                "n_variables": workload.n_qubits,
                "shots": workload.shots,
                "vartype": payload["vartype"],
            },
        )
        result = QuantumJobResult(
            job=job,
            status="completed",
            counts=counts,
            shots=observed_shots,
            provider_observation=observation,
            metadata={
                "approval_id": approval_id,
                "execution_mode": DWAVE_EXECUTION_MODE,
                "solver": self._solver,
                "vartype": payload["vartype"],
                "timestamp": _utc_now(),
            },
        )
        self._jobs[job.job_id] = job
        self._results[job.job_id] = result
        return job

    def status(self, job: QuantumJobRef) -> str:
        """Read the retained synchronous lifecycle after validating durable identity.

        Parameters
        ----------
        job
            Handle matching the original stored backend and workload.

        Returns
        -------
        str
            Stored completed status, or the legacy cancellation annotation.

        Raises
        ------
        KeyError
            If original submission is unavailable.
        ValueError
            If durable backend or workload identity differs.

        """
        return self._job(job).status

    def result(self, job: QuantumJobRef) -> QuantumJobResult:
        """Retrieve the original retained result without rerunning an annealing model.

        Parameters
        ----------
        job
            Handle matching the original stored backend and workload.

        Returns
        -------
        QuantumJobResult
            Original immutable compatibility histogram and optional typed
            observation. Returned native variable order and SPIN -1 values
            survive; missing energies remain unknown. Structured bytes retain
            the original dtype descriptor, shape and extra native fields.

        Raises
        ------
        KeyError
            If original submission or retained completed result is unavailable.
        ValueError
            If durable identity differs.

        """
        stored = self._job(job)
        result = self._results.get(stored.job_id)
        if result is None:
            raise KeyError(f"unknown job_id: {job.job_id}")
        return result

    def cancel(self, job: QuantumJobRef) -> QuantumJobRef:
        """Retain a legacy cancellation annotation without erasing completed samples.

        Parameters
        ----------
        job
            Handle matching the original stored backend and workload.

        Returns
        -------
        QuantumJobRef
            Original handle with cancelled lifecycle annotation and preserved
            companion. Synchronous samples remain retrievable; the annotation
            does not attest a physical provider cancellation.

        Raises
        ------
        KeyError
            If original submission is unavailable.
        ValueError
            If durable identity differs.

        """
        stored = self._job(job)
        cancelled = QuantumJobRef(
            job_id=stored.job_id,
            backend_id=stored.backend_id,
            workload_id=stored.workload_id,
            status="cancelled",
            submission=stored.submission,
            metadata=stored.metadata,
        )
        self._jobs[job.job_id] = cancelled
        return cancelled

    def _build_bqm(self, payload: dict[str, object]) -> Any:
        factory = self._bqm_factory if self._bqm_factory is not None else _default_bqm_factory
        return factory(payload)

    def _sampler_for(self) -> Any:
        if self._sampler is not None:
            return self._sampler
        factory = (
            self._sampler_factory
            if self._sampler_factory is not None
            else _default_sampler_factory
        )
        self._sampler = factory()
        return self._sampler

    def _job(self, job: QuantumJobRef) -> QuantumJobRef:
        return _resolve_stored_job(job, self._jobs)


def _default_bqm_factory(payload: dict[str, object]) -> Any:
    try:
        dimod = import_module("dimod")
    except Exception as exc:
        raise RuntimeError("dimod is required to construct D-Wave BQM workloads") from exc
    vartype = str(payload["vartype"])
    linear = cast(Mapping[str, float], payload["linear"])
    quadratic = {
        (str(edge["u"]), str(edge["v"])): _coerce_float(edge["bias"], field_name="quadratic bias")
        for edge in cast(Sequence[Mapping[str, object]], payload["quadratic"])
    }
    offset = _coerce_float(payload["offset"], field_name="offset")
    if vartype == "BINARY":
        qubo: dict[tuple[str, str], float] = {
            (str(variable), str(variable)): _coerce_float(bias, field_name="linear bias")
            for variable, bias in linear.items()
        }
        qubo.update(quadratic)
        return dimod.BinaryQuadraticModel.from_qubo(qubo, offset=offset)
    return dimod.BinaryQuadraticModel.from_ising(dict(linear), quadratic, offset=offset)


def _default_sampler_factory() -> Any:
    try:
        dwave_system = import_module("dwave.system")
    except Exception as exc:
        raise RuntimeError(
            "dwave-system with DWaveSampler is required for DWaveLeapHALAdapter"
        ) from exc
    sampler = dwave_system.DWaveSampler()
    embedding = getattr(dwave_system, "EmbeddingComposite", None)
    return embedding(sampler) if callable(embedding) else sampler


def _decode_bqm_payload(workload: QuantumWorkload) -> dict[str, object]:
    try:
        payload = json.loads(workload.program)
    except json.JSONDecodeError as exc:
        raise ValueError("D-Wave workload is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("D-Wave workload must be a JSON object")
    return dict(payload)


def _validate_bqm_payload(payload: Mapping[str, object], n_variables: int) -> None:
    if payload.get("schema") != DWAVE_BQM_SCHEMA:
        raise ValueError(f"unsupported D-Wave BQM schema; expected {DWAVE_BQM_SCHEMA}")
    variables = payload.get("variables")
    if not isinstance(variables, Sequence) or isinstance(variables, str):
        raise ValueError("D-Wave variables must be a sequence")
    variable_order = [str(variable) for variable in variables]
    if len(variable_order) != n_variables or len(set(variable_order)) != len(variable_order):
        raise ValueError("D-Wave variables must list every variable exactly once")
    _normalise_vartype(payload.get("vartype"))
    linear = payload.get("linear")
    if not isinstance(linear, Mapping):
        raise ValueError("D-Wave linear biases must be a mapping")
    if set(str(variable) for variable in linear) != set(variable_order):
        raise ValueError("D-Wave linear biases must cover every variable exactly once")
    for bias in linear.values():
        _coerce_float(bias, field_name="linear bias")
    quadratic = payload.get("quadratic")
    if not isinstance(quadratic, Sequence) or isinstance(quadratic, str):
        raise ValueError("D-Wave quadratic biases must be a sequence")
    for edge in quadratic:
        if not isinstance(edge, Mapping):
            raise ValueError("D-Wave quadratic entries must be mappings")
        u = str(edge.get("u"))
        v = str(edge.get("v"))
        if u not in variable_order or v not in variable_order or u == v:
            raise ValueError("D-Wave quadratic edge references invalid variables")
        _coerce_float(edge.get("bias"), field_name="quadratic bias")
    _coerce_float(payload.get("offset"), field_name="offset")


def _normalise_sample_counts(
    sample_set: object,
    variables: Sequence[str],
    *,
    vartype: Literal["SPIN", "BINARY"] | None = None,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample, occurrences in _sample_rows(sample_set):
        bitstring = _sample_bitstring(sample, variables, vartype=vartype)
        value = _coerce_int(occurrences, field_name="num_occurrences")
        if value < 0:
            raise ValueError("counts values must be non-negative integers")
        counts[bitstring] = counts.get(bitstring, 0) + value
    if not counts:
        raise ValueError("D-Wave sample set did not contain any samples")
    return counts


def _provider_job_id(sample_set: object) -> str:
    info = getattr(sample_set, "info", None)
    if isinstance(info, Mapping):
        for key in ("problem_id", "id", "task_id"):
            value = info.get(key)
            if value is not None and str(value).strip():
                return strict_provider_job_id(value, field_name="D-Wave provider job id")
    raise ValueError("D-Wave sample set does not expose a provider job id")


def _hal_job_id(backend_id: str, workload_id: str, provider_job_id: str) -> str:
    digest = hashlib.sha256(provider_job_id.encode("utf-8")).hexdigest()[:12]
    return f"{backend_id}:{workload_id}:{digest}"


def _sample_rows(sample_set: object) -> Iterable[tuple[Mapping[str, object], object]]:
    data = getattr(sample_set, "data", None)
    if callable(data):
        for row in data(["sample", "num_occurrences"]):
            sample = getattr(row, "sample", None)
            occurrences = getattr(row, "num_occurrences", None)
            if not isinstance(sample, Mapping):
                raise ValueError("D-Wave sample row does not contain a sample mapping")
            yield sample, occurrences
        return
    if isinstance(sample_set, Mapping):
        samples = sample_set.get("samples")
        counts = sample_set.get("num_occurrences", sample_set.get("counts"))
        if isinstance(samples, Sequence) and isinstance(counts, Sequence):
            for sample, occurrences in zip(samples, counts, strict=True):
                if not isinstance(sample, Mapping):
                    raise ValueError("D-Wave sample entry must be a mapping")
                yield sample, occurrences
            return
    raise ValueError("D-Wave sample set does not expose samples with occurrences")


def _sample_bitstring(
    sample: Mapping[str, object],
    variables: Sequence[str],
    *,
    vartype: Literal["SPIN", "BINARY"] | None = None,
) -> str:
    bits: list[str] = []
    for variable in variables:
        if variable not in sample:
            raise ValueError(f"D-Wave sample is missing variable {variable}")
        value = _coerce_int(sample[variable], field_name="sample value")
        if (vartype == "SPIN" and value not in (-1, 1)) or (
            vartype == "BINARY" and value not in (0, 1)
        ):
            raise ValueError("D-Wave sample values differ from the source-declared native domain")
        if value in (0, 1):
            bits.append(str(value))
        elif value == -1:
            bits.append("0")
        else:
            raise ValueError("D-Wave sample values must be binary or spin values")
    return "".join(bits)


def _normalise_vartype(value: object) -> Literal["BINARY", "SPIN"]:
    text = str(value).upper()
    if text == "BINARY":
        return "BINARY"
    if text == "SPIN":
        return "SPIN"
    raise ValueError("D-Wave vartype must be BINARY or SPIN")


def _annealing_observation(
    sample_set: object, request: ModalitySemantics, shots: int
) -> AnnealingObservation:
    """Retain actual record order, values, energies and bytes from native output."""
    record = getattr(sample_set, "record", None)
    native_domain = getattr(sample_set, "vartype", None)
    domain = (
        _normalise_vartype(getattr(native_domain, "name", native_domain))
        if native_domain is not None
        else None
    )
    samples: list[AnnealingSample] = []
    raw_record = None
    if record is not None:
        if (
            not isinstance(record, np.ndarray)
            or len(record.shape) != 1
            or record.dtype.hasobject
            or record.dtype.names is None
            or not {"sample", "energy", "num_occurrences"}.issubset(record.dtype.names)
            or record.dtype["sample"].base.kind not in "iu"
            or record.dtype["energy"].kind not in "fiu"
            or record.dtype["num_occurrences"].kind not in "iu"
        ):
            raise ValueError(
                "native D-Wave record must be a one-dimensional numeric structured array"
            )
        axes = tuple(getattr(sample_set, "variables", ()))
        raw_record = NativeAnnealingRecord(
            dtype_description=str(record.dtype.descr),
            shape=(int(record.shape[0]),),
            itemsize=int(record.dtype.itemsize),
            data=bytes(record.tobytes(order="C")),
        )
        for row in record:
            samples.append(
                AnnealingSample(
                    values=tuple(
                        strict_integer_value(value, field_name="native annealing value")
                        for value in row["sample"]
                    ),
                    occurrences=strict_integer_value(
                        row["num_occurrences"], field_name="native annealing occurrences"
                    ),
                    energy=row["energy"].item(),
                )
            )
    else:
        rows = list(_sample_rows(sample_set))
        axes = tuple(getattr(sample_set, "variables", tuple(rows[0][0]) if rows else ()))
        for sample, occurrences in rows:
            if set(sample) != set(axes):
                raise ValueError("native D-Wave sample labels differ from returned variable order")
            samples.append(
                AnnealingSample(
                    values=tuple(
                        strict_integer_value(sample[axis], field_name="native annealing value")
                        for axis in axes
                    ),
                    occurrences=strict_integer_value(
                        occurrences, field_name="native annealing occurrences"
                    ),
                    energy=None,
                )
            )
    return AnnealingObservation(
        request=request,
        returned_axes=axes,
        samples=tuple(samples),
        shots=shots,
        native_vartype=domain,
        raw_record=raw_record,
    )


def _coerce_int(value: object, *, field_name: str) -> int:
    return strict_integer_value(value, field_name=field_name)


def _coerce_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"{field_name} must be numeric")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "DWAVE_BQM_SCHEMA",
    "DWAVE_EXECUTION_MODE",
    "DWaveLeapHALAdapter",
    "dwave_bqm_workload",
]
