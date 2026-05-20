# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — D-Wave Leap adapter for the hardware HAL
"""Direct D-Wave Leap BQM adapter for the provider-neutral HAL."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from importlib import import_module
from typing import Any, cast

from .hal import BackendProfile, QuantumJobRef, QuantumJobResult, QuantumWorkload

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
) -> QuantumWorkload:
    """Encode an Ising/QUBO binary quadratic model as a D-Wave HAL workload."""

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
    return QuantumWorkload(
        workload_id=workload_id,
        ir_format="bqm",
        program=json.dumps(payload, sort_keys=True, separators=(",", ":")),
        n_qubits=n_variables,
        shots=reads,
        metadata=dict(metadata or {}),
    )


class DWaveLeapHALAdapter:
    """Synchronous D-Wave Leap sampler adapter for BQM workloads."""

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
        self._solver = solver or "default"
        self._jobs: dict[str, QuantumJobRef] = {}
        self._results: dict[str, QuantumJobResult] = {}

    def submit(
        self, workload: QuantumWorkload, *, approval_id: str | None = None
    ) -> QuantumJobRef:
        if not approval_id:
            raise PermissionError("approval_id is required for D-Wave Leap submission")
        if workload.ir_format != "bqm":
            raise ValueError("D-Wave Leap direct adapter requires bqm workloads")
        payload = _decode_bqm_payload(workload)
        _validate_bqm_payload(payload, workload.n_qubits)
        bqm = self._build_bqm(payload)
        sample_method = getattr(self._sampler_for(), "sample", None)
        if not callable(sample_method):
            raise TypeError("D-Wave sampler object does not provide sample()")
        sample_set = sample_method(bqm, num_reads=workload.shots, label=workload.workload_id)
        job = QuantumJobRef(
            job_id=f"{self.backend_id}:{workload.workload_id}",
            backend_id=self.backend_id,
            workload_id=workload.workload_id,
            status="completed",
            metadata={
                "approval_id": approval_id,
                "provider_job_id": f"{self.backend_id}:{workload.workload_id}",
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
            counts=_normalise_sample_counts(sample_set, cast(Sequence[str], payload["variables"])),
            shots=workload.shots,
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
        return self._job(job).status

    def result(self, job: QuantumJobRef) -> QuantumJobResult:
        result = self._results.get(job.job_id)
        if result is None:
            raise KeyError(f"unknown job_id: {job.job_id}")
        return result

    def cancel(self, job: QuantumJobRef) -> QuantumJobRef:
        stored = self._job(job)
        cancelled = QuantumJobRef(
            job_id=stored.job_id,
            backend_id=stored.backend_id,
            workload_id=stored.workload_id,
            status="cancelled",
            metadata=stored.metadata,
        )
        self._jobs[job.job_id] = cancelled
        return cancelled

    def _build_bqm(self, payload: dict[str, object]) -> Any:
        factory = self._bqm_factory or _default_bqm_factory
        return factory(payload)

    def _sampler_for(self) -> Any:
        if self._sampler is not None:
            return self._sampler
        factory = self._sampler_factory or _default_sampler_factory
        self._sampler = factory()
        return self._sampler

    def _job(self, job: QuantumJobRef) -> QuantumJobRef:
        stored = self._jobs.get(job.job_id)
        if stored is None:
            raise KeyError(f"unknown job_id: {job.job_id}")
        return stored


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
    vartype = _normalise_vartype(payload.get("vartype"))
    linear = payload.get("linear")
    if not isinstance(linear, Mapping):
        raise ValueError("D-Wave linear biases must be a mapping")
    if set(str(variable) for variable in linear) != set(variable_order):
        raise ValueError("D-Wave linear biases must cover every variable exactly once")
    for variable, bias in linear.items():
        if str(variable) not in variable_order:
            raise ValueError("D-Wave linear bias references unknown variable")
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
    if vartype not in {"BINARY", "SPIN"}:
        raise ValueError("D-Wave vartype must be BINARY or SPIN")


def _normalise_sample_counts(sample_set: object, variables: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample, occurrences in _sample_rows(sample_set):
        bitstring = _sample_bitstring(sample, variables)
        value = _coerce_int(occurrences, field_name="num_occurrences")
        if value < 0:
            raise ValueError("counts values must be non-negative integers")
        counts[bitstring] = counts.get(bitstring, 0) + value
    if not counts:
        raise ValueError("D-Wave sample set did not contain any samples")
    return counts


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


def _sample_bitstring(sample: Mapping[str, object], variables: Sequence[str]) -> str:
    bits: list[str] = []
    for variable in variables:
        if variable not in sample:
            raise ValueError(f"D-Wave sample is missing variable {variable}")
        value = _coerce_int(sample[variable], field_name="sample value")
        if value in (0, 1):
            bits.append(str(value))
        elif value == -1:
            bits.append("0")
        else:
            raise ValueError("D-Wave sample values must be binary or spin values")
    return "".join(bits)


def _normalise_vartype(value: object) -> str:
    text = str(value).upper()
    if text not in {"BINARY", "SPIN"}:
        raise ValueError("D-Wave vartype must be BINARY or SPIN")
    return text


def _coerce_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    try:
        return int(cast(int | str, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _coerce_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "DWAVE_BQM_SCHEMA",
    "DWAVE_EXECUTION_MODE",
    "DWaveLeapHALAdapter",
    "dwave_bqm_workload",
]
