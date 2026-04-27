# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control - QPU compute contracts
"""Provider-neutral QPU compute request/result contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

REQUEST_SCHEMA_VERSION = "scpn-quantum-control.qpu-compute-request.v1"
RESULT_SCHEMA_VERSION = "scpn-quantum-control.qpu-compute-result.v1"
NODE_SCHEMA_VERSION = "scpn-quantum-control.qpu-node-descriptor.v1"
STREAM_DELTA_SCHEMA_VERSION = "scpn-quantum-control.qpu-stream-delta.v1"
FUSION_SCHEMA_VERSION = "scpn-quantum-control.qpu-result-fusion.v1"
SUPPORTED_KERNELS = frozenset({"sync_witness", "dla_parity", "sync_dla"})
SUPPORTED_BACKEND_POLICIES = frozenset({"simulator_statevector"})
SUPPORTED_ACCESS_ROUTES = frozenset(
    {
        "aws_braket",
        "azure_quantum",
        "dwave",
        "ibm",
        "ionq_cloud",
        "local",
        "pasqal",
        "quantinuum",
        "research_programme",
    }
)
SUPPORTED_MODALITIES = frozenset(
    {
        "annealer",
        "neutral_atom",
        "photonic",
        "simulator",
        "superconducting",
        "trapped_ion",
    }
)
SUPPORTED_EXECUTION_MODELS = frozenset(
    {
        "analog_hamiltonian",
        "annealing",
        "emulator",
        "gate_model",
        "hybrid_solver",
    }
)
SUPPORTED_LATENCY_CLASSES = frozenset({"batch", "near_real_time", "real_time_candidate"})


def require_non_empty(value: str, name: str) -> str:
    """Return stripped string or raise if it is empty."""
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{name} must be non-empty")
    return cleaned


def json_sha256(payload: Mapping[str, Any]) -> str:
    """Stable SHA-256 over canonical JSON."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def counts_sha256(counts: Mapping[str, int]) -> str:
    """Stable SHA-256 over a count dictionary."""
    return json_sha256({str(key): int(value) for key, value in counts.items()})


@dataclass(frozen=True)
class QPUComputeRequest:
    """Auditable request for a QPU-compatible compute unit."""

    qpu_data_artifact_sha256: str
    kernel: str
    backend_policy: str = "simulator_statevector"
    shots: int = 1024
    kernel_params: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    circuit_limits: dict[str, Any] = field(default_factory=dict)
    mitigation: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str = ""
    output_dir: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        artifact_hash = str(self.qpu_data_artifact_sha256).strip()
        kernel = str(self.kernel).strip()
        backend_policy = str(self.backend_policy).strip()
        shots = int(self.shots)

        if not artifact_hash:
            raise ValueError("qpu_data_artifact_sha256 must be non-empty")
        if kernel not in SUPPORTED_KERNELS:
            raise ValueError(f"kernel must be one of {sorted(SUPPORTED_KERNELS)}")
        if backend_policy not in SUPPORTED_BACKEND_POLICIES:
            raise ValueError(f"backend_policy must be one of {sorted(SUPPORTED_BACKEND_POLICIES)}")
        if shots < 1:
            raise ValueError("shots must be >= 1")

        object.__setattr__(self, "qpu_data_artifact_sha256", artifact_hash)
        object.__setattr__(self, "kernel", kernel)
        object.__setattr__(self, "backend_policy", backend_policy)
        object.__setattr__(self, "shots", shots)
        object.__setattr__(self, "kernel_params", dict(self.kernel_params))
        object.__setattr__(self, "budget", dict(self.budget))
        object.__setattr__(self, "circuit_limits", dict(self.circuit_limits))
        object.__setattr__(self, "mitigation", dict(self.mitigation))
        object.__setattr__(self, "metadata", dict(self.metadata))

        if not self.idempotency_key:
            object.__setattr__(self, "idempotency_key", json_sha256(self._body_dict()))

    def _body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "qpu_data_artifact_sha256": self.qpu_data_artifact_sha256,
            "kernel": self.kernel,
            "backend_policy": self.backend_policy,
            "shots": self.shots,
            "kernel_params": self.kernel_params,
            "budget": self.budget,
            "circuit_limits": self.circuit_limits,
            "mitigation": self.mitigation,
            "output_dir": self.output_dir,
            "metadata": self.metadata,
        }

    @property
    def request_sha256(self) -> str:
        """Stable hash of the request body and idempotency key."""
        payload = self._body_dict()
        payload["idempotency_key"] = self.idempotency_key
        return json_sha256(payload)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the request to a JSON-compatible mapping."""
        payload = self._body_dict()
        payload["idempotency_key"] = self.idempotency_key
        payload["request_sha256"] = self.request_sha256
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise the request to JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QPUComputeRequest:
        """Load and validate a request mapping."""
        if data.get("schema_version") != REQUEST_SCHEMA_VERSION:
            raise ValueError("unsupported QPU compute request schema version")
        request = cls(
            qpu_data_artifact_sha256=str(data["qpu_data_artifact_sha256"]),
            kernel=str(data["kernel"]),
            backend_policy=str(data.get("backend_policy", "simulator_statevector")),
            shots=int(data.get("shots", 1024)),
            kernel_params=dict(data.get("kernel_params", {})),
            budget=dict(data.get("budget", {})),
            circuit_limits=dict(data.get("circuit_limits", {})),
            mitigation=dict(data.get("mitigation", {})),
            idempotency_key=str(data.get("idempotency_key", "")),
            output_dir=data.get("output_dir"),
            metadata=dict(data.get("metadata", {})),
        )
        if (
            data.get("request_sha256") is not None
            and data["request_sha256"] != request.request_sha256
        ):
            raise ValueError("request_sha256 does not match request payload")
        return request

    @classmethod
    def from_json(cls, payload: str) -> QPUComputeRequest:
        """Load and validate a request from JSON."""
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True)
class QPUComputeResult:
    """Auditable output of a QPU-compatible compute unit."""

    request_sha256: str
    qpu_data_artifact_sha256: str
    status: str
    backend_name: str
    backend_family: str
    execution_model: str
    kernel: str
    counts: dict[str, int] = field(default_factory=dict)
    observables: dict[str, float] = field(default_factory=dict)
    observable_classification: dict[str, str] = field(default_factory=dict)
    circuit_metadata: dict[str, Any] = field(default_factory=dict)
    mitigation: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    job_ids: list[str] = field(default_factory=list)
    simulator_seed: int | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.request_sha256).strip():
            raise ValueError("request_sha256 must be non-empty")
        if not str(self.qpu_data_artifact_sha256).strip():
            raise ValueError("qpu_data_artifact_sha256 must be non-empty")
        if not str(self.status).strip():
            raise ValueError("status must be non-empty")
        if self.kernel not in SUPPORTED_KERNELS:
            raise ValueError(f"kernel must be one of {sorted(SUPPORTED_KERNELS)}")
        counts = {str(key): int(value) for key, value in self.counts.items()}
        if any(value < 0 for value in counts.values()):
            raise ValueError("counts must be non-negative")
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "observables", dict(self.observables))
        object.__setattr__(self, "observable_classification", dict(self.observable_classification))
        object.__setattr__(self, "circuit_metadata", dict(self.circuit_metadata))
        object.__setattr__(self, "mitigation", dict(self.mitigation))
        object.__setattr__(self, "timings", dict(self.timings))
        object.__setattr__(self, "job_ids", [str(item) for item in self.job_ids])
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def counts_sha256(self) -> str:
        """Stable hash of the counts payload."""
        return counts_sha256(self.counts)

    @property
    def result_sha256(self) -> str:
        """Stable hash of the result payload."""
        return json_sha256(self._body_dict())

    def _body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "request_sha256": self.request_sha256,
            "qpu_data_artifact_sha256": self.qpu_data_artifact_sha256,
            "status": self.status,
            "backend_name": self.backend_name,
            "backend_family": self.backend_family,
            "execution_model": self.execution_model,
            "kernel": self.kernel,
            "counts": self.counts,
            "counts_sha256": self.counts_sha256,
            "observables": self.observables,
            "observable_classification": self.observable_classification,
            "circuit_metadata": self.circuit_metadata,
            "mitigation": self.mitigation,
            "timings": self.timings,
            "job_ids": self.job_ids,
            "simulator_seed": self.simulator_seed,
            "error": self.error,
            "metadata": self.metadata,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result to a JSON-compatible mapping."""
        payload = self._body_dict()
        payload["result_sha256"] = self.result_sha256
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise the result to JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QPUComputeResult:
        """Load and validate a result mapping."""
        if data.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise ValueError("unsupported QPU compute result schema version")
        result = cls(
            request_sha256=str(data["request_sha256"]),
            qpu_data_artifact_sha256=str(data["qpu_data_artifact_sha256"]),
            status=str(data["status"]),
            backend_name=str(data["backend_name"]),
            backend_family=str(data["backend_family"]),
            execution_model=str(data["execution_model"]),
            kernel=str(data["kernel"]),
            counts=dict(data.get("counts", {})),
            observables=dict(data.get("observables", {})),
            observable_classification=dict(data.get("observable_classification", {})),
            circuit_metadata=dict(data.get("circuit_metadata", {})),
            mitigation=dict(data.get("mitigation", {})),
            timings=dict(data.get("timings", {})),
            job_ids=list(data.get("job_ids", [])),
            simulator_seed=data.get("simulator_seed"),
            error=data.get("error"),
            metadata=dict(data.get("metadata", {})),
        )
        if data.get("counts_sha256") is not None and data["counts_sha256"] != result.counts_sha256:
            raise ValueError("counts_sha256 does not match counts payload")
        if data.get("result_sha256") is not None and data["result_sha256"] != result.result_sha256:
            raise ValueError("result_sha256 does not match result payload")
        return result

    @classmethod
    def from_json(cls, payload: str) -> QPUComputeResult:
        """Load and validate a result from JSON."""
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True)
class QPUNodeDescriptor:
    """Provider-neutral descriptor for one routable QPU or emulator node."""

    node_id: str
    access_route: str
    provider: str
    modality: str
    execution_model: str
    latency_class: str
    qubit_or_variable_limit: int
    native_features: dict[str, Any] = field(default_factory=dict)
    cost_model: dict[str, Any] = field(default_factory=dict)
    queue_model: str = "unknown"
    kernel_capabilities: list[str] = field(default_factory=list)
    calibration_snapshot: dict[str, Any] = field(default_factory=dict)
    verification_status: str = "unverified"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        node_id = require_non_empty(self.node_id, "node_id")
        access_route = require_non_empty(self.access_route, "access_route")
        provider = require_non_empty(self.provider, "provider")
        modality = require_non_empty(self.modality, "modality")
        execution_model = require_non_empty(self.execution_model, "execution_model")
        latency_class = require_non_empty(self.latency_class, "latency_class")
        queue_model = require_non_empty(self.queue_model, "queue_model")
        verification_status = require_non_empty(self.verification_status, "verification_status")
        qubit_or_variable_limit = int(self.qubit_or_variable_limit)
        kernel_capabilities = [str(item).strip() for item in self.kernel_capabilities]

        if access_route not in SUPPORTED_ACCESS_ROUTES:
            raise ValueError(f"access_route must be one of {sorted(SUPPORTED_ACCESS_ROUTES)}")
        if modality not in SUPPORTED_MODALITIES:
            raise ValueError(f"modality must be one of {sorted(SUPPORTED_MODALITIES)}")
        if execution_model not in SUPPORTED_EXECUTION_MODELS:
            raise ValueError(
                f"execution_model must be one of {sorted(SUPPORTED_EXECUTION_MODELS)}"
            )
        if latency_class not in SUPPORTED_LATENCY_CLASSES:
            raise ValueError(f"latency_class must be one of {sorted(SUPPORTED_LATENCY_CLASSES)}")
        if qubit_or_variable_limit < 1:
            raise ValueError("qubit_or_variable_limit must be >= 1")
        if not kernel_capabilities or any(not item for item in kernel_capabilities):
            raise ValueError("kernel_capabilities must contain at least one non-empty item")

        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "access_route", access_route)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "execution_model", execution_model)
        object.__setattr__(self, "latency_class", latency_class)
        object.__setattr__(self, "qubit_or_variable_limit", qubit_or_variable_limit)
        object.__setattr__(self, "native_features", dict(self.native_features))
        object.__setattr__(self, "cost_model", dict(self.cost_model))
        object.__setattr__(self, "queue_model", queue_model)
        object.__setattr__(self, "kernel_capabilities", kernel_capabilities)
        object.__setattr__(self, "calibration_snapshot", dict(self.calibration_snapshot))
        object.__setattr__(self, "verification_status", verification_status)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def descriptor_sha256(self) -> str:
        """Stable hash of the descriptor payload."""
        return json_sha256(self._body_dict())

    def _body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NODE_SCHEMA_VERSION,
            "node_id": self.node_id,
            "access_route": self.access_route,
            "provider": self.provider,
            "modality": self.modality,
            "execution_model": self.execution_model,
            "latency_class": self.latency_class,
            "qubit_or_variable_limit": self.qubit_or_variable_limit,
            "native_features": self.native_features,
            "cost_model": self.cost_model,
            "queue_model": self.queue_model,
            "kernel_capabilities": self.kernel_capabilities,
            "calibration_snapshot": self.calibration_snapshot,
            "verification_status": self.verification_status,
            "metadata": self.metadata,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialise the node descriptor to a JSON-compatible mapping."""
        payload = self._body_dict()
        payload["descriptor_sha256"] = self.descriptor_sha256
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise the node descriptor to JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QPUNodeDescriptor:
        """Load and validate a node descriptor mapping."""
        if data.get("schema_version") != NODE_SCHEMA_VERSION:
            raise ValueError("unsupported QPU node descriptor schema version")
        descriptor = cls(
            node_id=str(data["node_id"]),
            access_route=str(data["access_route"]),
            provider=str(data["provider"]),
            modality=str(data["modality"]),
            execution_model=str(data["execution_model"]),
            latency_class=str(data["latency_class"]),
            qubit_or_variable_limit=int(data["qubit_or_variable_limit"]),
            native_features=dict(data.get("native_features", {})),
            cost_model=dict(data.get("cost_model", {})),
            queue_model=str(data.get("queue_model", "unknown")),
            kernel_capabilities=list(data.get("kernel_capabilities", [])),
            calibration_snapshot=dict(data.get("calibration_snapshot", {})),
            verification_status=str(data.get("verification_status", "unverified")),
            metadata=dict(data.get("metadata", {})),
        )
        if (
            data.get("descriptor_sha256") is not None
            and data["descriptor_sha256"] != descriptor.descriptor_sha256
        ):
            raise ValueError("descriptor_sha256 does not match descriptor payload")
        return descriptor

    @classmethod
    def from_json(cls, payload: str) -> QPUNodeDescriptor:
        """Load and validate a node descriptor from JSON."""
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True)
class QPUStreamDelta:
    """Incremental live-state update for real-time QPU compute loops."""

    stream_id: str
    sequence_id: int
    event_time: str
    ingest_time: str
    artifact_base_sha256: str
    state_delta: dict[str, Any]
    deadline: str | None = None
    control_window: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        stream_id = require_non_empty(self.stream_id, "stream_id")
        event_time = require_non_empty(self.event_time, "event_time")
        ingest_time = require_non_empty(self.ingest_time, "ingest_time")
        artifact_hash = require_non_empty(self.artifact_base_sha256, "artifact_base_sha256")
        sequence_id = int(self.sequence_id)
        confidence = float(self.confidence)
        if sequence_id < 0:
            raise ValueError("sequence_id must be >= 0")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

        object.__setattr__(self, "stream_id", stream_id)
        object.__setattr__(self, "sequence_id", sequence_id)
        object.__setattr__(self, "event_time", event_time)
        object.__setattr__(self, "ingest_time", ingest_time)
        object.__setattr__(self, "artifact_base_sha256", artifact_hash)
        object.__setattr__(self, "state_delta", dict(self.state_delta))
        object.__setattr__(self, "control_window", dict(self.control_window))
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def delta_sha256(self) -> str:
        """Stable hash of the stream delta payload."""
        return json_sha256(self._body_dict())

    def _body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STREAM_DELTA_SCHEMA_VERSION,
            "stream_id": self.stream_id,
            "sequence_id": self.sequence_id,
            "event_time": self.event_time,
            "ingest_time": self.ingest_time,
            "artifact_base_sha256": self.artifact_base_sha256,
            "state_delta": self.state_delta,
            "deadline": self.deadline,
            "control_window": self.control_window,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialise the stream delta to a JSON-compatible mapping."""
        payload = self._body_dict()
        payload["delta_sha256"] = self.delta_sha256
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise the stream delta to JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QPUStreamDelta:
        """Load and validate a stream delta mapping."""
        if data.get("schema_version") != STREAM_DELTA_SCHEMA_VERSION:
            raise ValueError("unsupported QPU stream delta schema version")
        delta = cls(
            stream_id=str(data["stream_id"]),
            sequence_id=int(data["sequence_id"]),
            event_time=str(data["event_time"]),
            ingest_time=str(data["ingest_time"]),
            artifact_base_sha256=str(data["artifact_base_sha256"]),
            state_delta=dict(data["state_delta"]),
            deadline=data.get("deadline"),
            control_window=dict(data.get("control_window", {})),
            confidence=float(data.get("confidence", 1.0)),
            metadata=dict(data.get("metadata", {})),
        )
        if data.get("delta_sha256") is not None and data["delta_sha256"] != delta.delta_sha256:
            raise ValueError("delta_sha256 does not match stream delta payload")
        return delta

    @classmethod
    def from_json(cls, payload: str) -> QPUStreamDelta:
        """Load and validate a stream delta from JSON."""
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True)
class QPUFusionResult:
    """Decision-grade fusion of several QPU compute results."""

    fused_observables: dict[str, float]
    contributing_result_sha256: list[str]
    node_ids: list[str]
    weighting_rule: str
    agreement_metrics: dict[str, float] = field(default_factory=dict)
    excluded_nodes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        weighting_rule = require_non_empty(self.weighting_rule, "weighting_rule")
        result_hashes = [
            require_non_empty(item, "contributing_result_sha256")
            for item in self.contributing_result_sha256
        ]
        node_ids = [require_non_empty(item, "node_id") for item in self.node_ids]
        if not result_hashes:
            raise ValueError("contributing_result_sha256 must not be empty")
        if len(node_ids) != len(result_hashes):
            raise ValueError("node_ids length must match contributing results")
        object.__setattr__(
            self,
            "fused_observables",
            {str(key): float(value) for key, value in self.fused_observables.items()},
        )
        object.__setattr__(self, "contributing_result_sha256", result_hashes)
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "weighting_rule", weighting_rule)
        object.__setattr__(
            self,
            "agreement_metrics",
            {str(key): float(value) for key, value in self.agreement_metrics.items()},
        )
        object.__setattr__(self, "excluded_nodes", [str(item) for item in self.excluded_nodes])
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def fusion_sha256(self) -> str:
        """Stable hash of the fusion payload."""
        return json_sha256(self._body_dict())

    def _body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FUSION_SCHEMA_VERSION,
            "fused_observables": self.fused_observables,
            "contributing_result_sha256": self.contributing_result_sha256,
            "node_ids": self.node_ids,
            "weighting_rule": self.weighting_rule,
            "agreement_metrics": self.agreement_metrics,
            "excluded_nodes": self.excluded_nodes,
            "metadata": self.metadata,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialise the fusion result to a JSON-compatible mapping."""
        payload = self._body_dict()
        payload["fusion_sha256"] = self.fusion_sha256
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise the fusion result to JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QPUFusionResult:
        """Load and validate a fusion result mapping."""
        if data.get("schema_version") != FUSION_SCHEMA_VERSION:
            raise ValueError("unsupported QPU fusion schema version")
        fusion = cls(
            fused_observables=dict(data["fused_observables"]),
            contributing_result_sha256=list(data["contributing_result_sha256"]),
            node_ids=list(data["node_ids"]),
            weighting_rule=str(data["weighting_rule"]),
            agreement_metrics=dict(data.get("agreement_metrics", {})),
            excluded_nodes=list(data.get("excluded_nodes", [])),
            metadata=dict(data.get("metadata", {})),
        )
        if data.get("fusion_sha256") is not None and data["fusion_sha256"] != fusion.fusion_sha256:
            raise ValueError("fusion_sha256 does not match fusion payload")
        return fusion

    @classmethod
    def from_json(cls, payload: str) -> QPUFusionResult:
        """Load and validate a fusion result from JSON."""
        return cls.from_dict(json.loads(payload))


def fuse_compute_results(
    results: Sequence[QPUComputeResult],
    *,
    node_ids: Sequence[str] | None = None,
    weighting_rule: str = "shots",
    excluded_nodes: Sequence[str] = (),
) -> QPUFusionResult:
    """Fuse numeric observables from multiple compute results."""
    if not results:
        raise ValueError("results must not be empty")
    if weighting_rule != "shots":
        raise ValueError("only shots weighting is currently supported")
    resolved_node_ids = (
        [str(item) for item in node_ids]
        if node_ids is not None
        else [result.backend_name for result in results]
    )
    if len(resolved_node_ids) != len(results):
        raise ValueError("node_ids length must match results")

    observable_names = sorted(
        {
            name
            for result in results
            for name, value in result.observables.items()
            if isinstance(value, int | float)
        }
    )
    fused: dict[str, float] = {}
    agreement: dict[str, float] = {}
    weights = [max(1, sum(result.counts.values())) for result in results]
    for name in observable_names:
        values = [
            float(result.observables[name])
            for result in results
            if name in result.observables and isinstance(result.observables[name], int | float)
        ]
        value_weights = [
            float(weight)
            for result, weight in zip(results, weights, strict=True)
            if name in result.observables and isinstance(result.observables[name], int | float)
        ]
        denominator = sum(value_weights)
        if denominator > 0:
            fused[name] = (
                sum(value * weight for value, weight in zip(values, value_weights, strict=True))
                / denominator
            )
            agreement[f"{name}_max_minus_min"] = max(values) - min(values)

    return QPUFusionResult(
        fused_observables=fused,
        contributing_result_sha256=[result.result_sha256 for result in results],
        node_ids=resolved_node_ids,
        weighting_rule=weighting_rule,
        agreement_metrics=agreement,
        excluded_nodes=list(excluded_nodes),
        metadata={
            "n_results": len(results),
            "total_shots": sum(weights),
        },
    )


__all__ = [
    "FUSION_SCHEMA_VERSION",
    "NODE_SCHEMA_VERSION",
    "QPUComputeRequest",
    "QPUComputeResult",
    "QPUFusionResult",
    "QPUNodeDescriptor",
    "QPUStreamDelta",
    "REQUEST_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "STREAM_DELTA_SCHEMA_VERSION",
    "SUPPORTED_ACCESS_ROUTES",
    "SUPPORTED_BACKEND_POLICIES",
    "SUPPORTED_EXECUTION_MODELS",
    "SUPPORTED_KERNELS",
    "SUPPORTED_LATENCY_CLASSES",
    "SUPPORTED_MODALITIES",
    "counts_sha256",
    "fuse_compute_results",
    "json_sha256",
    "require_non_empty",
]
