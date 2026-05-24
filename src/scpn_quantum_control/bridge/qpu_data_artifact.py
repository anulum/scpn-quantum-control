# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — QPU Data Artifact
"""QPU-ready oscillator artifact with provenance gates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

SCHEMA_VERSION = "scpn-quantum-control.qpu-data-artifact.v1"
SC_NEUROCORE_STREAM_SCHEMA = "sc-neurocore.scpn.datastream.v1"
REAL_SOURCE_MODES = frozenset({"recorded", "replay", "curated", "derived"})
SYNTHETIC_SOURCE_MODES = frozenset({"synthetic", "simulation", "fixture"})
ALL_SOURCE_MODES = REAL_SOURCE_MODES | SYNTHETIC_SOURCE_MODES


def _array_sha256(array: NDArray[np.float64]) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_thaw_json_value(item) for item in value]
    return value


def _verify_or_set_array_hash(
    hashes: dict[str, str],
    key: str,
    array: NDArray[np.float64] | None,
) -> None:
    if array is None:
        if key in hashes:
            raise ValueError(f"{key} is present but the corresponding array is absent")
        return
    digest = _array_sha256(array)
    supplied = hashes.get(key)
    if supplied is not None and supplied != digest:
        raise ValueError(f"{key} does not match the artifact array payload")
    hashes.setdefault(key, digest)


def _finite_float_array(name: str, value: Any, *, ndim: int) -> NDArray[np.float64]:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class QPUDataArtifact:
    """Validated oscillator data ready for quantum-control compilation.

    ``K_nm`` follows the Phase Orchestrator convention: row ``i`` receives
    coupling from column ``j``. Current Kuramoto-XY circuits require a
    symmetric, non-negative, zero-diagonal matrix unless the caller explicitly
    opts into more specialised models.
    """

    domain: str
    source_name: str
    source_mode: str
    K_nm: NDArray[np.float64]
    omega: NDArray[np.float64]
    theta0: NDArray[np.float64] | None = None
    layer_assignments: list[str] = field(default_factory=list)
    normalization: str = ""
    extraction_method: str = ""
    source_timestamp: str | None = None
    replay_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    hashes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        domain = str(self.domain).strip()
        source_name = str(self.source_name).strip()
        source_mode = str(self.source_mode).strip()
        normalization = str(self.normalization).strip()
        extraction_method = str(self.extraction_method).strip()
        K_nm = _finite_float_array("K_nm", self.K_nm, ndim=2)
        omega = _finite_float_array("omega", self.omega, ndim=1)
        theta0 = (
            None if self.theta0 is None else _finite_float_array("theta0", self.theta0, ndim=1)
        )
        layer_assignments = [str(item) for item in self.layer_assignments]
        metadata = dict(self.metadata)
        hashes = dict(self.hashes)

        if not domain:
            raise ValueError("domain must be non-empty")
        if not source_name:
            raise ValueError("source_name must be non-empty")
        if source_mode not in ALL_SOURCE_MODES:
            raise ValueError(f"source_mode must be one of {sorted(ALL_SOURCE_MODES)}")
        if not normalization:
            raise ValueError("normalization must be non-empty")
        if not extraction_method:
            raise ValueError("extraction_method must be non-empty")
        if K_nm.shape[0] != K_nm.shape[1]:
            raise ValueError("K_nm must be square")
        if omega.shape != (K_nm.shape[0],):
            raise ValueError(f"omega shape must be ({K_nm.shape[0]},), got {omega.shape}")
        if theta0 is not None and theta0.shape != omega.shape:
            raise ValueError(f"theta0 shape must match omega shape, got {theta0.shape}")
        if layer_assignments and len(layer_assignments) != K_nm.shape[0]:
            raise ValueError("layer_assignments length must match K_nm dimension")
        if not np.allclose(np.diag(K_nm), 0.0, atol=1e-12):
            raise ValueError("K_nm diagonal must be zero")
        if np.any(K_nm < -1e-12):
            raise ValueError("K_nm must be non-negative; encode lag in alpha metadata")
        if not np.allclose(K_nm, K_nm.T, atol=1e-12):
            raise ValueError("K_nm must be symmetric for current Kuramoto-XY circuits")

        _verify_or_set_array_hash(hashes, "K_nm_sha256", K_nm)
        _verify_or_set_array_hash(hashes, "omega_sha256", omega)
        _verify_or_set_array_hash(hashes, "theta0_sha256", theta0)

        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "source_name", source_name)
        object.__setattr__(self, "source_mode", source_mode)
        object.__setattr__(self, "K_nm", K_nm)
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "theta0", theta0)
        object.__setattr__(self, "layer_assignments", layer_assignments)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "extraction_method", extraction_method)
        object.__setattr__(self, "metadata", _freeze_json_value(metadata))
        object.__setattr__(self, "hashes", MappingProxyType(hashes))

    @property
    def n_oscillators(self) -> int:
        """Number of oscillators/qubits implied by the artifact."""
        return int(self.K_nm.shape[0])

    @property
    def is_synthetic(self) -> bool:
        """Whether this artifact is non-publication synthetic data."""
        return self.source_mode in SYNTHETIC_SOURCE_MODES

    def require_publication_safe(self) -> None:
        """Reject synthetic or insufficiently traceable artifacts."""
        if self.is_synthetic:
            raise ValueError("synthetic artifacts are not publication-safe")
        if not (self.source_timestamp or self.replay_id):
            raise ValueError("publication artifacts require source_timestamp or replay_id")

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible mapping."""
        payload = {
            "schema_version": SCHEMA_VERSION,
            "domain": self.domain,
            "source_name": self.source_name,
            "source_mode": self.source_mode,
            "K_nm": self.K_nm.tolist(),
            "omega": self.omega.tolist(),
            "theta0": None if self.theta0 is None else self.theta0.tolist(),
            "layer_assignments": list(self.layer_assignments),
            "normalization": self.normalization,
            "extraction_method": self.extraction_method,
            "source_timestamp": self.source_timestamp,
            "replay_id": self.replay_id,
            "metadata": _thaw_json_value(self.metadata),
            "hashes": dict(self.hashes),
        }
        payload["artifact_sha256"] = _json_sha256(payload)
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise to JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QPUDataArtifact:
        """Load and validate an artifact from a mapping."""
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported QPU data artifact schema version")
        artifact = cls(
            domain=str(data["domain"]),
            source_name=str(data["source_name"]),
            source_mode=str(data["source_mode"]),
            K_nm=np.asarray(data["K_nm"], dtype=np.float64),
            omega=np.asarray(data["omega"], dtype=np.float64),
            theta0=(
                None
                if data.get("theta0") is None
                else np.asarray(data["theta0"], dtype=np.float64)
            ),
            layer_assignments=list(data.get("layer_assignments", [])),
            normalization=str(data["normalization"]),
            extraction_method=str(data["extraction_method"]),
            source_timestamp=data.get("source_timestamp"),
            replay_id=data.get("replay_id"),
            metadata=dict(data.get("metadata", {})),
            hashes=dict(data.get("hashes", {})),
        )
        supplied_artifact_sha256 = data.get("artifact_sha256")
        if supplied_artifact_sha256 is not None:
            computed_artifact_sha256 = artifact.to_dict()["artifact_sha256"]
            if supplied_artifact_sha256 != computed_artifact_sha256:
                raise ValueError("artifact_sha256 does not match the artifact payload")
        return artifact

    @classmethod
    def from_json(cls, payload: str) -> QPUDataArtifact:
        """Load and validate an artifact from JSON."""
        return cls.from_dict(json.loads(payload))

    @classmethod
    def from_scpn_datastream_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        source_mode: str = "synthetic",
        domain: str = "scpn",
        source_name: str = "sc-neurocore-datastream",
        normalization: str = "sc-neurocore canonical stream",
        extraction_method: str = "sc_neurocore.scpn.datastream",
    ) -> QPUDataArtifact:
        """Adapt the current SC-NeuroCore datastream payload for smoke tests.

        The SC-NeuroCore stream generated by commit ``52bd3649`` is deterministic
        and useful for interface tests, but it is not a recorded source artifact.
        It therefore defaults to ``source_mode='synthetic'``.
        """
        if payload.get("schema_version") != SC_NEUROCORE_STREAM_SCHEMA:
            raise ValueError("unsupported SC-NeuroCore datastream schema version")
        K_nm = np.asarray(payload["knm"], dtype=np.float64)
        omega = np.asarray(payload["omega_rad_s"], dtype=np.float64)
        layer_ids = [str(item) for item in payload.get("layer_ids", [])]
        metadata = {
            "source_project": payload.get("source_project", "sc-neurocore"),
            "dt_s": payload.get("dt_s"),
            "seed": payload.get("seed"),
            "n_steps": payload.get("n_steps"),
            "n_layers": payload.get("n_layers"),
            "payload_sha256": _json_sha256(payload),
        }
        return cls(
            domain=domain,
            source_name=source_name,
            source_mode=source_mode,
            K_nm=K_nm,
            omega=omega,
            layer_assignments=layer_ids,
            normalization=normalization,
            extraction_method=extraction_method,
            replay_id=f"seed:{payload.get('seed')}",
            metadata=metadata,
        )


def read_qpu_data_artifact(path: str | Path) -> QPUDataArtifact:
    """Read a QPU data artifact from JSON."""
    return QPUDataArtifact.from_json(Path(path).read_text(encoding="utf-8"))


def write_qpu_data_artifact(path: str | Path, artifact: QPUDataArtifact) -> None:
    """Write a QPU data artifact to JSON."""
    Path(path).write_text(artifact.to_json() + "\n", encoding="utf-8")


def validate_qpu_data_artifact(
    artifact: QPUDataArtifact | Mapping[str, Any],
    *,
    require_publication_safe: bool = True,
) -> QPUDataArtifact:
    """Validate a QPU artifact and optionally enforce publication safety."""
    parsed = (
        artifact if isinstance(artifact, QPUDataArtifact) else QPUDataArtifact.from_dict(artifact)
    )
    if require_publication_safe:
        parsed.require_publication_safe()
    return parsed


def artifact_from_arrays(
    *,
    domain: str,
    source_name: str,
    source_mode: str,
    K_nm: NDArray[np.float64] | Sequence[Sequence[float]],
    omega: NDArray[np.float64] | Sequence[float],
    normalization: str,
    extraction_method: str,
    theta0: NDArray[np.float64] | Sequence[float] | None = None,
    layer_assignments: Sequence[str] = (),
    source_timestamp: str | None = None,
    replay_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> QPUDataArtifact:
    """Convenience constructor for bridge tests and loaders."""
    return QPUDataArtifact(
        domain=domain,
        source_name=source_name,
        source_mode=source_mode,
        K_nm=np.asarray(K_nm, dtype=np.float64),
        omega=np.asarray(omega, dtype=np.float64),
        theta0=None if theta0 is None else np.asarray(theta0, dtype=np.float64),
        layer_assignments=list(layer_assignments),
        normalization=normalization,
        extraction_method=extraction_method,
        source_timestamp=source_timestamp,
        replay_id=replay_id,
        metadata=dict(metadata or {}),
    )


__all__ = [
    "ALL_SOURCE_MODES",
    "QPUDataArtifact",
    "REAL_SOURCE_MODES",
    "SCHEMA_VERSION",
    "SC_NEUROCORE_STREAM_SCHEMA",
    "SYNTHETIC_SOURCE_MODES",
    "artifact_from_arrays",
    "read_qpu_data_artifact",
    "validate_qpu_data_artifact",
    "write_qpu_data_artifact",
]
