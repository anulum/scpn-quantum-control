"""Shared phase-state artifact schema for SCPN classical/quantum interoperability."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _as_finite_float(name: str, value: Any) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return x


@dataclass(frozen=True)
class LockSignatureArtifact:
    """Pairwise lock metrics between source and target layers."""

    source_layer: int
    target_layer: int
    plv: float
    mean_lag: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_layer", int(self.source_layer))
        object.__setattr__(self, "target_layer", int(self.target_layer))
        object.__setattr__(self, "plv", _as_finite_float("plv", self.plv))
        object.__setattr__(self, "mean_lag", _as_finite_float("mean_lag", self.mean_lag))
        if self.source_layer < 0 or self.target_layer < 0:
            raise ValueError("source_layer and target_layer must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_layer": self.source_layer,
            "target_layer": self.target_layer,
            "plv": self.plv,
            "mean_lag": self.mean_lag,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LockSignatureArtifact:
        return cls(
            source_layer=int(data["source_layer"]),
            target_layer=int(data["target_layer"]),
            plv=float(data["plv"]),
            mean_lag=float(data["mean_lag"]),
        )


@dataclass(frozen=True)
class LayerStateArtifact:
    """Layer-local coherence metrics and lock signatures."""

    R: float
    psi: float
    lock_signatures: dict[str, LockSignatureArtifact] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "R", _as_finite_float("R", self.R))
        object.__setattr__(self, "psi", _as_finite_float("psi", self.psi))
        if not 0.0 <= self.R <= 1.0:
            raise ValueError(f"R must be in [0, 1], got {self.R}")
        for key in self.lock_signatures:
            if not isinstance(key, str):
                raise TypeError("lock_signatures keys must be strings.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "R": self.R,
            "psi": self.psi,
            "lock_signatures": {
                key: value.to_dict() for key, value in self.lock_signatures.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LayerStateArtifact:
        raw_locks = data.get("lock_signatures", {})
        locks = {
            str(key): (
                value
                if isinstance(value, LockSignatureArtifact)
                else LockSignatureArtifact.from_dict(value)
            )
            for key, value in dict(raw_locks).items()
        }
        return cls(R=float(data["R"]), psi=float(data["psi"]), lock_signatures=locks)


@dataclass(frozen=True)
class UPDEPhaseArtifact:
    """Portable UPDE state artifact shared across execution backends."""

    layers: list[LayerStateArtifact]
    cross_layer_alignment: NDArray[np.float64]
    stability_proxy: float
    regime_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        layers = list(self.layers)
        alignment = np.asarray(self.cross_layer_alignment, dtype=np.float64)
        stability = _as_finite_float("stability_proxy", self.stability_proxy)
        regime_id = str(self.regime_id).strip()
        metadata = dict(self.metadata)

        if not regime_id:
            raise ValueError("regime_id must be non-empty.")
        if alignment.ndim != 2:
            raise ValueError("cross_layer_alignment must be a 2-D matrix.")
        n_layers = len(layers)
        if alignment.shape != (n_layers, n_layers):
            raise ValueError(
                "cross_layer_alignment shape must match number of layers: "
                f"expected ({n_layers}, {n_layers}), got {alignment.shape}"
            )
        if not np.all(np.isfinite(alignment)):
            raise ValueError("cross_layer_alignment must contain only finite values.")

        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "cross_layer_alignment", alignment)
        object.__setattr__(self, "stability_proxy", stability)
        object.__setattr__(self, "regime_id", regime_id)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layers": [layer.to_dict() for layer in self.layers],
            "cross_layer_alignment": self.cross_layer_alignment.tolist(),
            "stability_proxy": self.stability_proxy,
            "regime_id": self.regime_id,
            "metadata": dict(self.metadata),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UPDEPhaseArtifact:
        raw_layers = list(data.get("layers", []))
        layers = [
            layer if isinstance(layer, LayerStateArtifact) else LayerStateArtifact.from_dict(layer)
            for layer in raw_layers
        ]
        alignment = np.asarray(data["cross_layer_alignment"], dtype=np.float64)
        metadata = dict(data.get("metadata", {}))
        return cls(
            layers=layers,
            cross_layer_alignment=alignment,
            stability_proxy=float(data["stability_proxy"]),
            regime_id=str(data["regime_id"]),
            metadata=metadata,
        )

    @classmethod
    def from_json(cls, payload: str) -> UPDEPhaseArtifact:
        return cls.from_dict(json.loads(payload))


__all__ = [
    "LockSignatureArtifact",
    "LayerStateArtifact",
    "UPDEPhaseArtifact",
]
