"""Adapters between scpn-phase-orchestrator state and quantum bridge artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .phase_artifact import LayerStateArtifact, LockSignatureArtifact, UPDEPhaseArtifact


def _read_field(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    if default is not None:
        return default
    fields = ", ".join(names)
    raise KeyError(f"Missing required field: one of [{fields}]")


def _infer_layer_pair(key: str, fallback_layer: int) -> tuple[int, int]:
    parts = key.split("_")
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1])
    return fallback_layer, fallback_layer


def _coerce_lock_signature(
    key: str,
    value: Any,
    *,
    fallback_layer: int,
) -> LockSignatureArtifact:
    source_layer, target_layer = _infer_layer_pair(str(key), fallback_layer)
    return LockSignatureArtifact(
        source_layer=int(_read_field(value, "source_layer", default=source_layer)),
        target_layer=int(_read_field(value, "target_layer", default=target_layer)),
        plv=float(_read_field(value, "plv")),
        mean_lag=float(_read_field(value, "mean_lag", "lag")),
    )


class PhaseOrchestratorAdapter:
    """Convert orchestrator state payloads to stable quantum bridge artifacts."""

    @staticmethod
    def from_orchestrator_state(
        state: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> UPDEPhaseArtifact:
        raw_layers = list(_read_field(state, "layers"))
        layers: list[LayerStateArtifact] = []
        for idx, layer in enumerate(raw_layers):
            raw_locks = dict(_read_field(layer, "lock_signatures", "locks", default={}))
            locks = {
                str(key): _coerce_lock_signature(str(key), value, fallback_layer=idx)
                for key, value in raw_locks.items()
            }
            layers.append(
                LayerStateArtifact(
                    R=float(_read_field(layer, "R", "r")),
                    psi=float(_read_field(layer, "psi")),
                    lock_signatures=locks,
                )
            )

        base_metadata = dict(metadata or {})
        base_metadata.setdefault("adapter", "scpn_quantum_control.bridge.orchestrator")
        return UPDEPhaseArtifact(
            layers=layers,
            cross_layer_alignment=_read_field(state, "cross_layer_alignment", "cross_alignment"),
            stability_proxy=float(_read_field(state, "stability_proxy", "stability")),
            regime_id=str(_read_field(state, "regime_id", "regime")),
            metadata=base_metadata,
        )

    @staticmethod
    def to_orchestrator_payload(artifact: UPDEPhaseArtifact) -> dict[str, Any]:
        """Emit payload using canonical orchestrator field names."""
        return artifact.to_dict()

    @staticmethod
    def to_scpn_control_telemetry(artifact: UPDEPhaseArtifact) -> dict[str, Any]:
        """Emit scpn-control telemetry layout for downstream compatibility."""
        return {
            "regime": artifact.regime_id,
            "stability": artifact.stability_proxy,
            "layers": [
                {
                    "R": layer.R,
                    "psi": layer.psi,
                    "locks": {
                        key: {"plv": sig.plv, "lag": sig.mean_lag}
                        for key, sig in layer.lock_signatures.items()
                    },
                }
                for layer in artifact.layers
            ],
            "cross_alignment": artifact.cross_layer_alignment.tolist(),
        }

    @staticmethod
    def build_knm_from_binding_spec(
        binding_spec: Any,
        *,
        zero_diagonal: bool = False,
    ) -> np.ndarray:
        """Build Knm from orchestrator BindingSpec coupling fields.

        This follows the orchestrator default contract:
        K_ij = base_strength * exp(-decay_alpha * |i-j|).
        """
        layers = list(_read_field(binding_spec, "layers"))
        n_osc = sum(len(list(_read_field(layer, "oscillator_ids", default=[]))) for layer in layers)
        if n_osc < 1:
            raise ValueError("BindingSpec must define at least one oscillator.")

        coupling = _read_field(binding_spec, "coupling")
        base_strength = float(_read_field(coupling, "base_strength"))
        decay_alpha = float(_read_field(coupling, "decay_alpha"))

        idx = np.arange(n_osc, dtype=np.float64)
        dist = np.abs(idx[:, None] - idx[None, :])
        knm = base_strength * np.exp(-decay_alpha * dist)
        if zero_diagonal:
            np.fill_diagonal(knm, 0.0)
        return knm

    @staticmethod
    def build_omega_from_binding_spec(
        binding_spec: Any,
        *,
        default_omega: float = 1.0,
    ) -> np.ndarray:
        """Build per-oscillator omega vector from binding spec layer metadata.

        If a layer contains ``natural_frequency`` metadata, that value is applied
        to all oscillators in that layer. Otherwise ``default_omega`` is used.
        """
        layers = list(_read_field(binding_spec, "layers"))
        omegas: list[float] = []
        for layer in layers:
            osc_ids = list(_read_field(layer, "oscillator_ids", default=[]))
            layer_omega = float(_read_field(layer, "natural_frequency", default=default_omega))
            omegas.extend([layer_omega] * len(osc_ids))
        if not omegas:
            raise ValueError("BindingSpec must define at least one oscillator.")
        return np.asarray(omegas, dtype=np.float64)


__all__ = ["PhaseOrchestratorAdapter"]
