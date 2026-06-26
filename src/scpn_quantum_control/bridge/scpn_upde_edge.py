# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — SPO UPDE edge
"""Bounded ``knm.scpn-upde`` edge payloads for SPO federation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from numbers import Real
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

SCPN_UPDE_EDGE_SCHEMA = "knm.scpn-upde.v1"
SCPN_UPDE_SCOPE_ENVELOPE = "computational-agreement"
PAPER27_PROVISIONAL_BOUNDARY = (
    "Paper-27 K_nm is provisional and non-canonical; this edge asserts "
    "computational agreement only."
)

JsonObject: TypeAlias = dict[str, Any]


def _finite_float(value: object, *, name: str) -> float:
    """Return ``value`` as a finite real float, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _positive_int(value: object, *, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer >= 1")
    if value < 1:
        raise ValueError(f"{name} must be an integer >= 1")
    return int(value)


def _finite_array(value: object, *, name: str) -> NDArray[np.float64]:
    """Return ``value`` as a finite ``float64`` array, else raise."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _canonical_json(payload: Mapping[str, Any]) -> str:
    """Return canonical JSON for digesting edge payloads."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest_mapping(payload: Mapping[str, Any]) -> str:
    """Return the SHA-256 digest of a canonical JSON mapping."""
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _numeric_digest(value: object, *, name: str) -> str:
    """Return a SHA-256 digest for a numeric JSON-compatible value."""
    return _digest_mapping({"name": name, "value": value})


def edge_content_digest(payload: Mapping[str, Any]) -> str:
    """Return the digest over an edge payload excluding ``edge_sha256``."""
    payload_copy = dict(payload)
    payload_copy.pop("edge_sha256", None)
    return _digest_mapping(payload_copy)


def _operation_counts(counts: Mapping[str, int]) -> dict[str, int]:
    """Return deterministic operation counts for a compiled circuit."""
    return {str(key): int(value) for key, value in sorted(counts.items())}


@dataclass(frozen=True)
class SCPNUPDEEdge:
    """A bounded QUANTUM-to-SPO ``knm.scpn-upde`` handoff.

    The edge carries validated ``K_nm`` and ``omega`` arrays plus enough
    deterministic compiler metadata for SPO to rebuild its own review manifest.
    It never authorises QPU execution or actuation.
    """

    K_nm: NDArray[np.float64]
    omega: NDArray[np.float64]
    time: float
    trotter_steps: int
    trotter_order: int
    circuit_depth: int
    operation_counts: Mapping[str, int]
    claim_boundary: str = PAPER27_PROVISIONAL_BOUNDARY

    def __post_init__(self) -> None:
        K_nm = _finite_array(self.K_nm, name="K_nm")
        omega = _finite_array(self.omega, name="omega")
        if K_nm.ndim != 2 or K_nm.shape[0] != K_nm.shape[1]:
            raise ValueError(f"K_nm must be a square matrix, got shape {K_nm.shape}")
        if omega.shape != (K_nm.shape[0],):
            raise ValueError(f"omega must have shape ({K_nm.shape[0]},), got {omega.shape}")
        if not np.allclose(K_nm, K_nm.T, atol=1e-12, rtol=1e-12):
            raise ValueError("K_nm must be symmetric")
        if not str(self.claim_boundary).strip():
            raise ValueError("claim_boundary must be non-empty")

        object.__setattr__(self, "K_nm", K_nm.copy())
        object.__setattr__(self, "omega", omega.copy())
        object.__setattr__(self, "time", _finite_float(self.time, name="time"))
        object.__setattr__(
            self,
            "trotter_steps",
            _positive_int(self.trotter_steps, name="trotter_steps"),
        )
        object.__setattr__(
            self,
            "trotter_order",
            _positive_int(self.trotter_order, name="trotter_order"),
        )
        object.__setattr__(
            self,
            "circuit_depth",
            _positive_int(self.circuit_depth, name="circuit_depth"),
        )
        object.__setattr__(self, "operation_counts", _operation_counts(self.operation_counts))
        object.__setattr__(self, "claim_boundary", str(self.claim_boundary).strip())

    @property
    def n_oscillators(self) -> int:
        """Return the oscillator/qubit count carried by the edge."""
        return int(self.omega.shape[0])

    @property
    def dt(self) -> float:
        """Return the implied per-step Trotter interval."""
        return float(self.time / self.trotter_steps)

    def to_payload(self) -> JsonObject:
        """Return the JSON-compatible edge payload with an integrity digest."""
        K_payload = self.K_nm.tolist()
        omega_payload = self.omega.tolist()
        payload: JsonObject = {
            "schema": SCPN_UPDE_EDGE_SCHEMA,
            "producer": "scpn-quantum-control",
            "consumer": "scpn-phase-orchestrator",
            "scope_envelope": SCPN_UPDE_SCOPE_ENVELOPE,
            "claim_boundary": self.claim_boundary,
            "n_oscillators": self.n_oscillators,
            "K_nm": K_payload,
            "omega": omega_payload,
            "trotter": {
                "time": self.time,
                "steps": self.trotter_steps,
                "order": self.trotter_order,
                "dt": self.dt,
            },
            "compiler": {
                "kind": "qiskit-pauli-evolution",
                "num_qubits": self.n_oscillators,
                "depth": self.circuit_depth,
                "operation_counts": dict(self.operation_counts),
            },
            "permissions": {
                "qpu_execution_permitted": False,
                "actuation_permitted": False,
            },
            "digests": {
                "K_nm_sha256": _numeric_digest(K_payload, name="K_nm"),
                "omega_sha256": _numeric_digest(omega_payload, name="omega"),
            },
        }
        payload["edge_sha256"] = edge_content_digest(payload)
        return payload


def build_scpn_upde_edge(
    K_nm: NDArray[np.float64],
    omega: NDArray[np.float64],
    *,
    time: float = 0.1,
    trotter_steps: int = 1,
    trotter_order: int = 1,
    claim_boundary: str = PAPER27_PROVISIONAL_BOUNDARY,
) -> SCPNUPDEEdge:
    """Build a bounded ``knm.scpn-upde`` edge from Kuramoto inputs."""
    from ..kuramoto_core import build_kuramoto_problem, compile_trotter_circuit

    problem = build_kuramoto_problem(
        K_nm,
        omega,
        metadata={
            "wire_format": SCPN_UPDE_EDGE_SCHEMA,
            "scope_envelope": SCPN_UPDE_SCOPE_ENVELOPE,
        },
    )
    circuit = compile_trotter_circuit(
        problem,
        time=time,
        trotter_steps=trotter_steps,
        trotter_order=trotter_order,
    )
    return SCPNUPDEEdge(
        K_nm=problem.K_nm,
        omega=problem.omega,
        time=time,
        trotter_steps=trotter_steps,
        trotter_order=trotter_order,
        circuit_depth=int(circuit.depth()),
        operation_counts=cast("Mapping[str, int]", circuit.count_ops()),
        claim_boundary=claim_boundary,
    )


def build_paper27_scpn_upde_edge(
    *,
    time: float = 0.1,
    trotter_steps: int = 1,
    trotter_order: int = 1,
) -> SCPNUPDEEdge:
    """Return the 16-oscillator Paper-27 QUANTUM→SPO edge.

    The emitted scope is deliberately limited to computational agreement; the
    Paper-27 coupling matrix remains provisional rather than canonical physics.
    """
    from .knm_hamiltonian import OMEGA_N_16, build_knm_paper27

    return build_scpn_upde_edge(
        build_knm_paper27(L=16),
        OMEGA_N_16,
        time=time,
        trotter_steps=trotter_steps,
        trotter_order=trotter_order,
    )


def validate_scpn_upde_edge_payload(payload: Mapping[str, Any]) -> None:
    """Validate a ``knm.scpn-upde`` payload emitted by this module."""
    if payload.get("schema") != SCPN_UPDE_EDGE_SCHEMA:
        raise ValueError("schema must be knm.scpn-upde.v1")
    if payload.get("scope_envelope") != SCPN_UPDE_SCOPE_ENVELOPE:
        raise ValueError("scope_envelope must be computational-agreement")
    if payload.get("producer") != "scpn-quantum-control":
        raise ValueError("producer must be scpn-quantum-control")
    if payload.get("consumer") != "scpn-phase-orchestrator":
        raise ValueError("consumer must be scpn-phase-orchestrator")

    permissions = payload.get("permissions")
    if not isinstance(permissions, Mapping):
        raise ValueError("permissions must be a mapping")
    if permissions.get("qpu_execution_permitted") is not False:
        raise ValueError("qpu_execution_permitted must be false")
    if permissions.get("actuation_permitted") is not False:
        raise ValueError("actuation_permitted must be false")

    n_oscillators = _positive_int(payload.get("n_oscillators"), name="n_oscillators")
    K_nm = _finite_array(payload.get("K_nm"), name="K_nm")
    omega = _finite_array(payload.get("omega"), name="omega")
    if K_nm.shape != (n_oscillators, n_oscillators):
        raise ValueError("K_nm shape must match n_oscillators")
    if omega.shape != (n_oscillators,):
        raise ValueError("omega shape must match n_oscillators")

    digests = payload.get("digests")
    if not isinstance(digests, Mapping):
        raise ValueError("digests must be a mapping")
    if digests.get("K_nm_sha256") != _numeric_digest(K_nm.tolist(), name="K_nm"):
        raise ValueError("K_nm_sha256 does not match K_nm")
    if digests.get("omega_sha256") != _numeric_digest(omega.tolist(), name="omega"):
        raise ValueError("omega_sha256 does not match omega")
    if payload.get("edge_sha256") != edge_content_digest(payload):
        raise ValueError("edge_sha256 does not match payload")


__all__ = [
    "PAPER27_PROVISIONAL_BOUNDARY",
    "SCPNUPDEEdge",
    "SCPN_UPDE_EDGE_SCHEMA",
    "SCPN_UPDE_SCOPE_ENVELOPE",
    "build_paper27_scpn_upde_edge",
    "build_scpn_upde_edge",
    "edge_content_digest",
    "validate_scpn_upde_edge_payload",
]
