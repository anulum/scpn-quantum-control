# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — analog-mapping analog mapping contracts
"""Typed contracts for bounded analog oscillator mapping feasibility."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
Topology: TypeAlias = Literal["ring", "all_to_all", "sparse"]
Measurement: TypeAlias = Literal["phase_proxy", "population", "z_basis", "quadrature"]
ProfilePosture: TypeAlias = Literal[
    "internal_compiler_model",
    "capability_sketch",
    "unsupported",
]
DiagnosticSeverity: TypeAlias = Literal["blocker", "warning", "info"]

ANALOG_MAPPING_SCHEMA = "analog_mapping_feasibility.v1"
ANALOG_MAPPING_CLAIM_BOUNDARY = (
    "Research-feasibility evidence over an internal compiler model and dated capability "
    "sketches only; no pulse calibration, provider submission, physical dynamical "
    "equivalence, device availability, hardware performance, or analog advantage claim"
)
_TOPOLOGIES = frozenset({"ring", "all_to_all", "sparse"})
_MEASUREMENTS = frozenset({"phase_proxy", "population", "z_basis", "quadrature"})
_POSTURES = frozenset({"internal_compiler_model", "capability_sketch", "unsupported"})
_SEVERITIES = frozenset({"blocker", "warning", "info"})


@dataclass(frozen=True, slots=True)
class AnalogPlatformProfile:
    """Static, source-dated platform capability profile; never a driver."""

    profile_id: str
    display_name: str
    platform_family: str
    posture: ProfilePosture
    supported_topologies: tuple[Topology, ...]
    max_nodes: int | None
    coupling_abs_min: float
    coupling_abs_max: float | None
    supports_signed_couplings: bool
    supports_local_detuning: bool
    supported_measurements: tuple[Measurement, ...]
    control_model: str
    compiler_platform: str | None
    arbitrary_pairwise_control_verified: bool
    source_url: str
    verified_at_source_utc: str
    ledger_ref: str
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject incomplete, promotional, or internally inconsistent profiles."""
        required = (
            self.profile_id,
            self.display_name,
            self.platform_family,
            self.control_model,
            self.source_url,
            self.verified_at_source_utc,
            self.ledger_ref,
        )
        if not all(value.strip() for value in required):
            raise ValueError("platform profile string fields must be non-empty")
        if self.posture not in _POSTURES:
            raise ValueError(f"unknown platform profile posture {self.posture!r}")
        if not self.supported_topologies or any(
            topology not in _TOPOLOGIES for topology in self.supported_topologies
        ):
            raise ValueError("platform profile requires known supported topologies")
        if self.max_nodes is not None and self.max_nodes < 2:
            raise ValueError("max_nodes must be None or an integer >= 2")
        if not math.isfinite(self.coupling_abs_min) or self.coupling_abs_min < 0.0:
            raise ValueError("coupling_abs_min must be finite and non-negative")
        if self.coupling_abs_max is not None and (
            not math.isfinite(self.coupling_abs_max)
            or self.coupling_abs_max <= self.coupling_abs_min
        ):
            raise ValueError("coupling_abs_max must be None or greater than coupling_abs_min")
        if not self.supported_measurements or any(
            measurement not in _MEASUREMENTS for measurement in self.supported_measurements
        ):
            raise ValueError("platform profile requires known supported measurements")
        if not self.source_url.startswith("https://"):
            raise ValueError("platform profile source_url must use HTTPS")
        if not self.limitations or any(not item.strip() for item in self.limitations):
            raise ValueError("platform profile requires non-empty limitations")
        if self.posture == "internal_compiler_model" and self.compiler_platform is None:
            raise ValueError("internal compiler profiles require compiler_platform")
        if self.posture != "internal_compiler_model" and self.compiler_platform is not None:
            raise ValueError("external capability sketches cannot select an internal compiler")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready capability row."""
        payload = asdict(self)
        payload["supported_topologies"] = list(self.supported_topologies)
        payload["supported_measurements"] = list(self.supported_measurements)
        payload["limitations"] = list(self.limitations)
        return payload


@dataclass(frozen=True, slots=True)
class MappingRequest:
    """Immutable analog mapping request in explicit design units."""

    couplings: tuple[tuple[float, ...], ...]
    detunings: tuple[float, ...]
    topology: Topology
    measurement: Measurement
    duration: float = 1.0
    coupling_scale: float = 1.0
    comparison_tolerance: float = 1e-6
    ledger_ref: str = "docs/qpu_provider_readiness.md"

    def __post_init__(self) -> None:
        """Validate symmetry, dimensions, declared topology, and units."""
        n_nodes = len(self.couplings)
        if n_nodes < 2 or any(len(row) != n_nodes for row in self.couplings):
            raise ValueError("couplings must be a square matrix with at least two nodes")
        if len(self.detunings) != n_nodes:
            raise ValueError("detunings length must match coupling dimension")
        matrix = self.coupling_matrix
        detunings = self.detuning_array
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(detunings)):
            raise ValueError("couplings and detunings must contain finite values")
        if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
            raise ValueError("couplings must be symmetric")
        if not np.allclose(np.diag(matrix), 0.0, atol=1e-12, rtol=0.0):
            raise ValueError("coupling diagonal must be zero")
        if self.topology not in _TOPOLOGIES:
            raise ValueError(f"unknown requested topology {self.topology!r}")
        if self.measurement not in _MEASUREMENTS:
            raise ValueError(f"unknown requested measurement {self.measurement!r}")
        for value, name in (
            (self.duration, "duration"),
            (self.coupling_scale, "coupling_scale"),
            (self.comparison_tolerance, "comparison_tolerance"),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.ledger_ref.strip():
            raise ValueError("ledger_ref must be non-empty")

    @classmethod
    def from_arrays(
        cls,
        couplings: FloatArray,
        detunings: FloatArray,
        *,
        topology: Topology,
        measurement: Measurement,
        duration: float = 1.0,
        coupling_scale: float = 1.0,
        comparison_tolerance: float = 1e-6,
        ledger_ref: str = "docs/qpu_provider_readiness.md",
    ) -> MappingRequest:
        """Build a request without retaining mutable NumPy inputs."""
        matrix = np.asarray(couplings, dtype=np.float64)
        frequencies = np.asarray(detunings, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("couplings must be a rank-2 matrix")
        if frequencies.ndim != 1:
            raise ValueError("detunings must be a rank-1 vector")
        return cls(
            couplings=tuple(tuple(float(value) for value in row) for row in matrix),
            detunings=tuple(float(value) for value in frequencies),
            topology=topology,
            measurement=measurement,
            duration=duration,
            coupling_scale=coupling_scale,
            comparison_tolerance=comparison_tolerance,
            ledger_ref=ledger_ref,
        )

    @property
    def n_nodes(self) -> int:
        """Return the oscillator count."""
        return len(self.detunings)

    @property
    def coupling_matrix(self) -> FloatArray:
        """Return a fresh coupling matrix."""
        return np.asarray(self.couplings, dtype=np.float64)

    @property
    def detuning_array(self) -> FloatArray:
        """Return a fresh detuning vector."""
        return np.asarray(self.detunings, dtype=np.float64)

    @property
    def digest(self) -> str:
        """Return the deterministic request digest."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return exact request assumptions in JSON-ready form."""
        return {
            "couplings": [list(row) for row in self.couplings],
            "detunings": list(self.detunings),
            "topology": self.topology,
            "measurement": self.measurement,
            "duration": self.duration,
            "coupling_scale": self.coupling_scale,
            "comparison_tolerance": self.comparison_tolerance,
            "ledger_ref": self.ledger_ref,
        }


@dataclass(frozen=True, slots=True)
class FeasibilityDiagnostic:
    """One fail-closed mapping diagnostic."""

    code: str
    severity: DiagnosticSeverity
    message: str

    def __post_init__(self) -> None:
        """Validate diagnostic vocabulary and prose."""
        if not self.code.strip() or not self.message.strip():
            raise ValueError("diagnostic code and message must be non-empty")
        if self.severity not in _SEVERITIES:
            raise ValueError(f"unknown diagnostic severity {self.severity!r}")

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-ready diagnostic."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MappingResult:
    """Compiler-model mapping result, never a hardware execution record."""

    compiler_platform: str
    n_nodes: int
    n_couplers: int
    reconstructed_coupling_rmse: float
    compiled_program_digest: str
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready compiler result."""
        payload = asdict(self)
        payload["limitations"] = list(self.limitations)
        return payload


@dataclass(frozen=True, slots=True)
class FeasibilityReport:
    """Fail-closed decision for one request/profile pair."""

    schema: str
    request_digest: str
    profile_id: str
    observed_topology: Topology
    supported: bool
    diagnostics: tuple[FeasibilityDiagnostic, ...]
    mapping_result: MappingResult | None
    source_url: str
    verified_at_source_utc: str
    claim_boundary: str = ANALOG_MAPPING_CLAIM_BOUNDARY
    hardware_submission_allowed: bool = False
    hardware_support_claim_allowed: bool = False
    analog_advantage_claim_allowed: bool = False

    def __post_init__(self) -> None:
        """Prevent report flags from outrunning diagnostics or evidence."""
        has_blocker = any(item.severity == "blocker" for item in self.diagnostics)
        if self.supported == has_blocker:
            raise ValueError("supported must be true exactly when no blocker is present")
        if self.supported != (self.mapping_result is not None):
            raise ValueError("supported reports require exactly one mapping_result")
        if (
            self.hardware_submission_allowed
            or self.hardware_support_claim_allowed
            or self.analog_advantage_claim_allowed
        ):
            raise ValueError(
                "analog-mapping reports must keep hardware and advantage claims blocked"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready feasibility report."""
        return {
            "schema": self.schema,
            "request_digest": self.request_digest,
            "profile_id": self.profile_id,
            "observed_topology": self.observed_topology,
            "supported": self.supported,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "mapping_result": (
                self.mapping_result.to_dict() if self.mapping_result is not None else None
            ),
            "source_url": self.source_url,
            "verified_at_source_utc": self.verified_at_source_utc,
            "claim_boundary": self.claim_boundary,
            "hardware_submission_allowed": self.hardware_submission_allowed,
            "hardware_support_claim_allowed": self.hardware_support_claim_allowed,
            "analog_advantage_claim_allowed": self.analog_advantage_claim_allowed,
        }


__all__ = [
    "ANALOG_MAPPING_CLAIM_BOUNDARY",
    "ANALOG_MAPPING_SCHEMA",
    "AnalogPlatformProfile",
    "FeasibilityDiagnostic",
    "FeasibilityReport",
    "MappingRequest",
    "MappingResult",
    "Measurement",
    "Topology",
]
