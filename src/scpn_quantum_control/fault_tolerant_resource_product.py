# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — conservative fault-tolerant resource product
"""Compose existing QEC resource primitives into a conservative fault-tolerant resource report.

The estimator allocates precision across Trotter, logical-failure, and rotation-
synthesis channels; selects an odd surface-code distance with a union bound;
counts rotated-patch and repetition-scaffold qubits using existing QEC helpers;
and applies a cited conservative Clifford+T rotation-synthesis formula. It is a
future-resource planning model, not proof of logical gates, decoding, available
hardware, runtime, or fault-tolerant execution.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Final, Literal

from .qec.error_budget import SURFACE_CODE_THRESHOLD, logical_error_rate
from .qec.logical_dla_parity import (
    repetition_scaffold_physical_qubits,
    surface_code_physical_qubits,
)

SupportPosture = Literal["supported", "research", "boundary"]
SensitivityStatus = Literal["estimated", "refused"]

FT_RESOURCE_PRODUCT_SCHEMA: Final[str] = "fault_tolerant_resource_product.v2"
FT_RESOURCE_CLAIM_BOUNDARY: Final[str] = (
    "Conservative future-resource planning only: rotated-patch register counts, "
    "phenomenological logical-error ansatz, union-bound opportunities, and "
    "Clifford+T rotation-synthesis upper estimate; no available FT hardware, "
    "validated logical RZ/RZZ, decoder integration, magic-state factory, total "
    "runtime, target precision attainment, or fault-tolerant execution claim"
)


class FaultTolerantResourceBoundaryError(ValueError):
    """Raised when the bounded estimator cannot issue an honest estimate."""


@dataclass(frozen=True, slots=True)
class FormulaReference:
    """Primary-source pin for one formula or architectural assumption."""

    formula_id: str
    title: str
    authors: str
    year: int
    url: str
    verified_at_source_utc: str

    def __post_init__(self) -> None:
        """Validate the immutable source pin."""
        if not all(
            value.strip()
            for value in (
                self.formula_id,
                self.title,
                self.authors,
                self.url,
                self.verified_at_source_utc,
            )
        ):
            raise ValueError("formula reference fields must be non-empty")
        if self.year < 1900 or not self.url.startswith("https://"):
            raise ValueError("formula references require a plausible year and HTTPS URL")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready reference record."""
        return asdict(self)


FORMULA_REFERENCES: Final[tuple[FormulaReference, ...]] = (
    FormulaReference(
        "rotated_surface_patch_register",
        "Surface code quantum computing by lattice surgery",
        "Horsman, Fowler, Devitt, and Van Meter",
        2012,
        "https://arxiv.org/abs/1111.4022",
        "2026-07-25T23:11:31Z",
    ),
    FormulaReference(
        "phenomenological_surface_logical_rate",
        "Surface codes: Towards practical large-scale quantum computation",
        "Fowler, Mariantoni, Martinis, and Cleland",
        2012,
        "https://arxiv.org/abs/1208.0928",
        "2026-07-25T23:11:31Z",
    ),
    FormulaReference(
        "clifford_t_z_rotation_upper_estimate",
        "Efficient Clifford+T approximation of single-qubit operators",
        "Peter Selinger",
        2012,
        "https://arxiv.org/abs/1212.6253",
        "2026-07-25T23:11:31Z",
    ),
    FormulaReference(
        "below_threshold_distance_scaling_evidence",
        "Quantum error correction below the surface code threshold",
        "Google Quantum AI and collaborators",
        2024,
        "https://arxiv.org/abs/2408.13687",
        "2026-07-25T23:11:31Z",
    ),
)


@dataclass(frozen=True, slots=True)
class SyncProblemResourceRequest:
    """Bounded Kuramoto/XY problem assumptions for a resource estimate."""

    n_oscillators: int
    evolution_time: float
    target_precision: float
    coupling_density: float
    trotter_steps: int
    physical_error_rate: float = 0.001
    syndrome_cycle_seconds: float = 1e-6
    nisq_shots: int = 4096
    max_code_distance: int = 51

    def __post_init__(self) -> None:
        """Validate request inputs without silently normalising them."""
        if isinstance(self.n_oscillators, bool) or self.n_oscillators < 2:
            raise ValueError("n_oscillators must be an integer >= 2")
        if not math.isfinite(self.evolution_time) or self.evolution_time <= 0.0:
            raise ValueError("evolution_time must be finite and positive")
        if not math.isfinite(self.target_precision) or not 0.0 < self.target_precision < 1.0:
            raise ValueError("target_precision must be finite with 0 < value < 1")
        if not math.isfinite(self.coupling_density) or not 0.0 <= self.coupling_density <= 1.0:
            raise ValueError("coupling_density must be finite with 0 <= value <= 1")
        if isinstance(self.trotter_steps, bool) or self.trotter_steps < 1:
            raise ValueError("trotter_steps must be an integer >= 1")
        if (
            not math.isfinite(self.physical_error_rate)
            or not 0.0 <= self.physical_error_rate < 1.0
        ):
            raise ValueError("physical_error_rate must be finite with 0 <= value < 1")
        if not math.isfinite(self.syndrome_cycle_seconds) or self.syndrome_cycle_seconds <= 0.0:
            raise ValueError("syndrome_cycle_seconds must be finite and positive")
        if isinstance(self.nisq_shots, bool) or self.nisq_shots < 1:
            raise ValueError("nisq_shots must be an integer >= 1")
        if self.max_code_distance < 3 or self.max_code_distance % 2 == 0:
            raise ValueError("max_code_distance must be an odd integer >= 3")

    def to_dict(self) -> dict[str, object]:
        """Return the exact request assumptions."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    """Conservative resource counts and their explicit formula identities."""

    logical_qubits: int
    interacting_pairs: int
    code_distance: int
    surface_code_physical_qubits: int
    repetition_scaffold_physical_qubits: int
    qec_rounds: int
    stabilizer_measurements: int
    arbitrary_rotation_count: int
    rotation_synthesis_precision: float
    t_count_per_rotation: int
    total_t_count: int
    syndrome_time_floor_seconds: float
    logical_failure_union_bound: float
    precision_allocation: dict[str, float]
    formula_ids: tuple[str, ...]
    assumptions: tuple[str, ...]
    hardware_availability_claim_allowed: bool = False
    fault_tolerant_execution_claim_allowed: bool = False

    def __post_init__(self) -> None:
        """Validate count arithmetic and blocked claim defaults."""
        if (
            min(
                self.logical_qubits,
                self.code_distance,
                self.surface_code_physical_qubits,
                self.qec_rounds,
                self.stabilizer_measurements,
                self.arbitrary_rotation_count,
                self.t_count_per_rotation,
                self.total_t_count,
            )
            <= 0
            or self.interacting_pairs < 0
        ):
            raise ValueError("resource counts must be positive except interacting_pairs >= 0")
        if self.total_t_count != self.arbitrary_rotation_count * self.t_count_per_rotation:
            raise ValueError("total_t_count arithmetic mismatch")
        if self.hardware_availability_claim_allowed or self.fault_tolerant_execution_claim_allowed:
            raise ValueError(
                "conservative resource estimates must keep hardware and execution claims blocked"
            )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready resource estimate."""
        payload = asdict(self)
        payload["formula_ids"] = list(self.formula_ids)
        payload["assumptions"] = list(self.assumptions)
        return payload


@dataclass(frozen=True, slots=True)
class SensitivityPoint:
    """One physical-error sensitivity decision."""

    physical_error_rate: float
    status: SensitivityStatus
    code_distance: int | None
    physical_qubits: int | None
    reason: str

    def __post_init__(self) -> None:
        """Keep estimated and refused rows structurally distinct."""
        if not self.reason.strip():
            raise ValueError("sensitivity reason must be non-empty")
        if self.status == "estimated" and (
            self.code_distance is None or self.physical_qubits is None
        ):
            raise ValueError("estimated sensitivity rows require distance and qubits")
        if self.status == "refused" and (
            self.code_distance is not None or self.physical_qubits is not None
        ):
            raise ValueError("refused sensitivity rows cannot carry invented counts")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready sensitivity row."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RegimeComparisonRow:
    """Claim-bounded NISQ, scaffold, analog, simulator, or FT regime row."""

    regime: str
    posture: SupportPosture
    physical_qubits: int | None
    resource_label: str
    claim_boundary: str

    def __post_init__(self) -> None:
        """Validate regime labels."""
        if not all(
            value.strip() for value in (self.regime, self.resource_label, self.claim_boundary)
        ):
            raise ValueError("regime comparison fields must be non-empty")
        if self.physical_qubits is not None and self.physical_qubits < 1:
            raise ValueError("physical_qubits must be positive when present")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready regime row."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FaultTolerantResourceProduct:
    """Complete deterministic fault-tolerant resource report."""

    schema: str
    request: SyncProblemResourceRequest
    estimate: ResourceEstimate
    sensitivity: tuple[SensitivityPoint, ...]
    regimes: tuple[RegimeComparisonRow, ...]
    references: tuple[FormulaReference, ...]
    payload_sha256: str
    claim_boundary: str = FT_RESOURCE_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate product completeness and digest shape."""
        if self.schema != FT_RESOURCE_PRODUCT_SCHEMA:
            raise ValueError(f"unknown product schema: {self.schema!r}")
        if len(self.sensitivity) < 3 or len(self.regimes) != 6 or len(self.references) != 4:
            raise ValueError("fault-tolerant resource report inventory is incomplete")
        if len(self.payload_sha256) != 64:
            raise ValueError("payload_sha256 must be a SHA-256 hex digest")

    def to_dict(self) -> dict[str, object]:
        """Return the deterministic evidence report."""
        return {
            "schema": self.schema,
            "request": self.request.to_dict(),
            "estimate": self.estimate.to_dict(),
            "sensitivity": [row.to_dict() for row in self.sensitivity],
            "regimes": [row.to_dict() for row in self.regimes],
            "references": [row.to_dict() for row in self.references],
            "payload_sha256": self.payload_sha256,
            "claim_boundary": self.claim_boundary,
        }


def _payload_digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _select_code_distance(request: SyncProblemResourceRequest, logical_budget: float) -> int:
    if request.physical_error_rate >= SURFACE_CODE_THRESHOLD:
        raise FaultTolerantResourceBoundaryError(
            "physical_error_rate is at or above the repository surface-code threshold ansatz"
        )
    for distance in range(3, request.max_code_distance + 1, 2):
        opportunities = request.n_oscillators * request.trotter_steps * distance
        union_bound = opportunities * logical_error_rate(distance, request.physical_error_rate)
        if union_bound <= logical_budget:
            return distance
    raise FaultTolerantResourceBoundaryError(
        "no code distance within max_code_distance satisfies the logical union-bound allocation"
    )


def estimate_ft_resources(request: SyncProblemResourceRequest) -> ResourceEstimate:
    """Estimate conservative counts for one bounded request."""
    allocation = {
        "trotter_error_budget": request.target_precision / 3.0,
        "logical_failure_budget": request.target_precision / 3.0,
        "rotation_synthesis_budget": request.target_precision / 3.0,
    }
    distance = _select_code_distance(request, allocation["logical_failure_budget"])
    possible_pairs = request.n_oscillators * (request.n_oscillators - 1) // 2
    interacting_pairs = math.ceil(request.coupling_density * possible_pairs)
    rotations = request.trotter_steps * (request.n_oscillators + interacting_pairs)
    rotation_precision = allocation["rotation_synthesis_budget"] / rotations
    t_per_rotation = math.ceil(10.0 + 4.0 * math.log2(1.0 / rotation_precision))
    qec_rounds = request.trotter_steps * distance
    logical_bound = (
        request.n_oscillators
        * qec_rounds
        * logical_error_rate(distance, request.physical_error_rate)
    )
    surface_qubits = surface_code_physical_qubits(request.n_oscillators, distance)
    repetition_qubits = repetition_scaffold_physical_qubits(request.n_oscillators, distance)
    return ResourceEstimate(
        logical_qubits=request.n_oscillators,
        interacting_pairs=interacting_pairs,
        code_distance=distance,
        surface_code_physical_qubits=surface_qubits,
        repetition_scaffold_physical_qubits=repetition_qubits,
        qec_rounds=qec_rounds,
        stabilizer_measurements=(distance * distance - 1) * request.n_oscillators * qec_rounds,
        arbitrary_rotation_count=rotations,
        rotation_synthesis_precision=rotation_precision,
        t_count_per_rotation=t_per_rotation,
        total_t_count=rotations * t_per_rotation,
        syndrome_time_floor_seconds=qec_rounds * request.syndrome_cycle_seconds,
        logical_failure_union_bound=logical_bound,
        precision_allocation=allocation,
        formula_ids=tuple(reference.formula_id for reference in FORMULA_REFERENCES[:3]),
        assumptions=(
            "one logical oscillator per rotated surface-code patch",
            "one arbitrary Z-axis synthesis per local RZ or pairwise RZZ term per Trotter step",
            "independent logical-error opportunities combined by a union bound",
            "syndrome_time_floor excludes decoding, feed-forward, routing, and magic-state factories",
        ),
    )


def build_ft_sensitivity(
    request: SyncProblemResourceRequest,
    physical_error_rates: tuple[float, ...] = (0.003, 0.001, 0.0001, 0.01),
) -> tuple[SensitivityPoint, ...]:
    """Evaluate the same request across explicit physical-error assumptions."""
    if not physical_error_rates:
        raise ValueError("physical_error_rates must not be empty")
    rows: list[SensitivityPoint] = []
    for rate in physical_error_rates:
        candidate = replace(request, physical_error_rate=rate)
        try:
            estimate = estimate_ft_resources(candidate)
        except FaultTolerantResourceBoundaryError as exc:
            rows.append(SensitivityPoint(rate, "refused", None, None, str(exc)))
        else:
            rows.append(
                SensitivityPoint(
                    rate,
                    "estimated",
                    estimate.code_distance,
                    estimate.surface_code_physical_qubits,
                    "Bounded phenomenological estimate; not hardware or decoder evidence.",
                )
            )
    return tuple(rows)


def build_regime_comparison(
    request: SyncProblemResourceRequest, estimate: ResourceEstimate
) -> tuple[RegimeComparisonRow, ...]:
    """Build six explicit, non-equivalent execution/resource regimes."""
    return (
        RegimeComparisonRow(
            "classical_reference",
            "supported",
            None,
            "Classical solver resource accounting is separate from quantum qubit counts.",
            "Reference computation only; no quantum-resource claim.",
        ),
        RegimeComparisonRow(
            "nisq_sampling",
            "research",
            request.n_oscillators,
            f"{request.nisq_shots} requested samples before provider/transpile overhead.",
            "No error-corrected precision guarantee or provider availability claim.",
        ),
        RegimeComparisonRow(
            "repetition_code_scaffold",
            "boundary",
            estimate.repetition_scaffold_physical_qubits,
            "Bit-flip-only repetition register from qec.fault_tolerant.",
            "Comparison scaffold; phase errors and logical universality are uncorrected.",
        ),
        RegimeComparisonRow(
            "surface_code_scaffold",
            "boundary",
            estimate.surface_code_physical_qubits,
            "Rotated-patch-shaped data plus ancilla register.",
            "No measured syndrome, decoder, or validated logical RZ/RZZ operation.",
        ),
        RegimeComparisonRow(
            "analog_mapping",
            "boundary",
            None,
            "Bounded analog-feasibility dependency; no digital-qubit equivalence invented.",
            "No named analog platform capacity or execution claim.",
        ),
        RegimeComparisonRow(
            "fault_tolerant_planning_model",
            "research",
            estimate.surface_code_physical_qubits,
            "Distance, T-count, stabilizer-measurement, and syndrome-time-floor estimate.",
            FT_RESOURCE_CLAIM_BOUNDARY,
        ),
    )


def build_fault_tolerant_resource_product(
    request: SyncProblemResourceRequest,
) -> FaultTolerantResourceProduct:
    """Build the deterministic resource product and bind its payload digest."""
    estimate = estimate_ft_resources(request)
    sensitivity = build_ft_sensitivity(request)
    regimes = build_regime_comparison(request, estimate)
    payload: dict[str, object] = {
        "schema": FT_RESOURCE_PRODUCT_SCHEMA,
        "request": request.to_dict(),
        "estimate": estimate.to_dict(),
        "sensitivity": [row.to_dict() for row in sensitivity],
        "regimes": [row.to_dict() for row in regimes],
        "references": [row.to_dict() for row in FORMULA_REFERENCES],
        "claim_boundary": FT_RESOURCE_CLAIM_BOUNDARY,
    }
    return FaultTolerantResourceProduct(
        schema=FT_RESOURCE_PRODUCT_SCHEMA,
        request=request,
        estimate=estimate,
        sensitivity=sensitivity,
        regimes=regimes,
        references=FORMULA_REFERENCES,
        payload_sha256=_payload_digest(payload),
    )


def render_ft_resource_markdown(product: FaultTolerantResourceProduct) -> str:
    """Render a compact evidence table without promoting availability claims."""
    estimate = product.estimate
    lines = [
        "# Fault-tolerant synchronisation resource estimate",
        "",
        product.claim_boundary,
        "",
        f"Payload SHA-256: `{product.payload_sha256}`",
        "",
        "## Estimate",
        "",
        "| Logical oscillators | d | Surface qubits | T count | QEC rounds | Stabilizer measurements | Syndrome-time floor (s) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {estimate.logical_qubits} | {estimate.code_distance} | "
        f"{estimate.surface_code_physical_qubits} | {estimate.total_t_count} | "
        f"{estimate.qec_rounds} | {estimate.stabilizer_measurements} | "
        f"{estimate.syndrome_time_floor_seconds:.9g} |",
        "",
        "## Sensitivity",
        "",
        "| Physical error rate | Status | d | Physical qubits | Reason |",
        "| ---: | --- | ---: | ---: | --- |",
    ]
    for row in product.sensitivity:
        lines.append(
            f"| {row.physical_error_rate:.6g} | {row.status} | "
            f"{row.code_distance if row.code_distance is not None else '—'} | "
            f"{row.physical_qubits if row.physical_qubits is not None else '—'} | "
            f"{row.reason} |"
        )
    lines.extend(["", "## Primary source pins", ""])
    lines.extend(f"- [{row.title}]({row.url}) — `{row.formula_id}`" for row in product.references)
    return "\n".join(lines) + "\n"


__all__ = [
    "FORMULA_REFERENCES",
    "FT_RESOURCE_CLAIM_BOUNDARY",
    "FT_RESOURCE_PRODUCT_SCHEMA",
    "FaultTolerantResourceBoundaryError",
    "FaultTolerantResourceProduct",
    "FormulaReference",
    "RegimeComparisonRow",
    "ResourceEstimate",
    "SensitivityPoint",
    "SyncProblemResourceRequest",
    "build_fault_tolerant_resource_product",
    "build_ft_sensitivity",
    "build_regime_comparison",
    "estimate_ft_resources",
    "render_ft_resource_markdown",
]
