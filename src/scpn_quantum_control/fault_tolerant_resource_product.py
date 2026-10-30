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
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
from numpy.typing import NDArray

from .qec.error_budget import SURFACE_CODE_THRESHOLD, logical_error_rate
from .qec.logical_dla_parity import (
    repetition_scaffold_physical_qubits,
    surface_code_physical_qubits,
)

if TYPE_CHECKING:
    from .qec.control_qec import MWPMDecoder, SurfaceCode

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


# --- Calibrated logical-failure confidence -------------------------------------------

LOGICAL_FAILURE_CONFIDENCE_METHOD: Final[str] = "clopper_pearson_exact"
"""Exact binomial interval; it never collapses to zero width at zero failures."""


@dataclass(frozen=True, slots=True)
class LogicalFailureConfidence:
    """A logical-failure rate stated only for the sample that produced it.

    Attributes
    ----------
    failures, trials
        Observed logical failures and the number of independent trials.
    confidence_level
        Two-sided confidence level of the exact interval.
    lower, upper
        Clopper–Pearson bounds on the per-trial failure probability.
    sample_model
        The exact code, decoder and noise model the trials were drawn from.
    method
        Interval method; always :data:`LOGICAL_FAILURE_CONFIDENCE_METHOD`.

    """

    failures: int
    trials: int
    confidence_level: float
    lower: float
    upper: float
    sample_model: str
    method: str = LOGICAL_FAILURE_CONFIDENCE_METHOD

    @property
    def rate(self) -> float:
        """Return the observed failure fraction."""
        return self.failures / self.trials

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready confidence statement including the observed rate."""
        return {**asdict(self), "rate": self.rate}


def logical_failure_confidence(
    failures: int,
    trials: int,
    *,
    sample_model: str,
    confidence_level: float = 0.95,
) -> LogicalFailureConfidence:
    """State an exact two-sided confidence interval for sampled logical failures.

    Parameters
    ----------
    failures
        Observed logical failures, ``0 <= failures <= trials``.
    trials
        Independent trials actually sampled, at least one.
    sample_model
        Non-empty description of the code, decoder and noise model sampled.
    confidence_level
        Two-sided level with ``0 < confidence_level < 1``.

    Returns
    -------
    LogicalFailureConfidence
        Clopper–Pearson bounds. Zero failures still give a positive upper bound,
        and all failures still give a lower bound below one.

    Raises
    ------
    ValueError
        If the counts, level or sample model are malformed.

    """
    from scipy.stats import beta

    for name, value in (("failures", failures), ("trials", trials)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if trials < 1 or not 0 <= failures <= trials:
        raise ValueError("counts must satisfy 0 <= failures <= trials and trials >= 1")
    if (
        isinstance(confidence_level, bool)
        or not math.isfinite(confidence_level)
        or not 0.0 < confidence_level < 1.0
    ):
        raise ValueError("confidence_level must be finite with 0 < value < 1")
    if not sample_model.strip():
        raise ValueError("sample_model must name the sampled code, decoder and noise model")
    alpha = 1.0 - confidence_level
    lower = 0.0 if failures == 0 else float(beta.ppf(alpha / 2.0, failures, trials - failures + 1))
    upper = (
        1.0
        if failures == trials
        else float(beta.ppf(1.0 - alpha / 2.0, failures + 1, trials - failures))
    )
    return LogicalFailureConfidence(failures, trials, confidence_level, lower, upper, sample_model)


# --- Structural code scaffold and decoder experiment ---------------------------------

ADMITTED_TORIC_DISTANCES: Final[tuple[int, ...]] = (3, 5)
"""Toric-code distances whose syndromes and decoding this module admits."""
TORIC_ERROR_MODEL: Final[str] = (
    "independent bit flips on every data qubit at the stated rate; one round of "
    "perfect vertex-syndrome extraction; no measurement, leakage or correlated errors"
)
DECODER_EXPERIMENT_CLAIM_BOUNDARY: Final[str] = (
    "Software Monte Carlo of a toric code under the stated error model only; no "
    "hardware syndrome, circuit-level noise, measured logical-error reduction or "
    "threshold claim"
)


def _admitted_toric_distance(distance: int) -> int:
    if isinstance(distance, bool) or distance not in ADMITTED_TORIC_DISTANCES:
        raise FaultTolerantResourceBoundaryError(
            f"toric distance must be one of {ADMITTED_TORIC_DISTANCES}"
        )
    return distance


def _toric_logical_parity(residual: NDArray[np.int8], distance: int) -> bool:
    """Return whether a syndrome-free residual winds around the torus."""
    horizontal_cut = [2 * row * distance for row in range(distance)]
    vertical_cut = [2 * column + 1 for column in range(distance)]
    return bool(residual[horizontal_cut].sum() % 2 or residual[vertical_cut].sum() % 2)


def _toric_trial_failed(code: SurfaceCode, decoder: MWPMDecoder, error: NDArray[np.int8]) -> bool:
    syndrome = ((code.Hx @ error) % 2).astype(np.int8)
    residual = (error ^ decoder.decode(syndrome)).astype(np.int8)
    # A correction that leaves any syndrome counts as a failure, never as a success.
    return bool(((code.Hx @ residual) % 2).any()) or _toric_logical_parity(residual, code.d)


@dataclass(frozen=True, slots=True)
class SingleErrorSyndrome:
    """The syndrome and decoding outcome of one data-qubit flip.

    Attributes
    ----------
    data_qubit
        Flipped data-qubit index.
    defects
        Vertex checks that fire, in increasing order.
    corrected
        Whether matching returns the code to a trivial logical class.

    """

    data_qubit: int
    defects: tuple[int, ...]
    corrected: bool


def enumerate_single_error_syndromes(distance: int = 3) -> tuple[SingleErrorSyndrome, ...]:
    """Flip each data qubit of an admitted toric code once and decode it.

    Parameters
    ----------
    distance
        Admitted toric distance from :data:`ADMITTED_TORIC_DISTANCES`.

    Returns
    -------
    tuple[SingleErrorSyndrome, ...]
        One row per data qubit, in data-qubit order.

    Raises
    ------
    FaultTolerantResourceBoundaryError
        If the distance is not admitted.

    """
    from .qec.control_qec import MWPMDecoder, SurfaceCode

    code = SurfaceCode(_admitted_toric_distance(distance))
    decoder = MWPMDecoder(distance)
    rows: list[SingleErrorSyndrome] = []
    for qubit in range(code.num_data):
        error = np.zeros(code.num_data, dtype=np.int8)
        error[qubit] = 1
        defects = tuple(int(index) for index in np.flatnonzero((code.Hx @ error) % 2))
        rows.append(
            SingleErrorSyndrome(qubit, defects, not _toric_trial_failed(code, decoder, error))
        )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class DecoderExperiment:
    """One seeded decoder Monte Carlo and the confidence its sample supports.

    Attributes
    ----------
    distance, physical_error_rate, shots, seed
        Exactly what was sampled.
    confidence
        Exact interval on the logical-failure probability of this sample model.
    code, decoder, error_model, claim_boundary
        Fixed identities of the scaffold, decoder, noise model and claim scope.
    measured_logical_error_reduction_claim_allowed
        Always false; a software sample is not a hardware measurement.

    """

    distance: int
    physical_error_rate: float
    shots: int
    seed: int
    confidence: LogicalFailureConfidence
    code: str = "toric_surface_code"
    decoder: str = "minimum_weight_perfect_matching"
    error_model: str = TORIC_ERROR_MODEL
    claim_boundary: str = DECODER_EXPERIMENT_CLAIM_BOUNDARY
    measured_logical_error_reduction_claim_allowed: bool = False

    def __post_init__(self) -> None:
        """Keep the experiment claim blocked and its sample consistent."""
        if self.measured_logical_error_reduction_claim_allowed:
            raise ValueError("a software decoder sample cannot claim measured error reduction")
        if self.confidence.trials != self.shots:
            raise ValueError("confidence must be stated for exactly the sampled shots")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready experiment record."""
        payload = asdict(self)
        payload["confidence"] = self.confidence.to_dict()
        return payload


def run_toric_decoder_experiment(
    distance: int,
    physical_error_rate: float,
    shots: int,
    seed: int,
    *,
    confidence_level: float = 0.95,
) -> DecoderExperiment:
    """Sample bit-flip errors on an admitted toric code and decode each shot.

    Parameters
    ----------
    distance
        Admitted toric distance.
    physical_error_rate
        Per-data-qubit flip probability, ``0 <= p < 1``.
    shots
        Independent trials, at least one.
    seed
        Non-negative seed of the NumPy generator; the same seed repeats the sample.
    confidence_level
        Two-sided level of the exact interval.

    Returns
    -------
    DecoderExperiment
        Counts and an interval stated only for this sample model.

    Raises
    ------
    FaultTolerantResourceBoundaryError
        If the distance is not admitted.
    ValueError
        If the rate, shots or seed are malformed.

    """
    from .qec.control_qec import MWPMDecoder, SurfaceCode

    code = SurfaceCode(_admitted_toric_distance(distance))
    if (
        isinstance(physical_error_rate, bool)
        or not math.isfinite(physical_error_rate)
        or not 0.0 <= physical_error_rate < 1.0
    ):
        raise ValueError("physical_error_rate must be finite with 0 <= value < 1")
    for name, value in (("shots", shots), ("seed", seed)):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (1 if name == "shots" else 0)
        ):
            raise ValueError(
                f"{name} must be a {'positive' if name == 'shots' else 'non-negative'} integer"
            )
    decoder = MWPMDecoder(distance)
    errors = (
        np.random.default_rng(seed).random((shots, code.num_data)) < physical_error_rate
    ).astype(np.int8)
    failures = sum(_toric_trial_failed(code, decoder, error) for error in errors)
    sample_model = (
        f"toric code d={distance}, minimum-weight perfect matching, {TORIC_ERROR_MODEL}, "
        f"p={physical_error_rate!r}, seed={seed}"
    )
    return DecoderExperiment(
        distance=distance,
        physical_error_rate=physical_error_rate,
        shots=shots,
        seed=seed,
        confidence=logical_failure_confidence(
            int(failures), shots, sample_model=sample_model, confidence_level=confidence_level
        ),
    )


# --- Logical-operation semantics ------------------------------------------------------

LogicalOperationStatus = Literal["counted_not_validated", "modelled_ideal", "not_modelled"]


@dataclass(frozen=True, slots=True)
class LogicalOperationSemantics:
    """What the estimate means, and does not mean, for one logical operation."""

    operation: str
    status: LogicalOperationStatus
    meaning: str


LOGICAL_OPERATION_SEMANTICS: Final[tuple[LogicalOperationSemantics, ...]] = (
    LogicalOperationSemantics(
        "logical_rz",
        "counted_not_validated",
        "T gates are counted from a Clifford+T synthesis upper estimate; no logical RZ is validated.",
    ),
    LogicalOperationSemantics(
        "logical_rzz",
        "counted_not_validated",
        "Each pairwise term is counted as one synthesised Z rotation; no lattice-surgery RZZ is validated.",
    ),
    LogicalOperationSemantics(
        "syndrome_extraction",
        "modelled_ideal",
        "Decoder experiments assume one perfect extraction round; circuit-level extraction is not modelled.",
    ),
    LogicalOperationSemantics(
        "magic_state_distillation",
        "not_modelled",
        "Factory footprint and throughput enter only through caller-stated overhead scenarios.",
    ),
    LogicalOperationSemantics(
        "lattice_surgery_routing",
        "not_modelled",
        "Routing enters only as a caller-stated overhead factor on the data patches.",
    ),
)


# --- Future resource scenarios with explicit overheads --------------------------------

QEC_CLAIM_QUALIFICATION_SCHEMA: Final[str] = "qec_claim_qualification.v1"
EXCLUDED_FROM_BASE_ESTIMATE: Final[tuple[str, ...]] = (
    "lattice-surgery routing space and time",
    "magic-state factory footprint and throughput",
    "decoder latency and feed-forward",
    "measurement, leakage and correlated noise",
)
OVERHEAD_SCENARIO_CLAIM_BOUNDARY: Final[str] = (
    "Time and space floors under caller-stated routing, factory and decoder "
    "assumptions; a scenario spread, not a runtime, device-availability or "
    "fault-tolerant-execution claim"
)


@dataclass(frozen=True, slots=True)
class OverheadAssumptions:
    """Caller-stated routing, factory and decoder assumptions for one scenario."""

    label: str
    routing_overhead_factor: float
    factory_count: int
    factory_physical_qubits: int
    factory_cycles_per_t_state: int
    decoder_latency_seconds_per_round: float = 0.0

    def __post_init__(self) -> None:
        """Refuse assumptions that would silently shrink or invent resources."""
        if not self.label.strip():
            raise ValueError("scenario label must be non-empty")
        if (
            isinstance(self.routing_overhead_factor, bool)
            or not math.isfinite(self.routing_overhead_factor)
            or self.routing_overhead_factor < 1.0
        ):
            raise ValueError("routing_overhead_factor must be finite and >= 1")
        for name in ("factory_count", "factory_physical_qubits", "factory_cycles_per_t_state"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        if (
            isinstance(self.decoder_latency_seconds_per_round, bool)
            or not math.isfinite(self.decoder_latency_seconds_per_round)
            or self.decoder_latency_seconds_per_round < 0.0
        ):
            raise ValueError("decoder_latency_seconds_per_round must be finite and >= 0")


@dataclass(frozen=True, slots=True)
class OverheadScenarioEstimate:
    """Space and time floors of the base estimate under one overhead scenario."""

    assumptions: OverheadAssumptions
    data_patch_physical_qubits: int
    routed_patch_physical_qubits: int
    factory_physical_qubits: int
    total_physical_qubits: int
    t_state_time_floor_seconds: float
    decoding_time_floor_seconds: float
    scenario_time_floor_seconds: float
    claim_boundary: str = OVERHEAD_SCENARIO_CLAIM_BOUNDARY
    hardware_availability_claim_allowed: bool = False
    total_runtime_claim_allowed: bool = False

    def __post_init__(self) -> None:
        """Keep scenario arithmetic exact and availability claims blocked."""
        if self.hardware_availability_claim_allowed or self.total_runtime_claim_allowed:
            raise ValueError(
                "overhead scenarios must keep availability and runtime claims blocked"
            )
        if self.total_physical_qubits != (
            self.routed_patch_physical_qubits + self.factory_physical_qubits
        ):
            raise ValueError("total_physical_qubits arithmetic mismatch")


def estimate_overhead_scenario(
    request: SyncProblemResourceRequest, assumptions: OverheadAssumptions
) -> OverheadScenarioEstimate:
    """Add caller-stated routing, factory and decoder overheads to the base estimate.

    Parameters
    ----------
    request
        The same bounded request the base estimate uses.
    assumptions
        Routing factor, factory count, size and cycle cost, and decoder latency.

    Returns
    -------
    OverheadScenarioEstimate
        Routed patches plus factories, and the larger of the syndrome and
        T-state time floors plus decoding latency.

    Raises
    ------
    FaultTolerantResourceBoundaryError
        If the base estimate itself is refused.

    """
    base = estimate_ft_resources(request)
    routed = math.ceil(base.surface_code_physical_qubits * assumptions.routing_overhead_factor)
    factories = assumptions.factory_count * assumptions.factory_physical_qubits
    t_state_floor = (
        math.ceil(base.total_t_count / assumptions.factory_count)
        * assumptions.factory_cycles_per_t_state
        * request.syndrome_cycle_seconds
    )
    decoding_floor = base.qec_rounds * assumptions.decoder_latency_seconds_per_round
    return OverheadScenarioEstimate(
        assumptions=assumptions,
        data_patch_physical_qubits=base.surface_code_physical_qubits,
        routed_patch_physical_qubits=routed,
        factory_physical_qubits=factories,
        total_physical_qubits=routed + factories,
        t_state_time_floor_seconds=t_state_floor,
        decoding_time_floor_seconds=decoding_floor,
        scenario_time_floor_seconds=max(base.syndrome_time_floor_seconds, t_state_floor)
        + decoding_floor,
    )


@dataclass(frozen=True, slots=True)
class QecClaimQualification:
    """Separate decoder evidence, code scaffold, operation semantics and future resources.

    The base resource product keeps its own schema and digest; this companion
    record refers to it by digest and never alters it.
    """

    schema: str
    resource_payload_sha256: str
    scenarios: tuple[OverheadScenarioEstimate, ...]
    total_physical_qubit_range: tuple[int, int]
    scenario_time_floor_range_seconds: tuple[float, float]
    excluded_from_base_estimate: tuple[str, ...]
    logical_operations: tuple[LogicalOperationSemantics, ...]
    decoder_experiment: DecoderExperiment | None
    payload_sha256: str
    device_available_claim_allowed: bool = False

    def __post_init__(self) -> None:
        """Refuse an incomplete or availability-claiming record."""
        if self.schema != QEC_CLAIM_QUALIFICATION_SCHEMA:
            raise ValueError(f"unknown qualification schema: {self.schema!r}")
        if self.device_available_claim_allowed:
            raise ValueError("resource estimates never establish device availability")
        if not self.scenarios:
            raise ValueError("at least one overhead scenario is required")

    def to_dict(self) -> dict[str, object]:
        """Return the deterministic qualification record."""
        return {
            **_qualification_payload(self),
            "payload_sha256": self.payload_sha256,
        }


def _qualification_payload(record: QecClaimQualification) -> dict[str, object]:
    return {
        "schema": record.schema,
        "resource_payload_sha256": record.resource_payload_sha256,
        "scenarios": [asdict(row) for row in record.scenarios],
        "total_physical_qubit_range": list(record.total_physical_qubit_range),
        "scenario_time_floor_range_seconds": list(record.scenario_time_floor_range_seconds),
        "excluded_from_base_estimate": list(record.excluded_from_base_estimate),
        "logical_operations": [asdict(row) for row in record.logical_operations],
        "decoder_experiment": (
            None if record.decoder_experiment is None else record.decoder_experiment.to_dict()
        ),
        "device_available_claim_allowed": record.device_available_claim_allowed,
    }


def build_qec_claim_qualification(
    request: SyncProblemResourceRequest,
    scenarios: tuple[OverheadAssumptions, ...],
    *,
    decoder_experiment: DecoderExperiment | None = None,
) -> QecClaimQualification:
    """Qualify a resource estimate against explicit overheads and decoder evidence.

    Parameters
    ----------
    request
        Bounded request of the base resource product.
    scenarios
        One or more uniquely labelled overhead scenarios; their spread is the
        reported scaling uncertainty.
    decoder_experiment
        Optional sampled decoder evidence, kept separate from the resource model.

    Returns
    -------
    QecClaimQualification
        Companion record bound to the base product by its payload digest.

    Raises
    ------
    ValueError
        If no scenario is given or labels repeat.
    FaultTolerantResourceBoundaryError
        If the base estimate is refused.

    """
    if not scenarios:
        raise ValueError("at least one overhead scenario is required")
    labels = [scenario.label for scenario in scenarios]
    if len(set(labels)) != len(labels):
        raise ValueError("overhead scenario labels must be unique")
    product = build_fault_tolerant_resource_product(request)
    rows = tuple(estimate_overhead_scenario(request, scenario) for scenario in scenarios)
    totals = [row.total_physical_qubits for row in rows]
    floors = [row.scenario_time_floor_seconds for row in rows]
    draft = QecClaimQualification(
        schema=QEC_CLAIM_QUALIFICATION_SCHEMA,
        resource_payload_sha256=product.payload_sha256,
        scenarios=rows,
        total_physical_qubit_range=(min(totals), max(totals)),
        scenario_time_floor_range_seconds=(min(floors), max(floors)),
        excluded_from_base_estimate=EXCLUDED_FROM_BASE_ESTIMATE,
        logical_operations=LOGICAL_OPERATION_SEMANTICS,
        decoder_experiment=decoder_experiment,
        payload_sha256="0" * 64,
    )
    return replace(draft, payload_sha256=_payload_digest(_qualification_payload(draft)))


__all__ = [
    "ADMITTED_TORIC_DISTANCES",
    "DECODER_EXPERIMENT_CLAIM_BOUNDARY",
    "EXCLUDED_FROM_BASE_ESTIMATE",
    "FORMULA_REFERENCES",
    "FT_RESOURCE_CLAIM_BOUNDARY",
    "FT_RESOURCE_PRODUCT_SCHEMA",
    "LOGICAL_FAILURE_CONFIDENCE_METHOD",
    "LOGICAL_OPERATION_SEMANTICS",
    "OVERHEAD_SCENARIO_CLAIM_BOUNDARY",
    "QEC_CLAIM_QUALIFICATION_SCHEMA",
    "TORIC_ERROR_MODEL",
    "DecoderExperiment",
    "FaultTolerantResourceBoundaryError",
    "FaultTolerantResourceProduct",
    "FormulaReference",
    "LogicalFailureConfidence",
    "LogicalOperationSemantics",
    "OverheadAssumptions",
    "OverheadScenarioEstimate",
    "QecClaimQualification",
    "RegimeComparisonRow",
    "ResourceEstimate",
    "SensitivityPoint",
    "SingleErrorSyndrome",
    "SyncProblemResourceRequest",
    "build_fault_tolerant_resource_product",
    "build_ft_sensitivity",
    "build_qec_claim_qualification",
    "build_regime_comparison",
    "enumerate_single_error_syndromes",
    "estimate_ft_resources",
    "estimate_overhead_scenario",
    "logical_failure_confidence",
    "render_ft_resource_markdown",
    "run_toric_decoder_experiment",
]
