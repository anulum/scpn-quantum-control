# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — bounded JAX neural-quantum-state baseline
"""Build claim-bounded JAX NQS evidence against exact diagonalisation.

This product composes the ambient exact-enumeration RBM runner with validated
small-system inputs, an exact dense reference, environment provenance, and the
advantage-language no-advantage default. It is a research baseline, not sampled VMC,
scalable many-body evidence, a performance benchmark, or hardware execution.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .advantage_language_protocol import NoAdvantageCertificate, issue_no_advantage_certificate
from .bridge.knm_hamiltonian import knm_to_dense_matrix
from .phase.jax_nqs import is_jax_available, jax_vmc_ground_state

JAX_NQS_BASELINE_PRODUCT_SCHEMA: Final[str] = "jax_nqs_baseline_product.v2"
JAX_NQS_BASELINE_MIN_QUBITS: Final[int] = 2
JAX_NQS_BASELINE_MAX_QUBITS: Final[int] = 6
JAX_NQS_BASELINE_MAX_ITERATIONS: Final[int] = 5_000
JAX_NQS_BASELINE_CLAIM_BOUNDARY: Final[str] = (
    "Research-only real-valued RBM baseline with exact enumeration and dense "
    "exact-diagonalisation comparison for 2 <= N <= 6; no sampled VMC, scalable "
    "many-body, hardware, provider, accuracy-guarantee, or performance-advantage claim"
)
_JAX_NQS_BASELINE_NO_ADVANTAGE_CONTEXT: Final[str] = "JAX NQS exact-reference baseline"


def _canonical_digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class JAXNQSBaselineSpec:
    """Validated immutable input for one exact-reference comparison."""

    coupling: tuple[tuple[float, ...], ...]
    omega: tuple[float, ...]
    n_hidden: int
    learning_rate: float = 0.03
    n_iterations: int = 200
    seed: int = 42
    relative_error_tolerance: float = 0.2
    variational_slack: float = 1e-5
    max_dense_gib: float | None = None

    def __post_init__(self) -> None:
        """Fail closed on malformed, non-finite, or unbounded requests."""
        n_qubits = len(self.coupling)
        if not JAX_NQS_BASELINE_MIN_QUBITS <= n_qubits <= JAX_NQS_BASELINE_MAX_QUBITS:
            raise ValueError(
                "JAX NQS exact-reference evidence requires "
                f"{JAX_NQS_BASELINE_MIN_QUBITS} <= N <= {JAX_NQS_BASELINE_MAX_QUBITS}"
            )
        if any(len(row) != n_qubits for row in self.coupling):
            raise ValueError("coupling must be a square matrix")
        if len(self.omega) != n_qubits:
            raise ValueError("omega length must match coupling dimension")
        coupling_array = np.asarray(self.coupling, dtype=float)
        omega_array = np.asarray(self.omega, dtype=float)
        if not np.all(np.isfinite(coupling_array)) or not np.all(np.isfinite(omega_array)):
            raise ValueError("coupling and omega must contain only finite values")
        if not np.allclose(coupling_array, coupling_array.T, rtol=0.0, atol=1e-12):
            raise ValueError("coupling must be symmetric")
        if (
            not isinstance(self.n_hidden, int)
            or isinstance(self.n_hidden, bool)
            or self.n_hidden <= 0
        ):
            raise ValueError("n_hidden must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if (
            not isinstance(self.n_iterations, int)
            or isinstance(self.n_iterations, bool)
            or not 1 <= self.n_iterations <= JAX_NQS_BASELINE_MAX_ITERATIONS
        ):
            raise ValueError(f"n_iterations must be in [1, {JAX_NQS_BASELINE_MAX_ITERATIONS}]")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not math.isfinite(self.relative_error_tolerance) or self.relative_error_tolerance < 0.0:
            raise ValueError("relative_error_tolerance must be finite and non-negative")
        if not math.isfinite(self.variational_slack) or self.variational_slack < 0.0:
            raise ValueError("variational_slack must be finite and non-negative")
        if self.max_dense_gib is not None and (
            not math.isfinite(self.max_dense_gib) or self.max_dense_gib <= 0.0
        ):
            raise ValueError("max_dense_gib must be finite and positive when supplied")

    @classmethod
    def from_arrays(
        cls,
        coupling: NDArray[np.float64],
        omega: NDArray[np.float64],
        *,
        n_hidden: int | None = None,
        learning_rate: float = 0.03,
        n_iterations: int = 200,
        seed: int = 42,
        relative_error_tolerance: float = 0.2,
        variational_slack: float = 1e-5,
        max_dense_gib: float | None = None,
    ) -> JAXNQSBaselineSpec:
        """Copy NumPy inputs into the immutable contract."""
        coupling_array = np.asarray(coupling, dtype=float)
        omega_array = np.asarray(omega, dtype=float)
        if coupling_array.ndim != 2:
            raise ValueError("coupling must be a rank-2 square matrix")
        if omega_array.ndim != 1:
            raise ValueError("omega must be a rank-1 vector")
        inferred_hidden = 2 * coupling_array.shape[0] if n_hidden is None else n_hidden
        return cls(
            coupling=tuple(tuple(float(value) for value in row) for row in coupling_array),
            omega=tuple(float(value) for value in omega_array),
            n_hidden=inferred_hidden,
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            seed=seed,
            relative_error_tolerance=relative_error_tolerance,
            variational_slack=variational_slack,
            max_dense_gib=max_dense_gib,
        )

    @property
    def n_qubits(self) -> int:
        """Visible-spin count."""
        return len(self.coupling)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready request contract."""
        return {
            "coupling": [list(row) for row in self.coupling],
            "omega": list(self.omega),
            "coupling_diagonal_used": False,
            "n_hidden": self.n_hidden,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "seed": self.seed,
            "relative_error_tolerance": self.relative_error_tolerance,
            "variational_slack": self.variational_slack,
            "max_dense_gib": self.max_dense_gib,
        }


@dataclass(frozen=True, slots=True)
class JAXNQSEnvironment:
    """Observed JAX execution environment without device identifiers."""

    jax_version: str
    backend: str
    device_kinds: tuple[str, ...]
    x64_enabled: bool

    def __post_init__(self) -> None:
        """Require complete environment provenance."""
        if not self.jax_version or not self.backend or not self.device_kinds:
            raise ValueError("JAX environment provenance is incomplete")
        if any(not kind for kind in self.device_kinds):
            raise ValueError("JAX device kinds must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready environment provenance."""
        return {
            "jax_version": self.jax_version,
            "backend": self.backend,
            "device_kinds": list(self.device_kinds),
            "x64_enabled": self.x64_enabled,
            "numeric_posture": "float64_enabled" if self.x64_enabled else "default_float32",
        }


@dataclass(frozen=True, slots=True)
class JAXNQSComparison:
    """Exact-reference comparison and optimisation diagnostics."""

    exact_ground_energy: float
    variational_energy: float
    absolute_gap: float
    relative_error: float
    variational_upper_bound_respected: bool
    within_declared_tolerance: bool
    initial_energy: float
    energy_decreased: bool
    n_parameters: int
    exact_configuration_count: int
    energy_history: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate result finiteness and structural consistency."""
        scalars = (
            self.exact_ground_energy,
            self.variational_energy,
            self.absolute_gap,
            self.relative_error,
            self.initial_energy,
            *self.energy_history,
        )
        if not scalars or not all(math.isfinite(value) for value in scalars):
            raise ValueError("comparison values must be finite")
        if self.absolute_gap < 0.0 or self.relative_error < 0.0:
            raise ValueError("comparison errors must be non-negative")
        if self.n_parameters <= 0 or self.exact_configuration_count <= 0:
            raise ValueError("comparison counts must be positive")
        if not self.energy_history or self.initial_energy != self.energy_history[0]:
            raise ValueError("energy history must start at initial_energy")
        if self.variational_energy != self.energy_history[-1]:
            raise ValueError("energy history must end at variational_energy")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready comparison evidence."""
        return {
            "exact_ground_energy": self.exact_ground_energy,
            "variational_energy": self.variational_energy,
            "absolute_gap": self.absolute_gap,
            "relative_error": self.relative_error,
            "variational_upper_bound_respected": self.variational_upper_bound_respected,
            "within_declared_tolerance": self.within_declared_tolerance,
            "initial_energy": self.initial_energy,
            "energy_decreased": self.energy_decreased,
            "n_parameters": self.n_parameters,
            "exact_configuration_count": self.exact_configuration_count,
            "energy_history": list(self.energy_history),
        }


@dataclass(frozen=True, slots=True)
class JAXNQSBaselineProduct:
    """Complete immutable JAX NQS exact-reference evidence record."""

    schema: str
    request: JAXNQSBaselineSpec
    environment: JAXNQSEnvironment
    comparison: JAXNQSComparison
    no_advantage: NoAdvantageCertificate
    evidence_sha256: str
    claim_boundary: str = JAX_NQS_BASELINE_CLAIM_BOUNDARY
    support_posture: str = "research"
    execution_mode: str = "exact_enumeration_autodiff"
    hardware_execution: bool = False
    performance_advantage_claimed: bool = False
    scalable_many_body_claimed: bool = False

    def __post_init__(self) -> None:
        """Enforce schema, digest, and fail-closed claim posture."""
        if self.schema != JAX_NQS_BASELINE_PRODUCT_SCHEMA:
            raise ValueError(f"unknown JAX NQS baseline schema: {self.schema!r}")
        if self.claim_boundary != JAX_NQS_BASELINE_CLAIM_BOUNDARY:
            raise ValueError("JAX NQS baseline claim boundary must not drift")
        if (
            self.support_posture != "research"
            or self.execution_mode != "exact_enumeration_autodiff"
        ):
            raise ValueError("JAX NQS support and execution posture must remain bounded")
        if (
            self.hardware_execution
            or self.performance_advantage_claimed
            or self.scalable_many_body_claimed
        ):
            raise ValueError(
                "JAX NQS baseline may not promote hardware, performance, or scale claims"
            )
        expected_no_advantage = issue_no_advantage_certificate(
            context=_JAX_NQS_BASELINE_NO_ADVANTAGE_CONTEXT
        )
        if self.no_advantage != expected_no_advantage:
            raise ValueError("JAX NQS baseline must retain its canonical no-advantage certificate")
        if len(self.evidence_sha256) != 64:
            raise ValueError("evidence_sha256 must be a SHA-256 hex digest")
        if self.evidence_sha256 != _canonical_digest(self.payload_dict()):
            raise ValueError("evidence_sha256 does not match the canonical payload")

    def payload_dict(self) -> dict[str, object]:
        """Return the canonical payload covered by the evidence digest."""
        return {
            "schema": self.schema,
            "request": self.request.to_dict(),
            "environment": self.environment.to_dict(),
            "comparison": self.comparison.to_dict(),
            "no_advantage": self.no_advantage.to_dict(),
            "claim_boundary": self.claim_boundary,
            "support_posture": self.support_posture,
            "execution_mode": self.execution_mode,
            "hardware_execution": self.hardware_execution,
            "performance_advantage_claimed": self.performance_advantage_claimed,
            "scalable_many_body_claimed": self.scalable_many_body_claimed,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON-ready evidence record."""
        return {**self.payload_dict(), "evidence_sha256": self.evidence_sha256}


def _observe_jax_environment() -> JAXNQSEnvironment:
    """Capture public JAX environment fields after availability is proven."""
    import jax

    return JAXNQSEnvironment(
        jax_version=str(jax.__version__),
        backend=str(jax.default_backend()),
        device_kinds=tuple(sorted({str(device.device_kind) for device in jax.devices()})),
        x64_enabled=bool(jax.config.read("jax_enable_x64")),
    )


def run_jax_nqs_baseline(spec: JAXNQSBaselineSpec) -> JAXNQSBaselineProduct:
    """Run the bounded ambient JAX RBM and compare it with exact diagonalisation."""
    if not is_jax_available():
        raise ImportError(
            "JAX NQS exact-reference execution requires the optional JAX runtime; "
            "no NumPy fallback is used"
        )
    coupling = np.asarray(spec.coupling, dtype=float)
    omega = np.asarray(spec.omega, dtype=float)
    hamiltonian = knm_to_dense_matrix(coupling, omega, max_dense_gib=spec.max_dense_gib)
    exact_ground_energy = float(np.linalg.eigvalsh(hamiltonian)[0])
    ambient = jax_vmc_ground_state(
        coupling,
        omega,
        n_hidden=spec.n_hidden,
        learning_rate=spec.learning_rate,
        n_iterations=spec.n_iterations,
        seed=spec.seed,
        max_dense_gib=spec.max_dense_gib,
    )
    energy_history = tuple(float(value) for value in ambient["energy_history"])
    if len(energy_history) != spec.n_iterations + 1:
        raise RuntimeError("ambient JAX NQS energy-history length violates its public contract")
    variational_energy = float(ambient["energy"])
    n_parameters = int(ambient["n_params"])
    expected_parameters = spec.n_qubits + spec.n_hidden + spec.n_qubits * spec.n_hidden
    if n_parameters != expected_parameters:
        raise RuntimeError("ambient JAX NQS parameter count violates its public contract")
    signed_gap = variational_energy - exact_ground_energy
    absolute_gap = abs(signed_gap)
    energy_scale = max(abs(exact_ground_energy), np.finfo(float).eps)
    relative_error = absolute_gap / energy_scale
    comparison = JAXNQSComparison(
        exact_ground_energy=exact_ground_energy,
        variational_energy=variational_energy,
        absolute_gap=absolute_gap,
        relative_error=relative_error,
        variational_upper_bound_respected=signed_gap >= -spec.variational_slack,
        within_declared_tolerance=relative_error <= spec.relative_error_tolerance,
        initial_energy=energy_history[0],
        energy_decreased=variational_energy <= energy_history[0],
        n_parameters=n_parameters,
        exact_configuration_count=2**spec.n_qubits,
        energy_history=energy_history,
    )
    no_advantage = issue_no_advantage_certificate(context=_JAX_NQS_BASELINE_NO_ADVANTAGE_CONTEXT)
    environment = _observe_jax_environment()
    payload = {
        "schema": JAX_NQS_BASELINE_PRODUCT_SCHEMA,
        "request": spec.to_dict(),
        "environment": environment.to_dict(),
        "comparison": comparison.to_dict(),
        "no_advantage": no_advantage.to_dict(),
        "claim_boundary": JAX_NQS_BASELINE_CLAIM_BOUNDARY,
        "support_posture": "research",
        "execution_mode": "exact_enumeration_autodiff",
        "hardware_execution": False,
        "performance_advantage_claimed": False,
        "scalable_many_body_claimed": False,
    }
    return JAXNQSBaselineProduct(
        schema=JAX_NQS_BASELINE_PRODUCT_SCHEMA,
        request=spec,
        environment=environment,
        comparison=comparison,
        no_advantage=no_advantage,
        evidence_sha256=_canonical_digest(payload),
    )


def render_jax_nqs_baseline_markdown(product: JAXNQSBaselineProduct) -> str:
    """Render a compact deterministic review report."""
    comparison = product.comparison
    environment = product.environment
    return "\n".join(
        (
            "# JAX NQS baseline evidence",
            "",
            f"- Schema: `{product.schema}`",
            f"- Evidence SHA-256: `{product.evidence_sha256}`",
            f"- Support posture: `{product.support_posture}`",
            f"- System size: `{product.request.n_qubits}` visible spins",
            f"- Exact configurations: `{comparison.exact_configuration_count}`",
            "- Coupling diagonal used: `false`",
            f"- Exact ground energy: `{comparison.exact_ground_energy:.12g}`",
            f"- Variational energy: `{comparison.variational_energy:.12g}`",
            f"- Absolute gap: `{comparison.absolute_gap:.12g}`",
            f"- Relative error: `{comparison.relative_error:.12g}`",
            "- Variational upper bound respected: "
            f"`{str(comparison.variational_upper_bound_respected).lower()}`",
            f"- Within declared tolerance: `{str(comparison.within_declared_tolerance).lower()}`",
            f"- JAX: `{environment.jax_version}` on `{environment.backend}`",
            f"- JAX X64 enabled: `{str(environment.x64_enabled).lower()}`",
            "- Hardware execution: `false`",
            "- Performance advantage claimed: `false`",
            "- Scalable many-body claim: `false`",
            "",
            f"> {product.claim_boundary}",
            "",
        )
    )


def write_jax_nqs_baseline_evidence(
    product: JAXNQSBaselineProduct,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write deterministic JSON and Markdown evidence files."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(product.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_output.write_text(render_jax_nqs_baseline_markdown(product), encoding="utf-8")
    return json_output, markdown_output


__all__ = [
    "JAX_NQS_BASELINE_CLAIM_BOUNDARY",
    "JAX_NQS_BASELINE_MAX_ITERATIONS",
    "JAX_NQS_BASELINE_MAX_QUBITS",
    "JAX_NQS_BASELINE_MIN_QUBITS",
    "JAX_NQS_BASELINE_PRODUCT_SCHEMA",
    "JAXNQSBaselineProduct",
    "JAXNQSBaselineSpec",
    "JAXNQSComparison",
    "JAXNQSEnvironment",
    "render_jax_nqs_baseline_markdown",
    "run_jax_nqs_baseline",
    "write_jax_nqs_baseline_evidence",
]
