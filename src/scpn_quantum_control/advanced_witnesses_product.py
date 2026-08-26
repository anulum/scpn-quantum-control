# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Advanced witnesses product
"""Fail-closed **advanced witnesses** product surface.

Productises estimator-aware scientific diagnostics over ambient Krylov,
OTOC, and classical-shadow surfaces:

* versioned witness capability catalogue (Krylov complexity, OTOC probe,
  classical shadows, small-system tomography cap, ambient inventory,
  harmonic synchronisation-witness compose);
* common :class:`WitnessEstimate` with estimator id, mean, uncertainty,
  support status, and provenance;
* local probes over ambient
  :func:`~scpn_quantum_control.analysis.krylov_complexity.krylov_complexity`,
  :func:`~scpn_quantum_control.analysis.otoc.compute_otoc`, and
  :func:`~scpn_quantum_control.analysis.shadow_tomography.classical_shadow_estimation`
  with hard dimension / shot caps;
* refuse invent-green OTOC advantage, topology certification, live QPU
  witness campaigns, and unrestricted shadow tomography outside support
  profiles.

Does **not** claim OTOC quantum advantage, certify topological phases,
submit to QPU hardware, or replace synchronisation-witness order-parameter / Vietoris–Rips
witnesses.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess  # nosec B404
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, cast

import numpy as np
from numpy.typing import NDArray

from .analysis.krylov_complexity import KrylovResult, krylov_complexity
from .analysis.otoc import OTOCResult, compute_otoc
from .bridge.knm_hamiltonian import knm_to_dense_matrix
from .phase.synchronisation_witness import harmonic_order_parameter

# Subprocess use is limited to the current interpreter and an in-memory fixed script.
WitnessCapabilityKind = Literal[
    "krylov_complexity",
    "otoc_probe",
    "classical_shadows",
    "small_tomography_cap",
    "ambient_inventory",
    "synchronisation_witness_compose",
]
"""Witness capability kinds on the product catalogue."""

SupportPosture = Literal[
    "local_research",
    "live_hardware_gated",
    "policy_only",
    "metadata_only",
]
"""Support posture badges."""

PathDecisionOutcome = Literal["allowed", "refused"]
"""Structured path-eligibility outcomes."""

SupportStatus = Literal[
    "supported",
    "under_sampled",
    "out_of_support",
    "refused",
]
"""Estimator support status for a single witness estimate."""

BoundaryKind = Literal[
    "otoc_advantage_claim",
    "topology_certification",
    "live_qpu_witness",
    "unrestricted_shadow_tomography",
    "under_sampled_silent_green",
]
"""Hard-gap boundary kinds for advanced-witness honesty."""

ADVANCED_WITNESSES_PRODUCT_SCHEMA: Final[str] = "advanced_witnesses_product.v1"
"""JSON schema identifier for serialised product payloads."""

ADVANCED_WITNESSES_CLAIM_BOUNDARY: Final[str] = (
    "Advanced witnesses product surface only; catalogues Krylov/OTOC/"
    "classical-shadow estimators with uncertainty and provenance over ambient "
    "analysis/*; small-system probes with hard qubit/shot caps; refuse "
    "invent-green OTOC advantage, topology certification, live QPU witness "
    "campaigns, and unrestricted shadow tomography; compose harmonic "
    "synchronisation order parameters; residual dashboard panel hooks and "
    "full-suite artefact depth remain open honestly"
)
"""Shared claim boundary for advanced witnesses product payloads."""

# Hard local-probe caps; every path fails closed beyond these bounds.
MAX_WITNESS_QUBITS: Final[int] = 6
"""Maximum qubit count for dense local witness probes."""

MAX_DEMO_SHADOW_SHOTS: Final[int] = 200
"""Maximum classical-shadow shots on the product demo path."""

MIN_SHADOW_SHOTS: Final[int] = 16
"""Minimum shots before a shadow estimate is considered under-sampled."""

WITNESS_GLOSSARY: Final[Mapping[str, str]] = {
    "Krylov": (
        "Krylov complexity K(t) measures operator spreading in the Lanczos "
        "basis of the Liouvillian L = [H, ·]; exponential then linear growth "
        "signals chaotic operator growth (research diagnostic, not advantage)."
    ),
    "OTOC": (
        "Out-of-time-order correlator F(t) = Re⟨W†(t) V† W(t) V⟩; decay rate "
        "estimates a quantum Lyapunov exponent bounded by MSS thermal "
        "arguments — local sim probe, not quantum advantage certification."
    ),
    "classical_shadow": (
        "Huang–Kueng–Preskill classical shadows: random single-qubit Clifford "
        "measurements with classical post-processing to estimate many Pauli "
        "observables with uncertainty bounds; hardware-scale estimator, not "
        "full state tomography."
    ),
    "WitnessEstimate": (
        "Product estimate envelope: estimator_id, mean, uncertainty, "
        "support_status, provenance (backend, shots/n_times, caps)."
    ),
}
"""Machine-readable glossary labels for capabilities and estimates."""


@dataclass(frozen=True, slots=True)
class WitnessCapabilityRow:
    """One advanced-witness capability catalogue row.

    Attributes
    ----------
    capability_id
        Stable capability identifier.
    kind
        Capability kind enum.
    title
        Human-readable title.
    summary
        Short description.
    ambient_module
        Ambient module path.
    ambient_symbol
        Primary ambient symbol.
    hardware_submit_allowed
        Must remain False.
    support_posture
        Support posture badge.
    as_of
        Inventory date label.
    claim_boundary
        Non-promotional claim boundary.

    """

    capability_id: str
    kind: WitnessCapabilityKind
    title: str
    summary: str
    ambient_module: str
    ambient_symbol: str
    hardware_submit_allowed: bool = False
    support_posture: SupportPosture = "local_research"
    as_of: str = "2026-07-24"
    claim_boundary: str = ADVANCED_WITNESSES_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate capability row invariants."""
        if not self.capability_id or not self.capability_id.strip():
            raise ValueError("capability_id must be non-empty")
        if self.kind not in {
            "krylov_complexity",
            "otoc_probe",
            "classical_shadows",
            "small_tomography_cap",
            "ambient_inventory",
            "synchronisation_witness_compose",
        }:
            raise ValueError(f"unknown capability kind: {self.kind!r}")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if not self.ambient_module or not self.ambient_module.strip():
            raise ValueError("ambient_module must be non-empty")
        if not self.ambient_symbol or not self.ambient_symbol.strip():
            raise ValueError("ambient_symbol must be non-empty")
        if self.hardware_submit_allowed:
            raise ValueError("hardware_submit_allowed must be False on product surface")
        if self.support_posture not in {
            "local_research",
            "live_hardware_gated",
            "policy_only",
            "metadata_only",
        }:
            raise ValueError(f"unknown support_posture: {self.support_posture!r}")
        if not self.as_of or not self.as_of.strip():
            raise ValueError("as_of must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this row."""
        return {
            "capability_id": self.capability_id,
            "kind": self.kind,
            "title": self.title,
            "summary": self.summary,
            "ambient_module": self.ambient_module,
            "ambient_symbol": self.ambient_symbol,
            "hardware_submit_allowed": self.hardware_submit_allowed,
            "support_posture": self.support_posture,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class WitnessBoundaryRow:
    """One hard-gap boundary row for out-of-scope advanced witnesses."""

    boundary_id: str
    kind: BoundaryKind
    title: str
    summary: str
    fail_closed: bool = True
    claim_boundary: str = ADVANCED_WITNESSES_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate boundary row invariants."""
        if not self.boundary_id or not self.boundary_id.strip():
            raise ValueError("boundary_id must be non-empty")
        if self.kind not in {
            "otoc_advantage_claim",
            "topology_certification",
            "live_qpu_witness",
            "unrestricted_shadow_tomography",
            "under_sampled_silent_green",
        }:
            raise ValueError(f"unknown boundary kind: {self.kind!r}")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if self.fail_closed is not True:
            raise ValueError("fail_closed must be True")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this boundary row."""
        return {
            "boundary_id": self.boundary_id,
            "kind": self.kind,
            "title": self.title,
            "summary": self.summary,
            "fail_closed": self.fail_closed,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class WitnessEstimate:
    """Common witness estimate envelope.

    Attributes
    ----------
    estimator_id
        Stable estimator identifier (e.g. ``krylov_peak``).
    mean
        Point estimate (finite float).
    uncertainty
        Non-negative uncertainty radius (0 when exact/deterministic).
    support_status
        Support posture for this estimate.
    backend
        Backend label (local sim / ambient module).
    n_qubits
        System size used for the estimate.
    n_shots_or_times
        Shot count (shadows) or time-sample count (Krylov/OTOC).
    invent_green_live_qpu
        Must remain False.
    claim_boundary
        Honesty boundary string.

    """

    estimator_id: str
    mean: float
    uncertainty: float
    support_status: SupportStatus
    backend: str
    n_qubits: int
    n_shots_or_times: int
    invent_green_live_qpu: bool = False
    claim_boundary: str = ADVANCED_WITNESSES_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate estimate envelope invariants."""
        if not self.estimator_id or not self.estimator_id.strip():
            raise ValueError("estimator_id must be non-empty")
        if not math.isfinite(self.mean):
            raise ValueError("mean must be a finite float")
        if not math.isfinite(self.uncertainty) or self.uncertainty < 0.0:
            raise ValueError("uncertainty must be a non-negative finite float")
        if self.support_status not in {
            "supported",
            "under_sampled",
            "out_of_support",
            "refused",
        }:
            raise ValueError(f"unknown support_status: {self.support_status!r}")
        if not self.backend or not self.backend.strip():
            raise ValueError("backend must be non-empty")
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        if self.n_shots_or_times < 0:
            raise ValueError("n_shots_or_times must be non-negative")
        if self.invent_green_live_qpu:
            raise ValueError("invent_green_live_qpu must be False")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this estimate."""
        return {
            "estimator_id": self.estimator_id,
            "mean": self.mean,
            "uncertainty": self.uncertainty,
            "support_status": self.support_status,
            "backend": self.backend,
            "n_qubits": self.n_qubits,
            "n_shots_or_times": self.n_shots_or_times,
            "invent_green_live_qpu": self.invent_green_live_qpu,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class PathEligibilityDecision:
    """Structured path eligibility for advanced-witness routes."""

    path_id: str
    outcome: PathDecisionOutcome
    reason: str
    invent_green_refused: bool = False

    def __post_init__(self) -> None:
        """Validate decision invariants."""
        if not self.path_id or not self.path_id.strip():
            raise ValueError("path_id must be non-empty")
        if self.outcome not in {"allowed", "refused"}:
            raise ValueError(f"unknown outcome: {self.outcome!r}")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")

    @property
    def allowed(self) -> bool:
        """Return True when the path is allowed."""
        return self.outcome == "allowed"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this decision."""
        return {
            "path_id": self.path_id,
            "outcome": self.outcome,
            "reason": self.reason,
            "invent_green_refused": self.invent_green_refused,
            "allowed": self.allowed,
        }


@dataclass(frozen=True, slots=True)
class MaterialisedKrylovProbe:
    """Materialised Krylov complexity probe result."""

    estimate: WitnessEstimate
    peak_complexity: float
    n_lanczos: int
    n_times: int
    digest: str
    invent_green_live_qpu: bool = False
    claim_boundary: str = ADVANCED_WITNESSES_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate Krylov probe invariants.

        ``n_lanczos`` may be zero when the ambient Lanczos basis terminates
        immediately (e.g. single-qubit seed operator commuting with a
        diagonal Hamiltonian): that is a supported finite peak of 0, not a
        construction error.
        """
        if self.n_lanczos < 0:
            raise ValueError("n_lanczos must be non-negative")
        if self.n_times < 1:
            raise ValueError("n_times must be >= 1")
        if not math.isfinite(self.peak_complexity):
            raise ValueError("peak_complexity must be finite")
        if len(self.digest) != 64:
            raise ValueError("digest must be a 64-char hex SHA-256")
        if self.invent_green_live_qpu:
            raise ValueError("invent_green_live_qpu must be False")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe."""
        return {
            "estimate": self.estimate.to_dict(),
            "peak_complexity": self.peak_complexity,
            "n_lanczos": self.n_lanczos,
            "n_times": self.n_times,
            "digest": self.digest,
            "invent_green_live_qpu": self.invent_green_live_qpu,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class MaterialisedOtocProbe:
    """Materialised OTOC probe result."""

    estimate: WitnessEstimate
    final_otoc: float
    lyapunov_estimate: float | None
    scrambling_time: float | None
    n_times: int
    digest: str
    invent_green_otoc_advantage: bool = False
    invent_green_live_qpu: bool = False
    claim_boundary: str = ADVANCED_WITNESSES_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate OTOC probe invariants."""
        if self.n_times < 1:
            raise ValueError("n_times must be >= 1")
        if not math.isfinite(self.final_otoc):
            raise ValueError("final_otoc must be finite")
        if self.lyapunov_estimate is not None and not math.isfinite(self.lyapunov_estimate):
            raise ValueError("lyapunov_estimate must be finite when present")
        if self.scrambling_time is not None and not math.isfinite(self.scrambling_time):
            raise ValueError("scrambling_time must be finite when present")
        if len(self.digest) != 64:
            raise ValueError("digest must be a 64-char hex SHA-256")
        if self.invent_green_otoc_advantage:
            raise ValueError("invent_green_otoc_advantage must be False")
        if self.invent_green_live_qpu:
            raise ValueError("invent_green_live_qpu must be False")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe."""
        return {
            "estimate": self.estimate.to_dict(),
            "final_otoc": self.final_otoc,
            "lyapunov_estimate": self.lyapunov_estimate,
            "scrambling_time": self.scrambling_time,
            "n_times": self.n_times,
            "digest": self.digest,
            "invent_green_otoc_advantage": self.invent_green_otoc_advantage,
            "invent_green_live_qpu": self.invent_green_live_qpu,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class MaterialisedShadowProbe:
    """Materialised classical-shadow probe result."""

    estimate: WitnessEstimate
    observables: Mapping[str, float]
    shadow_norm_bound: float
    n_shots: int
    digest: str
    invent_green_live_qpu: bool = False
    claim_boundary: str = ADVANCED_WITNESSES_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate shadow probe invariants."""
        if self.n_shots < 1:
            raise ValueError("n_shots must be >= 1")
        if not math.isfinite(self.shadow_norm_bound) or self.shadow_norm_bound < 0.0:
            raise ValueError("shadow_norm_bound must be a non-negative finite float")
        if not self.observables:
            raise ValueError("observables must be non-empty")
        for key, value in self.observables.items():
            if not key or not str(key).strip():
                raise ValueError("observable name must be non-empty")
            if not math.isfinite(float(value)):
                raise ValueError(f"observable {key!r} value must be finite")
        if len(self.digest) != 64:
            raise ValueError("digest must be a 64-char hex SHA-256")
        if self.invent_green_live_qpu:
            raise ValueError("invent_green_live_qpu must be False")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe."""
        return {
            "estimate": self.estimate.to_dict(),
            "observables": dict(self.observables),
            "shadow_norm_bound": self.shadow_norm_bound,
            "n_shots": self.n_shots,
            "digest": self.digest,
            "invent_green_live_qpu": self.invent_green_live_qpu,
            "claim_boundary": self.claim_boundary,
        }


def _canonical_digest(payload: Mapping[str, object]) -> str:
    """Return a stable SHA-256 hex digest of a JSON-serialisable payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _require_qubit_cap(n_qubits: int, *, label: str) -> None:
    """Fail-closed when a probe exceeds the product qubit cap."""
    if n_qubits < 1:
        raise ValueError(f"{label}: n_qubits must be >= 1")
    if n_qubits > MAX_WITNESS_QUBITS:
        raise ValueError(
            f"{label}: n_qubits={n_qubits} exceeds product cap "
            f"MAX_WITNESS_QUBITS={MAX_WITNESS_QUBITS}"
        )


def _pauli_z_on_first_qubit(n_qubits: int) -> NDArray[np.complex128]:
    """Build Z⊗I⊗… as the default Krylov seed operator."""
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    eye = np.eye(2, dtype=np.complex128)
    op: NDArray[np.complex128] = z
    for _ in range(1, n_qubits):
        op = cast(NDArray[np.complex128], np.kron(op, eye))
    return op


def _run_ambient_shadow_json(
    *,
    n_qubits: int,
    n_shots: int,
    seed: int,
    observables: Mapping[str, str],
) -> dict[str, object]:
    """Run ambient classical shadows in a clean subprocess (pytest-cov safe).

    pytest-cov reloads NumPy and breaks in-process shadow measurement sampling;
    isolate ambient ``classical_shadow_estimation`` in a fresh interpreter.
    """
    obs = {str(k): str(v) for k, v in observables.items()}
    script = f"""
import json
import numpy as np
from scpn_quantum_control.analysis.shadow_tomography import classical_shadow_estimation

n_qubits = {int(n_qubits)}
n_shots = {int(n_shots)}
seed = {int(seed)}
observables = {obs!r}
dim = 2 ** n_qubits
psi = np.zeros(dim, dtype=np.complex128)
psi[0] = 1.0
result = classical_shadow_estimation(
    psi, n_qubits, observables, n_shots=n_shots, seed=seed
)
out = {{
    "n_qubits": int(result.n_qubits),
    "n_shots": int(result.n_shots),
    "estimated_observables": {{
        str(k): float(v) for k, v in result.estimated_observables.items()
    }},
    "shadow_norm_bound": float(result.shadow_norm_bound),
}}
print(json.dumps(out))
    """
    try:
        # The executable and flags are fixed; inputs are integers or repr-escaped strings.
        completed = subprocess.run(  # nosec B603
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or str(exc)).strip()
        raise ValueError(f"ambient shadow subprocess failed: {err}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("ambient shadow subprocess timed out") from exc
    line = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ambient shadow subprocess returned non-JSON: {line!r}") from exc
    if not isinstance(payload, dict):
        raise ValueError("ambient shadow subprocess payload must be an object")
    return cast(dict[str, object], payload)


def iter_witness_capabilities() -> tuple[WitnessCapabilityRow, ...]:
    """Return the fixed advanced-witness capability catalogue."""
    return (
        WitnessCapabilityRow(
            capability_id="krylov_complexity",
            kind="krylov_complexity",
            title="Krylov complexity diagnostic",
            summary=(
                "Operator Lanczos Krylov complexity K(t) on bounded unitary "
                "evolution via ambient krylov_complexity."
            ),
            ambient_module="scpn_quantum_control.analysis.krylov_complexity",
            ambient_symbol="krylov_complexity",
        ),
        WitnessCapabilityRow(
            capability_id="otoc_probe",
            kind="otoc_probe",
            title="OTOC scrambling probe",
            summary=(
                "Out-of-time-order correlator F(t) and Lyapunov estimate on "
                "supported gate-model / Kuramoto-XY sims via ambient compute_otoc."
            ),
            ambient_module="scpn_quantum_control.analysis.otoc",
            ambient_symbol="compute_otoc",
        ),
        WitnessCapabilityRow(
            capability_id="classical_shadows",
            kind="classical_shadows",
            title="Classical shadow Pauli estimator",
            summary=(
                "Huang–Kueng–Preskill classical shadows for Pauli observables "
                "with shadow-norm uncertainty bounds."
            ),
            ambient_module="scpn_quantum_control.analysis.shadow_tomography",
            ambient_symbol="classical_shadow_estimation",
        ),
        WitnessCapabilityRow(
            capability_id="small_tomography_cap",
            kind="small_tomography_cap",
            title="Small-system tomography bound",
            summary=(
                "Tomography comparison path only when dimension permits; "
                f"product qubit cap is MAX_WITNESS_QUBITS={MAX_WITNESS_QUBITS}."
            ),
            ambient_module="scpn_quantum_control.analysis.shadow_tomography",
            ambient_symbol="estimate_pauli_expectation",
            support_posture="policy_only",
        ),
        WitnessCapabilityRow(
            capability_id="ambient_inventory",
            kind="ambient_inventory",
            title="Ambient advanced-witness inventory",
            summary=(
                "Inventory of ambient Krylov/OTOC/shadow/sync-witness modules "
                "without reimplementation."
            ),
            ambient_module="scpn_quantum_control.analysis",
            ambient_symbol="krylov_complexity|otoc|shadow_tomography",
            support_posture="metadata_only",
        ),
        WitnessCapabilityRow(
            capability_id="synchronisation_witness_compose",
            kind="synchronisation_witness_compose",
            title="Harmonic synchronisation witness compose",
            summary=(
                "Compose with the ambient harmonic_order_parameter for "
                "order-parameter diagnostics alongside advanced estimators."
            ),
            ambient_module="scpn_quantum_control.phase.synchronisation_witness",
            ambient_symbol="harmonic_order_parameter",
            support_posture="local_research",
        ),
    )


def list_witness_capability_ids() -> tuple[str, ...]:
    """Return ordered capability identifiers."""
    return tuple(row.capability_id for row in iter_witness_capabilities())


def get_witness_capability(capability_id: str) -> WitnessCapabilityRow:
    """Return one capability row by id or raise KeyError."""
    for row in iter_witness_capabilities():
        if row.capability_id == capability_id:
            return row
    raise KeyError(f"unknown witness capability_id: {capability_id!r}")


def iter_witness_boundaries() -> tuple[WitnessBoundaryRow, ...]:
    """Return the fixed hard-gap boundary catalogue."""
    return (
        WitnessBoundaryRow(
            boundary_id="otoc_advantage_claim",
            kind="otoc_advantage_claim",
            title="OTOC quantum advantage claim",
            summary=(
                "Refuse invent-green claims that OTOC decay proves quantum "
                "advantage; local Lyapunov estimates are research diagnostics."
            ),
        ),
        WitnessBoundaryRow(
            boundary_id="topology_certification",
            kind="topology_certification",
            title="Topology / topological-phase certification",
            summary=(
                "Refuse invent-green topological certification from witnesses; "
                "Vietoris–Rips persistence remains a synthetic diagnostic, not a cert."
            ),
        ),
        WitnessBoundaryRow(
            boundary_id="live_qpu_witness",
            kind="live_qpu_witness",
            title="Live QPU witness campaign",
            summary=(
                "Refuse live hardware submission from this product surface; "
                "probes are local sim only."
            ),
        ),
        WitnessBoundaryRow(
            boundary_id="unrestricted_shadow_tomography",
            kind="unrestricted_shadow_tomography",
            title="Unrestricted shadow tomography",
            summary=(
                "Refuse unbounded shot/weight shadow campaigns without a "
                "support profile and product shot/qubit caps."
            ),
        ),
        WitnessBoundaryRow(
            boundary_id="under_sampled_silent_green",
            kind="under_sampled_silent_green",
            title="Under-sampled silent green",
            summary=(
                "Refuse silent green on under-sampled shadow estimates; "
                "support_status must surface under_sampled when shots are low."
            ),
        ),
    )


def list_witness_boundary_ids() -> tuple[str, ...]:
    """Return ordered boundary identifiers."""
    return tuple(row.boundary_id for row in iter_witness_boundaries())


def get_witness_boundary(boundary_id: str) -> WitnessBoundaryRow:
    """Return one boundary row by id or raise KeyError."""
    for row in iter_witness_boundaries():
        if row.boundary_id == boundary_id:
            return row
    raise KeyError(f"unknown witness boundary_id: {boundary_id!r}")


def list_witness_glossary_keys() -> tuple[str, ...]:
    """Return ordered glossary keys."""
    return tuple(WITNESS_GLOSSARY.keys())


def get_witness_glossary_entry(key: str) -> str:
    """Return one glossary definition or raise KeyError."""
    try:
        return WITNESS_GLOSSARY[key]
    except KeyError as exc:
        raise KeyError(f"unknown glossary key: {key!r}") from exc


def list_witness_ambient_inventory() -> tuple[dict[str, str], ...]:
    """Return ambient module inventory rows (do not reimplement)."""
    return (
        {
            "module": "scpn_quantum_control.analysis.krylov_complexity",
            "symbol": "krylov_complexity",
            "role": "Krylov complexity K(t) + Lanczos coefficients",
        },
        {
            "module": "scpn_quantum_control.analysis.otoc",
            "symbol": "compute_otoc",
            "role": "OTOC F(t) + Lyapunov / scrambling-time estimates",
        },
        {
            "module": "scpn_quantum_control.analysis.otoc_sync_probe",
            "symbol": "otoc_sync_scan",
            "role": "OTOC vs coupling scan (sync-transition probe)",
        },
        {
            "module": "scpn_quantum_control.analysis.shadow_tomography",
            "symbol": "classical_shadow_estimation",
            "role": "Classical shadows for Pauli observables",
        },
        {
            "module": "scpn_quantum_control.phase.synchronisation_witness",
            "symbol": "harmonic_order_parameter",
            "role": "harmonic synchronisation order-parameter compose",
        },
    )


def map_advanced_witnesses_public_surfaces() -> dict[str, str]:
    """Map public product surface names to ambient entry points."""
    return {
        "krylov_probe": "scpn_quantum_control.analysis.krylov_complexity.krylov_complexity",
        "otoc_probe": "scpn_quantum_control.analysis.otoc.compute_otoc",
        "shadow_probe": (
            "scpn_quantum_control.analysis.shadow_tomography.classical_shadow_estimation"
        ),
        "harmonic_order_parameter": (
            "scpn_quantum_control.phase.synchronisation_witness.harmonic_order_parameter"
        ),
        "product_registry": "advanced_witnesses_product.v1",
    }


def decide_witness_path(
    path_id: str,
    *,
    invent_green_otoc_advantage: bool = False,
    invent_green_topology_cert: bool = False,
    invent_green_live_qpu: bool = False,
    unrestricted_shadow: bool = False,
    n_qubits: int | None = None,
) -> PathEligibilityDecision:
    """Decide whether a witness path is allowed under product policy.

    Parameters
    ----------
    path_id
        Logical path identifier (``krylov``, ``otoc``, ``shadow``,
        ``tomography_small``, ``synchronisation_witness_compose``).
    invent_green_otoc_advantage
        When True, refuse with invent-green flag.
    invent_green_topology_cert
        When True, refuse topology-certification invent-green.
    invent_green_live_qpu
        When True, refuse live QPU invent-green.
    unrestricted_shadow
        When True, refuse unrestricted shadow campaigns.
    n_qubits
        Optional qubit count to enforce the product cap.

    """
    if not path_id or not path_id.strip():
        raise ValueError("path_id must be non-empty")
    pid = path_id.strip()
    if invent_green_otoc_advantage:
        return PathEligibilityDecision(
            path_id=pid,
            outcome="refused",
            reason="refuse invent-green OTOC quantum advantage claim",
            invent_green_refused=True,
        )
    if invent_green_topology_cert:
        return PathEligibilityDecision(
            path_id=pid,
            outcome="refused",
            reason="refuse invent-green topology / topological-phase certification",
            invent_green_refused=True,
        )
    if invent_green_live_qpu:
        return PathEligibilityDecision(
            path_id=pid,
            outcome="refused",
            reason="refuse invent-green live QPU witness campaign",
            invent_green_refused=True,
        )
    if unrestricted_shadow:
        return PathEligibilityDecision(
            path_id=pid,
            outcome="refused",
            reason="refuse unrestricted shadow tomography without support profile",
            invent_green_refused=True,
        )
    if n_qubits is not None and n_qubits > MAX_WITNESS_QUBITS:
        return PathEligibilityDecision(
            path_id=pid,
            outcome="refused",
            reason=(
                f"n_qubits={n_qubits} exceeds product cap MAX_WITNESS_QUBITS={MAX_WITNESS_QUBITS}"
            ),
            invent_green_refused=False,
        )
    allowed_paths = {
        "krylov",
        "otoc",
        "shadow",
        "tomography_small",
        "synchronisation_witness_compose",
        "ambient_inventory",
    }
    if pid not in allowed_paths:
        return PathEligibilityDecision(
            path_id=pid,
            outcome="refused",
            reason=f"unknown witness path_id {pid!r}",
            invent_green_refused=False,
        )
    return PathEligibilityDecision(
        path_id=pid,
        outcome="allowed",
        reason=f"local research path {pid!r} allowed under product caps",
        invent_green_refused=False,
    )


def materialise_krylov_probe(
    *,
    n_qubits: int = 2,
    coupling: float = 0.5,
    t_max: float = 1.0,
    n_times: int = 8,
    max_lanczos: int = 12,
    invent_green_live_qpu: bool = False,
    invent_green_topology_cert: bool = False,
) -> MaterialisedKrylovProbe:
    """Materialise a bounded Krylov complexity probe over ambient APIs.

    Builds a small Kuramoto-XY Hamiltonian, runs ambient
    :func:`krylov_complexity` with a Z⊗I… seed operator, and packages a
    :class:`WitnessEstimate` for the peak complexity.

    Invent-green flags are forwarded to :func:`decide_witness_path` and refuse
    before any ambient work (fail-closed product policy).
    """
    _require_qubit_cap(n_qubits, label="krylov_probe")
    if n_times < 2:
        raise ValueError("n_times must be >= 2")
    if max_lanczos < 2:
        raise ValueError("max_lanczos must be >= 2")
    if t_max <= 0.0 or not math.isfinite(t_max):
        raise ValueError("t_max must be a positive finite float")
    decision = decide_witness_path(
        "krylov",
        n_qubits=n_qubits,
        invent_green_live_qpu=invent_green_live_qpu,
        invent_green_topology_cert=invent_green_topology_cert,
    )
    if not decision.allowed:
        raise ValueError(decision.reason)

    omega = np.linspace(-0.1, 0.1, n_qubits, dtype=np.float64)
    k_mat = np.zeros((n_qubits, n_qubits), dtype=np.float64)
    for i in range(n_qubits - 1):
        k_mat[i, i + 1] = coupling
        k_mat[i + 1, i] = coupling
    hamiltonian = knm_to_dense_matrix(k_mat, omega)
    seed = _pauli_z_on_first_qubit(n_qubits)
    result: KrylovResult = krylov_complexity(
        hamiltonian,
        seed,
        t_max=t_max,
        n_times=n_times,
        max_lanczos=max_lanczos,
    )
    peak = float(result.peak_complexity)
    if not math.isfinite(peak):
        raise ValueError("ambient krylov_complexity returned non-finite peak_complexity")
    # n_lanczos may be 0 on trivial commuting single-qubit cases; still supported.
    n_lanczos = int(result.n_lanczos)
    if n_lanczos < 0:
        raise ValueError(f"ambient krylov_complexity returned invalid n_lanczos={n_lanczos}")
    estimate = WitnessEstimate(
        estimator_id="krylov_peak",
        mean=peak,
        uncertainty=0.0,
        support_status="supported",
        backend="local_sim.krylov_complexity",
        n_qubits=n_qubits,
        n_shots_or_times=int(result.times.size),
    )
    digest = _canonical_digest(
        {
            "estimator": "krylov_peak",
            "peak": peak,
            "n_lanczos": n_lanczos,
            "n_times": int(result.times.size),
            "n_qubits": n_qubits,
        }
    )
    return MaterialisedKrylovProbe(
        estimate=estimate,
        peak_complexity=peak,
        n_lanczos=n_lanczos,
        n_times=int(result.times.size),
        digest=digest,
    )


def materialise_demo_krylov_probe() -> MaterialisedKrylovProbe:
    """Materialise the fixed two-qubit demo Krylov probe."""
    return materialise_krylov_probe(n_qubits=2, coupling=0.5, t_max=1.0, n_times=6, max_lanczos=8)


def materialise_otoc_probe(
    *,
    n_qubits: int = 2,
    coupling: float = 0.5,
    t_max: float = 0.5,
    n_times: int = 6,
    invent_green_otoc_advantage: bool = False,
    invent_green_live_qpu: bool = False,
) -> MaterialisedOtocProbe:
    """Materialise a bounded OTOC probe over ambient :func:`compute_otoc`.

    Invent-green flags are forwarded to :func:`decide_witness_path` and refuse
    before any ambient work (no silent advantage or live-QPU materialisation).
    """
    _require_qubit_cap(n_qubits, label="otoc_probe")
    if n_times < 2:
        raise ValueError("n_times must be >= 2")
    if t_max <= 0.0 or not math.isfinite(t_max):
        raise ValueError("t_max must be a positive finite float")
    decision = decide_witness_path(
        "otoc",
        n_qubits=n_qubits,
        invent_green_otoc_advantage=invent_green_otoc_advantage,
        invent_green_live_qpu=invent_green_live_qpu,
    )
    if not decision.allowed:
        raise ValueError(decision.reason)

    omega = np.linspace(-0.1, 0.1, n_qubits, dtype=np.float64)
    k_mat = np.zeros((n_qubits, n_qubits), dtype=np.float64)
    for i in range(n_qubits - 1):
        k_mat[i, i + 1] = coupling
        k_mat[i + 1, i] = coupling
    times = np.linspace(0.0, t_max, n_times, dtype=np.float64)
    result: OTOCResult = compute_otoc(
        k_mat,
        omega,
        times=times,
        max_dense_gib=1.0,
    )
    final = float(result.otoc_values[-1])
    lyap = result.lyapunov_estimate
    scramble = result.scrambling_time
    estimate = WitnessEstimate(
        estimator_id="otoc_final",
        mean=final,
        uncertainty=0.0,
        support_status="supported",
        backend="local_sim.compute_otoc",
        n_qubits=int(result.n_qubits),
        n_shots_or_times=int(result.times.size),
    )
    digest = _canonical_digest(
        {
            "estimator": "otoc_final",
            "final": final,
            "lyapunov": lyap,
            "scrambling_time": scramble,
            "n_times": int(result.times.size),
            "n_qubits": int(result.n_qubits),
        }
    )
    return MaterialisedOtocProbe(
        estimate=estimate,
        final_otoc=final,
        lyapunov_estimate=float(lyap) if lyap is not None else None,
        scrambling_time=float(scramble) if scramble is not None else None,
        n_times=int(result.times.size),
        digest=digest,
    )


def materialise_demo_otoc_probe() -> MaterialisedOtocProbe:
    """Materialise the fixed two-qubit demo OTOC probe."""
    return materialise_otoc_probe(n_qubits=2, coupling=0.5, t_max=0.5, n_times=5)


def materialise_shadow_probe(
    *,
    n_qubits: int = 2,
    n_shots: int = 80,
    seed: int = 7,
    observables: Mapping[str, str] | None = None,
    invent_green_live_qpu: bool = False,
    unrestricted_shadow: bool = False,
) -> MaterialisedShadowProbe:
    """Materialise a bounded classical-shadow probe over ambient APIs.

    Under-sampled runs (``n_shots < MIN_SHADOW_SHOTS``) still return a probe
    but mark ``support_status="under_sampled"`` under the fail-closed sampling
    policy.

    Invent-green / unrestricted flags are forwarded to :func:`decide_witness_path`
    and refuse before ambient shadow work.
    """
    _require_qubit_cap(n_qubits, label="shadow_probe")
    if n_shots < 1:
        raise ValueError("n_shots must be >= 1")
    if n_shots > MAX_DEMO_SHADOW_SHOTS:
        raise ValueError(
            f"n_shots={n_shots} exceeds product cap MAX_DEMO_SHADOW_SHOTS={MAX_DEMO_SHADOW_SHOTS}"
        )
    decision = decide_witness_path(
        "shadow",
        n_qubits=n_qubits,
        invent_green_live_qpu=invent_green_live_qpu,
        unrestricted_shadow=unrestricted_shadow,
    )
    if not decision.allowed:
        raise ValueError(decision.reason)

    obs: dict[str, str]
    if observables is None:
        if n_qubits == 1:
            obs = {"z": "Z"}
        else:
            obs = {
                "zz": "Z" * n_qubits,
                "zi": "Z" + "I" * (n_qubits - 1),
            }
    else:
        obs = {str(k): str(v) for k, v in observables.items()}
        if not obs:
            raise ValueError("observables must be non-empty when provided")
        for name, label in obs.items():
            if len(label) != n_qubits:
                raise ValueError(
                    f"observable {name!r} label length {len(label)} != n_qubits={n_qubits}"
                )

    payload = _run_ambient_shadow_json(
        n_qubits=n_qubits,
        n_shots=n_shots,
        seed=seed,
        observables=obs,
    )
    raw_obs = payload.get("estimated_observables")
    if not isinstance(raw_obs, Mapping) or not raw_obs:
        raise ValueError("ambient shadow payload missing estimated_observables")
    estimated = {str(k): float(v) for k, v in raw_obs.items()}
    raw_bound = payload.get("shadow_norm_bound", float("nan"))
    if not isinstance(raw_bound, (int, float)):
        raise ValueError("ambient shadow payload has invalid shadow_norm_bound")
    bound = float(raw_bound)
    if not math.isfinite(bound) or bound < 0.0:
        raise ValueError("ambient shadow payload has invalid shadow_norm_bound")
    raw_nq = payload.get("n_qubits", n_qubits)
    raw_ns = payload.get("n_shots", n_shots)
    if not isinstance(raw_nq, (int, float)):
        raise ValueError("ambient shadow payload has invalid n_qubits")
    if not isinstance(raw_ns, (int, float)):
        raise ValueError("ambient shadow payload has invalid n_shots")
    n_q = int(raw_nq)
    n_s = int(raw_ns)
    # Primary mean: first observable in sorted order for stability.
    primary_key = sorted(estimated.keys())[0]
    primary_mean = float(estimated[primary_key])
    support: SupportStatus = "under_sampled" if n_shots < MIN_SHADOW_SHOTS else "supported"
    estimate = WitnessEstimate(
        estimator_id=f"shadow_{primary_key}",
        mean=primary_mean,
        uncertainty=bound,
        support_status=support,
        backend="local_sim.classical_shadow_estimation",
        n_qubits=n_q,
        n_shots_or_times=n_s,
    )
    digest = _canonical_digest(
        {
            "estimator": "classical_shadow",
            "observables": dict(sorted(estimated.items())),
            "bound": bound,
            "n_shots": n_s,
            "n_qubits": n_q,
            "seed": seed,
        }
    )
    return MaterialisedShadowProbe(
        estimate=estimate,
        observables=estimated,
        shadow_norm_bound=bound,
        n_shots=n_s,
        digest=digest,
    )


def materialise_demo_shadow_probe() -> MaterialisedShadowProbe:
    """Materialise the fixed two-qubit demo classical-shadow probe."""
    return materialise_shadow_probe(n_qubits=2, n_shots=80, seed=7)


def materialise_harmonic_order_parameter_compose(
    phases: Sequence[float] | None = None,
    *,
    harmonic: int = 1,
    invent_green_topology_cert: bool = False,
    invent_green_live_qpu: bool = False,
) -> WitnessEstimate:
    """Compose the harmonic order parameter into a :class:`WitnessEstimate`.

    Parameters
    ----------
    phases
        Phase samples; default is a tightly synchronised cloud.
    harmonic
        Harmonic index for the ambient order-parameter estimator.
    invent_green_topology_cert
        When True, refuse invent-green topology certification because the harmonic
        order parameter is a synthetic diagnostic, not a certificate.
    invent_green_live_qpu
        When True, refuse live-QPU invent-green on this compose path.

    """
    if harmonic < 1:
        raise ValueError("harmonic must be >= 1")
    decision = decide_witness_path(
        "synchronisation_witness_compose",
        invent_green_topology_cert=invent_green_topology_cert,
        invent_green_live_qpu=invent_green_live_qpu,
    )
    if not decision.allowed:
        raise ValueError(decision.reason)
    if phases is None:
        phase_arr = np.array([0.01, -0.02, 0.0, 0.015], dtype=np.float64)
    else:
        phase_arr = np.asarray(list(phases), dtype=np.float64)
        if phase_arr.ndim != 1 or phase_arr.size < 1:
            raise ValueError("phases must be a non-empty 1-D sequence")
    r_value = float(harmonic_order_parameter(phase_arr, harmonic=harmonic))
    return WitnessEstimate(
        estimator_id=f"harmonic_order_parameter_R_h{harmonic}",
        mean=r_value,
        uncertainty=0.0,
        support_status="supported",
        backend="local_sim.harmonic_order_parameter",
        n_qubits=1,
        n_shots_or_times=int(phase_arr.size),
    )


def build_advanced_witnesses_product_registry() -> dict[str, object]:
    """Build the versioned advanced-witnesses product registry (v1)."""
    capabilities = [row.to_dict() for row in iter_witness_capabilities()]
    boundaries = [row.to_dict() for row in iter_witness_boundaries()]
    return {
        "schema": ADVANCED_WITNESSES_PRODUCT_SCHEMA,
        "claim_boundary": ADVANCED_WITNESSES_CLAIM_BOUNDARY,
        "capabilities": capabilities,
        "boundaries": boundaries,
        "capability_count": len(capabilities),
        "boundary_count": len(boundaries),
        "blank_entry_count": 0,
        "glossary": dict(WITNESS_GLOSSARY),
        "ambient_inventory": list(list_witness_ambient_inventory()),
        "public_surfaces": map_advanced_witnesses_public_surfaces(),
        "max_witness_qubits": MAX_WITNESS_QUBITS,
        "max_demo_shadow_shots": MAX_DEMO_SHADOW_SHOTS,
        "min_shadow_shots": MIN_SHADOW_SHOTS,
        "hardware_submit_allowed_policy": False,
        "otoc_advantage_claim_policy": False,
        "topology_certification_policy": False,
        "live_qpu_witness_policy": False,
        "unrestricted_shadow_tomography_policy": False,
        "under_sampled_silent_green_policy": False,
    }


def assert_advanced_witnesses_product_integrity(
    registry: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert structural integrity of the advanced-witnesses product registry.

    Parameters
    ----------
    registry
        Registry mapping; when None, builds a fresh registry.

    Returns
    -------
    dict[str, object]
        The validated registry (same object if provided and valid).

    """
    if registry is None:
        reg: dict[str, object] = build_advanced_witnesses_product_registry()
    else:
        reg = dict(registry)

    if reg.get("schema") != ADVANCED_WITNESSES_PRODUCT_SCHEMA:
        raise ValueError(
            f"schema must be {ADVANCED_WITNESSES_PRODUCT_SCHEMA!r}, got {reg.get('schema')!r}"
        )
    claim = reg.get("claim_boundary")
    if not isinstance(claim, str) or "Advanced witnesses product" not in claim:
        raise ValueError("claim_boundary missing or invalid")

    capabilities = reg.get("capabilities")
    boundaries = reg.get("boundaries")
    if not isinstance(capabilities, list) or not capabilities:
        raise ValueError(
            "advanced witnesses product registry must contain a non-empty capabilities list"
        )
    if not isinstance(boundaries, list) or not boundaries:
        raise ValueError(
            "advanced witnesses product registry must contain a non-empty boundaries list"
        )

    seen: set[str] = set()
    blank = 0
    required_caps = {
        "krylov_complexity",
        "otoc_probe",
        "classical_shadows",
        "synchronisation_witness_compose",
    }
    found_required: set[str] = set()
    for index, row in enumerate(capabilities):
        if not isinstance(row, Mapping):
            raise ValueError(f"capability row {index} must be a mapping")
        capability_id = row.get("capability_id")
        hw = row.get("hardware_submit_allowed")
        symbol = row.get("ambient_symbol")
        if not capability_id or not str(capability_id).strip():
            blank += 1
            continue
        cid = str(capability_id).strip()
        if cid in seen:
            raise ValueError(f"duplicate capability_id in registry: {cid!r}")
        seen.add(cid)
        if cid in required_caps:
            found_required.add(cid)
        if hw is not False:
            raise ValueError(f"capability {cid!r} hardware_submit_allowed must be False")
        if not symbol or not str(symbol).strip():
            raise ValueError(f"capability {cid!r} must have non-empty ambient_symbol")
    if blank:
        raise ValueError(
            f"advanced witnesses product registry has {blank} blank or invalid entries"
        )
    missing_req = required_caps - found_required
    if missing_req:
        raise ValueError(f"registry missing required capabilities: {sorted(missing_req)!r}")
    expected = set(list_witness_capability_ids())
    if seen != expected:
        raise ValueError(
            f"registry capability set drift (missing={expected - seen!r}, "
            f"extra={seen - expected!r})"
        )

    seen_b: set[str] = set()
    for index, row in enumerate(boundaries):
        if not isinstance(row, Mapping):
            raise ValueError(f"boundary row {index} must be a mapping")
        boundary_id = row.get("boundary_id")
        fail_closed = row.get("fail_closed")
        if not boundary_id or not str(boundary_id).strip():
            raise ValueError(f"boundary row {index} blank or invalid boundary_id")
        bid = str(boundary_id).strip()
        if bid in seen_b:
            raise ValueError(f"duplicate boundary_id in registry: {bid!r}")
        seen_b.add(bid)
        if fail_closed is not True:
            raise ValueError(f"boundary {bid!r} fail_closed must be True")
    expected_b = set(list_witness_boundary_ids())
    if seen_b != expected_b:
        raise ValueError(
            f"registry boundary set drift (missing={expected_b - seen_b!r}, "
            f"extra={seen_b - expected_b!r})"
        )

    blank_entry_count = reg.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    capability_count = reg.get("capability_count", -1)
    if not isinstance(capability_count, int) or capability_count != len(capabilities):
        raise ValueError("capability_count does not match capabilities list length")
    boundary_count = reg.get("boundary_count", -1)
    if not isinstance(boundary_count, int) or boundary_count != len(boundaries):
        raise ValueError("boundary_count does not match boundaries list length")

    for policy_key in (
        "hardware_submit_allowed_policy",
        "otoc_advantage_claim_policy",
        "topology_certification_policy",
        "live_qpu_witness_policy",
        "unrestricted_shadow_tomography_policy",
        "under_sampled_silent_green_policy",
    ):
        if reg.get(policy_key, True) is not False:
            raise ValueError(f"{policy_key} must be False")

    glossary = reg.get("glossary")
    if not isinstance(glossary, Mapping) or not glossary:
        raise ValueError("glossary must be a non-empty mapping")
    for key in list_witness_glossary_keys():
        if key not in glossary:
            raise ValueError(f"glossary missing key {key!r}")

    inventory = reg.get("ambient_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("ambient_inventory must be a non-empty list")

    if reg.get("max_witness_qubits") != MAX_WITNESS_QUBITS:
        raise ValueError("max_witness_qubits must match MAX_WITNESS_QUBITS")
    if reg.get("max_demo_shadow_shots") != MAX_DEMO_SHADOW_SHOTS:
        raise ValueError("max_demo_shadow_shots must match MAX_DEMO_SHADOW_SHOTS")
    if reg.get("min_shadow_shots") != MIN_SHADOW_SHOTS:
        raise ValueError("min_shadow_shots must match MIN_SHADOW_SHOTS")

    return reg


__all__ = [
    "ADVANCED_WITNESSES_CLAIM_BOUNDARY",
    "ADVANCED_WITNESSES_PRODUCT_SCHEMA",
    "MAX_DEMO_SHADOW_SHOTS",
    "MAX_WITNESS_QUBITS",
    "MIN_SHADOW_SHOTS",
    "WITNESS_GLOSSARY",
    "BoundaryKind",
    "MaterialisedKrylovProbe",
    "MaterialisedOtocProbe",
    "MaterialisedShadowProbe",
    "PathDecisionOutcome",
    "PathEligibilityDecision",
    "SupportPosture",
    "SupportStatus",
    "WitnessBoundaryRow",
    "WitnessCapabilityKind",
    "WitnessCapabilityRow",
    "WitnessEstimate",
    "assert_advanced_witnesses_product_integrity",
    "build_advanced_witnesses_product_registry",
    "decide_witness_path",
    "get_witness_boundary",
    "get_witness_capability",
    "get_witness_glossary_entry",
    "iter_witness_boundaries",
    "iter_witness_capabilities",
    "list_witness_ambient_inventory",
    "list_witness_boundary_ids",
    "list_witness_capability_ids",
    "list_witness_glossary_keys",
    "map_advanced_witnesses_public_surfaces",
    "materialise_harmonic_order_parameter_compose",
    "materialise_demo_krylov_probe",
    "materialise_demo_otoc_probe",
    "materialise_demo_shadow_probe",
    "materialise_krylov_probe",
    "materialise_otoc_probe",
    "materialise_shadow_probe",
]
