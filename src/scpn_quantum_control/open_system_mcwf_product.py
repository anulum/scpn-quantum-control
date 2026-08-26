# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Open-system MCWF completeness product
"""Fail-closed **open-system MCWF completeness** product surface.

Productises documented open-system dynamics completeness over ambient
:mod:`scpn_quantum_control.phase.tensor_jump` (MCWF trajectories/ensembles) and
:mod:`scpn_quantum_control.phase.open_system_objectives` (seeded variance
certificates, open-system-objective boundary rows):

* versioned open-system surface catalogue (Lindblad density, MCWF trajectory,
  ensemble, noise-model I/O, gradient boundary);
* hard-gap boundary catalogue (non-CP, non-Markovian, adjoint Lindblad,
  hardware noise fidelity, process-tensor AD);
* real MCWF ensemble probe + same-seed reproducibility certificate;
* noise-model schema import/export for **simulation rates only**;
* refuse invent-green hardware noise claims, adjoint Lindblad, non-Markovian
  process tensors, and unseeded variance certificates.

Does **not** contact QPU providers, claim hardware noise fidelity, or execute
non-Markovian process-tensor AD.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

from .phase.open_system_objectives import (
    OPEN_SYSTEM_OBJECTIVE_CLAIM_BOUNDARY,
    default_open_system_objective_cases,
    open_system_objective_boundary_rows,
)

OpenSystemSurfaceKind = Literal[
    "lindblad_density",
    "mcwf_trajectory",
    "mcwf_ensemble",
    "noise_model_io",
    "gradient_boundary",
]
"""Open-system product surface kinds."""

BoundaryKind = Literal[
    "non_cp",
    "non_markovian",
    "adjoint_lindblad",
    "hardware_noise_fidelity",
    "process_tensor_ad",
]
"""Hard-gap boundary kinds for open-system completeness."""

SupportPosture = Literal[
    "local_research",
    "live_hardware_gated",
    "policy_only",
    "metadata_only",
]
"""Support posture badges."""

PathDecisionOutcome = Literal["allowed", "refused"]
"""Structured path-eligibility outcomes."""

OPEN_SYSTEM_MCWF_PRODUCT_SCHEMA: Final[str] = "open_system_mcwf_product.v1"
"""JSON schema identifier for serialised product payloads."""

OPEN_SYSTEM_MCWF_CLAIM_BOUNDARY: Final[str] = (
    "Open-system MCWF completeness product surface only; catalogues Lindblad/"
    "MCWF/noise-model/sim-first surfaces over ambient phase.tensor_jump and "
    "open_system_objectives; seeded ensemble variance certificates; refuse "
    "invent-green hardware noise fidelity, adjoint Lindblad gradients, "
    "non-Markovian process-tensor AD, and unseeded variance claims; residual "
    "committed evidence breadth and comparative closed/open/hardware-noisy "
    "gradient documentation remain explicitly incomplete"
)
"""Shared claim boundary for open-system MCWF product payloads."""

NOISE_MODEL_SCHEMA_ID: Final[str] = "open_system_sim_noise_model.v1"
"""Simulation-only noise-model schema identifier."""

_DEMO_SEED: Final[int] = 51
_DEMO_N_TRAJECTORIES: Final[int] = 4
_DEMO_T_MAX: Final[float] = 0.2
_DEMO_DT: Final[float] = 0.05


@dataclass(frozen=True, slots=True)
class OpenSystemSurfaceRow:
    """One open-system completeness surface catalogue row.

    Attributes
    ----------
    surface_id
        Stable surface identifier.
    kind
        Surface kind enum.
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

    surface_id: str
    kind: OpenSystemSurfaceKind
    title: str
    summary: str
    ambient_module: str
    ambient_symbol: str
    hardware_submit_allowed: bool = False
    support_posture: SupportPosture = "local_research"
    as_of: str = "2026-07-24"
    claim_boundary: str = OPEN_SYSTEM_MCWF_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate open-system surface invariants."""
        if not self.surface_id or not self.surface_id.strip():
            raise ValueError("surface_id must be non-empty")
        if self.kind not in {
            "lindblad_density",
            "mcwf_trajectory",
            "mcwf_ensemble",
            "noise_model_io",
            "gradient_boundary",
        }:
            raise ValueError(f"unknown surface kind: {self.kind!r}")
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
            "surface_id": self.surface_id,
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
class OpenSystemBoundaryRow:
    """One hard-gap boundary row for open-system completeness.

    Attributes
    ----------
    boundary_id
        Stable boundary identifier.
    kind
        Boundary kind enum.
    title
        Human-readable title.
    failure_class
        Machine-oriented failure class.
    summary
        Short description / setup guidance.
    fail_closed
        Must remain True.
    claim_boundary
        Non-promotional claim boundary.

    """

    boundary_id: str
    kind: BoundaryKind
    title: str
    failure_class: str
    summary: str
    fail_closed: bool = True
    claim_boundary: str = OPEN_SYSTEM_MCWF_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate boundary-row invariants."""
        if not self.boundary_id or not self.boundary_id.strip():
            raise ValueError("boundary_id must be non-empty")
        if self.kind not in {
            "non_cp",
            "non_markovian",
            "adjoint_lindblad",
            "hardware_noise_fidelity",
            "process_tensor_ad",
        }:
            raise ValueError(f"unknown boundary kind: {self.kind!r}")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.failure_class or not self.failure_class.strip():
            raise ValueError("failure_class must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if self.fail_closed is not True:
            raise ValueError("fail_closed must be True on product boundary rows")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this row."""
        return {
            "boundary_id": self.boundary_id,
            "kind": self.kind,
            "title": self.title,
            "failure_class": self.failure_class,
            "summary": self.summary,
            "fail_closed": self.fail_closed,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class PathEligibilityDecision:
    """Fail-closed path eligibility for open-system MCWF product use.

    Attributes
    ----------
    outcome
        Allowed or refused.
    allowed
        Whether the path may proceed under this product.
    reason
        Human-readable reason.
    blockers
        Non-empty when refused.
    claim_boundary
        Non-promotional claim boundary.

    """

    outcome: PathDecisionOutcome
    allowed: bool
    reason: str
    blockers: tuple[str, ...]
    claim_boundary: str = OPEN_SYSTEM_MCWF_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate path eligibility invariants."""
        if self.outcome not in {"allowed", "refused"}:
            raise ValueError(f"unknown outcome: {self.outcome!r}")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if self.allowed and self.outcome != "allowed":
            raise ValueError("allowed decisions must use outcome=allowed")
        if not self.allowed and self.outcome != "refused":
            raise ValueError("refused decisions must use outcome=refused")
        if self.allowed and self.blockers:
            raise ValueError("allowed decisions cannot list blockers")
        if not self.allowed and not self.blockers:
            raise ValueError("refused decisions require blockers")
        if any(not item or not item.strip() for item in self.blockers):
            raise ValueError("blockers entries must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this decision."""
        return {
            "outcome": self.outcome,
            "allowed": self.allowed,
            "reason": self.reason,
            "blockers": list(self.blockers),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class MaterialisedMcwfEnsembleProbe:
    """Materialised MCWF probe through the seeded ambient ensemble surface.

    Attributes
    ----------
    surface_id
        Surface used for the probe.
    n_trajectories
        Ensemble size.
    seed
        RNG seed (required for product probes).
    time_steps
        Number of time samples.
    final_mean_order_parameter
        Final mean order parameter R.
    final_std_order_parameter
        Final ensemble std of R.
    total_jumps
        Sum of jump counts across trajectories.
    probe_digest
        Canonical SHA-256 over key ensemble fields.
    invent_green_hardware_noise
        Always False.
    invent_green_adjoint_lindblad
        Always False.
    demo_label
        Demo fixture label.
    claim_boundary
        Product claim boundary.

    """

    surface_id: str
    n_trajectories: int
    seed: int
    time_steps: int
    final_mean_order_parameter: float
    final_std_order_parameter: float
    total_jumps: int
    probe_digest: str
    invent_green_hardware_noise: bool
    invent_green_adjoint_lindblad: bool
    demo_label: str
    claim_boundary: str = OPEN_SYSTEM_MCWF_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate MCWF ensemble probe invariants."""
        if not self.surface_id or not self.surface_id.strip():
            raise ValueError("surface_id must be non-empty")
        if self.n_trajectories < 1:
            raise ValueError("n_trajectories must be positive")
        if self.time_steps < 1:
            raise ValueError("time_steps must be positive")
        if self.total_jumps < 0:
            raise ValueError("total_jumps must be non-negative")
        if not math.isfinite(self.final_mean_order_parameter):
            raise ValueError("final_mean_order_parameter must be finite")
        if not math.isfinite(self.final_std_order_parameter):
            raise ValueError("final_std_order_parameter must be finite")
        if not self.probe_digest or len(self.probe_digest) != 64:
            raise ValueError("probe_digest must be a 64-char hex SHA-256")
        if self.invent_green_hardware_noise:
            raise ValueError("invent_green_hardware_noise must be False")
        if self.invent_green_adjoint_lindblad:
            raise ValueError("invent_green_adjoint_lindblad must be False")
        if not self.demo_label or not self.demo_label.strip():
            raise ValueError("demo_label must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe."""
        return {
            "surface_id": self.surface_id,
            "n_trajectories": self.n_trajectories,
            "seed": self.seed,
            "time_steps": self.time_steps,
            "final_mean_order_parameter": self.final_mean_order_parameter,
            "final_std_order_parameter": self.final_std_order_parameter,
            "total_jumps": self.total_jumps,
            "probe_digest": self.probe_digest,
            "invent_green_hardware_noise": self.invent_green_hardware_noise,
            "invent_green_adjoint_lindblad": self.invent_green_adjoint_lindblad,
            "demo_label": self.demo_label,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class MaterialisedReproducibilityProbe:
    """Same-seed MCWF reproducibility probe through the ambient certificate.

    Attributes
    ----------
    surface_id
        Surface identifier.
    certificate
        Ambient :class:`MCWFReproducibilityCertificate` as dict.
    ambient_claim_boundary
        Bounded open-system objective claim boundary.
    probe_digest
        Digest over certificate payload.
    demo_label
        Demo fixture label.
    claim_boundary
        Product claim boundary.

    """

    surface_id: str
    certificate: dict[str, object]
    ambient_claim_boundary: str
    probe_digest: str
    demo_label: str
    claim_boundary: str = OPEN_SYSTEM_MCWF_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate reproducibility probe invariants."""
        if not self.surface_id or not self.surface_id.strip():
            raise ValueError("surface_id must be non-empty")
        if not self.certificate:
            raise ValueError("certificate must be non-empty")
        if self.certificate.get("passed") is not True:
            raise ValueError("reproducibility certificate must have passed=True")
        if not self.ambient_claim_boundary or not self.ambient_claim_boundary.strip():
            raise ValueError("ambient_claim_boundary must be non-empty")
        if not self.probe_digest or len(self.probe_digest) != 64:
            raise ValueError("probe_digest must be a 64-char hex SHA-256")
        if not self.demo_label or not self.demo_label.strip():
            raise ValueError("demo_label must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe."""
        return {
            "surface_id": self.surface_id,
            "certificate": dict(self.certificate),
            "ambient_claim_boundary": self.ambient_claim_boundary,
            "probe_digest": self.probe_digest,
            "demo_label": self.demo_label,
            "claim_boundary": self.claim_boundary,
        }


def _build_surfaces() -> tuple[OpenSystemSurfaceRow, ...]:
    """Build the open-system surface catalogue."""
    return (
        OpenSystemSurfaceRow(
            surface_id="lindblad_density",
            kind="lindblad_density",
            title="Lindblad density-matrix path",
            summary=(
                "Scipy Lindblad density-matrix evolution on small local systems; "
                "objective evidence comes from the bounded open-system objective suite."
            ),
            ambient_module="scpn_quantum_control.phase.lindblad",
            ambient_symbol="LindbladKuramotoSolver",
            support_posture="local_research",
        ),
        OpenSystemSurfaceRow(
            surface_id="mcwf_trajectory",
            kind="mcwf_trajectory",
            title="Single MCWF trajectory",
            summary=(
                "Sparse MCWF statevector trajectory with quantum jumps; "
                "seeded replay; O(2^n) memory (not MPS)."
            ),
            ambient_module="scpn_quantum_control.phase.tensor_jump",
            ambient_symbol="mcwf_trajectory",
            support_posture="local_research",
        ),
        OpenSystemSurfaceRow(
            surface_id="mcwf_ensemble",
            kind="mcwf_ensemble",
            title="MCWF trajectory ensemble",
            summary=(
                "Seeded ensemble mean/std of order-parameter histories with "
                "reproducibility and variance certificates."
            ),
            ambient_module="scpn_quantum_control.phase.tensor_jump",
            ambient_symbol="mcwf_ensemble",
            support_posture="local_research",
        ),
        OpenSystemSurfaceRow(
            surface_id="noise_model_io",
            kind="noise_model_io",
            title="Simulation noise-model import/export",
            summary=(
                "Schema v1 for amplitude-damping and dephasing rates (sim first); "
                "not a hardware noise fidelity claim."
            ),
            ambient_module="scpn_quantum_control.open_system_mcwf_product",
            ambient_symbol="export_sim_noise_model",
            support_posture="metadata_only",
        ),
        OpenSystemSurfaceRow(
            surface_id="gradient_boundary",
            kind="gradient_boundary",
            title="Differentiable expectation boundary",
            summary=(
                "Central finite differences over scalar coupling/damping scales "
                "only; adjoint Lindblad and hardware gradients are hard gaps."
            ),
            ambient_module="scpn_quantum_control.phase.open_system_objectives",
            ambient_symbol="open_system_objective_boundary_rows",
            support_posture="policy_only",
        ),
    )


def _build_boundaries() -> tuple[OpenSystemBoundaryRow, ...]:
    """Build the hard-gap boundary catalogue."""
    return (
        OpenSystemBoundaryRow(
            boundary_id="non_cp_map",
            kind="non_cp",
            title="Non-completely-positive map",
            failure_class="non_cp_channel_refused",
            summary=(
                "Product refuses non-CP channel invent-green; ambient Lindblad/"
                "MCWF paths assume CP Markovian jump operators."
            ),
        ),
        OpenSystemBoundaryRow(
            boundary_id="non_markovian_dynamics",
            kind="non_markovian",
            title="Non-Markovian open dynamics",
            failure_class="non_markovian_refused",
            summary=(
                "Full non-Markovian process-tensor AD is out of v1 scope; "
                "fail closed until a validated process-tensor path exists."
            ),
        ),
        OpenSystemBoundaryRow(
            boundary_id="adjoint_lindblad_gradient",
            kind="adjoint_lindblad",
            title="Adjoint Lindblad gradient",
            failure_class="unsupported_adjoint_lindblad_gradient",
            summary=(
                "Only central finite differences over bounded scalar scales are "
                "executed on the objective suite; continuous adjoint Lindblad "
                "sensitivities require a separate validated solver."
            ),
        ),
        OpenSystemBoundaryRow(
            boundary_id="hardware_noise_fidelity",
            kind="hardware_noise_fidelity",
            title="Hardware noise fidelity claim",
            failure_class="no_live_provider_attestation",
            summary=(
                "No hardware-submitted open-system gradient or provider noise "
                "fidelity claim; the no-submit safety policy remains binding."
            ),
        ),
        OpenSystemBoundaryRow(
            boundary_id="process_tensor_ad",
            kind="process_tensor_ad",
            title="Process-tensor automatic differentiation",
            failure_class="process_tensor_ad_out_of_scope",
            summary=(
                "Process-tensor AD is outside the current product scope; residual for a future "
                "promotion package with validated invariants."
            ),
        ),
    )


_SURFACES: Final[tuple[OpenSystemSurfaceRow, ...]] = _build_surfaces()
_BOUNDARIES: Final[tuple[OpenSystemBoundaryRow, ...]] = _build_boundaries()


def _surface_map() -> dict[str, OpenSystemSurfaceRow]:
    """Return surface_id → row map; refuse blanks/duplicates."""
    mapping: dict[str, OpenSystemSurfaceRow] = {}
    for row in _SURFACES:
        key = row.surface_id.strip()
        if not key:
            raise RuntimeError("open-system surface catalogue contains blank surface_id")
        if key in mapping:
            raise RuntimeError(f"duplicate surface_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("open-system surface catalogue must be non-empty")
    return mapping


_SURFACE_BY_ID: Final[Mapping[str, OpenSystemSurfaceRow]] = _surface_map()


def list_open_system_surface_ids() -> tuple[str, ...]:
    """Return all open-system surface identifiers in catalogue order."""
    return tuple(row.surface_id for row in _SURFACES)


def list_open_system_boundary_ids() -> tuple[str, ...]:
    """Return all hard-gap boundary identifiers in catalogue order."""
    return tuple(row.boundary_id for row in _BOUNDARIES)


def get_open_system_surface(surface_id: str) -> OpenSystemSurfaceRow:
    """Return one surface row; fail closed on blank/unknown."""
    if not surface_id or not str(surface_id).strip():
        raise ValueError("surface_id must be non-empty")
    key = str(surface_id).strip()
    try:
        return _SURFACE_BY_ID[key]
    except KeyError as exc:
        raise ValueError(f"unknown surface_id: {key!r}") from exc


def get_open_system_boundary(boundary_id: str) -> OpenSystemBoundaryRow:
    """Return one boundary row; fail closed on blank/unknown."""
    if not boundary_id or not str(boundary_id).strip():
        raise ValueError("boundary_id must be non-empty")
    key = str(boundary_id).strip()
    for row in _BOUNDARIES:
        if row.boundary_id == key:
            return row
    raise ValueError(f"unknown boundary_id: {key!r}")


def iter_open_system_surfaces(
    *,
    kind: OpenSystemSurfaceKind | None = None,
) -> tuple[OpenSystemSurfaceRow, ...]:
    """Return filtered surface rows in stable order."""
    rows: Sequence[OpenSystemSurfaceRow] = _SURFACES
    if kind is not None:
        rows = tuple(row for row in rows if row.kind == kind)
    return tuple(rows)


def iter_open_system_boundaries(
    *,
    kind: BoundaryKind | None = None,
) -> tuple[OpenSystemBoundaryRow, ...]:
    """Return filtered boundary rows in stable order."""
    rows: Sequence[OpenSystemBoundaryRow] = _BOUNDARIES
    if kind is not None:
        rows = tuple(row for row in rows if row.kind == kind)
    return tuple(rows)


def decide_open_system_path(
    surface_id: str,
    *,
    invent_green_hardware_noise: bool = False,
    invent_green_adjoint_lindblad: bool = False,
    invent_green_non_markovian: bool = False,
    invent_green_non_cp: bool = False,
    unseeded_variance_claim: bool = False,
) -> PathEligibilityDecision:
    """Decide whether a catalogued open-system product path may proceed."""
    row = get_open_system_surface(surface_id)
    blockers: list[str] = []
    if invent_green_hardware_noise:
        blockers.append(
            "invent-green hardware noise fidelity refused "
            f"(surface={row.surface_id}; no-submit policy / no provider attestation)"
        )
    if invent_green_adjoint_lindblad:
        blockers.append(
            "invent-green adjoint Lindblad gradient refused "
            f"(surface={row.surface_id}; FD scalar scales only)"
        )
    if invent_green_non_markovian:
        blockers.append(
            "invent-green non-Markovian process-tensor path refused "
            f"(surface={row.surface_id}; outside current product scope)"
        )
    if invent_green_non_cp:
        blockers.append(
            "invent-green non-CP channel refused "
            f"(surface={row.surface_id}; CP Markovian jump ops only)"
        )
    if unseeded_variance_claim:
        blockers.append(
            "unseeded variance certificate claim refused "
            f"(surface={row.surface_id}; seeded replay required for variance evidence)"
        )
    if blockers:
        return PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="open-system path refused under product honesty gates",
            blockers=tuple(blockers),
        )
    return PathEligibilityDecision(
        outcome="allowed",
        allowed=True,
        reason=(
            f"surface {row.surface_id!r} may proceed as local open-system "
            "simulation completeness only"
        ),
        blockers=(),
    )


def _digest_payload(payload: Mapping[str, object]) -> str:
    """Canonical SHA-256 over a JSON-serialisable mapping."""
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_ambient_mcwf_json(
    mode: Literal["trajectory", "ensemble", "reproducibility"],
    *,
    n_trajectories: int,
    seed: int,
    t_max: float,
    dt: float,
    gamma_amp: float,
    gamma_deph: float,
) -> dict[str, Any]:
    """Run ambient MCWF in a clean subprocess (isolates pytest-cov numpy reloads)."""
    script = f"""
import json
import numpy as np
from scpn_quantum_control.phase.tensor_jump import mcwf_ensemble, mcwf_trajectory
from scpn_quantum_control.phase.open_system_objectives import (
    BoundedOpenSystemObjectiveCase,
    certify_mcwf_reproducibility,
)

K = np.array([[0.0, 0.1], [0.1, 0.0]], dtype=np.float64)
omega = np.array([1.0, -1.0], dtype=np.float64)
mode = {mode!r}
n_trajectories = {int(n_trajectories)}
seed = {int(seed)}
t_max = {float(t_max)}
dt = {float(dt)}
gamma_amp = {float(gamma_amp)}
gamma_deph = {float(gamma_deph)}

if mode == "trajectory":
    traj = mcwf_trajectory(K, omega, gamma_amp=gamma_amp, gamma_deph=gamma_deph,
                           t_max=t_max, dt=dt, seed=seed)
    R = np.asarray(traj["R"], dtype=np.float64)
    out = {{
        "n_trajectories": 1,
        "time_steps": int(R.shape[0]),
        "final_mean_order_parameter": float(R[-1]),
        "final_std_order_parameter": 0.0,
        "total_jumps": int(traj["n_jumps"]),
    }}
elif mode == "ensemble":
    ens = mcwf_ensemble(K, omega, gamma_amp=gamma_amp, gamma_deph=gamma_deph,
                        t_max=t_max, dt=dt, n_trajectories=n_trajectories, seed=seed)
    mean = np.asarray(ens["R_mean"], dtype=np.float64)
    std = np.asarray(ens["R_std"], dtype=np.float64)
    out = {{
        "n_trajectories": int(ens["n_trajectories"]),
        "time_steps": int(mean.shape[0]),
        "final_mean_order_parameter": float(mean[-1]),
        "final_std_order_parameter": float(std[-1]),
        "total_jumps": int(ens["total_jumps"]),
    }}
elif mode == "reproducibility":
    case = BoundedOpenSystemObjectiveCase(
        case_id="product_mcwf_repro_demo",
        n_oscillators=2,
        coupling_matrix=K,
        omega=omega,
        gamma_amp=gamma_amp,
        gamma_deph=gamma_deph,
        initial_params=np.array([1.0, 1.0], dtype=np.float64),
        target_order_parameter=0.5,
        target_purity=0.8,
        t_max=t_max,
        dt=dt,
        n_trajectories=n_trajectories,
        seed=seed,
    )
    first = mcwf_ensemble(K, omega, gamma_amp=gamma_amp, gamma_deph=gamma_deph,
                          t_max=t_max, dt=dt, n_trajectories=n_trajectories, seed=seed)
    second = mcwf_ensemble(K, omega, gamma_amp=gamma_amp, gamma_deph=gamma_deph,
                           t_max=t_max, dt=dt, n_trajectories=n_trajectories, seed=seed)
    cert = certify_mcwf_reproducibility(case, first, second)
    out = {{"certificate": cert.to_dict()}}
else:
    raise SystemExit(f"unknown mode {{mode!r}}")
print(json.dumps(out))
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or str(exc)).strip()
        raise ValueError(f"ambient MCWF subprocess failed: {err}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("ambient MCWF subprocess timed out") from exc
    line = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ambient MCWF subprocess returned non-JSON: {line!r}") from exc
    if not isinstance(payload, dict):
        raise ValueError("ambient MCWF subprocess payload must be an object")
    return payload


def materialise_mcwf_ensemble_probe(
    surface_id: str = "mcwf_ensemble",
    *,
    n_trajectories: int = _DEMO_N_TRAJECTORIES,
    seed: int = _DEMO_SEED,
    t_max: float = _DEMO_T_MAX,
    dt: float = _DEMO_DT,
    gamma_amp: float = 0.05,
    gamma_deph: float = 0.02,
    invent_green_hardware_noise: bool = False,
    invent_green_adjoint_lindblad: bool = False,
    demo_label: str = "mcwf_ensemble_demo",
) -> MaterialisedMcwfEnsembleProbe:
    """Materialise a real seeded probe through the ambient MCWF ensemble."""
    decision = decide_open_system_path(
        surface_id,
        invent_green_hardware_noise=invent_green_hardware_noise,
        invent_green_adjoint_lindblad=invent_green_adjoint_lindblad,
    )
    if not decision.allowed:
        raise ValueError("mcwf ensemble probe refused: " + "; ".join(decision.blockers))
    row = get_open_system_surface(surface_id)
    if row.kind not in {"mcwf_ensemble", "mcwf_trajectory"}:
        raise ValueError(
            f"materialise_mcwf_ensemble_probe requires MCWF surface, got {row.kind!r}"
        )
    if n_trajectories < 1:
        raise ValueError("n_trajectories must be positive")
    mode: Literal["trajectory", "ensemble"] = (
        "trajectory" if row.kind == "mcwf_trajectory" else "ensemble"
    )
    raw = _run_ambient_mcwf_json(
        mode,
        n_trajectories=n_trajectories,
        seed=seed,
        t_max=t_max,
        dt=dt,
        gamma_amp=gamma_amp,
        gamma_deph=gamma_deph,
    )
    try:
        n_traj = int(raw["n_trajectories"])
        time_steps = int(raw["time_steps"])
        final_mean = float(raw["final_mean_order_parameter"])
        final_std = float(raw["final_std_order_parameter"])
        total_jumps = int(raw["total_jumps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"ambient MCWF payload missing fields: {raw!r}") from exc
    if time_steps < 1 or n_traj < 1:
        raise ValueError("ambient MCWF result must be non-empty")
    if not math.isfinite(final_mean) or not math.isfinite(final_std):
        raise ValueError("ambient MCWF result must be finite")
    if total_jumps < 0:
        raise ValueError("total_jumps must be non-negative")
    digest = _digest_payload(
        {
            "schema": "open_system_mcwf_ensemble_probe.v1",
            "surface_id": row.surface_id,
            "n_trajectories": n_traj,
            "seed": seed,
            "time_steps": time_steps,
            "final_mean": final_mean,
            "final_std": final_std,
            "total_jumps": total_jumps,
            "product_schema": OPEN_SYSTEM_MCWF_PRODUCT_SCHEMA,
        }
    )
    return MaterialisedMcwfEnsembleProbe(
        surface_id=row.surface_id,
        n_trajectories=n_traj,
        seed=seed,
        time_steps=time_steps,
        final_mean_order_parameter=final_mean,
        final_std_order_parameter=final_std,
        total_jumps=total_jumps,
        probe_digest=digest,
        invent_green_hardware_noise=False,
        invent_green_adjoint_lindblad=False,
        demo_label=demo_label.strip() or "mcwf_ensemble_demo",
    )


def materialise_demo_mcwf_ensemble_probe() -> MaterialisedMcwfEnsembleProbe:
    """Materialise the default offline MCWF ensemble demo probe."""
    return materialise_mcwf_ensemble_probe("mcwf_ensemble")


def materialise_reproducibility_probe(
    *,
    seed: int = _DEMO_SEED,
    n_trajectories: int = _DEMO_N_TRAJECTORIES,
    demo_label: str = "mcwf_reproducibility_demo",
) -> MaterialisedReproducibilityProbe:
    """Run the same seeded ensemble twice and certify reproducibility."""
    decision = decide_open_system_path("mcwf_ensemble", unseeded_variance_claim=False)
    if not decision.allowed:
        raise ValueError("reproducibility probe refused: " + "; ".join(decision.blockers))
    if n_trajectories < 1:
        raise ValueError("n_trajectories must be positive")
    raw = _run_ambient_mcwf_json(
        "reproducibility",
        n_trajectories=n_trajectories,
        seed=seed,
        t_max=_DEMO_T_MAX,
        dt=_DEMO_DT,
        gamma_amp=0.05,
        gamma_deph=0.02,
    )
    cert_obj = raw.get("certificate")
    if not isinstance(cert_obj, dict):
        raise ValueError("ambient reproducibility payload missing certificate object")
    if cert_obj.get("passed") is not True:
        raise ValueError("ambient MCWF reproducibility certificate did not pass")
    cert_dict: dict[str, object] = dict(cert_obj)
    digest = _digest_payload(
        {
            "schema": "open_system_mcwf_reproducibility_probe.v1",
            "certificate": cert_dict,
            "product_schema": OPEN_SYSTEM_MCWF_PRODUCT_SCHEMA,
        }
    )
    return MaterialisedReproducibilityProbe(
        surface_id="mcwf_ensemble",
        certificate=cert_dict,
        ambient_claim_boundary=OPEN_SYSTEM_OBJECTIVE_CLAIM_BOUNDARY,
        probe_digest=digest,
        demo_label=demo_label.strip() or "mcwf_reproducibility_demo",
    )


def export_sim_noise_model(
    *,
    gamma_amp: float,
    gamma_deph: float,
    label: str = "sim_noise",
) -> dict[str, object]:
    """Export a simulation noise-model payload without hardware-fidelity claims."""
    if not math.isfinite(gamma_amp) or gamma_amp < 0.0:
        raise ValueError("gamma_amp must be finite and non-negative")
    if not math.isfinite(gamma_deph) or gamma_deph < 0.0:
        raise ValueError("gamma_deph must be finite and non-negative")
    if not label or not str(label).strip():
        raise ValueError("label must be non-empty")
    payload = {
        "schema": NOISE_MODEL_SCHEMA_ID,
        "label": str(label).strip(),
        "gamma_amp": float(gamma_amp),
        "gamma_deph": float(gamma_deph),
        "domain": "simulation_only",
        "hardware_noise_fidelity_claim": False,
        "claim_boundary": OPEN_SYSTEM_MCWF_CLAIM_BOUNDARY,
    }
    payload["digest"] = _digest_payload(
        {
            "schema": payload["schema"],
            "label": payload["label"],
            "gamma_amp": payload["gamma_amp"],
            "gamma_deph": payload["gamma_deph"],
            "domain": payload["domain"],
        }
    )
    return payload


def import_sim_noise_model(payload: Mapping[str, object]) -> dict[str, object]:
    """Import and validate a simulation-only noise-model payload."""
    if not isinstance(payload, Mapping):
        raise ValueError("noise model payload must be a mapping")
    schema = payload.get("schema")
    if schema != NOISE_MODEL_SCHEMA_ID:
        raise ValueError(f"unknown noise model schema: {schema!r}")
    if payload.get("domain") != "simulation_only":
        raise ValueError("noise model domain must be simulation_only")
    if payload.get("hardware_noise_fidelity_claim") is not False:
        raise ValueError("hardware_noise_fidelity_claim must be False")
    gamma_amp = payload.get("gamma_amp")
    gamma_deph = payload.get("gamma_deph")
    label = payload.get("label")
    if not isinstance(gamma_amp, (int, float)) or isinstance(gamma_amp, bool):
        raise ValueError("gamma_amp must be a finite non-negative number")
    if not isinstance(gamma_deph, (int, float)) or isinstance(gamma_deph, bool):
        raise ValueError("gamma_deph must be a finite non-negative number")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("label must be a non-empty string")
    return export_sim_noise_model(
        gamma_amp=float(gamma_amp),
        gamma_deph=float(gamma_deph),
        label=label,
    )


def map_open_system_mcwf_public_surfaces() -> tuple[dict[str, object], ...]:
    """Map public surfaces composing the open-system MCWF product."""
    return (
        {
            "surface_id": "open_system_mcwf_product",
            "module_path": "scpn_quantum_control.open_system_mcwf_product",
            "role": "product_facade",
            "claim_boundary": OPEN_SYSTEM_MCWF_CLAIM_BOUNDARY,
        },
        {
            "surface_id": "tensor_jump_mcwf",
            "module_path": "scpn_quantum_control.phase.tensor_jump",
            "role": "ambient_mcwf",
            "claim_boundary": OPEN_SYSTEM_MCWF_CLAIM_BOUNDARY,
        },
        {
            "surface_id": "open_system_objectives",
            "module_path": "scpn_quantum_control.phase.open_system_objectives",
            "role": "ambient_objectives_trainability_evidence",
            "claim_boundary": OPEN_SYSTEM_OBJECTIVE_CLAIM_BOUNDARY,
        },
    )


def list_ambient_objective_boundary_ids() -> tuple[str, ...]:
    """Return case identifiers from the bounded objective boundary catalogue."""
    return tuple(row.case_id for row in open_system_objective_boundary_rows())


def list_default_objective_case_ids() -> tuple[str, ...]:
    """Return ambient default open-system objective case ids."""
    return tuple(case.case_id for case in default_open_system_objective_cases())


def build_open_system_mcwf_product_registry() -> dict[str, object]:
    """Build the versioned open-system MCWF product registry payload."""
    surfaces = [row.to_dict() for row in _SURFACES]
    boundaries = [row.to_dict() for row in _BOUNDARIES]
    return {
        "schema": OPEN_SYSTEM_MCWF_PRODUCT_SCHEMA,
        "claim_boundary": OPEN_SYSTEM_MCWF_CLAIM_BOUNDARY,
        "surface_count": len(surfaces),
        "boundary_count": len(boundaries),
        "blank_entry_count": 0,
        "hardware_submit_allowed_policy": False,
        "hardware_noise_fidelity_claim_policy": False,
        "adjoint_lindblad_allowed_policy": False,
        "non_markovian_process_tensor_allowed_policy": False,
        "public_surfaces": list(map_open_system_mcwf_public_surfaces()),
        "surfaces": surfaces,
        "boundaries": boundaries,
        "ambient_objective_boundary_ids": list(list_ambient_objective_boundary_ids()),
        "ambient_objective_case_ids": list(list_default_objective_case_ids()),
        "noise_model_schema": NOISE_MODEL_SCHEMA_ID,
        "policy_note": (
            "Open-system MCWF completeness is local simulation + certificates only; "
            "seeded ensembles; sim noise I/O; hard gaps for non-CP, non-Markovian "
            "process tensor, adjoint Lindblad, and hardware noise fidelity."
        ),
    }


def assert_open_system_mcwf_product_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert registry covers surfaces/boundaries without invent-green policies."""
    registry = dict(payload) if payload is not None else build_open_system_mcwf_product_registry()
    surfaces = registry.get("surfaces")
    boundaries = registry.get("boundaries")
    if not isinstance(surfaces, list) or not surfaces:
        raise ValueError(
            "open-system MCWF product registry must contain a non-empty surfaces list"
        )
    if not isinstance(boundaries, list) or not boundaries:
        raise ValueError(
            "open-system MCWF product registry must contain a non-empty boundaries list"
        )
    seen: set[str] = set()
    blank = 0
    mcwf_found = False
    for index, row in enumerate(surfaces):
        if not isinstance(row, Mapping):
            raise ValueError(f"surface row {index} must be a mapping")
        surface_id = row.get("surface_id")
        hw = row.get("hardware_submit_allowed")
        symbol = row.get("ambient_symbol")
        if not surface_id or not str(surface_id).strip():
            blank += 1
            continue
        sid = str(surface_id).strip()
        if sid in seen:
            raise ValueError(f"duplicate surface_id in registry: {sid!r}")
        seen.add(sid)
        if sid == "mcwf_ensemble":
            mcwf_found = True
        if hw is not False:
            raise ValueError(f"surface {sid!r} hardware_submit_allowed must be False")
        if not symbol or not str(symbol).strip():
            raise ValueError(f"surface {sid!r} must have non-empty ambient_symbol")
    if blank:
        raise ValueError(f"open-system MCWF product registry has {blank} blank or invalid entries")
    if not mcwf_found:
        raise ValueError("open-system MCWF product registry missing mcwf_ensemble")
    expected = set(list_open_system_surface_ids())
    if seen != expected:
        raise ValueError(
            f"registry surface set drift (missing={expected - seen!r}, extra={seen - expected!r})"
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
    expected_b = set(list_open_system_boundary_ids())
    if seen_b != expected_b:
        raise ValueError(
            f"registry boundary set drift (missing={expected_b - seen_b!r}, "
            f"extra={seen_b - expected_b!r})"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    surface_count = registry.get("surface_count", -1)
    if not isinstance(surface_count, int) or surface_count != len(surfaces):
        raise ValueError("surface_count does not match surfaces list length")
    boundary_count = registry.get("boundary_count", -1)
    if not isinstance(boundary_count, int) or boundary_count != len(boundaries):
        raise ValueError("boundary_count does not match boundaries list length")
    if registry.get("hardware_submit_allowed_policy", True) is not False:
        raise ValueError("hardware_submit_allowed_policy must be False")
    if registry.get("hardware_noise_fidelity_claim_policy", True) is not False:
        raise ValueError("hardware_noise_fidelity_claim_policy must be False")
    if registry.get("adjoint_lindblad_allowed_policy", True) is not False:
        raise ValueError("adjoint_lindblad_allowed_policy must be False")
    if registry.get("non_markovian_process_tensor_allowed_policy", True) is not False:
        raise ValueError("non_markovian_process_tensor_allowed_policy must be False")
    return registry


__all__ = [
    "BoundaryKind",
    "MaterialisedMcwfEnsembleProbe",
    "MaterialisedReproducibilityProbe",
    "NOISE_MODEL_SCHEMA_ID",
    "OPEN_SYSTEM_MCWF_CLAIM_BOUNDARY",
    "OPEN_SYSTEM_MCWF_PRODUCT_SCHEMA",
    "OpenSystemBoundaryRow",
    "OpenSystemSurfaceKind",
    "OpenSystemSurfaceRow",
    "PathDecisionOutcome",
    "PathEligibilityDecision",
    "SupportPosture",
    "assert_open_system_mcwf_product_integrity",
    "build_open_system_mcwf_product_registry",
    "decide_open_system_path",
    "export_sim_noise_model",
    "get_open_system_boundary",
    "get_open_system_surface",
    "import_sim_noise_model",
    "iter_open_system_boundaries",
    "iter_open_system_surfaces",
    "list_ambient_objective_boundary_ids",
    "list_default_objective_case_ids",
    "list_open_system_boundary_ids",
    "list_open_system_surface_ids",
    "map_open_system_mcwf_public_surfaces",
    "materialise_demo_mcwf_ensemble_probe",
    "materialise_mcwf_ensemble_probe",
    "materialise_reproducibility_probe",
]
