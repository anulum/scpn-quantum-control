# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — PGBO quantum geometric tensor product
"""Fail-closed **PGBO quantum geometric tensor** product surface.

Productises metric + Berry curvature on coupling-parameter space over ambient
:mod:`scpn_quantum_control.pgbo.quantum_bridge`:

* versioned QGT capability catalogue (tensor compute, Fubini–Study metric,
  Berry curvature, size-cap policy, geometric-control compose);
* real :func:`~scpn_quantum_control.pgbo.quantum_bridge.compute_pgbo_tensor`
  probe on small XY systems with fail-closed oscillator-count caps;
* hard-gap boundaries: invent-green experimental geometry, live QPU, unbounded N;
* compose geometric-control geometric control catalogue; refuse invent-green.

Does **not** claim experimental quantum geometry on hardware, submit to QPU,
or remove ambient finite-difference approximation of state derivatives.
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

import numpy as np
from numpy.typing import NDArray

QgtCapabilityKind = Literal[
    "pgbo_tensor",
    "fubini_study_metric",
    "berry_curvature",
    "size_cap_policy",
    "geometric_control_compose",
]
"""QGT capability kinds on the product catalogue."""

SupportPosture = Literal[
    "local_research",
    "live_hardware_gated",
    "policy_only",
    "metadata_only",
]
"""Support posture badges."""

PathDecisionOutcome = Literal["allowed", "refused"]
"""Structured path-eligibility outcomes."""

BoundaryKind = Literal[
    "experimental_geometry_claim",
    "live_qpu_qgt",
    "unbounded_system_size",
    "fd_derivative_as_exact",
]
"""Hard-gap boundary kinds for PGBO QGT honesty."""

PGBO_QGT_PRODUCT_SCHEMA: Final[str] = "pgbo_qgt_product.v2"
"""JSON schema identifier for serialised product payloads."""

PGBO_QGT_CLAIM_BOUNDARY: Final[str] = (
    "PGBO quantum geometric tensor product surface only; catalogues QGT/"
    "Fubini–Study/Berry-curvature capabilities over ambient "
    "pgbo.quantum_bridge.compute_pgbo_tensor; small-system probes with "
    "fail-closed N caps; parameter-shift FD derivatives on K (not exact "
    "analytic AD); refuse invent-green experimental geometry and live QPU; "
    "compose the geometric-control catalogue; dashboard integration and "
    "metamorphic registration depth remain unresolved"
)
"""Shared claim boundary for PGBO QGT product payloads."""

_PGBO_QGT_POLICY_NOTE: Final[str] = (
    "PGBO QGT is small-system simulation geometry on K-space with finite-difference "
    "state derivatives; experimental geometry and live-QPU claims remain refused; "
    "system size is capped; geometric-control catalogue composition is metadata-only; "
    "dashboard integration and metamorphic registration depth remain unresolved."
)
_PGBO_QGT_REGISTRY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "claim_boundary",
        "capability_count",
        "boundary_count",
        "blank_entry_count",
        "max_oscillators",
        "default_epsilon",
        "hardware_submit_allowed_policy",
        "experimental_geometry_claim_policy",
        "unbounded_system_size_policy",
        "fd_derivative_as_exact_policy",
        "public_surfaces",
        "capabilities",
        "boundaries",
        "policy_note",
    }
)

# Fail-closed dense ground-state size cap. Ambient exact diagonalisation is O(2^n).
MAX_OSCILLATORS: Final[int] = 6
"""Maximum oscillators allowed on product probes (fail closed above)."""

DEFAULT_EPSILON: Final[float] = 0.005
"""Default finite-difference step for ambient parameter-shift derivatives."""


@dataclass(frozen=True, slots=True)
class QgtCapabilityRow:
    """One PGBO QGT capability catalogue row.

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
    kind: QgtCapabilityKind
    title: str
    summary: str
    ambient_module: str
    ambient_symbol: str
    hardware_submit_allowed: bool = False
    support_posture: SupportPosture = "local_research"
    as_of: str = "2026-07-24"
    claim_boundary: str = PGBO_QGT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate QGT capability invariants."""
        if not self.capability_id or not self.capability_id.strip():
            raise ValueError("capability_id must be non-empty")
        if self.kind not in {
            "pgbo_tensor",
            "fubini_study_metric",
            "berry_curvature",
            "size_cap_policy",
            "geometric_control_compose",
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
class QgtBoundaryRow:
    """One hard-gap boundary row for PGBO QGT honesty."""

    boundary_id: str
    kind: BoundaryKind
    title: str
    failure_class: str
    summary: str
    fail_closed: bool = True
    claim_boundary: str = PGBO_QGT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate boundary-row invariants."""
        if not self.boundary_id or not self.boundary_id.strip():
            raise ValueError("boundary_id must be non-empty")
        if self.kind not in {
            "experimental_geometry_claim",
            "live_qpu_qgt",
            "unbounded_system_size",
            "fd_derivative_as_exact",
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
    """Fail-closed path eligibility for PGBO QGT product use."""

    outcome: PathDecisionOutcome
    allowed: bool
    reason: str
    blockers: tuple[str, ...]
    claim_boundary: str = PGBO_QGT_CLAIM_BOUNDARY

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
class MaterialisedPgboTensorProbe:
    """Materialised ambient ``compute_pgbo_tensor`` probe."""

    capability_id: str
    n_oscillators: int
    n_parameters: int
    metric_determinant: float
    total_curvature: float
    parameter_labels: tuple[str, ...]
    metric_frobenius: float
    epsilon: float
    probe_digest: str
    invent_green_experimental_geometry: bool
    invent_green_live_qpu: bool
    demo_label: str
    claim_boundary: str = PGBO_QGT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate PGBO tensor probe invariants."""
        if not self.capability_id or not self.capability_id.strip():
            raise ValueError("capability_id must be non-empty")
        if self.n_oscillators < 2:
            raise ValueError("n_oscillators must be at least 2")
        if self.n_oscillators > MAX_OSCILLATORS:
            raise ValueError(f"n_oscillators must be <= {MAX_OSCILLATORS} on product probes")
        if self.n_parameters < 1:
            raise ValueError("n_parameters must be positive")
        expected_params = self.n_oscillators * (self.n_oscillators - 1) // 2
        if self.n_parameters != expected_params:
            raise ValueError(f"n_parameters must equal upper-triangle size {expected_params}")
        if not math.isfinite(self.metric_determinant):
            raise ValueError("metric_determinant must be finite")
        if not math.isfinite(self.total_curvature) or self.total_curvature < 0.0:
            raise ValueError("total_curvature must be finite and non-negative")
        if len(self.parameter_labels) != self.n_parameters:
            raise ValueError("parameter_labels length must equal n_parameters")
        if any(not item or not str(item).strip() for item in self.parameter_labels):
            raise ValueError("parameter_labels entries must be non-empty")
        if not math.isfinite(self.metric_frobenius) or self.metric_frobenius < 0.0:
            raise ValueError("metric_frobenius must be finite and non-negative")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        if not self.probe_digest or len(self.probe_digest) != 64:
            raise ValueError("probe_digest must be a 64-char hex SHA-256")
        if self.invent_green_experimental_geometry:
            raise ValueError("invent_green_experimental_geometry must be False")
        if self.invent_green_live_qpu:
            raise ValueError("invent_green_live_qpu must be False")
        if not self.demo_label or not self.demo_label.strip():
            raise ValueError("demo_label must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe."""
        return {
            "capability_id": self.capability_id,
            "n_oscillators": self.n_oscillators,
            "n_parameters": self.n_parameters,
            "metric_determinant": self.metric_determinant,
            "total_curvature": self.total_curvature,
            "parameter_labels": list(self.parameter_labels),
            "metric_frobenius": self.metric_frobenius,
            "epsilon": self.epsilon,
            "probe_digest": self.probe_digest,
            "invent_green_experimental_geometry": self.invent_green_experimental_geometry,
            "invent_green_live_qpu": self.invent_green_live_qpu,
            "demo_label": self.demo_label,
            "claim_boundary": self.claim_boundary,
        }


def _build_capabilities() -> tuple[QgtCapabilityRow, ...]:
    """Build the QGT capability catalogue."""
    return (
        QgtCapabilityRow(
            capability_id="pgbo_tensor",
            kind="pgbo_tensor",
            title="PGBO quantum geometric tensor",
            summary=(
                "Ambient compute_pgbo_tensor: Q_μν from ground-state parameter-shift "
                "derivatives w.r.t. upper-triangle K_ij (simulation only)."
            ),
            ambient_module="scpn_quantum_control.pgbo.quantum_bridge",
            ambient_symbol="compute_pgbo_tensor",
        ),
        QgtCapabilityRow(
            capability_id="fubini_study_metric",
            kind="fubini_study_metric",
            title="Fubini–Study metric (Re Q)",
            summary=(
                "h_μν = Re(Q_μν): quantum distance metric on coupling-parameter "
                "space extracted from the PGBO tensor."
            ),
            ambient_module="scpn_quantum_control.pgbo.quantum_bridge",
            ambient_symbol="PGBOResult.metric_tensor",
        ),
        QgtCapabilityRow(
            capability_id="berry_curvature",
            kind="berry_curvature",
            title="Berry curvature (Im Q)",
            summary=(
                "F_μν = -2 Im(Q_μν): geometric phase curvature on K-space; "
                "total_curvature aggregates absolute entries."
            ),
            ambient_module="scpn_quantum_control.pgbo.quantum_bridge",
            ambient_symbol="PGBOResult.berry_curvature",
        ),
        QgtCapabilityRow(
            capability_id="size_cap_policy",
            kind="size_cap_policy",
            title="Fail-closed oscillator count cap",
            summary=(
                f"Product probes refuse n > {MAX_OSCILLATORS} oscillators "
                "(dense ground-state cost); no invent-green unbounded N."
            ),
            ambient_module="scpn_quantum_control.pgbo_qgt_product",
            ambient_symbol="MAX_OSCILLATORS",
            support_posture="policy_only",
        ),
        QgtCapabilityRow(
            capability_id="geometric_control_compose",
            kind="geometric_control_compose",
            title="Geometric-control catalogue composition",
            summary=(
                "Pairs with geometric_control_product (McLachlan/QNG on ansatz "
                "params); this product owns QGT on coupling K-space."
            ),
            ambient_module="scpn_quantum_control.geometric_control_product",
            ambient_symbol="build_geometric_control_product_registry",
            support_posture="metadata_only",
        ),
    )


def _build_boundaries() -> tuple[QgtBoundaryRow, ...]:
    """Build hard-gap boundary catalogue."""
    return (
        QgtBoundaryRow(
            boundary_id="experimental_geometry_claim",
            kind="experimental_geometry_claim",
            title="Experimental quantum geometry claim",
            failure_class="experimental_geometry_claim_refused",
            summary=(
                "Product refuses invent-green claims that PGBO QGT was measured "
                "on hardware or validates experimental geometry."
            ),
        ),
        QgtBoundaryRow(
            boundary_id="live_qpu_qgt",
            kind="live_qpu_qgt",
            title="Live QPU QGT submit",
            failure_class="live_qpu_qgt_refused",
            summary=(
                "Compose the hardware-safe no-submit policy; probes use local "
                "classical exact-diagonalisation ground states only."
            ),
        ),
        QgtBoundaryRow(
            boundary_id="unbounded_system_size",
            kind="unbounded_system_size",
            title="Unbounded system size",
            failure_class="unbounded_system_size_refused",
            summary=(
                f"Fail closed for n > {MAX_OSCILLATORS}; no invent-green large-N "
                "QGT without a validated sparse path."
            ),
        ),
        QgtBoundaryRow(
            boundary_id="fd_derivative_as_exact",
            kind="fd_derivative_as_exact",
            title="FD state derivatives sold as exact AD",
            failure_class="fd_derivative_as_exact_refused",
            summary=(
                "Ambient ∂|ψ>/∂K uses central finite differences with phase "
                "alignment; product refuses invent-green exact analytic AD claim."
            ),
        ),
    )


_CAPABILITIES: Final[tuple[QgtCapabilityRow, ...]] = _build_capabilities()
_BOUNDARIES: Final[tuple[QgtBoundaryRow, ...]] = _build_boundaries()


def _capability_map() -> dict[str, QgtCapabilityRow]:
    """Return capability_id → row map; refuse blanks/duplicates."""
    mapping: dict[str, QgtCapabilityRow] = {}
    for row in _CAPABILITIES:
        key = row.capability_id.strip()
        if not key:
            raise RuntimeError("QGT capability catalogue contains blank capability_id")
        if key in mapping:
            raise RuntimeError(f"duplicate capability_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("QGT capability catalogue must be non-empty")
    return mapping


_CAPABILITY_BY_ID: Final[Mapping[str, QgtCapabilityRow]] = _capability_map()


def list_qgt_capability_ids() -> tuple[str, ...]:
    """Return all QGT capability identifiers in catalogue order."""
    return tuple(row.capability_id for row in _CAPABILITIES)


def list_qgt_boundary_ids() -> tuple[str, ...]:
    """Return all hard-gap boundary identifiers in catalogue order."""
    return tuple(row.boundary_id for row in _BOUNDARIES)


def get_qgt_capability(capability_id: str) -> QgtCapabilityRow:
    """Return one capability row; fail closed on blank/unknown."""
    if not capability_id or not str(capability_id).strip():
        raise ValueError("capability_id must be non-empty")
    key = str(capability_id).strip()
    try:
        return _CAPABILITY_BY_ID[key]
    except KeyError as exc:
        raise ValueError(f"unknown capability_id: {key!r}") from exc


def get_qgt_boundary(boundary_id: str) -> QgtBoundaryRow:
    """Return one boundary row; fail closed on blank/unknown."""
    if not boundary_id or not str(boundary_id).strip():
        raise ValueError("boundary_id must be non-empty")
    key = str(boundary_id).strip()
    for row in _BOUNDARIES:
        if row.boundary_id == key:
            return row
    raise ValueError(f"unknown boundary_id: {key!r}")


def iter_qgt_capabilities(
    *,
    kind: QgtCapabilityKind | None = None,
) -> tuple[QgtCapabilityRow, ...]:
    """Return filtered capability rows in stable order."""
    rows: Sequence[QgtCapabilityRow] = _CAPABILITIES
    if kind is not None:
        rows = tuple(row for row in rows if row.kind == kind)
    return tuple(rows)


def iter_qgt_boundaries(
    *,
    kind: BoundaryKind | None = None,
) -> tuple[QgtBoundaryRow, ...]:
    """Return filtered boundary rows in stable order."""
    rows: Sequence[QgtBoundaryRow] = _BOUNDARIES
    if kind is not None:
        rows = tuple(row for row in rows if row.kind == kind)
    return tuple(rows)


def decide_qgt_path(
    capability_id: str,
    *,
    invent_green_experimental_geometry: bool = False,
    invent_green_live_qpu: bool = False,
    invent_green_unbounded_n: bool = False,
    invent_green_fd_as_exact: bool = False,
) -> PathEligibilityDecision:
    """Decide whether a PGBO QGT product path may proceed."""
    row = get_qgt_capability(capability_id)
    blockers: list[str] = []
    if invent_green_experimental_geometry:
        blockers.append(
            "invent-green experimental geometry claim refused "
            f"(capability={row.capability_id}; simulation QGT only)"
        )
    if invent_green_live_qpu:
        blockers.append(
            "invent-green live QPU QGT refused "
            f"(capability={row.capability_id}; hardware-safe no-submit policy)"
        )
    if invent_green_unbounded_n:
        blockers.append(
            "invent-green unbounded system size refused "
            f"(capability={row.capability_id}; max_oscillators={MAX_OSCILLATORS})"
        )
    if invent_green_fd_as_exact:
        blockers.append(
            "invent-green exact-AD claim for FD state derivatives refused "
            f"(capability={row.capability_id}; ambient uses finite differences)"
        )
    if blockers:
        return PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="QGT path refused under product honesty gates",
            blockers=tuple(blockers),
        )
    return PathEligibilityDecision(
        outcome="allowed",
        allowed=True,
        reason=(
            f"capability {row.capability_id!r} may proceed as local small-system "
            "QGT simulation only"
        ),
        blockers=(),
    )


def _digest_payload(payload: Mapping[str, object]) -> str:
    """Canonical SHA-256 over a JSON-serialisable mapping."""
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _demo_coupling_system(
    n_oscillators: int = 2,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a small deterministic coupling system for demos."""
    if n_oscillators < 2:
        raise ValueError("n_oscillators must be at least 2")
    if n_oscillators > MAX_OSCILLATORS:
        raise ValueError(f"n_oscillators must be <= {MAX_OSCILLATORS}")
    if n_oscillators == 2:
        coupling = np.array([[0.0, 0.3], [0.3, 0.0]], dtype=np.float64)
        omega = np.array([1.0, -1.0], dtype=np.float64)
        return coupling, omega
    rng = np.random.default_rng(71 + n_oscillators)
    raw = rng.normal(size=(n_oscillators, n_oscillators))
    coupling = 0.5 * (raw + raw.T)
    np.fill_diagonal(coupling, 0.0)
    omega = rng.normal(size=n_oscillators).astype(np.float64)
    return coupling.astype(np.float64), omega


def _run_ambient_pgbo_json(
    *,
    n_oscillators: int,
    epsilon: float,
) -> dict[str, Any]:
    """Run ambient compute_pgbo_tensor in a clean subprocess (pytest-cov safe)."""
    if n_oscillators == 2:
        k_lit = "[[0.0, 0.3], [0.3, 0.0]]"
        o_lit = "[1.0, -1.0]"
    else:
        coupling, omega = _demo_coupling_system(n_oscillators)
        k_lit = json.dumps(coupling.tolist())
        o_lit = json.dumps(omega.tolist())
    script = f"""
import json
import numpy as np
from scpn_quantum_control.pgbo.quantum_bridge import compute_pgbo_tensor
K = np.asarray({k_lit}, dtype=np.float64)
omega = np.asarray({o_lit}, dtype=np.float64)
r = compute_pgbo_tensor(K, omega, epsilon={float(epsilon)})
metric = np.asarray(r.metric_tensor, dtype=np.float64)
out = {{
    "n_parameters": int(r.n_parameters),
    "metric_determinant": float(r.metric_determinant),
    "total_curvature": float(r.total_curvature),
    "parameter_labels": [str(x) for x in r.parameter_labels],
    "metric_frobenius": float(np.linalg.norm(metric, ord="fro")),
    "metric_shape": list(metric.shape),
    "metric_finite": bool(np.all(np.isfinite(metric))),
}}
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
        raise ValueError(f"ambient PGBO subprocess failed: {err}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("ambient PGBO subprocess timed out") from exc
    line = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ambient PGBO subprocess returned non-JSON: {line!r}") from exc
    if not isinstance(payload, dict):
        raise ValueError("ambient PGBO subprocess payload must be an object")
    return payload


def materialise_pgbo_tensor_probe(
    capability_id: str = "pgbo_tensor",
    *,
    n_oscillators: int = 2,
    epsilon: float = DEFAULT_EPSILON,
    invent_green_experimental_geometry: bool = False,
    invent_green_live_qpu: bool = False,
    invent_green_unbounded_n: bool = False,
    invent_green_fd_as_exact: bool = False,
    demo_label: str = "pgbo_tensor_demo",
) -> MaterialisedPgboTensorProbe:
    """Materialise a real ambient PGBO QGT probe."""
    decision = decide_qgt_path(
        capability_id,
        invent_green_experimental_geometry=invent_green_experimental_geometry,
        invent_green_live_qpu=invent_green_live_qpu,
        invent_green_unbounded_n=invent_green_unbounded_n,
        invent_green_fd_as_exact=invent_green_fd_as_exact,
    )
    if not decision.allowed:
        raise ValueError("pgbo tensor probe refused: " + "; ".join(decision.blockers))
    row = get_qgt_capability(capability_id)
    if row.kind not in {"pgbo_tensor", "fubini_study_metric", "berry_curvature"}:
        raise ValueError(
            f"materialise_pgbo_tensor_probe requires tensor-family capability, got {row.kind!r}"
        )
    if n_oscillators > MAX_OSCILLATORS:
        raise ValueError(f"n_oscillators={n_oscillators} exceeds product cap {MAX_OSCILLATORS}")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    raw = _run_ambient_pgbo_json(n_oscillators=n_oscillators, epsilon=epsilon)
    try:
        n_parameters = int(raw["n_parameters"])
        metric_determinant = float(raw["metric_determinant"])
        total_curvature = float(raw["total_curvature"])
        labels_raw = raw["parameter_labels"]
        frobenius = float(raw["metric_frobenius"])
        metric_finite = bool(raw["metric_finite"])
        shape = raw["metric_shape"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"ambient PGBO payload missing fields: {raw!r}") from exc
    if n_parameters < 1:
        raise ValueError("ambient PGBO result must have positive n_parameters")
    if not isinstance(labels_raw, list) or len(labels_raw) != n_parameters:
        raise ValueError("ambient parameter_labels must match n_parameters")
    if not metric_finite:
        raise ValueError("ambient metric_tensor must be finite")
    if not isinstance(shape, list) or len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("ambient metric_tensor must be square")
    if not math.isfinite(metric_determinant):
        raise ValueError("metric_determinant must be finite")
    if not math.isfinite(total_curvature) or total_curvature < 0.0:
        raise ValueError("total_curvature must be finite and non-negative")
    if not math.isfinite(frobenius) or frobenius < 0.0:
        raise ValueError("metric_frobenius must be finite and non-negative")
    labels = tuple(str(item) for item in labels_raw)
    digest = _digest_payload(
        {
            "schema": "pgbo_qgt_tensor_probe.v1",
            "capability_id": row.capability_id,
            "n_oscillators": n_oscillators,
            "n_parameters": n_parameters,
            "metric_determinant": metric_determinant,
            "total_curvature": total_curvature,
            "parameter_labels": list(labels),
            "metric_frobenius": frobenius,
            "epsilon": float(epsilon),
            "product_schema": PGBO_QGT_PRODUCT_SCHEMA,
        }
    )
    return MaterialisedPgboTensorProbe(
        capability_id=row.capability_id,
        n_oscillators=n_oscillators,
        n_parameters=n_parameters,
        metric_determinant=metric_determinant,
        total_curvature=total_curvature,
        parameter_labels=labels,
        metric_frobenius=frobenius,
        epsilon=float(epsilon),
        probe_digest=digest,
        invent_green_experimental_geometry=False,
        invent_green_live_qpu=False,
        demo_label=demo_label.strip() or "pgbo_tensor_demo",
    )


def materialise_demo_pgbo_tensor_probe() -> MaterialisedPgboTensorProbe:
    """Materialise the default offline two-oscillator PGBO QGT demo probe."""
    return materialise_pgbo_tensor_probe("pgbo_tensor")


def map_pgbo_qgt_public_surfaces() -> tuple[dict[str, object], ...]:
    """Map public surfaces composing the PGBO QGT product."""
    return (
        {
            "surface_id": "pgbo_qgt_product",
            "module_path": "scpn_quantum_control.pgbo_qgt_product",
            "role": "product_facade",
            "claim_boundary": PGBO_QGT_CLAIM_BOUNDARY,
        },
        {
            "surface_id": "pgbo_quantum_bridge",
            "module_path": "scpn_quantum_control.pgbo.quantum_bridge",
            "role": "ambient_qgt",
            "claim_boundary": PGBO_QGT_CLAIM_BOUNDARY,
        },
        {
            "surface_id": "geometric_control_product",
            "module_path": "scpn_quantum_control.geometric_control_product",
            "role": "geometric_control_compose",
            "claim_boundary": PGBO_QGT_CLAIM_BOUNDARY,
        },
    )


def build_pgbo_qgt_product_registry() -> dict[str, object]:
    """Build the versioned PGBO QGT product registry payload."""
    capabilities = [row.to_dict() for row in _CAPABILITIES]
    boundaries = [row.to_dict() for row in _BOUNDARIES]
    return {
        "schema": PGBO_QGT_PRODUCT_SCHEMA,
        "claim_boundary": PGBO_QGT_CLAIM_BOUNDARY,
        "capability_count": len(capabilities),
        "boundary_count": len(boundaries),
        "blank_entry_count": 0,
        "max_oscillators": MAX_OSCILLATORS,
        "default_epsilon": DEFAULT_EPSILON,
        "hardware_submit_allowed_policy": False,
        "experimental_geometry_claim_policy": False,
        "unbounded_system_size_policy": False,
        "fd_derivative_as_exact_policy": False,
        "public_surfaces": list(map_pgbo_qgt_public_surfaces()),
        "capabilities": capabilities,
        "boundaries": boundaries,
        "policy_note": _PGBO_QGT_POLICY_NOTE,
    }


def assert_pgbo_qgt_product_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert registry covers capabilities/boundaries without invent-green policies."""
    registry = dict(payload) if payload is not None else build_pgbo_qgt_product_registry()
    if registry.get("schema") != PGBO_QGT_PRODUCT_SCHEMA:
        raise ValueError("unexpected PGBO QGT product schema")
    if set(registry) != _PGBO_QGT_REGISTRY_KEYS:
        raise ValueError("PGBO QGT product registry keys drift")
    if registry.get("claim_boundary") != PGBO_QGT_CLAIM_BOUNDARY:
        raise ValueError("PGBO QGT product claim boundary drift")
    if registry.get("public_surfaces") != list(map_pgbo_qgt_public_surfaces()):
        raise ValueError("PGBO QGT product public surface map drift")
    if registry.get("policy_note") != _PGBO_QGT_POLICY_NOTE:
        raise ValueError("PGBO QGT product policy note drift")
    capabilities = registry.get("capabilities")
    boundaries = registry.get("boundaries")
    if not isinstance(capabilities, list) or not capabilities:
        raise ValueError("PGBO QGT product registry must contain a non-empty capabilities list")
    if not isinstance(boundaries, list) or not boundaries:
        raise ValueError("PGBO QGT product registry must contain a non-empty boundaries list")
    seen: set[str] = set()
    blank = 0
    tensor_found = False
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
        if cid == "pgbo_tensor":
            tensor_found = True
        if hw is not False:
            raise ValueError(f"capability {cid!r} hardware_submit_allowed must be False")
        if not symbol or not str(symbol).strip():
            raise ValueError(f"capability {cid!r} must have non-empty ambient_symbol")
        canonical = _CAPABILITY_BY_ID.get(cid)
        if canonical is not None and dict(row) != canonical.to_dict():
            raise ValueError(f"capability {cid!r} catalogue row drift")
    if blank:
        raise ValueError(f"PGBO QGT product registry has {blank} blank or invalid entries")
    if not tensor_found:
        raise ValueError("PGBO QGT product registry missing pgbo_tensor")
    expected = set(list_qgt_capability_ids())
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
        canonical_boundary = next(
            (candidate for candidate in _BOUNDARIES if candidate.boundary_id == bid),
            None,
        )
        if canonical_boundary is not None and dict(row) != canonical_boundary.to_dict():
            raise ValueError(f"boundary {bid!r} catalogue row drift")
    expected_b = set(list_qgt_boundary_ids())
    if seen_b != expected_b:
        raise ValueError(
            f"registry boundary set drift (missing={expected_b - seen_b!r}, "
            f"extra={seen_b - expected_b!r})"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    capability_count = registry.get("capability_count", -1)
    if not isinstance(capability_count, int) or capability_count != len(capabilities):
        raise ValueError("capability_count does not match capabilities list length")
    boundary_count = registry.get("boundary_count", -1)
    if not isinstance(boundary_count, int) or boundary_count != len(boundaries):
        raise ValueError("boundary_count does not match boundaries list length")
    max_n = registry.get("max_oscillators", -1)
    if not isinstance(max_n, int) or max_n != MAX_OSCILLATORS:
        raise ValueError(f"max_oscillators must equal {MAX_OSCILLATORS}")
    if registry.get("default_epsilon") != DEFAULT_EPSILON:
        raise ValueError(f"default_epsilon must equal {DEFAULT_EPSILON}")
    if registry.get("hardware_submit_allowed_policy", True) is not False:
        raise ValueError("hardware_submit_allowed_policy must be False")
    if registry.get("experimental_geometry_claim_policy", True) is not False:
        raise ValueError("experimental_geometry_claim_policy must be False")
    if registry.get("unbounded_system_size_policy", True) is not False:
        raise ValueError("unbounded_system_size_policy must be False")
    if registry.get("fd_derivative_as_exact_policy", True) is not False:
        raise ValueError("fd_derivative_as_exact_policy must be False")
    return registry


__all__ = [
    "BoundaryKind",
    "DEFAULT_EPSILON",
    "MAX_OSCILLATORS",
    "MaterialisedPgboTensorProbe",
    "PGBO_QGT_CLAIM_BOUNDARY",
    "PGBO_QGT_PRODUCT_SCHEMA",
    "PathDecisionOutcome",
    "PathEligibilityDecision",
    "QgtBoundaryRow",
    "QgtCapabilityKind",
    "QgtCapabilityRow",
    "SupportPosture",
    "assert_pgbo_qgt_product_integrity",
    "build_pgbo_qgt_product_registry",
    "decide_qgt_path",
    "get_qgt_boundary",
    "get_qgt_capability",
    "iter_qgt_boundaries",
    "iter_qgt_capabilities",
    "list_qgt_boundary_ids",
    "list_qgt_capability_ids",
    "map_pgbo_qgt_public_surfaces",
    "materialise_demo_pgbo_tensor_probe",
    "materialise_pgbo_tensor_probe",
]
