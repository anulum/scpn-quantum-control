# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Theory-hook promotion registry
"""Evidence-gated promotion records for experimental theory hooks.

This module is the theory-hook boundary between an importable research routine and a
promoted control or product capability.  The registry covers quantum speed
limits, Hamiltonian learning, the finite Koopman closure, the legacy
``quantum_phi`` mutual-information diagnostic, stabilizer Rényi entropy, and
spectral form-factor diagnostics.

The registry is deliberately conservative.  A passing local evidence probe
shows that a bounded software route works on its stated synthetic fixture; it
does not establish hardware validity, differentiability, critical scaling,
quantum advantage, consciousness, or operational control authority.  All
records are immutable and JSON-ready.  Evidence execution is local,
deterministic, credential-free, and never submits provider work.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

THEORY_HOOK_PROMOTION_SCHEMA = "scpn.theory-hook-promotion.v1"
THEORY_HOOK_PROMOTION_BOUNDARY = (
    "bounded local synthetic evidence only; not hardware validation, differentiable "
    "support, criticality certification, quantum advantage, consciousness evidence, "
    "clinical interpretation, or actuation authority"
)


class TheoryHookTier(str, Enum):
    """Evidence tier assigned to a theory hook.

    ``BOUNDED`` identifies a small, testable research diagnostic with an
    explicit claim boundary. ``RESEARCH_ONLY`` identifies a route that must not
    be promoted beyond exploratory analysis under the current semantics.
    """

    BOUNDED = "B"
    RESEARCH_ONLY = "D"


class TheoryHookRole(str, Enum):
    """Permitted role of a hook in the SCPN Quantum Control stack."""

    OPTIONAL_CONTROL_CONSTRAINT = "optional_control_constraint"
    SYNTHETIC_INVERSE_PROBLEM = "synthetic_inverse_problem"
    CLASSICAL_LOCAL_BASELINE = "classical_local_baseline"
    MUTUAL_INFORMATION_DIAGNOSTIC = "mutual_information_diagnostic"
    RESOURCE_THEORY_DIAGNOSTIC = "resource_theory_diagnostic"
    SPECTRAL_DIAGNOSTIC = "spectral_diagnostic"


class TheoryHookStatus(str, Enum):
    """Promotion state after applying the theory-hook evidence checklist."""

    BOUNDED_CANDIDATE = "bounded_candidate"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    RESEARCH_ONLY = "research_only"


@dataclass(frozen=True, slots=True)
class TheoryHookPromotionRecord:
    """Immutable promotion decision for one experimental theory hook.

    Parameters
    ----------
    hook_id
        Stable machine identifier used by evidence records.
    title
        Human-readable name of the hook.
    module
        Import path containing the bounded implementation.
    tier
        Evidence tier after theory-hook review.
    role
        Only role for which the current implementation may be used.
    status
        Current promotion state. None of the reviewed records is a production
        control capability.
    differentiable
        Whether a documented, tested derivative contract exists.  This is
        ``False`` for every current hook.
    evidence_fixture
        Exact local fixture exercised by :func:`run_theory_hook_evidence`.
    allowed_claims
        Narrow statements supported by the local software evidence.
    forbidden_claims
        Statements that remain prohibited even when the evidence probe passes.
    promotion_requirements
        Additional evidence required before a future status change.
    references
        Primary literature identifiers supporting the mathematical label.

    Notes
    -----
    The record is policy metadata, not scientific evidence by itself.  Pair it
    with a passing :class:`TheoryHookEvidenceRecord` from the same schema.

    """

    hook_id: str
    title: str
    module: str
    tier: TheoryHookTier
    role: TheoryHookRole
    status: TheoryHookStatus
    differentiable: bool
    evidence_fixture: str
    allowed_claims: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    promotion_requirements: tuple[str, ...]
    references: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject incomplete or internally inconsistent policy records."""
        for name, value in (
            ("hook_id", self.hook_id),
            ("title", self.title),
            ("module", self.module),
            ("evidence_fixture", self.evidence_fixture),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")
        for name, values in (
            ("allowed_claims", self.allowed_claims),
            ("forbidden_claims", self.forbidden_claims),
            ("promotion_requirements", self.promotion_requirements),
            ("references", self.references),
        ):
            if not values or any(not value.strip() for value in values):
                raise ValueError(f"{name} must contain non-empty strings")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        if self.differentiable:
            raise ValueError("theory hooks have no admitted differentiable contract")
        if (
            self.tier is TheoryHookTier.RESEARCH_ONLY
            and self.status is not TheoryHookStatus.RESEARCH_ONLY
        ):
            raise ValueError("tier-D hooks must remain research_only")

    @property
    def admitted_for_control(self) -> bool:
        """Always false; this registry admits no hook for actuation."""
        return False

    @property
    def admitted_for_publication_claim(self) -> bool:
        """Always false; local fixture evidence is not publication proof."""
        return False

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready record with explicit negative capabilities."""
        return {
            "hook_id": self.hook_id,
            "title": self.title,
            "module": self.module,
            "tier": self.tier.value,
            "role": self.role.value,
            "status": self.status.value,
            "differentiable": self.differentiable,
            "admitted_for_control": self.admitted_for_control,
            "admitted_for_publication_claim": self.admitted_for_publication_claim,
            "evidence_fixture": self.evidence_fixture,
            "allowed_claims": list(self.allowed_claims),
            "forbidden_claims": list(self.forbidden_claims),
            "promotion_requirements": list(self.promotion_requirements),
            "references": list(self.references),
        }


@dataclass(frozen=True, slots=True)
class TheoryHookEvidenceRecord:
    """Result of one bounded local theory-hook fixture.

    Parameters
    ----------
    hook_id
        Identifier of the corresponding promotion record.
    passed
        Whether every invariant in ``checks`` passed.
    fixture
        Human-readable fixture description.
    checks
        Named boolean invariants evaluated by the probe.
    metrics
        Small JSON-ready numerical or categorical observations.  These values
        describe the fixture only and are not extrapolation claims.

    """

    hook_id: str
    passed: bool
    fixture: str
    checks: tuple[tuple[str, bool], ...]
    metrics: tuple[tuple[str, bool | float | int | str], ...]

    def __post_init__(self) -> None:
        """Validate evidence identity, uniqueness, and aggregate status."""
        if not self.hook_id.strip() or not self.fixture.strip():
            raise ValueError("hook_id and fixture must be non-empty")
        if not self.checks:
            raise ValueError("checks must not be empty")
        check_names = [name for name, _value in self.checks]
        metric_names = [name for name, _value in self.metrics]
        if any(not name.strip() for name in (*check_names, *metric_names)):
            raise ValueError("check and metric names must be non-empty")
        if len(set(check_names)) != len(check_names):
            raise ValueError("check names must be unique")
        if len(set(metric_names)) != len(metric_names):
            raise ValueError("metric names must be unique")
        if any(isinstance(value, np.generic) for _name, value in self.metrics):
            raise ValueError("metrics must use JSON-native scalar values")
        if self.passed is not all(value for _name, value in self.checks):
            raise ValueError("passed must equal the conjunction of checks")

    def as_dict(self) -> dict[str, Any]:
        """Return the evidence record as deterministic JSON-ready data."""
        return {
            "hook_id": self.hook_id,
            "passed": self.passed,
            "fixture": self.fixture,
            "checks": {name: value for name, value in self.checks},
            "metrics": {name: value for name, value in self.metrics},
        }


@dataclass(frozen=True, slots=True)
class TheoryHookPromotionReport:
    """Complete theory-hook registry plus its local evidence results.

    Parameters
    ----------
    schema
        Versioned serialization schema.
    claim_boundary
        Global non-claim that applies to every record.
    records
        Promotion decisions in canonical order.
    evidence
        One local evidence result for each promotion decision.
    content_digest
        SHA-256 digest over the report payload excluding the digest itself.

    """

    schema: str
    claim_boundary: str
    records: tuple[TheoryHookPromotionRecord, ...]
    evidence: tuple[TheoryHookEvidenceRecord, ...]
    content_digest: str

    @property
    def passed(self) -> bool:
        """Whether all bounded fixture checks passed."""
        return all(item.passed for item in self.evidence)

    def as_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready report."""
        return {
            "schema": self.schema,
            "claim_boundary": self.claim_boundary,
            "passed": self.passed,
            "content_digest": self.content_digest,
            "records": [record.as_dict() for record in self.records],
            "evidence": [item.as_dict() for item in self.evidence],
        }


_COMMON_FORBIDDEN = (
    "hardware validation or provider performance",
    "differentiable or gradient support",
    "quantum advantage or computational supremacy",
    "operational actuation or safety authority",
)

_RECORDS = (
    TheoryHookPromotionRecord(
        hook_id="quantum_speed_limit",
        title="Quantum speed-limit threshold diagnostic",
        module="scpn_quantum_control.analysis.quantum_speed_limit",
        tier=TheoryHookTier.BOUNDED,
        role=TheoryHookRole.OPTIONAL_CONTROL_CONSTRAINT,
        status=TheoryHookStatus.BOUNDED_CANDIDATE,
        differentiable=False,
        evidence_fixture="two-qubit closed-system threshold evolution",
        allowed_claims=(
            "computes finite-system Mandelstam-Tamm and legacy Margolus-Levitin diagnostics",
            "may inform an offline optional lower-bound check after independent control validation",
        ),
        forbidden_claims=_COMMON_FORBIDDEN
        + ("measured synchronization time or critical exponent", "BKT certification"),
        promotion_requirements=(
            "define a control-objective contract that consumes the bound",
            "validate arbitrary-fidelity bound semantics and numerical tolerance",
            "add held-out controller comparisons before control admission",
        ),
        references=("doi:10.1103/PhysRevLett.103.160502",),
    ),
    TheoryHookPromotionRecord(
        hook_id="hamiltonian_learning",
        title="Ground-state correlator inverse problem",
        module="scpn_quantum_control.analysis.hamiltonian_learning",
        tier=TheoryHookTier.BOUNDED,
        role=TheoryHookRole.SYNTHETIC_INVERSE_PROBLEM,
        status=TheoryHookStatus.BOUNDED_CANDIDATE,
        differentiable=False,
        evidence_fixture="two-qubit exact correlators with the generating coupling as initial point",
        allowed_claims=(
            "solves a small non-negative symmetric coupling fit on exact synthetic correlators",
            "reports reconstruction loss and correlator residual for the fitted fixture",
        ),
        forbidden_claims=_COMMON_FORBIDDEN
        + (
            "identifiable recovery from arbitrary measurements",
            "noise-robust experimental learning",
        ),
        promotion_requirements=(
            "add identifiability and uncertainty analysis",
            "evaluate held-out noisy and misspecified synthetic systems",
            "validate measured-data calibration without reusing fit data",
        ),
        references=("doi:10.1103/PhysRevA.89.042314",),
    ),
    TheoryHookPromotionRecord(
        hook_id="koopman_local_closure",
        title="Finite local Koopman-style closure",
        module="scpn_quantum_control.analysis.koopman",
        tier=TheoryHookTier.BOUNDED,
        role=TheoryHookRole.CLASSICAL_LOCAL_BASELINE,
        status=TheoryHookStatus.DIAGNOSTIC_ONLY,
        differentiable=False,
        evidence_fixture="two-oscillator finite observable matrix and Hermitian projection",
        allowed_claims=(
            "constructs a finite local observable-space matrix for the documented basis",
            "provides a classical local spectral baseline and Hermitian projection",
        ),
        forbidden_claims=_COMMON_FORBIDDEN
        + (
            "exact finite Koopman invariant subspace",
            "full nonlinear dynamics",
            "BQP-completeness",
        ),
        promotion_requirements=(
            "quantify closure residual against nonlinear trajectories",
            "compare held-out reference points and observable dictionaries",
            "separate approximation error from eigensolver error",
        ),
        references=("doi:10.1007/s00332-015-9258-5",),
    ),
    TheoryHookPromotionRecord(
        hook_id="bipartite_mutual_information",
        title="Minimum bipartite quantum mutual information",
        module="scpn_quantum_control.analysis.quantum_phi",
        tier=TheoryHookTier.RESEARCH_ONLY,
        role=TheoryHookRole.MUTUAL_INFORMATION_DIAGNOSTIC,
        status=TheoryHookStatus.RESEARCH_ONLY,
        differentiable=False,
        evidence_fixture="two-qubit Bell-state mutual information",
        allowed_claims=(
            "computes bipartite von Neumann mutual information",
            "retains a legacy quantum_phi field for serialization compatibility",
        ),
        forbidden_claims=_COMMON_FORBIDDEN
        + (
            "Integrated Information Theory Phi",
            "consciousness, sentience, cognition, or clinical state",
            "minimum mutual information as causal irreducibility",
        ),
        promotion_requirements=(
            "implement an explicit causal model and intervention semantics",
            "validate the chosen IIT formulation independently",
            "rename or migrate all legacy phi fields before any promotion",
        ),
        references=("doi:10.1371/journal.pcbi.1003588",),
    ),
    TheoryHookPromotionRecord(
        hook_id="stabilizer_renyi_entropy",
        title="Stabilizer Rényi entropy resource diagnostic",
        module="scpn_quantum_control.analysis.magic_nonstabilizerness",
        tier=TheoryHookTier.BOUNDED,
        role=TheoryHookRole.RESOURCE_THEORY_DIAGNOSTIC,
        status=TheoryHookStatus.DIAGNOSTIC_ONLY,
        differentiable=False,
        evidence_fixture="single-qubit stabilizer and T-state contrast",
        allowed_claims=(
            "computes the documented pure-state stabilizer Rényi-2 diagnostic",
            "labels non-stabilizerness as a resource-theory observable",
        ),
        forbidden_claims=_COMMON_FORBIDDEN
        + ("critical-point estimator", "fault-tolerant resource cost", "classical hardness"),
        promotion_requirements=(
            "add finite-size preregistration before criticality studies",
            "add uncertainty-aware measurement protocol for non-exact inputs",
            "validate resource monotone conventions across supported states",
        ),
        references=("doi:10.1103/PhysRevLett.128.050402",),
    ),
    TheoryHookPromotionRecord(
        hook_id="spectral_form_factor",
        title="Finite-size spectral diagnostics",
        module="scpn_quantum_control.analysis.spectral_form_factor",
        tier=TheoryHookTier.BOUNDED,
        role=TheoryHookRole.SPECTRAL_DIAGNOSTIC,
        status=TheoryHookStatus.DIAGNOSTIC_ONLY,
        differentiable=False,
        evidence_fixture="four-qubit exact spectrum with magnetisation-sector spacing ratio",
        allowed_claims=(
            "computes normalized finite-spectrum form factors",
            "reports symmetry-resolved adjacent-gap-ratio diagnostics",
        ),
        forbidden_claims=_COMMON_FORBIDDEN
        + ("quantum-chaos certification", "Poisson-to-RMT transition", "BKT-chaos coincidence"),
        promotion_requirements=(
            "preregister ensemble, unfolding, symmetry, and finite-size protocol",
            "establish uncertainty and null distributions",
            "replicate scaling on held-out model families",
        ),
        references=("doi:10.1103/PhysRevLett.110.084101",),
    ),
)


def list_theory_hook_promotions() -> tuple[TheoryHookPromotionRecord, ...]:
    """Return all theory-hook promotion decisions in stable canonical order.

    Returns
    -------
    tuple[TheoryHookPromotionRecord, ...]
        Immutable registry containing exactly one record per reviewed hook.

    """
    return _RECORDS


def get_theory_hook_promotion(hook_id: str) -> TheoryHookPromotionRecord:
    """Look up one promotion decision by stable identifier.

    Parameters
    ----------
    hook_id
        Identifier from :func:`list_theory_hook_promotions`.

    Returns
    -------
    TheoryHookPromotionRecord
        Matching immutable policy record.

    Raises
    ------
    KeyError
        If ``hook_id`` is not registered.

    """
    for record in _RECORDS:
        if record.hook_id == hook_id:
            return record
    raise KeyError(f"unknown theory hook: {hook_id!r}")


def _round(value: float) -> float:
    return round(float(value), 12)


def _qsl_evidence() -> TheoryHookEvidenceRecord:
    from .quantum_speed_limit import compute_qsl

    coupling = np.array([[0.0, 0.25], [0.25, 0.0]], dtype=np.float64)
    frequencies = np.array([0.8, 1.1], dtype=np.float64)
    result = compute_qsl(coupling, frequencies, t_target=0.2, dt=0.05, R_threshold=0.5)
    checks = (
        ("finite_bounds", bool(np.isfinite(result.tau_MT) and np.isfinite(result.tau_ML))),
        ("mt_not_above_simulated_time", bool(result.tau_MT <= result.tau_actual + 0.05)),
        ("two_qubit_fixture", result.n_qubits == 2),
    )
    return TheoryHookEvidenceRecord(
        hook_id="quantum_speed_limit",
        passed=all(value for _name, value in checks),
        fixture="two-qubit closed-system threshold evolution",
        checks=checks,
        metrics=(
            ("tau_mt", _round(result.tau_MT)),
            ("tau_ml_legacy", _round(result.tau_ML)),
            ("tau_actual", _round(result.tau_actual)),
        ),
    )


def _hamiltonian_learning_evidence() -> TheoryHookEvidenceRecord:
    from .hamiltonian_learning import learn_hamiltonian, measure_correlators

    coupling = np.array([[0.0, 0.35], [0.35, 0.0]], dtype=np.float64)
    frequencies = np.array([0.8, 1.1], dtype=np.float64)
    correlators = measure_correlators(coupling, frequencies)
    result = learn_hamiltonian(correlators, frequencies, K_init=coupling, maxiter=5)
    checks = (
        ("symmetric_fit", bool(np.allclose(result.K_learned, result.K_learned.T))),
        ("non_negative_fit", bool(np.all(result.K_learned >= 0.0))),
        ("correlator_error_below_fixture_gate", bool(result.correlator_error < 0.1)),
    )
    return TheoryHookEvidenceRecord(
        hook_id="hamiltonian_learning",
        passed=all(value for _name, value in checks),
        fixture="two-qubit exact correlators initialized at the generating coupling",
        checks=checks,
        metrics=(
            ("correlator_error", _round(result.correlator_error)),
            ("loss", _round(result.loss)),
            ("optimizer_evaluations", int(result.n_iterations)),
        ),
    )


def _koopman_evidence() -> TheoryHookEvidenceRecord:
    from .koopman import build_koopman_generator, koopman_to_hamiltonian

    coupling = np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64)
    frequencies = np.array([0.8, 1.1], dtype=np.float64)
    generator, labels = build_koopman_generator(coupling, frequencies)
    hamiltonian = koopman_to_hamiltonian(generator)
    checks = (
        ("finite_four_by_four_closure", generator.shape == (4, 4) and len(labels) == 4),
        ("finite_entries", bool(np.all(np.isfinite(generator)))),
        ("hermitian_projection", bool(np.allclose(hamiltonian, hamiltonian.conj().T))),
    )
    return TheoryHookEvidenceRecord(
        hook_id="koopman_local_closure",
        passed=all(value for _name, value in checks),
        fixture="two-oscillator finite observable closure at the zero-phase reference",
        checks=checks,
        metrics=(("observable_dimension", generator.shape[0]), ("reference_phase", "zero")),
    )


def _mutual_information_evidence() -> TheoryHookEvidenceRecord:
    from .quantum_phi import mutual_information

    bell = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    rho = np.outer(bell, bell.conj())
    qmi = mutual_information(rho, [0], [1], 2)
    policy = get_theory_hook_promotion("bipartite_mutual_information")
    checks = (
        ("bell_qmi_is_two_bits", bool(np.isclose(qmi, 2.0, atol=1e-12))),
        ("tier_d_research_only", policy.tier is TheoryHookTier.RESEARCH_ONLY),
        ("not_admitted_for_control", not policy.admitted_for_control),
    )
    return TheoryHookEvidenceRecord(
        hook_id="bipartite_mutual_information",
        passed=all(value for _name, value in checks),
        fixture="two-qubit Bell-state bipartite mutual information",
        checks=checks,
        metrics=(("mutual_information_bits", _round(qmi)), ("iit_phi", "not_computed")),
    )


def _magic_evidence() -> TheoryHookEvidenceRecord:
    from .magic_nonstabilizerness import _compute_sre_m2

    stabilizer = np.array([1.0, 0.0], dtype=np.complex128)
    t_state = np.array([1.0, np.exp(1j * np.pi / 4.0)], dtype=np.complex128) / np.sqrt(2.0)
    stabilizer_sre, _ = _compute_sre_m2(stabilizer, 1)
    t_sre, _ = _compute_sre_m2(t_state, 1)
    checks = (
        ("stabilizer_zero", bool(np.isclose(stabilizer_sre, 0.0, atol=1e-12))),
        ("t_state_positive", bool(t_sre > 0.0)),
        ("finite_results", bool(np.isfinite(stabilizer_sre) and np.isfinite(t_sre))),
    )
    return TheoryHookEvidenceRecord(
        hook_id="stabilizer_renyi_entropy",
        passed=all(value for _name, value in checks),
        fixture="single-qubit computational stabilizer and T state",
        checks=checks,
        metrics=(
            ("stabilizer_sre_m2", _round(stabilizer_sre)),
            ("t_state_sre_m2", _round(t_sre)),
        ),
    )


def _spectral_evidence() -> TheoryHookEvidenceRecord:
    from .spectral_form_factor import compute_sff

    coupling = np.zeros((4, 4), dtype=np.float64)
    for index in range(4):
        neighbour = (index + 1) % 4
        coupling[index, neighbour] = coupling[neighbour, index] = 0.5
    frequencies = np.array([0.8, 1.1, 0.9, 1.2], dtype=np.float64)
    result = compute_sff(coupling, frequencies, t_max=0.5, n_times=4)
    checks = (
        ("sff_normalized_at_zero", bool(np.isclose(result.sff[0], 1.0, atol=1e-12))),
        ("sff_bounded", bool(np.all((result.sff >= 0.0) & (result.sff <= 1.0 + 1e-12)))),
        ("magnetisation_sector_default", result.level_spacing_basis == "magnetisation"),
    )
    return TheoryHookEvidenceRecord(
        hook_id="spectral_form_factor",
        passed=all(value for _name, value in checks),
        fixture="four-qubit exact spectrum with magnetisation-sector level statistics",
        checks=checks,
        metrics=(
            ("time_points", len(result.times)),
            ("sector_dimension", result.level_spacing_sector_dim),
            ("sff_at_zero", _round(result.sff[0])),
        ),
    )


def run_theory_hook_evidence() -> tuple[TheoryHookEvidenceRecord, ...]:
    """Execute every theory-hook local fixture in canonical registry order.

    Returns
    -------
    tuple[TheoryHookEvidenceRecord, ...]
        One immutable evidence result for each promotion record.

    Notes
    -----
    The fixtures use exact local simulators and tiny deterministic arrays.  The
    function does not read credentials, connect to a provider, submit hardware
    work, or grant control/publication authority.

    """
    evidence = (
        _qsl_evidence(),
        _hamiltonian_learning_evidence(),
        _koopman_evidence(),
        _mutual_information_evidence(),
        _magic_evidence(),
        _spectral_evidence(),
    )
    if tuple(item.hook_id for item in evidence) != tuple(record.hook_id for record in _RECORDS):
        raise RuntimeError("theory-hook evidence order does not match the promotion registry")
    return evidence


def build_theory_hook_promotion_report() -> TheoryHookPromotionReport:
    """Build the digest-locked theory-hook promotion and evidence report.

    Returns
    -------
    TheoryHookPromotionReport
        Complete registry, freshly executed local evidence, and a SHA-256
        content digest over the report payload.

    """
    evidence = run_theory_hook_evidence()
    payload = {
        "schema": THEORY_HOOK_PROMOTION_SCHEMA,
        "claim_boundary": THEORY_HOOK_PROMOTION_BOUNDARY,
        "records": [record.as_dict() for record in _RECORDS],
        "evidence": [item.as_dict() for item in evidence],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return TheoryHookPromotionReport(
        schema=THEORY_HOOK_PROMOTION_SCHEMA,
        claim_boundary=THEORY_HOOK_PROMOTION_BOUNDARY,
        records=_RECORDS,
        evidence=evidence,
        content_digest=digest,
    )


def render_theory_hook_promotion_markdown(report: TheoryHookPromotionReport) -> str:
    """Render a concise Markdown custody record for a promotion report.

    Parameters
    ----------
    report
        Report returned by :func:`build_theory_hook_promotion_report`.

    Returns
    -------
    str
        Deterministic Markdown ending in a newline.

    """
    spdx_header = "<!-- SPDX-License-" + "Identifier: AGPL-3.0-or-later -->"
    lines = [
        spdx_header,
        "",
        "# Theory-Hook Promotion Evidence",
        "",
        f"Schema: `{report.schema}`",
        "",
        f"Content digest: `{report.content_digest}`",
        "",
        f"All local fixtures passed: **{str(report.passed).lower()}**",
        "",
        f"> Boundary: {report.claim_boundary}.",
        "",
        "| Hook | Tier | Status | Permitted role | Fixture | Passed |",
        "|---|---:|---|---|---|---:|",
    ]
    evidence_by_id = {item.hook_id: item for item in report.evidence}
    for record in report.records:
        item = evidence_by_id[record.hook_id]
        lines.append(
            f"| `{record.hook_id}` | {record.tier.value} | `{record.status.value}` | "
            f"`{record.role.value}` | {item.fixture} | {str(item.passed).lower()} |"
        )
    lines.extend(
        [
            "",
            "A passing row proves only that its committed small local fixture satisfies the named "
            "software invariants. See `docs/theory_hook_promotion.md` for API semantics, "
            "forbidden claims, and future promotion gates.",
            "",
        ]
    )
    return "\n".join(lines)
