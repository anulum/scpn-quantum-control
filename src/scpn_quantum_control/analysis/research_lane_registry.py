# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Deep-analysis research-lane registry
"""Governed catalogue of the package's analysis and gauge research lanes.

The research-lane registry makes the existing deep-analysis stack visible
without promoting every importable module into a product or scientific claim.
Each immutable row records the module's human-reviewed maturity, relevance to
differentiable work, current claim status, optional promotion route, and
evidence pointers.

The inventory gate is intentionally strict: every ordinary ``analysis`` or
``gauge`` module must have exactly one row.  Package ``__init__`` modules and
this registry's own governance implementation are the only exclusions.  A new
module therefore fails :func:`assert_research_lane_inventory` until a reviewer
classifies it explicitly.

Registry membership is catalogue evidence only.  It grants no productisation,
control, hardware, differentiability, advantage, criticality, topology,
consciousness, clinical, or publication claim.  Promotions remain governed by
their own backlog and evidence packages.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

RESEARCH_LANE_REGISTRY_SCHEMA = "scpn.research-lane-registry.v1"
RESEARCH_LANE_REGISTRY_BOUNDARY = (
    "catalogue metadata only; registry membership does not grant productisation, "
    "differentiability, control, hardware, advantage, criticality, topology, "
    "consciousness, clinical, or publication claims"
)
_REGISTRY_MODULE = "scpn_quantum_control.analysis.research_lane_registry"
_PACKAGE_PREFIX = "scpn_quantum_control."


class ResearchLaneMaturity(str, Enum):
    """Human-reviewed implementation maturity.

    ``RESEARCH`` marks exploratory scientific code. ``PROTOTYPE`` marks a
    bounded reusable diagnostic whose public contract or evidence is not yet a
    product gate. ``PRODUCT_CANDIDATE`` marks a stable candidate or a module
    already composed by a separately governed product; it is not a promotion
    by this registry.
    """

    RESEARCH = "research"
    PROTOTYPE = "prototype"
    PRODUCT_CANDIDATE = "product_candidate"


class ResearchLaneDiffHook(str, Enum):
    """Relationship between a lane and differentiable-control work."""

    NONE = "none"
    DIAGNOSTIC = "diagnostic_only"
    CANDIDATE = "candidate_requires_evidence"
    BOUNDED_COMPOSITION = "bounded_composition"
    DEFERRED = "deferred_owner_gate"


class ResearchLaneClaimStatus(str, Enum):
    """Strongest claim class currently carried by a lane's own evidence."""

    RESEARCH_ONLY = "research_only"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    EVIDENCE_BOUNDED = "evidence_bounded"
    REFUSE_ONLY = "refuse_only"


@dataclass(frozen=True, slots=True)
class ResearchLaneRecord:
    """Immutable human classification for one importable research module.

    Parameters
    ----------
    module
        Fully qualified module path under ``scpn_quantum_control.analysis`` or
        ``scpn_quantum_control.gauge``.
    summary
        Narrow description of what the current implementation can provide.
        The summary is not a scientific validation claim.
    maturity
        Human-reviewed implementation maturity.
    diff_hook
        Relationship to separately governed differentiable-control work.
    claim_status
        Strongest claim class admitted by the module's current evidence.
    promotion_targets
        Backlog routes that may consume the lane. A route suffixed ``planned``
        or ``deferred-owner-gate`` is explicitly not a completed promotion.
    evidence_refs
        Repository-relative evidence or governance pointers. Empty tuples are
        expected for research-only and diagnostic-only lanes.

    Notes
    -----
    ``PRODUCT_CANDIDATE`` and ``EVIDENCE_BOUNDED`` remain non-promotional here.
    Callers must consult the referenced product/evidence package before making
    any stronger claim.

    """

    module: str
    summary: str
    maturity: ResearchLaneMaturity
    diff_hook: ResearchLaneDiffHook
    claim_status: ResearchLaneClaimStatus
    promotion_targets: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject blank, out-of-scope, duplicate, or permissive records."""
        if not self.module.startswith(
            ("scpn_quantum_control.analysis.", "scpn_quantum_control.gauge.")
        ):
            raise ValueError("module must be under the analysis or gauge package")
        if self.module == _REGISTRY_MODULE:
            raise ValueError("the governance registry must not catalogue itself")
        if not self.summary.strip():
            raise ValueError("summary must be non-empty")
        for name, values in (
            ("promotion_targets", self.promotion_targets),
            ("evidence_refs", self.evidence_refs),
        ):
            if any(not value.strip() for value in values):
                raise ValueError(f"{name} must not contain blank values")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
        if (
            self.claim_status is ResearchLaneClaimStatus.EVIDENCE_BOUNDED
            and not self.evidence_refs
        ):
            raise ValueError("evidence_bounded lanes require evidence_refs")
        if (
            self.diff_hook
            in {
                ResearchLaneDiffHook.BOUNDED_COMPOSITION,
                ResearchLaneDiffHook.DEFERRED,
            }
            and not self.promotion_targets
        ):
            raise ValueError("composed or deferred diff hooks require a promotion target")
        if (
            self.claim_status is ResearchLaneClaimStatus.REFUSE_ONLY
            and self.maturity is ResearchLaneMaturity.RESEARCH
        ):
            raise ValueError("refuse_only lanes must be explicit non-research interfaces")

    @property
    def family(self) -> str:
        """Module family, either ``analysis`` or ``gauge``."""
        return self.module.split(".", 2)[1]

    @property
    def registry_grants_productisation(self) -> bool:
        """Always false because catalogue membership is non-promotional."""
        return False

    @property
    def registry_grants_control(self) -> bool:
        """Always false; catalogue membership grants no actuation authority."""
        return False

    @property
    def registry_grants_publication_claim(self) -> bool:
        """Always false; catalogue membership is not publication evidence."""
        return False

    def as_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready row with explicit denials."""
        return {
            "module": self.module,
            "family": self.family,
            "summary": self.summary,
            "maturity": self.maturity.value,
            "diff_hook": self.diff_hook.value,
            "claim_status": self.claim_status.value,
            "promotion_targets": list(self.promotion_targets),
            "evidence_refs": list(self.evidence_refs),
            "registry_grants_productisation": self.registry_grants_productisation,
            "registry_grants_control": self.registry_grants_control,
            "registry_grants_publication_claim": (self.registry_grants_publication_claim),
        }


@dataclass(frozen=True, slots=True)
class ResearchLaneInventoryReport:
    """Comparison between registered rows and modules found on disk.

    ``missing_modules`` are importable modules without a human classification.
    ``orphaned_records`` are registry rows whose implementation no longer
    exists. Both conditions fail the gate.
    """

    registered_modules: tuple[str, ...]
    discovered_modules: tuple[str, ...]
    missing_modules: tuple[str, ...]
    orphaned_records: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Whether discovery and the immutable registry match exactly."""
        return not self.missing_modules and not self.orphaned_records

    def as_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-ready inventory evidence."""
        return {
            "passed": self.passed,
            "registered_count": len(self.registered_modules),
            "discovered_count": len(self.discovered_modules),
            "registered_modules": list(self.registered_modules),
            "discovered_modules": list(self.discovered_modules),
            "missing_modules": list(self.missing_modules),
            "orphaned_records": list(self.orphaned_records),
        }


@dataclass(frozen=True, slots=True)
class ResearchLaneRegistryReport:
    """Complete research-lane catalogue, inventory gate, counts, and digest."""

    schema: str
    claim_boundary: str
    records: tuple[ResearchLaneRecord, ...]
    inventory: ResearchLaneInventoryReport
    content_digest: str

    def as_dict(self) -> dict[str, Any]:
        """Return the complete deterministic report as JSON-ready data."""
        maturity_counts = _enum_counts(self.records, "maturity")
        diff_hook_counts = _enum_counts(self.records, "diff_hook")
        claim_status_counts = _enum_counts(self.records, "claim_status")
        return {
            "schema": self.schema,
            "claim_boundary": self.claim_boundary,
            "record_count": len(self.records),
            "maturity_counts": maturity_counts,
            "diff_hook_counts": diff_hook_counts,
            "claim_status_counts": claim_status_counts,
            "inventory": self.inventory.as_dict(),
            "records": [record.as_dict() for record in self.records],
            "content_digest": self.content_digest,
        }


def _lane(
    module: str,
    summary: str,
    maturity: ResearchLaneMaturity = ResearchLaneMaturity.RESEARCH,
    diff_hook: ResearchLaneDiffHook = ResearchLaneDiffHook.NONE,
    claim_status: ResearchLaneClaimStatus = ResearchLaneClaimStatus.RESEARCH_ONLY,
    *,
    promotion_targets: tuple[str, ...] = (),
    evidence_refs: tuple[str, ...] = (),
) -> ResearchLaneRecord:
    """Build one explicit registry row from a package-relative module path."""
    return ResearchLaneRecord(
        module=f"{_PACKAGE_PREFIX}{module}",
        summary=summary,
        maturity=maturity,
        diff_hook=diff_hook,
        claim_status=claim_status,
        promotion_targets=promotion_targets,
        evidence_refs=evidence_refs,
    )


_R = ResearchLaneMaturity.RESEARCH
_P = ResearchLaneMaturity.PROTOTYPE
_C = ResearchLaneMaturity.PRODUCT_CANDIDATE
_N = ResearchLaneDiffHook.NONE
_D = ResearchLaneDiffHook.DIAGNOSTIC
_K = ResearchLaneDiffHook.CANDIDATE
_B = ResearchLaneDiffHook.BOUNDED_COMPOSITION
_X = ResearchLaneDiffHook.DEFERRED
_RO = ResearchLaneClaimStatus.RESEARCH_ONLY
_DO = ResearchLaneClaimStatus.DIAGNOSTIC_ONLY
_EB = ResearchLaneClaimStatus.EVIDENCE_BOUNDED
_FO = ResearchLaneClaimStatus.REFUSE_ONLY

_RESEARCH_LANES = tuple(
    sorted(
        (
            _lane(
                "analysis.adaptive_fim_evidence",
                "Digest-bound offline calibration and historical replay custody "
                "for adaptive FIM feedback.",
                _C,
                _B,
                _EB,
                promotion_targets=("adaptive-fim:complete",),
                evidence_refs=("data/adaptive_fim_product/adaptive_fim_evidence.json",),
            ),
            _lane(
                "analysis.adaptive_fim_feedback",
                "Count-aware, decrease-only next-experiment proposals with hardware refusal.",
                _C,
                _B,
                _EB,
                promotion_targets=("adaptive-fim:complete",),
                evidence_refs=("docs/adaptive_fim_feedback.md",),
            ),
            _lane("analysis.berry_phase", "Finite-size Berry-phase scan diagnostic.", _R, _D, _RO),
            _lane(
                "analysis.bkt_analysis", "Finite-graph BKT proxy and transition scan.", _R, _D, _RO
            ),
            _lane("analysis.bkt_universals", "Candidate BKT-universal ratio check.", _R, _N, _RO),
            _lane(
                "analysis.critical_concordance",
                "Finite-size concordance between critical probes.",
                _R,
                _D,
                _RO,
            ),
            _lane(
                "analysis.dla_parity_exact_baseline",
                "Exact noiseless statevector baseline for DLA parity circuits.",
                _C,
                _B,
                _EB,
                promotion_targets=("topology-control:complete",),
                evidence_refs=("docs/dla_topology_constrained_control.md",),
            ),
            _lane(
                "analysis.dla_parity_theorem",
                "Exact parity-sector dimension theorem and verification helpers.",
                _C,
                _B,
                _EB,
                promotion_targets=("topology-control:complete",),
                evidence_refs=("data/dla_topology_control/evidence.json",),
            ),
            _lane(
                "analysis.dla_parity_witness",
                "Bitstring-count parity witness consumed by the bounded DLA lane.",
                _P,
                _B,
                _DO,
                promotion_targets=("topology-control:complete",),
            ),
            _lane(
                "analysis.dla_truncated_tn",
                "Fail-fast placeholder for an unavailable DLA tensor-network route.",
                _P,
                _N,
                _FO,
            ),
            _lane(
                "analysis.dynamical_lie_algebra",
                "Finite-system DLA closure and generator diagnostics.",
                _P,
                _D,
                _DO,
                promotion_targets=("topology-control:complete",),
            ),
            _lane(
                "analysis.enaqt",
                "Bounded finite-horizon environment-assisted transport scan.",
                _C,
                _D,
                _EB,
                evidence_refs=("data/enaqt_product/enaqt_evidence.json",),
            ),
            _lane(
                "analysis.enaqt_evidence",
                "Digest-bound ENAQT fixture and negative-control custody.",
                _C,
                _N,
                _EB,
                evidence_refs=("data/enaqt_product/enaqt_evidence.json",),
            ),
            _lane(
                "analysis.entanglement_enhanced_sync",
                "Bounded initial-state coherence comparison under finite evolution.",
                _C,
                _D,
                _EB,
                evidence_refs=("data/entanglement_sync_product/entanglement_sync_evidence.json",),
            ),
            _lane(
                "analysis.entanglement_entropy",
                "Finite-size entropy and Schmidt-gap diagnostic.",
                _R,
                _D,
                _RO,
            ),
            _lane(
                "analysis.entanglement_percolation",
                "Finite-size concurrence/percolation diagnostic.",
                _R,
                _D,
                _RO,
            ),
            _lane(
                "analysis.entanglement_spectrum",
                "Finite-size entanglement-spectrum and CFT fit diagnostic.",
                _R,
                _D,
                _RO,
            ),
            _lane(
                "analysis.entanglement_sync_evidence",
                "Digest-bound controls for the initial-state coherence comparison.",
                _C,
                _N,
                _EB,
                evidence_refs=("data/entanglement_sync_product/entanglement_sync_evidence.json",),
            ),
            _lane(
                "analysis.fim_hamiltonian",
                "Offline FIM-Hamiltonian spectrum and gap diagnostics.",
                _P,
                _K,
                _DO,
                promotion_targets=("geometric-control:planned",),
            ),
            _lane(
                "analysis.finite_size_scaling",
                "Small-system critical-coupling scaling fit.",
                _P,
                _D,
                _DO,
            ),
            _lane(
                "analysis.graph_topology_scan",
                "Graph-topology scan for the open p_h1 target.",
                _R,
                _N,
                _RO,
            ),
            _lane(
                "analysis.h1_persistence",
                "H1-persistence check at a candidate BKT transition.",
                _R,
                _N,
                _RO,
            ),
            _lane(
                "analysis.hamiltonian_learning",
                "Synthetic exact-correlator inverse fit for coupling recovery.",
                _P,
                _D,
                _EB,
                evidence_refs=("data/theory_hook_promotion/evidence.json",),
            ),
            _lane(
                "analysis.hamiltonian_self_consistency",
                "Synthetic correlator-to-coupling self-consistency loop.",
                _P,
                _D,
                _DO,
            ),
            _lane(
                "analysis.integrated_information_phi",
                "Fail-closed IIT request boundary with an explicit QMI proxy opt-in.",
                _C,
                _N,
                _FO,
            ),
            _lane(
                "analysis.koopman",
                "Finite local Koopman-style closure and classical baseline.",
                _P,
                _D,
                _EB,
                evidence_refs=("data/theory_hook_promotion/evidence.json",),
            ),
            _lane(
                "analysis.krylov_complexity",
                "Finite-system Krylov-complexity diagnostic.",
                _R,
                _D,
                _RO,
            ),
            _lane(
                "analysis.lindblad_ness",
                "Finite driven-dissipative steady-state scan.",
                _R,
                _D,
                _RO,
            ),
            _lane(
                "analysis.logical_sync_witness",
                "DLA-sector logical synchronisation witness.",
                _P,
                _D,
                _DO,
            ),
            _lane(
                "analysis.loschmidt_echo", "Finite-quench Loschmidt-echo diagnostic.", _R, _D, _RO
            ),
            _lane(
                "analysis.magic_nonstabilizerness",
                "Exact small-system stabilizer-Renyi diagnostic.",
                _P,
                _D,
                _EB,
                evidence_refs=("data/theory_hook_promotion/evidence.json",),
            ),
            _lane(
                "analysis.magnetisation_sectors",
                "U(1)-sector exact-diagonalisation utilities.",
                _P,
                _D,
                _DO,
            ),
            _lane(
                "analysis.monte_carlo_xy",
                "Classical XY Monte Carlo and finite-size fit baseline.",
                _P,
                _N,
                _DO,
            ),
            _lane(
                "analysis.otoc",
                "Finite-size OTOC diagnostic; not a chaos certificate.",
                _R,
                _D,
                _RO,
            ),
            _lane(
                "analysis.otoc_sync_probe",
                "Comparison of OTOC and synchronisation proxies.",
                _R,
                _D,
                _RO,
            ),
            _lane(
                "analysis.p_h1_derivation",
                "Audit record for the failed p_h1 derivation.",
                _R,
                _N,
                _RO,
            ),
            _lane(
                "analysis.p_h1_open_guard",
                "Machine guard that blocks unsupported public p_h1 closure claims.",
                _C,
                _N,
                _EB,
                evidence_refs=("tests/test_p_h1_open_guard.py",),
            ),
            _lane(
                "analysis.pairing_correlator",
                "Finite-system pairing-correlator diagnostic.",
                _R,
                _D,
                _RO,
            ),
            _lane(
                "analysis.persistent_homology",
                "Classical phase-configuration persistence diagnostic.",
                _P,
                _D,
                _DO,
            ),
            _lane("analysis.phase_diagram", "Finite-graph phase-boundary estimate.", _R, _D, _RO),
            _lane(
                "analysis.qfi",
                "Finite-difference QFI and gap trade-off diagnostic.",
                _P,
                _K,
                _DO,
                promotion_targets=("geometric-control:planned",),
            ),
            _lane(
                "analysis.qfi_criticality",
                "Finite-size QFI criticality probe.",
                _P,
                _K,
                _DO,
                promotion_targets=("geometric-control:planned",),
            ),
            _lane(
                "analysis.qfi_geometric_crosscheck",
                "Spectral-QFI versus geometric-tensor cross-check.",
                _P,
                _K,
                _DO,
                promotion_targets=("geometric-control:planned",),
            ),
            _lane(
                "analysis.qrc_phase_detector",
                "Finite-feature linear QRC-style phase detector.",
                _R,
                _D,
                _RO,
            ),
            _lane(
                "analysis.quantum_fisher_information",
                "Observable wrapper for bounded QFI estimation.",
                _P,
                _K,
                _DO,
                promotion_targets=("geometric-control:planned",),
            ),
            _lane(
                "analysis.quantum_mpemba",
                "Finite-system relaxation-ordering experiment.",
                _R,
                _D,
                _RO,
            ),
            _lane(
                "analysis.quantum_persistent_homology",
                "Persistence diagnostics over finite measurement correlations.",
                _R,
                _D,
                _RO,
            ),
            _lane(
                "analysis.quantum_phi",
                "Minimum bipartite quantum mutual information; not IIT phi.",
                _R,
                _N,
                _RO,
                evidence_refs=("data/theory_hook_promotion/evidence.json",),
            ),
            _lane(
                "analysis.quantum_speed_limit",
                "Bounded local-phase-threshold timing and legacy orthogonalisation reference.",
                _P,
                _D,
                _EB,
                evidence_refs=("data/theory_hook_promotion/evidence.json",),
            ),
            _lane(
                "analysis.rl_discovery_agent",
                "Compatibility wrapper for research witness discovery.",
                _P,
                _N,
                _RO,
            ),
            _lane(
                "analysis.rl_research_governance",
                "Fail-closed preregistration, seed, budget, and no-control policy for RL-adjacent research.",
                _C,
                _N,
                _EB,
                evidence_refs=("data/rl_research_governance/evidence.json",),
            ),
            _lane(
                "analysis.rl_pulse_optimizer",
                "Fail-fast placeholder for unavailable RL pulse optimisation.",
                _P,
                _N,
                _FO,
            ),
            _lane(
                "analysis.sensing",
                "No-submit synchronisation-order sensing readiness model.",
                _C,
                _D,
                _EB,
                evidence_refs=(
                    "data/s11_quantum_sensing/quantum_sensing_readiness_2026-05-20.json",
                ),
            ),
            _lane(
                "analysis.shadow_tomography",
                "Finite classical-shadow expectation estimator.",
                _P,
                _D,
                _DO,
            ),
            _lane(
                "analysis.spectral_form_factor",
                "Finite spectral-form-factor and gap-ratio diagnostic; not a chaos certificate.",
                _P,
                _D,
                _EB,
                evidence_refs=("data/theory_hook_promotion/evidence.json",),
            ),
            _lane(
                "analysis.symmetry_sectors",
                "Parity-sector exact-diagonalisation utilities.",
                _P,
                _D,
                _DO,
            ),
            _lane(
                "analysis.sync_entanglement_witness",
                "Order-parameter entanglement-witness diagnostic.",
                _P,
                _D,
                _DO,
            ),
            _lane(
                "analysis.sync_order_parameter",
                "Z-basis synchronisation proxy over counts.",
                _P,
                _D,
                _DO,
            ),
            _lane(
                "analysis.sync_uncertainty",
                "Shot-noise and bootstrap uncertainty utilities.",
                _C,
                _D,
                _DO,
            ),
            _lane(
                "analysis.sync_witness",
                "Finite-count synchronisation witness operators.",
                _C,
                _D,
                _DO,
            ),
            _lane(
                "analysis.tcbo_weighted_complex",
                "Coupling-weighted simplicial-complex diagnostic without topology certification.",
                _P,
                _X,
                _DO,
                promotion_targets=("coherence-observer:deferred-owner-gate",),
            ),
            _lane(
                "analysis.theory_hook_promotion",
                "Fail-closed theory-hook promotion decisions and local fixture evidence.",
                _C,
                _N,
                _EB,
                evidence_refs=("data/theory_hook_promotion/evidence.json",),
            ),
            _lane(
                "analysis.thermodynamic_witness",
                "Calibrated-work thermodynamic witness interface.",
                _P,
                _D,
                _DO,
            ),
            _lane(
                "analysis.translation_symmetry",
                "Homogeneous-chain momentum-sector utilities.",
                _P,
                _D,
                _DO,
            ),
            _lane(
                "analysis.two_colour_schedule",
                "Exact width-two scheduling audit for a 1-D XY chain.",
                _C,
                _N,
                _DO,
            ),
            _lane(
                "analysis.vortex_binding",
                "Finite vortex-pair and Kosterlitz-RG diagnostic.",
                _R,
                _D,
                _RO,
            ),
            _lane(
                "analysis.witness_discovery",
                "Bounded Bayesian/bandit search over witness candidates.",
                _P,
                _D,
                _DO,
            ),
            _lane(
                "analysis.xxz_phase_diagram",
                "Finite-size XXZ anisotropy crossover diagnostic.",
                _R,
                _D,
                _RO,
            ),
            _lane(
                "gauge.cft_analysis", "Finite-size CFT central-charge fit diagnostic.", _R, _D, _RO
            ),
            _lane(
                "gauge.confinement",
                "Finite U(1) confinement/deconfinement diagnostic.",
                _R,
                _D,
                _RO,
            ),
            _lane(
                "gauge.lattice_crosscheck",
                "Quantum/classical lattice confinement cross-check.",
                _R,
                _D,
                _RO,
            ),
            _lane(
                "gauge.universality",
                "Finite BKT/noisy-Kuramoto universality diagnostic.",
                _R,
                _D,
                _RO,
            ),
            _lane(
                "gauge.vortex_detector", "Finite-plaquette vortex-density diagnostic.", _R, _D, _RO
            ),
            _lane("gauge.wilson_loop", "Finite U(1) Wilson-loop diagnostic.", _R, _D, _RO),
        ),
        key=lambda record: record.module,
    )
)


def list_research_lanes() -> tuple[ResearchLaneRecord, ...]:
    """Return all human-reviewed lanes in canonical module order.

    Returns
    -------
    tuple[ResearchLaneRecord, ...]
        The immutable registry. The tuple and its records may be safely shared
        between callers.

    """
    return _RESEARCH_LANES


def get_research_lane(module: str) -> ResearchLaneRecord:
    """Return the exact row for ``module`` or fail closed.

    Parameters
    ----------
    module
        Fully qualified module path. Package-relative values are not expanded
        implicitly because that could hide namespace mistakes.

    Raises
    ------
    KeyError
        If no human-reviewed registry row matches ``module``.

    """
    for record in _RESEARCH_LANES:
        if record.module == module:
            return record
    raise KeyError(f"unregistered research lane: {module!r}")


def discover_research_lane_modules(package_root: Path | None = None) -> tuple[str, ...]:
    """Discover ordinary ``analysis`` and ``gauge`` modules from source files.

    Parameters
    ----------
    package_root
        Directory containing the ``analysis`` and ``gauge`` packages. When
        omitted, discovery uses the installed ``scpn_quantum_control`` package
        containing this module.

    Returns
    -------
    tuple[str, ...]
        Sorted fully qualified module paths. ``__init__.py`` and this registry
        implementation are excluded by policy.

    Raises
    ------
    FileNotFoundError
        If either required package directory is absent.

    """
    root = package_root if package_root is not None else Path(__file__).resolve().parents[1]
    discovered: list[str] = []
    for family in ("analysis", "gauge"):
        directory = root / family
        if not directory.is_dir():
            raise FileNotFoundError(f"research-lane package directory is missing: {directory}")
        for path in directory.glob("*.py"):
            if path.name == "__init__.py":
                continue
            module = f"{_PACKAGE_PREFIX}{family}.{path.stem}"
            if module != _REGISTRY_MODULE:
                discovered.append(module)
    return tuple(sorted(discovered))


def validate_research_lane_inventory(
    discovered_modules: Iterable[str] | None = None,
) -> ResearchLaneInventoryReport:
    """Compare discovered modules with the human-reviewed registry.

    Parameters
    ----------
    discovered_modules
        Optional explicit discovery result for testing or packaged consumers.
        When omitted, :func:`discover_research_lane_modules` scans the current
        package. Duplicate values are normalized before comparison.

    Returns
    -------
    ResearchLaneInventoryReport
        Exact registered/discovered sets plus missing and orphaned entries.
        Inspect :attr:`ResearchLaneInventoryReport.passed` or call
        :func:`assert_research_lane_inventory` for exception semantics.

    """
    registered = tuple(record.module for record in _RESEARCH_LANES)
    discovered = tuple(
        sorted(
            set(
                discover_research_lane_modules()
                if discovered_modules is None
                else discovered_modules
            )
        )
    )
    registered_set = set(registered)
    discovered_set = set(discovered)
    return ResearchLaneInventoryReport(
        registered_modules=registered,
        discovered_modules=discovered,
        missing_modules=tuple(sorted(discovered_set - registered_set)),
        orphaned_records=tuple(sorted(registered_set - discovered_set)),
    )


def assert_research_lane_inventory(
    discovered_modules: Iterable[str] | None = None,
) -> ResearchLaneInventoryReport:
    """Return a passing inventory report or raise a drift error.

    Raises
    ------
    RuntimeError
        If a discovered module lacks a row or a row has no implementation.

    """
    report = validate_research_lane_inventory(discovered_modules)
    if not report.passed:
        raise RuntimeError(
            "research-lane registry drift: "
            f"missing={report.missing_modules!r}, orphaned={report.orphaned_records!r}"
        )
    return report


def _enum_counts(records: tuple[ResearchLaneRecord, ...], field: str) -> dict[str, int]:
    """Count enum-valued record fields in stable lexical order."""
    counts: dict[str, int] = {}
    for record in records:
        value = getattr(record, field)
        if not isinstance(value, Enum):
            raise TypeError(f"{field} is not an enum-valued record field")
        counts[value.value] = counts.get(value.value, 0) + 1
    return dict(sorted(counts.items()))


def _report_payload(
    inventory: ResearchLaneInventoryReport,
) -> dict[str, Any]:
    """Build the digest input without the self-referential digest field."""
    return {
        "schema": RESEARCH_LANE_REGISTRY_SCHEMA,
        "claim_boundary": RESEARCH_LANE_REGISTRY_BOUNDARY,
        "record_count": len(_RESEARCH_LANES),
        "maturity_counts": _enum_counts(_RESEARCH_LANES, "maturity"),
        "diff_hook_counts": _enum_counts(_RESEARCH_LANES, "diff_hook"),
        "claim_status_counts": _enum_counts(_RESEARCH_LANES, "claim_status"),
        "inventory": inventory.as_dict(),
        "records": [record.as_dict() for record in _RESEARCH_LANES],
    }


def build_research_lane_registry_report() -> ResearchLaneRegistryReport:
    """Build the deterministic research-lane report after inventory validation.

    Returns
    -------
    ResearchLaneRegistryReport
        Complete catalogue and SHA-256 content digest.

    Raises
    ------
    RuntimeError
        If the source inventory and reviewed rows differ.

    """
    inventory = assert_research_lane_inventory()
    payload = _report_payload(inventory)
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return ResearchLaneRegistryReport(
        schema=RESEARCH_LANE_REGISTRY_SCHEMA,
        claim_boundary=RESEARCH_LANE_REGISTRY_BOUNDARY,
        records=_RESEARCH_LANES,
        inventory=inventory,
        content_digest=hashlib.sha256(canonical).hexdigest(),
    )


def render_research_lane_registry_markdown(
    report: ResearchLaneRegistryReport | None = None,
) -> str:
    """Render a reviewable Markdown catalogue from registry state.

    Parameters
    ----------
    report
        Optional prebuilt report. When omitted, the inventory gate runs before
        rendering.

    Returns
    -------
    str
        Deterministic Markdown ending with a newline.

    """
    current = report if report is not None else build_research_lane_registry_report()
    lines = [
        "# Research-lane registry evidence",
        "",
        f"- Schema: `{current.schema}`",
        f"- Content digest: `{current.content_digest}`",
        f"- Inventory: **{'PASS' if current.inventory.passed else 'FAIL'}** "
        f"({len(current.records)} registered / "
        f"{len(current.inventory.discovered_modules)} discovered)",
        f"- Boundary: {current.claim_boundary}.",
        "",
        "| Module | Maturity | Diff hook | Claim status | Promotion route | Evidence |",
        "|---|---|---|---|---|---|",
    ]
    for record in current.records:
        targets = "<br>".join(record.promotion_targets) or "—"
        evidence = "<br>".join(f"`{ref}`" for ref in record.evidence_refs) or "—"
        lines.append(
            f"| `{record.module}` | {record.maturity.value} | "
            f"{record.diff_hook.value} | {record.claim_status.value} | "
            f"{targets} | {evidence} |"
        )
    lines.extend(("", "Every row remains subject to the global non-promotion boundary.", ""))
    return "\n".join(lines)


__all__ = [
    "RESEARCH_LANE_REGISTRY_BOUNDARY",
    "RESEARCH_LANE_REGISTRY_SCHEMA",
    "ResearchLaneClaimStatus",
    "ResearchLaneDiffHook",
    "ResearchLaneInventoryReport",
    "ResearchLaneMaturity",
    "ResearchLaneRecord",
    "ResearchLaneRegistryReport",
    "assert_research_lane_inventory",
    "build_research_lane_registry_report",
    "discover_research_lane_modules",
    "get_research_lane",
    "list_research_lanes",
    "render_research_lane_registry_markdown",
    "validate_research_lane_inventory",
]
