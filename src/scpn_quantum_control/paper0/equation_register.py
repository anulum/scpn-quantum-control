# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Paper 0 equation register
"""Source-anchored Paper 0 equation records.

The register is intentionally conservative: it records only equations that have
already been extracted from the Paper 0 manuscript inventory, and it does not
promote any downstream helper as canonical unless the source anchor is explicit.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Paper0EquationRecord:
    """Canonical source record for a Paper 0 mathematical statement."""

    key: str
    manuscript: str
    source_equation_ids: tuple[str, ...]
    section_path: str
    canonical_latex: str
    variables: dict[str, str]
    assumptions: tuple[str, ...]
    validation_targets: tuple[str, ...]
    themes: tuple[str, ...]
    provenance_status: str = "paper0_extracted"


_MANUSCRIPT = "Paper 0 foundational framework"


_RECORDS: tuple[Paper0EquationRecord, ...] = (
    Paper0EquationRecord(
        key="upde.base_phase",
        manuscript=_MANUSCRIPT,
        source_equation_ids=("EQ0003", "EQ0032", "EQ0037", "EQ0039", "EQ0129"),
        section_path=(
            "Part I > 1.2 Foundational Viability > Unified Phase Dynamics; "
            "Part III > 3.2 Dynamic Spine"
        ),
        canonical_latex=(
            "\\frac{d\\theta_i^L}{dt}=\\omega_i^L+"
            "\\sum_j K_{ij}^{L}\\sin(\\theta_j^L-\\theta_i^L)+"
            "C_{\\mathrm{InterLayer}}+C_{\\mathrm{Field}}+\\eta_i^L(t)"
        ),
        variables={
            "theta_i_L": "phase of oscillator i in layer L",
            "omega_i_L": "intrinsic angular frequency of oscillator i in layer L",
            "K_ij_L": "intra-layer coupling strength from oscillator j to i",
            "C_interlayer": "hierarchical coupling contribution between adjacent layers",
            "C_field": "global field-coupling contribution",
            "eta_i_L": "stochastic drive or noise term for oscillator i in layer L",
        },
        assumptions=(
            "finite oscillator layer",
            "phase-reduced limit-cycle dynamics",
            "real-valued coupling weights",
            "explicitly modelled inter-layer and field terms",
        ),
        validation_targets=(
            "UPDE-to-XY gradient check",
            "phase-locking order-parameter scan",
            "off-onset topology control",
        ),
        themes=("UPDE", "Kuramoto", "synchronisation", "topology"),
    ),
    Paper0EquationRecord(
        key="upde.interlayer_coupling",
        manuscript=_MANUSCRIPT,
        source_equation_ids=("EQ0033", "EQ0040"),
        section_path="Part III > 3.2 Dynamic Spine > Inter-layer coupling",
        canonical_latex=(
            "C_{\\mathrm{InterLayer}}="
            "\\epsilon_{L-1}F_D(\\langle\\theta^{L-1}\\rangle,\\theta_i^L)+"
            "\\epsilon_{L+1}G_U(\\theta_i^L,\\langle\\theta^{L+1}\\rangle)"
        ),
        variables={
            "epsilon_L_minus_1": "downward-causation coupling gain",
            "epsilon_L_plus_1": "upward-feedback coupling gain",
            "F_D": "downward projection or prediction operator",
            "G_U": "upward aggregation or prediction-error operator",
            "theta_i_L": "local layer-L oscillator phase",
        },
        assumptions=(
            "adjacent-layer hierarchy",
            "separable downward and upward coupling channels",
            "well-defined layer-level phase averages",
        ),
        validation_targets=(
            "inter-layer coupling decomposition",
            "disconnected-layer null control",
            "directional perturbation response",
        ),
        themes=("UPDE", "hierarchy", "inter-layer coupling"),
    ),
    Paper0EquationRecord(
        key="upde.field_coupling",
        manuscript=_MANUSCRIPT,
        source_equation_ids=("EQ0034", "EQ0041", "EQ0043"),
        section_path="Part III > 3.2 Dynamic Spine > Global field coupling",
        canonical_latex=(
            "C_{\\mathrm{Field}}=\\zeta_L\\Psi_{\\mathrm{Global}}"
            "\\cos(\\theta_i^L-\\Theta_{\\Psi})"
        ),
        variables={
            "zeta_L": "layer-specific global field coupling gain",
            "Psi_Global": "global field amplitude used by the model",
            "Theta_Psi": "global field phase",
            "theta_i_L": "local oscillator phase",
        },
        assumptions=(
            "finite global phase reference",
            "bounded field-coupling gain",
            "cosine phase alignment term",
        ),
        validation_targets=(
            "field-coupling control",
            "randomised-global-phase null control",
            "zero-field baseline",
        ),
        themes=("UPDE", "field coupling", "synchronisation"),
    ),
    Paper0EquationRecord(
        key="upde.natural_gradient",
        manuscript=_MANUSCRIPT,
        source_equation_ids=("EQ0042",),
        section_path="Part III > 3.2 Dynamic Spine > Information-geometric lift",
        canonical_latex=(
            "\\dot{\\theta}_i^L=-\\eta_L"
            "\\left(g_F^{-1}\\nabla_{\\theta}\\mathcal{F}_L\\right)_i+"
            "C_{\\mathrm{InterLayer}}+C_{\\mathrm{Field}}+\\eta_i^L(t)"
        ),
        variables={
            "eta_L": "layer-level descent rate",
            "g_F": "Fisher information metric on the phase/statistical manifold",
            "mathcal_F_L": "layer-level free-energy functional",
            "theta_i_L": "local oscillator phase",
            "C_interlayer": "hierarchical coupling contribution",
            "C_field": "global field-coupling contribution",
        },
        assumptions=(
            "positive-definite or regularised Fisher information metric",
            "differentiable free-energy functional",
            "finite-dimensional statistical manifold chart",
        ),
        validation_targets=(
            "FIM natural-gradient check",
            "singular-FIM fail-fast guard",
            "Euclidean-vs-natural-gradient comparison",
        ),
        themes=("UPDE", "FIM", "free energy", "natural gradient"),
    ),
    Paper0EquationRecord(
        key="upde.adaptive_coupling",
        manuscript=_MANUSCRIPT,
        source_equation_ids=("EQ0045",),
        section_path="Part III > 3.2 Dynamic Spine > Formalising SOC as a control law",
        canonical_latex=(
            "\\dot{K}_{ij}^{L}=\\gamma_L(R_L-R_L^*)-\\lambda_LK_{ij}^{L}+"
            "\\xi_{ij}^{L}(t),\\quad "
            "\\dot{\\eta}^{L}=-\\alpha_L(\\sigma_L-1)"
        ),
        variables={
            "K_ij_L": "adaptive intra-layer coupling",
            "gamma_L": "order-parameter feedback gain",
            "R_L": "layer synchronisation order parameter",
            "R_L_star": "target layer synchronisation order parameter",
            "lambda_L": "coupling decay or regularisation rate",
            "xi_ij_L": "coupling-process noise",
            "eta_L": "layer noise or gain control variable",
            "alpha_L": "criticality feedback gain",
            "sigma_L": "layer criticality parameter",
        },
        assumptions=(
            "bounded adaptive gains",
            "observable layer order parameter",
            "criticality target sigma_L equals one",
            "finite stochastic perturbations",
        ),
        validation_targets=(
            "adaptive coupling and quasicritical controller",
            "wrong-sign feedback null control",
            "bounded-perturbation convergence",
        ),
        themes=("UPDE", "SOC", "adaptive coupling", "quasicriticality"),
    ),
    Paper0EquationRecord(
        key="nths.spin_glass_hamiltonian",
        manuscript=_MANUSCRIPT,
        source_equation_ids=("EQ0113",),
        section_path=(
            "Structural Overview (by Domain) > The Genesis of Life - "
            "Abiogenesis as a Guided Phase Transition > Domain III & IV: "
            "Memory, Control, and Collective Coherence (Layers 9-12)"
        ),
        canonical_latex=("H=-\\sum_{i<j}J_{ij}\\sigma_i\\sigma_j-\\sum_i h_i\\sigma_i"),
        variables={
            "sigma_i": "binary belief or agent state in {-1,+1}",
            "J_ij": "effective social coupling between agents i and j",
            "h_i": "media or AI-driven external field acting on agent i",
            "H": "spin-glass Hamiltonian or social dissonance energy",
        },
        assumptions=(
            "finite agent set",
            "binary belief-state reduction",
            "symmetric pair couplings for Hamiltonian evaluation",
            "external fields separated from endogenised social couplings",
        ),
        validation_targets=(
            "NTHS spin-glass Hamiltonian energy evaluation",
            "magnetisation and Edwards-Anderson order-parameter contrast",
            "ultrametric clustering test under matched disorder replicas",
        ),
        themes=(
            "macro-transition",
            "NTHS",
            "spin glass",
            "social phase transition",
        ),
    ),
    Paper0EquationRecord(
        key="macro_transition.effective_coupling_rg",
        manuscript=_MANUSCRIPT,
        source_equation_ids=("EQ0114",),
        section_path=(
            "Structural Overview (by Domain) > The Genesis of Life - "
            "Abiogenesis as a Guided Phase Transition > 1. Renormalisation "
            "Group (RG) Flow Across Domains"
        ),
        canonical_latex=("\\mu\\frac{dK_{\\mathrm{eff}}}{d\\mu}=\\beta_K(K,\\omega,\\ldots)"),
        variables={
            "mu": "coarse-graining or renormalisation scale",
            "K_eff": "effective coupling after coarse-graining",
            "beta_K": "beta function for the effective coupling",
            "K": "microscopic or lower-domain coupling parameters",
            "omega": "frequency or heterogeneity parameters entering the flow",
        },
        assumptions=(
            "explicit coarse-graining scale",
            "finite beta-function inputs",
            "effective coupling remains within declared physical bounds",
            "fixed-point claims require sign-change or zero-crossing evidence",
        ),
        validation_targets=(
            "effective-coupling flow integration",
            "fixed point and stability classification",
            "null beta-function and constant beta controls",
        ),
        themes=(
            "macro-transition",
            "renormalisation group",
            "effective coupling",
            "scale transition",
        ),
    ),
    Paper0EquationRecord(
        key="embodied.neurovascular_phase_coupling",
        manuscript=_MANUSCRIPT,
        source_equation_ids=("EQ0093",),
        section_path=(
            "5.2 Embodied SCPN: Cellular, Neural, & Systemic Implementation > "
            "II. Neuro-Vascular Coupling and Hemodynamics: The Energetics of "
            "Consciousness > 2. Hemodynamics and the UPDE"
        ),
        canonical_latex=(
            "\\dot{\\theta}_{\\mathrm{Neural}}=\\omega_N+"
            "K_{NH}\\sin(\\theta_{\\mathrm{Hemo}}-"
            "\\theta_{\\mathrm{Neural}})"
        ),
        variables={
            "theta_neural": "phase of the neural oscillator or rhythm",
            "theta_hemo": "phase of the hemodynamic or Mayer-wave oscillator",
            "omega_N": "intrinsic neural angular frequency",
            "K_NH": "hemodynamic-to-neural phase-coupling gain",
        },
        assumptions=(
            "two-oscillator phase-reduced neurovascular model",
            "finite real-valued coupling gain",
            "hemodynamic driver phase is observable or explicitly simulated",
            "Mayer-wave entrainment claim is separated from metabolic pathology claims",
        ),
        validation_targets=(
            "Mayer-wave hemodynamic entrainment phase-locking scan",
            "zero K_NH uncoupled neural-frequency null control",
            "detuned hemodynamic-frequency off-resonance control",
            "impaired energy-supply subcriticality boundary test",
        ),
        themes=(
            "embodied",
            "neurovascular",
            "hemodynamics",
            "UPDE",
            "phase coupling",
        ),
    ),
)

_RECORDS_BY_KEY = {record.key: record for record in _RECORDS}


def iter_paper0_equation_records(*, theme: str | None = None) -> Iterable[Paper0EquationRecord]:
    """Iterate over canonical Paper 0 equation records, optionally by theme."""
    if theme is None:
        return iter(_RECORDS)
    selected = tuple(record for record in _RECORDS if theme in record.themes)
    return iter(selected)


def paper0_upde_records() -> tuple[Paper0EquationRecord, ...]:
    """Return the canonical Paper 0 UPDE equation family."""
    return tuple(record for record in _RECORDS if record.key.startswith("upde."))


def get_paper0_equation_record(key: str) -> Paper0EquationRecord:
    """Return a source-anchored Paper 0 equation record by stable key."""
    try:
        return _RECORDS_BY_KEY[key]
    except KeyError as exc:
        raise KeyError(f"unknown Paper 0 equation record: {key}") from exc
