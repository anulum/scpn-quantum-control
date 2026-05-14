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
    Paper0EquationRecord(
        key="embodied.quantum_immune_interface",
        manuscript=_MANUSCRIPT,
        source_equation_ids=("EQ0105",),
        section_path=(
            "Structural Overview (by Domain) > The Genesis of Life - "
            "Abiogenesis as a Guided Phase Transition > The Slow Control Layer - "
            "Glial and Immune Modulation > II. The Quantum-Immune Interface "
            "(L1/L2/L5 Integration)"
        ),
        canonical_latex=(
            "H_{\\mathrm{int}}=-\\lambda(\\Psi_s,C_{\\mathrm{cyto}})\\sum_i\\sigma_x^{(i)}"
        ),
        variables={
            "H_int": "immune-state dependent interaction Hamiltonian term",
            "lambda": "coupling amplitude as a function of substrate state and cytokines",
            "Psi_s": "biological substrate field state used by the Paper 0 source",
            "C_cyto": "systemic cytokine concentration or cytokine-state vector",
            "sigma_x": "Pauli-X operator on the affected two-state quantum-biological mode",
        },
        assumptions=(
            "finite-dimensional two-state subsystem representation",
            "cytokine state enters only through declared coupling lambda",
            "operator support and units are declared before numerical simulation",
            "immune-interface claims remain mechanism-level until empirical data exist",
        ),
        validation_targets=(
            "cytokine-dependent Hamiltonian parameter scan",
            "zero lambda immune-decoupled null control",
            "fixed cytokine-state sensitivity and sign-convention control",
            "operator-norm boundedness and Hermiticity checks",
        ),
        themes=(
            "embodied",
            "immune-interface",
            "abiogenesis",
            "cellular",
            "Hamiltonian",
        ),
    ),
    Paper0EquationRecord(
        key="embodied.glial_sigma_control",
        manuscript=_MANUSCRIPT,
        source_equation_ids=(
            "EQ0106",
            "EQ0107",
            "EQ0108",
            "EQ0109",
            "EQ0110",
            "EQ0111",
            "EQ0112",
        ),
        section_path=(
            "Structural Overview (by Domain) > The Genesis of Life - "
            "Abiogenesis as a Guided Phase Transition > The Glial-Neuronal "
            "Coupling Mechanism: Slow Control of Neuronal Criticality"
        ),
        canonical_latex=(
            "\\dot{\\sigma}=-\\kappa\\left(\\sigma-(1+\\gamma G(t))\\right)+"
            "\\eta(t),\\quad "
            "\\dot{G}=\\alpha[Ca^{2+}]_A(t)-\\beta G(t)"
        ),
        variables={
            "sigma": "average neuronal branching parameter",
            "G": "local gliotransmitter concentration",
            "Ca_A": "astrocyte intracellular calcium signal",
            "kappa": "homeostatic relaxation rate of the neuronal criticality variable",
            "gamma": "sensitivity of neuronal excitation-inhibition balance to G",
            "alpha": "calcium-dependent gliotransmitter release rate",
            "beta": "gliotransmitter clearance or degradation rate",
            "eta": "stochastic fluctuation term in branching-parameter dynamics",
        },
        assumptions=(
            "sigma is a coarse-grained finite branching observable",
            "G is non-negative and cleared with positive beta",
            "astrocyte calcium drive is externally measured or explicitly generated",
            "timescale separation between fast neuronal avalanches and slow glial control",
        ),
        validation_targets=(
            "sigma relaxation to one under baseline homeostasis",
            "calcium-driven gliotransmitter response with positive alpha and beta",
            "glial slow-control shift in sigma set-point under finite gamma",
            "gliotransmitter blockade falsifier with attenuated sigma response",
        ),
        themes=(
            "embodied",
            "glial-control",
            "abiogenesis",
            "cellular",
            "quasicriticality",
            "slow control",
        ),
    ),
    Paper0EquationRecord(
        key="computational.cyclic_operator_boundary",
        manuscript=_MANUSCRIPT,
        source_equation_ids=("EQ0115",),
        section_path=(
            "Structural Overview (by Domain) > Computational Unifier. > "
            "1. The Cyclic Operator and Reversibility"
        ),
        canonical_latex=(
            "O_{\\mathrm{MMC}}|\\Psi_{\\mathrm{Source}}(t)\\rangle="
            "|\\Psi_{\\mathrm{Source}}(t+T_{\\mathrm{MMC}})\\rangle"
        ),
        variables={
            "O_MMC": "Meta Metatron Cycle evolution operator",
            "Psi_Source": "source-field state vector",
            "t": "time parameter",
            "T_MMC": "cycle period",
        },
        assumptions=(
            "finite or explicitly regularised state space",
            "periodic boundary condition declared before simulation",
            "unitarity and periodicity are testable operator properties",
            "retrocausal interpretation remains boundary-only without external evidence",
        ),
        validation_targets=(
            "boundary-only periodicity and unitarity check",
            "cycle-closure residual for O_MMC over one declared period",
            "identity-period and non-unitary operator null controls",
        ),
        themes=(
            "computational-unifier",
            "temporal-boundary",
            "cyclic operator",
            "boundary-only",
        ),
    ),
    Paper0EquationRecord(
        key="computational.tsvf_abl_boundary",
        manuscript=_MANUSCRIPT,
        source_equation_ids=("EQ0116",),
        section_path=(
            "Structural Overview (by Domain) > Computational Unifier. > "
            "3. Retrocausality via the Two-State Vector Formalism (TSVF)"
        ),
        canonical_latex=(
            "P(A=a|t)=\\frac{|\\langle\\phi(t)|P_a|\\psi(t)\\rangle|^2}"
            "{\\sum_j|\\langle\\phi(t)|P_j|\\psi(t)\\rangle|^2}"
        ),
        variables={
            "psi": "forward-evolving pre-selected state",
            "phi": "backward-evolving post-selected state",
            "P_a": "projector for intermediate outcome a",
            "P_j": "complete projective measurement family",
        },
        assumptions=(
            "normalisable pre- and post-selected states",
            "projectors form a complete orthogonal resolution of identity",
            "non-zero ABL denominator",
            "retrocausal interpretation remains boundary-only without external evidence",
        ),
        validation_targets=(
            "ABL probability normalisation check",
            "Born-rule reduction under trivial post-selection",
            "zero-denominator and non-projector rejection controls",
            "boundary-only interpretation label",
        ),
        themes=(
            "computational-unifier",
            "temporal-boundary",
            "TSVF",
            "ABL",
            "boundary-only",
        ),
    ),
    Paper0EquationRecord(
        key="computational.info_thermodynamics",
        manuscript=_MANUSCRIPT,
        source_equation_ids=("EQ0117", "EQ0118"),
        section_path=(
            "Structural Overview (by Domain) > Computational Unifier. > "
            "II. The Thermodynamics of Consciousness: Negentropy and Information"
        ),
        canonical_latex=(
            "\\frac{dS_{\\mathrm{Total}}}{dt}=\\frac{dS_{\\mathrm{Thermo}}}{dt}+"
            "\\frac{dS_{\\mathrm{Info}}}{dt}\\ge 0,\\quad "
            "N_{\\Psi}=-\\frac{dS_{\\mathrm{Thermo}}}{dt}\\propto I(\\Psi;B)"
        ),
        variables={
            "S_total": "total entropy budget",
            "S_thermo": "thermodynamic entropy contribution",
            "S_info": "information-processing entropy contribution",
            "N_Psi": "negentropy injection rate attributed to the modelled field channel",
            "I_Psi_B": "mutual information between Psi channel and biological substrate",
        },
        assumptions=(
            "entropy rates are measured on a shared time base",
            "local entropy reduction is paired with explicit information-processing cost",
            "mutual information estimator and units are declared",
            "GSL claims require non-negative total entropy-rate evidence",
        ),
        validation_targets=(
            "GSL entropy-rate budget check",
            "mutual information versus negentropy proportionality scan",
            "Landauer cost lower-bound control",
            "independent-channel zero-mutual-information null control",
        ),
        themes=(
            "computational-unifier",
            "information-thermodynamics",
            "entropy",
            "mutual information",
            "GSL",
        ),
    ),
    Paper0EquationRecord(
        key="computational.iit_or_threshold",
        manuscript=_MANUSCRIPT,
        source_equation_ids=("EQ0119",),
        section_path=(
            "Structural Overview (by Domain) > Computational Unifier. > "
            "1. The Threshold of Consciousness (IIT-OR)"
        ),
        canonical_latex="E_{\\Phi}=\\alpha_{\\Phi}\\Phi",
        variables={
            "E_Phi": "informational self-energy proxy assigned to a candidate state split",
            "alpha_Phi": "declared proportionality constant carrying the required units",
            "Phi": "integrated-information observable or estimator",
            "Phi_crit": "critical integrated-information threshold used for event labelling",
        },
        assumptions=(
            "Phi estimator, units, and coarse graining are declared before thresholding",
            "alpha_Phi is calibrated or explicitly scanned rather than inferred post hoc",
            "threshold labels are classifier outputs, not empirical collapse evidence",
            "OR interpretation remains a falsifiable boundary hypothesis",
        ),
        validation_targets=(
            "linear E_Phi versus Phi proportionality check with positive alpha_Phi",
            "threshold crossing classifier with pre-registered Phi_crit",
            "label-shuffle and alpha-zero null controls",
            "dimension and estimator-sensitivity scan",
        ),
        themes=(
            "computational-unifier",
            "IIT",
            "OR",
            "threshold",
            "classifier-boundary",
        ),
    ),
    Paper0EquationRecord(
        key="computational.coherence_noether_current",
        manuscript=_MANUSCRIPT,
        source_equation_ids=("EQ0120",),
        section_path=(
            "Structural Overview (by Domain) > Computational Unifier. > "
            "X. Symmetry, Conservation Laws, and the Coherence Current"
        ),
        canonical_latex=(
            "J_{\\Psi}^{\\mu}=i\\left(\\Psi^{*}\\partial^{\\mu}\\Psi-"
            "\\Psi\\partial^{\\mu}\\Psi^{*}\\right),\\quad "
            "\\partial_{\\mu}J_{\\Psi}^{\\mu}=0"
        ),
        variables={
            "J_Psi_mu": "Noether coherence-current four-vector",
            "Psi": "complex scalar field or finite lattice field",
            "partial_mu": "declared derivative operator on the chosen spacetime grid",
            "alpha": "global U(1) phase transformation parameter",
        },
        assumptions=(
            "field boundary conditions are declared before computing divergence",
            "derivative discretisation and metric signature are explicit",
            "current conservation is tested on equations of motion, not arbitrary fields",
            "gauge or global U(1) status is labelled before interpretation",
        ),
        validation_targets=(
            "global phase invariance of the finite action density",
            "discrete divergence residual for conserved-current trajectories",
            "boundary-flux accounting under non-periodic boundaries",
            "phase-broken and random-field null controls",
        ),
        themes=(
            "computational-unifier",
            "Noether",
            "coherence-current",
            "U1-symmetry",
            "conservation-law",
        ),
    ),
    Paper0EquationRecord(
        key="computational.information_energy_transduction",
        manuscript=_MANUSCRIPT,
        source_equation_ids=("EQ0121", "EQ0122"),
        section_path=(
            "Structural Overview (by Domain) > Computational Unifier. > "
            "XIV. The Physics of Information-Energy Transduction (IET)"
        ),
        canonical_latex=("Q=-\\frac{\\hbar^2}{2m}\\frac{\\nabla^2\\sqrt{\\rho}}{\\sqrt{\\rho}}"),
        variables={
            "Q": "Bohm quantum-potential energy density proxy",
            "hbar": "reduced Planck constant in the chosen unit system",
            "m": "effective mass parameter",
            "rho": "positive probability or density field",
            "nabla2": "declared Laplacian operator",
        },
        assumptions=(
            "rho is strictly positive or regularised before division",
            "grid spacing and boundary conditions are declared",
            "Q is interpreted as a quantum-potential diagnostic, not direct energy output",
            "singular-density points are rejected or handled by a stated regulariser",
        ),
        validation_targets=(
            "constant-density zero-potential null control",
            "Gaussian-density analytic quantum-potential residual",
            "grid-refinement convergence of the Laplacian term",
            "non-positive rho rejection control",
        ),
        themes=(
            "computational-unifier",
            "IET",
            "quantum-potential",
            "Bohm",
            "density-field",
        ),
    ),
    Paper0EquationRecord(
        key="computational.ethical_yang_mills_action",
        manuscript=_MANUSCRIPT,
        source_equation_ids=("EQ0123", "EQ0124"),
        section_path=(
            "Structural Overview (by Domain) > Computational Unifier. > "
            "The Physics of Teleology and the Origin of Ethics"
        ),
        canonical_latex=(
            "\\mathcal{L}_{\\mathrm{Ethical}}=\\lambda_E\\,"
            "\\mathrm{Tr}(F\\wedge\\star F),\\quad "
            "S_{\\mathrm{Ethical}}=\\int \\mathcal{L}_{\\mathrm{Ethical}}"
            "(\\mathrm{SCPN}(t))\\,dt,\\quad "
            "\\delta S_{\\mathrm{Ethical}}=0"
        ),
        variables={
            "L_Ethical": "ethical Yang-Mills-style action density proxy",
            "lambda_E": "declared proportionality or unit-conversion constant",
            "F": "curvature two-form or finite curvature tensor",
            "star": "Hodge dual under the declared metric and orientation",
            "S_Ethical": "integrated finite action functional",
        },
        assumptions=(
            "connection, curvature, metric, orientation, and integration domain are declared",
            "lambda_E is calibrated or explicitly scanned before comparison",
            "stationarity is a mathematical boundary condition, not empirical moral evidence",
            "finite discretisation and gauge representation are recorded with the result",
        ),
        validation_targets=(
            "finite Yang-Mills action non-negativity under positive metric",
            "gauge-action-boundary invariance under admissible gauge transforms",
            "stationary-action finite-difference residual around declared extrema",
            "wrong-sign metric and non-curvature tensor rejection controls",
        ),
        themes=(
            "computational-unifier",
            "ethical-gauge",
            "Yang-Mills",
            "least-action",
            "gauge-action-boundary",
        ),
    ),
    Paper0EquationRecord(
        key="computational.ethical_connection_boundary",
        manuscript=_MANUSCRIPT,
        source_equation_ids=("EQ0125", "EQ0126", "EQ0127"),
        section_path=(
            "Structural Overview (by Domain) > Computational Unifier. > "
            "III. Euler-Lagrange for the Ethical Connection"
        ),
        canonical_latex=(
            "D^{\\dagger}F=J_{\\mathrm{CEF}},\\quad "
            "J_{\\mathrm{CEF}}\\propto\\nabla_X S_C(X,\\tau),\\quad "
            "\\delta S_{\\mathrm{Ethical}}\\big|_{\\partial M}="
            "\\int_{\\partial M}\\mathrm{Tr}(\\delta A\\wedge\\star F),\\quad "
            "\\frac{d\\mathcal{C}_{\\mathrm{L10}}}{dt}\\le "
            "-\\kappa_{\\mathrm{Eth}}\\Phi_{\\partial M}+\\xi(t)"
        ),
        variables={
            "D_dagger": "adjoint covariant derivative under the declared connection",
            "J_CEF": "causal-entropic source current",
            "S_C": "future causal-pathway entropy functional",
            "delta_A": "variation of the connection at the boundary",
            "Phi_boundary": "boundary flux integral of the dual curvature",
            "C_L10": "boundary rendering-complexity observable",
            "kappa_Eth": "non-negative complexity-flux coupling",
            "xi": "declared stochastic or residual forcing term",
        },
        assumptions=(
            "boundary orientation and flux convention are explicit",
            "CEF source current is computed from a declared entropy functional",
            "complexity inequality uses measured or simulated C_L10 on a shared time grid",
            "stochastic residual xi is bounded or separately estimated",
        ),
        validation_targets=(
            "Euler-Lagrange residual for D_dagger F minus J_CEF",
            "boundary-flux term consistency under orientation reversal",
            "complexity-flux inequality margin under non-negative kappa_Eth",
            "zero-flux, wrong-sign-kappa, and shuffled-boundary null controls",
        ),
        themes=(
            "computational-unifier",
            "ethical-connection",
            "Euler-Lagrange",
            "boundary-flux",
            "complexity-flux",
        ),
    ),
    Paper0EquationRecord(
        key="computational.causal_entropic_force",
        manuscript=_MANUSCRIPT,
        source_equation_ids=("EQ0128",),
        section_path=(
            "Structural Overview (by Domain) > Computational Unifier. > "
            "IV. Ethics as Causal Entropic Forces (CEF)"
        ),
        canonical_latex="F_{\\mathrm{Causal}}=T_C\\nabla_X S_C(X,\\tau)",
        variables={
            "F_Causal": "causal-entropic force vector field",
            "T_C": "causal temperature or force-scale parameter",
            "S_C": "future causal-pathway entropy functional",
            "X": "configuration-space coordinate",
            "tau": "future horizon used for causal-pathway counting",
        },
        assumptions=(
            "configuration coordinates and metric are declared",
            "S_C estimator and future horizon tau are fixed before gradient evaluation",
            "T_C is non-negative and carries the declared force units",
            "CEF interpretation remains a force-field proxy until empirical coupling evidence exists",
        ),
        validation_targets=(
            "causal entropy gradient finite-difference agreement",
            "force direction increases S_C for positive T_C under small steps",
            "zero-temperature and flat-entropy null controls",
            "coordinate-rescaling sensitivity and unit audit",
        ),
        themes=(
            "computational-unifier",
            "CEF",
            "causal-entropic-force",
            "entropy-gradient",
            "force-field-boundary",
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
