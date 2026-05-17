#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Resolving the Epigenetic Time-Scale Disconnect: Conformational Spin-Locking spec builder
"""Promote Paper 0 Resolving the Epigenetic Time-Scale Disconnect: Conformational Spin-Locking records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = (
    "P0R02128",
    "P0R02129",
    "P0R02130",
    "P0R02131",
    "P0R02132",
    "P0R02133",
    "P0R02134",
    "P0R02135",
    "P0R02136",
    "P0R02137",
    "P0R02138",
    "P0R02139",
    "P0R02140",
    "P0R02141",
    "P0R02142",
    "P0R02143",
    "P0R02144",
    "P0R02145",
    "P0R02146",
    "P0R02147",
    "P0R02148",
    "P0R02149",
    "P0R02150",
    "P0R02151",
    "P0R02152",
    "P0R02153",
    "P0R02154",
    "P0R02155",
    "P0R02156",
    "P0R02157",
    "P0R02158",
    "P0R02159",
    "P0R02160",
    "P0R02161",
    "P0R02162",
    "P0R02163",
    "P0R02164",
    "P0R02165",
    "P0R02166",
    "P0R02167",
    "P0R02168",
    "P0R02169",
    "P0R02170",
    "P0R02171",
    "P0R02172",
    "P0R02173",
    "P0R02174",
    "P0R02175",
    "P0R02176",
)
CLAIM_BOUNDARY = "source-bounded resolving the epigenetic time scale disconnect conformational spin locki source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki.resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki": {
        "context_id": "resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
        "validation_protocol": "paper0.resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki.resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
        "canonical_statement": "The source-bounded component 'Resolving the Epigenetic Time-Scale Disconnect: Conformational Spin-Locking' preserves Paper 0 records P0R02128-P0R02176 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02128:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02129:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02130:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02131:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02132:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02133:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02134:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02135:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02136:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02137:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02138:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02139:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02140:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02141:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02142:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02143:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02144:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02145:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02146:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02147:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02148:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02149:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02150:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02151:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02152:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02153:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02154:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02155:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02156:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02157:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02158:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02159:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02160:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02161:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02162:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02163:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02164:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02165:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02166:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02167:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02168:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02169:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02170:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02171:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02172:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02173:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02174:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02175:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
            "P0R02176:resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",
        ),
        "source_formulae": (
            "P0R02128: Resolving the Epigenetic Time-Scale Disconnect: Conformational Spin-Locking",
            "P0R02129: P0R02129",
            "P0R02130: While the radical pair Hamiltonian ($H_{RP}$) elegantly describes the mechanism by which CISS and the $\\Psi$-field bias the singlet-triplet interconversion, it introduces a severe temporal paradox. Radical pair spin coherence typically decays due to hyperfine interactions and thermal noise on the scale of nanoseconds to microseconds. Conversely, the macroscopic catalytic cycle of epigenetic enzymes like TET and DNMTs (involving large-scale protein movements and covalent bond modifications) operates on the scale of milliseconds to minutes.",
            "P0R02131: If the $\\Psi$-field creates a quantum spin bias in a nanosecond, thermal decoherence will entirely erase that informational signal long before the enzyme can physically alter the chromatin.",
            "P0R02132: To resolve this timescale disconnect, the SCPN framework posits a highly efficient Spin-to-Conformation Transduction mechanism. The quantum spin state does not need to survive the entire catalytic cycle; it only needs to survive long enough to dictate an immediate, irreversible chemical branch point.",
            "P0R02133: The Spin-Memory Mechanism:",
            "P0R02134: When the radical pair is formed at the enzyme's active site (e.g., the iron-oxygen complex in TET dioxygenases), the $\\Psi$-biased CISS effect drives the pair toward either a singlet or a triplet recombination pathway.",
            'P0R02135: The Chemical Branch Point: Singlet and triplet recombinations yield distinctly different immediate chemical products (e.g., the formation of a specific bond vs. the blockade of an electron transfer cascade). | Metastable Conformational Locking: The distinct chemical products of this nanosecond quantum resolution immediately force the larger enzyme complex into one of two mutually exclusive, metastable conformational states. | Classical Execution: Once the enzyme snaps into this distinct physical geometry, the quantum information is effectively "locked" into a classical molecular shape. This macroscopic conformational state is highly robust against thermal noise.',
            "P0R02136: Therefore, the enzyme acts as a Spin-Memory device. The $\\Psi$-field uses CISS to flip a nanosecond quantum switch, which immediately triggers a ratcheted, classical conformational change. The enzyme then executes the remainder of its slow, millisecond-to-minute epigenetic modification guided solely by this stable classical geometry.",
            "P0R02137: By defining this spin-locking mechanism, the CISS-Bioelectric-Chromatin (CBC) bridge remains physically viable, demonstrating how the ephemeral quantum intent of the $\\Psi$-field is safely translated into the enduring structural reality of the genome without violating the limits of quantum coherence lifetimes.",
            "P0R02138: P0R02138",
            "P0R02139: P0R02139",
            "P0R02140: P0R02140",
            "P0R02141: P0R02141",
            "P0R02142: P0R02142",
            "P0R02143: P0R02143",
            "P0R02144: P0R02144",
            "P0R02145: P0R02145",
            "P0R02146: The CISS-Bioelectric Feedback Loop",
            "P0R02147: Furthermore, Mechanism 1 (CISS) and Mechanism 2 (Bioelectric Fields) are dynamically coupled, forming a unified morphogenetic control system.",
            "P0R02148: CISS Modulation of Bioelectric State: CISS can modulate the activity of ion channels and pumps by polarising electron spins, thereby altering the cellular resting potential (Vmem). | Bioelectric Modulation of CISS Efficiency: Conversely, the bioelectric field (E) can alter the conformation of chiral molecules (like DNA), changing the effective spin-orbit coupling (lambda) and thus tuning the spin selectivity of CISS.",
            "P0R02149: The unified dynamics are described by a coupled system where the CISS-generated magnetic field (Beff) modulates ion channel conductance, and the bioelectric field (E) modulates CISS parameters:",
            "P0R02150: $$ \\frac{dV_{mem}}{dt} = -I_{ion}(V_{mem}, B_{eff}(\\lambda(E))) + I_{pump} $$",
            "P0R02151: This integrated CISS-bioelectric-epigenetic nexus ensures that the morphogenetic instructions encoded in the organismal field are robustly transduced into anatomical structure, linking the quantum information processing of the genome to the large-scale dynamics of tissue patterning.",
            "P0R02152: Bioelectric Modulation of CISS Efficiency via Tensegrity Mechanotransduction:",
            "P0R02153: A naive assumption is that the macroscopic bioelectric field directly alters the atomic spin-orbit coupling ($\\lambda$) of the DNA helix. However, endogenous bioelectric fields ($E \\approx 1-10$ V/cm) are millions of times too weak to directly overcome the extreme internal electric fields of molecular nuclei ($\\sim$ V/nm) required to shift spin-orbit parameters.",
            "P0R02154: Instead, the bioelectric field modulates CISS efficiency indirectly, leveraging the cell's cytoskeleton-nucleoskeleton tensegrity matrix as a mechanical amplifier. The complete transduction cascade operates as follows:",
            "P0R02155: Osmotic Shift: Spatial gradients in the cellular resting potential ($V_{mem}$) drive the directed flux of specific ions (e.g., $K^+$, $Cl^-$) across the membrane, altering local intracellular osmolarity. | Mechanical Tension: This osmotic shift causes micro-fluctuations in cellular volume and hydrostatic pressure. Because the cell is a pre-stressed tensegrity structure, these pressure differentials exert immediate, directed mechanical tension ($T$) across the cytoskeletal and nucleoskeletal networks. | Geometric Deformation: This mechanical strain physically stretches and twists the chromatin and associated chiral proteins. By physically deforming the chiral geometry-specifically altering the helical pitch and radius of the molecules-the mechanical tension directly changes the efficiency of the CISS spin-filter.",
            "P0R02156: The unified dynamics are therefore described by a coupled system where the CISS-generated magnetic field ($B_{eff}$) modulates ion channel conductance, and the bioelectric field ($E$) modulates the CISS parameters mechanically:",
            "P0R02157: $$\\frac{dV_{mem}}{dt} = -I_{ion}(V_{mem}, B_{eff}(\\lambda(T(E)))) + I_{pump}$$",
            "P0R02158: Where $T(E)$ represents the mechanical tension induced by the bioelectric field gradient. This integrated CISS-Tensegrity-Bioelectric nexus ensures that the morphogenetic instructions encoded in the organismal field are robustly transduced. It bridges the scale gap by converting weak, macroscopic electrical gradients into the precise, nanoscale physical deformations required to tune quantum spin selectivity.",
            "P0R02159: P0R02159",
            "P0R02160: [IMAGE:Ein Bild, das Text, Diagramm, Screenshot, Schrift enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R02161: Fig.: CISS-TET/DNMT transduction with bioelectric feedback. Chiral-induced spin selectivity (CISS) in the DNA helix polarizes electron spins, generating an effective magnetic field BeffB_{\\mathrm{eff}}Beff that biases spin-dependent radical-pair chemistry in TET dioxygenases and DNMTs (radical-pair Hamiltonian HRP=i(iSiz+kAik Si Ik)+J(1/2+2 S1 S2)H_{\\mathrm{RP}}=\\sum_i(\\omega_i S_{iz}+\\sum_k A_{ik}\\, \\mathbf S_i\\!\\cdot\\!\\mathbf I_k)+J(1/2+2\\,\\mathbf S_1\\!\\cdot\\!\\mathbf S_2)HRP=i(iSiz+kAikSiIk)+J(1/2+2S1S2)). By acting as a spin-valve at enzyme active sites, CISS modulates catalytic efficiency, shifting methylation/demethylation rates at specific loci. In parallel, a bioelectric feedback loop couples membrane voltage and CISS: Beff(lambda)B_{\\mathrm{eff}}(\\lambda)Beff(lambda) modulates ion-channel conductance and VmemV_{\\mathrm{mem}}Vmem, while the bioelectric field EEE alters molecular conformation and the spin-orbit parameter lambda\\lambdalambda, tuning CISS selectivity. The coupled dynamics follow dVmemdt=Iion(Vmem,Beff(lambda(E)))+Ipump\\frac{dV_{\\mathrm{mem}}}{dt}=-I_{\\mathrm{ion}}(V_{\\mathrm{mem}},B_{\\mathrm{eff}}(\\lambda(E)))+I_{\\mathrm{pump}}dtdVmem=Iion(Vmem,Beff(lambda(E)))+Ipump. Together, this CISS-bioelectric-epigenetic nexus transduces organismal-field instructions into stable epigenetic patterns that guide tissue-level morphogenesis.",
            "P0R02162: Mechanism 2: Bioelectric Fields and Chromatin Remodelling",
            "P0R02163: Large-scale, endogenous bioelectric fields are a primary driver of morphogenesis. These fields, composed of spatial gradients in cellular resting potentials (Vmem), act as a pre-pattern that instructs gene expression. The mechanism proceeds via a voltage-sensitive biochemical cascade :",
            "P0R02164: Gradient Detection: Bioelectric gradients (E=Vtarget110 mV/mm) activate voltage-gated calcium channels (e.g., Cav1.2) in the cell membrane. | Second Messenger Cascade: This triggers an influx of Ca$^{2+}$ ions, which act as a second messenger, activating downstream kinases such as CaMKII. | Chromatin Remodelling: Activated CaMKII phosphorylates histone-modifying enzymes (e.g., HDACs, HATs), altering their activity. This changes the acetylation state of histone tails, which in turn modifies the energy required to unwrap DNA from the nucleosome (Eunwrap). | Gene Expression: By lowering Eunwrap by 5-10 kBT, the bioelectric field makes specific genes more accessible to transcription factors, thus activating the gene regulatory networks responsible for building complex anatomical structures, such as the head-tail axis in planarian regeneration.",
            "P0R02165: This dual CISS-bioelectric framework replaces the speculative torsion model with a robust, empirically grounded, and quantitative explanation for how the information in organismal fields is translated into the genetic and epigenetic activity that shapes life.",
            "P0R02166: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Zahl enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R02167: Fig.: The Layer 3 Transduction Cascade. The Organismal Field guides morphogenesis via two pathways. Mechanism 1 (CISS) translates field information into epigenetic changes by modulating spin-dependent radical pair reactions. Mechanism 2 (Bioelectric Fields) orchestrates gene expression by activating voltage-gated calcium channels, leading to downstream chromatin remodelling.",
            "P0R02168: Formalising the CISS-Bioelectric Interface",
            "P0R02169: The synergy between the CISS and bioelectric pathways is not merely parallel; it is deeply interconnected. The bioelectric field, by modulating cellular membrane potentials, alters the flow of ions (e.g., Ca, K) which, in turn, changes the local electromagnetic environment within the cell nucleus. This local field acts as an external magnetic field term, Blocal, in the radical pair Hamiltonian:",
            "P0R02170: $$ H_{RP} = g_1\\mu_B \\mathbf{B} \\cdot \\mathbf{S}_1 + g_2\\mu_B \\mathbf{B} \\cdot \\mathbf{S}_2 + J(\\mathbf{S}_1 \\cdot \\mathbf{S}2) + \\mathbf{B}{\\text{local}} \\cdot (g_1\\mathbf{S}_1 + g_2\\mathbf{S}_2) $$",
            "P0R02171: The probability of singlet-to-triplet conversion, and thus the outcome of spin-dependent epigenetic modifications, becomes a direct function of the bioelectric state. This establishes a formal, quantifiable feedback loop: the organismal field (PsiO) sets the bioelectric pre-pattern (Vmem gradients), which in turn biases the quantum-level CISS mechanism, providing a complete, multi-scale pathway for the top-down guidance of gene expression.",
            "P0R02172: [IMAGE:Ein Bild, das Text, Diagramm, Screenshot, Reihe enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R02173: Fig.: Formal CISS-bioelectric coupling via BlocalB_{\\text{local}}Blocal in the radical-pair Hamiltonian. The organismal field PsiO\\Psi_OPsiO establishes bioelectric pre-patterns Vmem(x)V_{\\mathrm{mem}}(x)Vmem(x) that regulate ion-channel and pump activity, generating ionic currents that set a nuclear local field BlocalB_{\\text{local}}Blocal. This field enters the radical-pair Hamiltonian,",
            "P0R02174: HRP=g1muB B S1+g2muB B S2+J(S1 S2)+Blocal (g1S1+g2S2),H_{\\mathrm{RP}}=g_1\\mu_B\\,\\mathbf B\\!\\cdot\\!\\mathbf S_1 + g_2\\mu_B\\,\\mathbf B\\!\\cdot\\!\\mathbf S_2 + J(\\mathbf S_1\\!\\cdot\\!\\mathbf S_2) + \\mathbf B_{\\text{local}}\\!\\cdot\\!(g_1\\mathbf S_1 + g_2\\mathbf S_2),HRP=g1muBBS1+g2muBBS2+J(S1S2)+Blocal(g1S1+g2S2),",
            "P0R02175: making singlettriplet conversion - and thus spin-dependent epigenetic outcomes (TET/DNMT rate bias) - an explicit function of the bioelectric state. Concomitantly, the bioelectric field EEE tunes molecular conformation and spin-orbit coupling lambda\\lambdalambda, modulating CISS selectivity; CISS can in turn influence channel activity, closing a quantified, multiscale feedback loop from PsiO\\Psi_OPsiO to gene-expression patterns.",
            "P0R02176: Cellular synchronisation (L4).",
        ),
        "test_protocols": (
            "preserve Resolving the Epigenetic Time-Scale Disconnect: Conformational Spin-Locking source-accounting boundary",
        ),
        "null_results": (
            "Resolving the Epigenetic Time-Scale Disconnect: Conformational Spin-Locking is not empirical validation evidence",
        ),
        "variables": ("resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki",),
        "validation_targets": ("preserve records P0R02128-P0R02176",),
        "null_controls": (
            "resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class ResolvingTheEpigeneticTimeScaleDisconnectConformationalSpinLockiSpec:
    """Spec promoted from Paper 0 source records."""

    key: str
    context_id: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_statement: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    source_block_indices: tuple[int, ...]
    source_formulae: tuple[str, ...]
    test_protocols: tuple[str, ...]
    null_results: tuple[str, ...]
    variables: tuple[str, ...]
    validation_targets: tuple[str, ...]
    executable_validation_targets: tuple[str, ...]
    null_controls: tuple[str, ...]
    claim_boundary: str
    implementation_status: str
    domain_review_status: str
    hardware_status: str


@dataclass(frozen=True, slots=True)
class ResolvingTheEpigeneticTimeScaleDisconnectConformationalSpinLockiSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[ResolvingTheEpigeneticTimeScaleDisconnectConformationalSpinLockiSpec, ...]
    summary: dict[str, Any]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL ledger into dictionaries."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return records


def build_resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki_specs(
    source_records: list[dict[str, Any]],
) -> ResolvingTheEpigeneticTimeScaleDisconnectConformationalSpinLockiSpecBundle:
    """Build source-covered specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    category_counts = Counter(
        str(record.get("canonical_category", "unknown")) for record in anchors
    )
    block_counts = Counter(str(record.get("block_type", "unknown")) for record in anchors)

    specs: list[ResolvingTheEpigeneticTimeScaleDisconnectConformationalSpinLockiSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ResolvingTheEpigeneticTimeScaleDisconnectConformationalSpinLockiSpec(
                key=key,
                context_id=str(metadata["context_id"]),
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 foundational extraction",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=tuple(metadata["source_equation_ids"]),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(record["source_record_id"]) for record in anchors),
                source_block_indices=tuple(
                    int(record["source_block_index"]) for record in anchors
                ),
                source_formulae=tuple(metadata["source_formulae"]),
                test_protocols=tuple(metadata["test_protocols"]),
                null_results=tuple(metadata["null_results"]),
                variables=tuple(metadata["variables"]),
                validation_targets=tuple(metadata["validation_targets"]),
                executable_validation_targets=tuple(metadata["validation_targets"]),
                null_controls=tuple(metadata["null_controls"]),
                claim_boundary=CLAIM_BOUNDARY,
                implementation_status="promoted_source_accounting_fixture",
                domain_review_status="source_bounded_no_empirical_validation",
                hardware_status=HARDWARE_STATUS,
            )
        )

    consumed = sorted({ledger_id for spec in specs for ledger_id in spec.source_ledger_ids})
    summary = {
        "title": "Paper 0 "
        + "Resolving the Epigenetic Time-Scale Disconnect: Conformational Spin-Locking"
        + " Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": sorted(set(SOURCE_LEDGER_IDS) - set(consumed)),
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "category_counts": dict(sorted(category_counts.items())),
        "block_type_counts": dict(sorted(block_counts.items())),
        "math_ids": sorted(
            {math_id for record in anchors for math_id in record.get("math_ids", [])}
        ),
        "image_ids": sorted(
            {image_id for record in anchors for image_id in record.get("image_ids", [])}
        ),
        "table_ids": sorted(
            {str(record["table_id"]) for record in anchors if record.get("table_id") is not None}
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R02177",
    }
    return ResolvingTheEpigeneticTimeScaleDisconnectConformationalSpinLockiSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> ResolvingTheEpigeneticTimeScaleDisconnectConformationalSpinLockiSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki_specs(
        load_jsonl(ledger_path)
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(
    bundle: ResolvingTheEpigeneticTimeScaleDisconnectConformationalSpinLockiSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "Resolving the Epigenetic Time-Scale Disconnect: Conformational Spin-Locking"
        + " Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Consumed source records: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Spec count: {bundle.summary['spec_count']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        f"- Next source boundary: {bundle.summary['next_source_boundary']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                f"### `{spec.key}`",
                "",
                spec.canonical_statement,
                "",
                f"- Context: `{spec.context_id}`",
                f"- Protocol: `{spec.validation_protocol}`",
                f"- Source equations: {', '.join(spec.source_equation_ids)}",
                f"- Null controls: {', '.join(spec.null_controls)}",
                "",
            ]
        )
    return "\n".join(lines)


def write_outputs(
    bundle: ResolvingTheEpigeneticTimeScaleDisconnectConformationalSpinLockiSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki_validation_specs_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-17")
    args = parser.parse_args()
    outputs = write_outputs(
        build_from_ledger(args.ledger), output_dir=args.output_dir, date_tag=args.date_tag
    )
    print(outputs["json"])
    print(outputs["report"])


if __name__ == "__main__":
    main()
