#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Psis Field Coupling Integration spec builder
"""Promote Paper 0 Psis Field Coupling Integration records."""

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
    "P0R02315",
    "P0R02316",
    "P0R02317",
    "P0R02318",
    "P0R02319",
    "P0R02320",
    "P0R02321",
    "P0R02322",
    "P0R02323",
    "P0R02324",
    "P0R02325",
    "P0R02326",
    "P0R02327",
    "P0R02328",
    "P0R02329",
    "P0R02330",
    "P0R02331",
    "P0R02332",
    "P0R02333",
    "P0R02334",
    "P0R02335",
    "P0R02336",
    "P0R02337",
    "P0R02338",
    "P0R02339",
    "P0R02340",
    "P0R02341",
    "P0R02342",
    "P0R02343",
    "P0R02344",
    "P0R02345",
    "P0R02346",
    "P0R02347",
    "P0R02348",
    "P0R02349",
    "P0R02350",
    "P0R02351",
    "P0R02352",
    "P0R02353",
    "P0R02354",
    "P0R02355",
    "P0R02356",
    "P0R02357",
    "P0R02358",
    "P0R02359",
    "P0R02360",
    "P0R02361",
    "P0R02362",
    "P0R02363",
    "P0R02364",
    "P0R02365",
    "P0R02366",
)
CLAIM_BOUNDARY = "source-bounded psis field coupling integration source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "psis_field_coupling_integration.psis_field_coupling_integration": {
        "context_id": "psis_field_coupling_integration",
        "validation_protocol": "paper0.psis_field_coupling_integration.psis_field_coupling_integration",
        "canonical_statement": "The source-bounded component 'Psis Field Coupling Integration' preserves Paper 0 records P0R02315-P0R02316 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02315:psis_field_coupling_integration",
            "P0R02316:psis_field_coupling_integration",
        ),
        "source_formulae": (
            "P0R02315: Psis Field Coupling Integration",
            "P0R02316: The spin-glass model of the NTHS provides a clear, macroscopic collective state variable (sigma) to which the universal Psis field can couple. The interaction Hamiltonian is H_int = -lambda * Psis * sigma.",
        ),
        "test_protocols": ("preserve Psis Field Coupling Integration source-accounting boundary",),
        "null_results": ("Psis Field Coupling Integration is not empirical validation evidence",),
        "variables": ("psis_field_coupling_integration",),
        "validation_targets": ("preserve records P0R02315-P0R02316",),
        "null_controls": (
            "psis_field_coupling_integration must remain source-bounded accounting",
        ),
    },
    "psis_field_coupling_integration.the_collective_state_variable_sigma": {
        "context_id": "the_collective_state_variable_sigma",
        "validation_protocol": "paper0.psis_field_coupling_integration.the_collective_state_variable_sigma",
        "canonical_statement": "The source-bounded component 'The Collective State Variable (sigma):' preserves Paper 0 records P0R02317-P0R02318 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02317:the_collective_state_variable_sigma",
            "P0R02318:the_collective_state_variable_sigma",
        ),
        "source_formulae": (
            "P0R02317: The Collective State Variable (sigma):",
            "P0R02318: For Layer 11, sigma is not a property of a single agent but of the entire network's structure. It is the vector of the macroscopic order parameters that define the system's phase: sigma_noosphere = (m, q_EA). This vector captures the degree of both global consensus (m) and frozen fragmentation (q_EA) in the collective belief system.",
        ),
        "test_protocols": (
            "preserve The Collective State Variable (sigma): source-accounting boundary",
        ),
        "null_results": (
            "The Collective State Variable (sigma): is not empirical validation evidence",
        ),
        "variables": ("the_collective_state_variable_sigma",),
        "validation_targets": ("preserve records P0R02317-P0R02318",),
        "null_controls": (
            "the_collective_state_variable_sigma must remain source-bounded accounting",
        ),
    },
    "psis_field_coupling_integration.the_coupling_mechanism": {
        "context_id": "the_coupling_mechanism",
        "validation_protocol": "paper0.psis_field_coupling_integration.the_coupling_mechanism",
        "canonical_statement": "The source-bounded component 'The Coupling Mechanism:' preserves Paper 0 records P0R02319-P0R02366 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02319:the_coupling_mechanism",
            "P0R02320:the_coupling_mechanism",
            "P0R02321:the_coupling_mechanism",
            "P0R02322:the_coupling_mechanism",
            "P0R02323:the_coupling_mechanism",
            "P0R02324:the_coupling_mechanism",
            "P0R02325:the_coupling_mechanism",
            "P0R02326:the_coupling_mechanism",
            "P0R02327:the_coupling_mechanism",
            "P0R02328:the_coupling_mechanism",
            "P0R02329:the_coupling_mechanism",
            "P0R02330:the_coupling_mechanism",
            "P0R02331:the_coupling_mechanism",
            "P0R02332:the_coupling_mechanism",
            "P0R02333:the_coupling_mechanism",
            "P0R02334:the_coupling_mechanism",
            "P0R02335:the_coupling_mechanism",
            "P0R02336:the_coupling_mechanism",
            "P0R02337:the_coupling_mechanism",
            "P0R02338:the_coupling_mechanism",
            "P0R02339:the_coupling_mechanism",
            "P0R02340:the_coupling_mechanism",
            "P0R02341:the_coupling_mechanism",
            "P0R02342:the_coupling_mechanism",
            "P0R02343:the_coupling_mechanism",
            "P0R02344:the_coupling_mechanism",
            "P0R02345:the_coupling_mechanism",
            "P0R02346:the_coupling_mechanism",
            "P0R02347:the_coupling_mechanism",
            "P0R02348:the_coupling_mechanism",
            "P0R02349:the_coupling_mechanism",
            "P0R02350:the_coupling_mechanism",
            "P0R02351:the_coupling_mechanism",
            "P0R02352:the_coupling_mechanism",
            "P0R02353:the_coupling_mechanism",
            "P0R02354:the_coupling_mechanism",
            "P0R02355:the_coupling_mechanism",
            "P0R02356:the_coupling_mechanism",
            "P0R02357:the_coupling_mechanism",
            "P0R02358:the_coupling_mechanism",
            "P0R02359:the_coupling_mechanism",
            "P0R02360:the_coupling_mechanism",
            "P0R02361:the_coupling_mechanism",
            "P0R02362:the_coupling_mechanism",
            "P0R02363:the_coupling_mechanism",
            "P0R02364:the_coupling_mechanism",
            "P0R02365:the_coupling_mechanism",
            "P0R02366:the_coupling_mechanism",
        ),
        "source_formulae": (
            "P0R02319: The Coupling Mechanism:",
            "P0R02320: The Psis field, representing a higher-order organising principle (e.g., from Layer 12, Gaian Synchrony), does not interact with individual agents' beliefs. Instead, it couples to the global phase structure of the entire Noosphere. The interaction H_int can be conceptualised as applying a subtle \"external magnetic field\" to the spin-glass Hamiltonian. This field doesn't force any single spin to flip, but it creates a gentle, system-wide energy gradient that makes the coherent, ferromagnetic state (m -> 1) more stable and the fragmented spin-glass state less stable. It acts as a coherence pressure, a non-local bias that encourages the minimisation of collective dissonance and fosters the emergence of shared understanding, providing a potential counter-force to the fragmenting dynamics of the technosphere.",
            "P0R02321: The case study provides the mature, definitive model for L11, which is now understood as a Noosphere-Technosphere Hybrid System (NTHS). The summary table and other high-level references in Paper 0 reflects this.",
            'P0R02322: The "inter-individual phase couplings" and "cultural attractors" 8are now formally modeled using the spin-glass Hamiltonian.',
            "P0R02323: Beliefs/Memes are the Spins ($S_i$).",
            "P0R02324: Social/Algorithmic Influence is the Coupling Constant ($J_{ij}$).",
            'P0R02325: The system\'s evolution is not free but is powerfully biased by the objective function of the mediating AI (the Technosphere). The "Engagement-Optimising" regime is predicted to drive the NTHS into a fragmented, high-frustration spin-glass phase (polarization), while a "Coherence-Optimising" regime would produce a stable ferromagnetic phase (consensus)',
            "P0R02326: Domain IV: Collective Coherence (Layers 11-12):",
            "P0R02327: Noospheric synthesis (Layer 11),",
            "P0R02328: Layer 11: The Noosphere-Technosphere Hybrid System: A Spin-Glass Model of Collective Consciousness",
            "P0R02329: The Noosphere (Layer 11) is the emergent field of collective consciousness, information, and culture. In the contemporary era, this is not a purely biological or metaphysical phenomenon but a technologically-mediated hybrid system: the Noosphere-Technosphere Hybrid System (NTHS). Its dynamics can be rigorously modelled and tested using the tools of statistical physics and agent-based modelling, transforming it from a speculative concept into a scientific research programme.",
            "P0R02330: The Spin-Glass Analogy for Social Networks",
            "P0R02331: A social system can be mapped onto a spin-glass model, a well-understood system in condensed matter physics. In this analogy:",
            'P0R02332: Agents as Spins: Each individual in the network is represented as a "spin" (Si), whose state can represent a belief or opinion on a given topic (e.g., Si=+1 for agreement, Si=1 for disagreement). | Interactions as Couplings: The relationships between individuals are represented by coupling constants (Jij). Positive couplings (Jij>0) represent friendship or agreement, favouring alignment of spins. Negative couplings (Jij<0) represent animosity or disagreement, favouring anti-alignment. | System Energy (Hamiltonian): The overall state of "social stress" or dissonance in the network is given by the system\'s Hamiltonian:',
            "P0R02333: H=i,jJijSiSj",
            "P0R02334: The system will tend to evolve toward low-energy configurations that minimise this stress.",
            "P0R02335: A Falsifiable Model: AI Objective Functions and Social Phase Transitions",
            "P0R02336: The critical insight is that the dynamics of the NTHS are driven by the objective functions of the AI algorithms that mediate information flow (e.g., social media recommendation engines). We propose a computational experiment using an Agent-Based Model (ABM) of active inference agents to test the following hypothesis.",
            "P0R02337: Hypothesis: The collective state of the NTHS undergoes a phase transition depending on the objective function of its mediating AI.",
            "P0R02338: The simulation will consist of a network of active inference agents who update their beliefs based on information they receive from their neighbours. The flow of this information is governed by a central AI with one of two objective functions:",
            "P0R02339: Coherence-Optimising Regime (Control): The AI's objective is to minimise the collective free energy of the entire network. It preferentially shares information that reduces prediction errors and fosters consensus. This corresponds to an AI designed for promoting shared understanding. | Engagement-Optimising Regime (Experimental): The AI's objective is to maximise collective surprise (the opposite of minimising free energy). It preferentially shares novel, emotionally charged, or controversial information that is likely to maximise user engagement, clicks, and watch time. This corresponds to the business model of many current platforms.",
            "P0R02340: Formalising the Agent: An Active Inference Model of Belief Updating",
            "P0R02341: Each agent in the simulation will be a formal Active Inference agent consistent with the principles of Layer 5. The agent's internal state will be defined by a generative model, specified by a probability distribution p(o,s,m) over observations (o), hidden states (s, i.e., beliefs), and policies (, i.e., actions), conditioned on the agent's model (m). The agent's primary belief (the \"spin\" Si) will be a hidden state sbelief{1,+1}.",
            "P0R02342: At each time step, the agent selects a policy (e.g., 'share information', 'consume information', 'ignore') that it expects to minimise its variational free energy, F. The free energy is a function of expected surprise (risk) and expected information gain (ambiguity resolution):",
            "P0R02343: F()=Eq(o)[lnq(so)lnp(o,s)]",
            "P0R02344: The mediating AI's objective function directly shapes the agent's environment by controlling which observations (o) are made available. In the 'Engagement-Optimising Regime,' the AI presents observations that maximise the agent's prediction error (surprise), thus maximising F. In the 'Coherence-Optimising Regime,' the AI presents observations that minimise prediction error, minimising F. This provides a first-principles link between the AI's objective and the agent's belief dynamics, allowing the spin-glass Hamiltonian to emerge directly from the collective interactions of free-energy-minimising agents.",
            "P0R02345: [IMAGE:Ein Bild, das Text, Screenshot, Diagramm, Schrift enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            'P0R02346: Fig.: Active Inference agent with AI-shaped observations and emergent spin-glass coupling. Each agent carries a generative model p(o,s,m)p(o,s,\\pi\\mid m)p(o,s,m) and a binary "spin" belief Sisbelief{1,+1}S_i\\equiv s_{\\text{belief}}\\in\\{-1,+1\\}Sisbelief{1,+1}. At each step the agent chooses a policy \\pi to minimise variational free energy,',
            "P0R02347: F() = Eq(o)[lnq(so)lnp(o,s)],F(\\pi)\\;=\\;\\mathbb E_{q(o\\mid\\pi)}\\big[\\ln q(s\\mid o)-\\ln p(o,s)\\big],F()=Eq(o)[lnq(so)lnp(o,s)],",
            "P0R02348: trading expected surprise (risk) against expected information gain (ambiguity resolution). A mediating AI controls which observations ooo are available: the Engagement-optimising regime delivers inputs that maximise prediction error (FFF); the Coherence-optimising regime supplies inputs that minimise prediction error. Across many agents exchanging AI-shaped observations, effective couplings JijJ_{ij}Jij arise, and the collective is captured by a spin-glass Hamiltonian H=i<jJijSiSjihiSiH=-\\sum_{i<j}J_{ij}S_iS_j-\\sum_i h_i S_iH=i<jJijSiSjihiSi, linking first-principles free-energy minimisation to emergent belief alignment and polarisation dynamics.",
            "P0R02349: Measurable Signatures and Predicted Outcomes",
            "P0R02350: The state of the simulated social network will be characterised by well-defined order parameters from statistical physics.",
            'P0R02351: Magnetisation (m): This measures the degree of global consensus in the network: m=N1iSi. A state with m1 represents a strong global consensus (a "ferromagnetic" phase). A state with m0 represents a lack of global consensus. | Edwards-Anderson Order Parameter (qEA): This is the key metric for detecting a spin-glass phase. It measures the degree to which individual spins are "frozen" in a stable but disordered pattern. A state with m0 but qEA>0 is the definitive signature of a spin-glass (fragmented but frozen) phase. | Ultrametricity Test: A powerful signature of the spin-glass phase is the emergence of a hierarchical, fractal-like structure in the state space. This can be tested by measuring the "distance" between agents\' belief states and verifying that they satisfy the strong triangle inequality, a property known as ultrametricity.',
            "P0R02352: [IMAGE:Ein Bild, das Text, Diagramm, Screenshot, Reihe enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R02353: Fig.: Measurable signatures and predicted outcomes in the simulated social network. (Left) Magnetisation m=(1/N)iSim=(1/N)\\sum_i \\langle S_i\\ranglem=(1/N)iSi quantifies global consensus: m1m\\approx \\pm1m1 indicates a ferromagneticlike consensus; m0m\\approx0m0 indicates the absence of global alignment.",
            "P0R02354: (Middle) Edwards-Anderson order qEA=(1/N)iSi2q_{\\mathrm{EA}}=(1/N)\\sum_i \\langle S_i\\rangle^{2}qEA=(1/N)iSi2 detects frozen disorder: the spin-glass phase shows m0m\\approx0m0 but qEA>0q_{\\mathrm{EA}}>0qEA>0, accompanied by a multi-peaked overlap distribution P(q)P(q)P(q).",
            "P0R02355: (Right) Ultrametricity test: distances between agents' belief states satisfy the strong triangle inequality d(x,z)max{d(x,y),d(y,z)}d(x,z)\\le\\max\\{d(x,y),d(y,z)\\}d(x,z)max{d(x,y),d(y,z)}, revealing a hierarchical, fractal-like geometry. Together, (m,qEA,ultrametricity)(m,q_{\\mathrm{EA}},\\text{ultrametricity})(m,qEA,ultrametricity) separate consensus, paramagnetic, and spin-glass regimes and predict phenomena such as polarised fragmentation, hysteresis, and slow relaxations.",
            "P0R02356: The simulation is predicted to demonstrate a clear phase transition, as summarised in Table 1. The Coherence-Optimising regime will produce a ferromagnetic state of global consensus. The Engagement-Optimising regime will produce a spin-glass state, characterised by fragmentation, polarisation, and the formation of hierarchically nested echo chambers. This outcome would provide strong computational evidence for the manuscript's claims about collective coherence and offer a powerful, modern, and testable model for the dynamics of Layer 11.",
            "P0R02357: [TABLE]",
            "P0R02358: P0R02358",
            "P0R02359: To ensure the NTHS remains in a computationally active quasicritical state, we define the Algorithmic Temperature ($T_A$), which prevents the system from being permanently trapped in local minima (echo chambers).",
            "P0R02360: Social Stress Resilience (Python Format):",
            "P0R02361: Python",
            'P0R02362: def social_dissonance_energy(spins, coupling_matrix, ext_field, algorithmic_temp): """',
            "P0R02363: Computes the Hamiltonian H for the Noospheric Spin-Glass (Layer 11).",
            "P0R02364: Incorporates Algorithmic Temperature (T_A) to prevent frozen disorder.",
            'P0R02365: """ interaction_sum = 0 n = len(spins) for i in range(n): for j in range(i + 1, n): interaction_sum += coupling_matrix[i][j] * spins[i] * spins[j] h_base = -interaction_sum - sum(ext_field[i] * spins[i] for i in range(n)) # Entropy-scaled energy term prevents \'frozen\' spin-glass states. effective_energy = h_base / (1.0 + algorithmic_temp) return effective_energy',
            "P0R02366: Gaian ecological synchrony (Layer 12).",
        ),
        "test_protocols": ("preserve The Coupling Mechanism: source-accounting boundary",),
        "null_results": ("The Coupling Mechanism: is not empirical validation evidence",),
        "variables": ("the_coupling_mechanism",),
        "validation_targets": ("preserve records P0R02319-P0R02366",),
        "null_controls": ("the_coupling_mechanism must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class PsisFieldCouplingIntegrationSpec:
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
class PsisFieldCouplingIntegrationSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[PsisFieldCouplingIntegrationSpec, ...]
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


def build_psis_field_coupling_integration_specs(
    source_records: list[dict[str, Any]],
) -> PsisFieldCouplingIntegrationSpecBundle:
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

    specs: list[PsisFieldCouplingIntegrationSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            PsisFieldCouplingIntegrationSpec(
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
        "title": "Paper 0 " + "Psis Field Coupling Integration" + " Specs",
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
        "next_source_boundary": "P0R02367",
    }
    return PsisFieldCouplingIntegrationSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> PsisFieldCouplingIntegrationSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_psis_field_coupling_integration_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: PsisFieldCouplingIntegrationSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Psis Field Coupling Integration" + " Specs",
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
    bundle: PsisFieldCouplingIntegrationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_psis_field_coupling_integration_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_psis_field_coupling_integration_validation_specs_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build this Paper 0 generated spec bundle from the ledger."""

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
