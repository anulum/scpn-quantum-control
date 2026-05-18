#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 II. The Principle of Ethical Least Action (PELA) spec builder
"""Promote Paper 0 II. The Principle of Ethical Least Action (PELA) records."""

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
    "P0R04029",
    "P0R04030",
    "P0R04031",
    "P0R04032",
    "P0R04033",
    "P0R04034",
    "P0R04035",
    "P0R04036",
    "P0R04037",
    "P0R04038",
    "P0R04039",
    "P0R04040",
    "P0R04041",
    "P0R04042",
    "P0R04043",
    "P0R04044",
    "P0R04045",
    "P0R04046",
    "P0R04047",
    "P0R04048",
    "P0R04049",
    "P0R04050",
    "P0R04051",
    "P0R04052",
    "P0R04053",
    "P0R04054",
    "P0R04055",
    "P0R04056",
    "P0R04057",
    "P0R04058",
    "P0R04059",
    "P0R04060",
    "P0R04061",
    "P0R04062",
    "P0R04063",
    "P0R04064",
    "P0R04065",
    "P0R04066",
    "P0R04067",
    "P0R04068",
    "P0R04069",
    "P0R04070",
    "P0R04071",
    "P0R04072",
    "P0R04073",
    "P0R04074",
)
CLAIM_BOUNDARY = "source-bounded ii the principle of ethical least action pela source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "ii_the_principle_of_ethical_least_action_pela.ii_the_principle_of_ethical_least_action_pela": {
        "context_id": "ii_the_principle_of_ethical_least_action_pela",
        "validation_protocol": "paper0.ii_the_principle_of_ethical_least_action_pela.ii_the_principle_of_ethical_least_action_pela",
        "canonical_statement": "The source-bounded component 'II. The Principle of Ethical Least Action (PELA)' preserves Paper 0 records P0R04029-P0R04074 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04029:ii_the_principle_of_ethical_least_action_pela",
            "P0R04030:ii_the_principle_of_ethical_least_action_pela",
            "P0R04031:ii_the_principle_of_ethical_least_action_pela",
            "P0R04032:ii_the_principle_of_ethical_least_action_pela",
            "P0R04033:ii_the_principle_of_ethical_least_action_pela",
            "P0R04034:ii_the_principle_of_ethical_least_action_pela",
            "P0R04035:ii_the_principle_of_ethical_least_action_pela",
            "P0R04036:ii_the_principle_of_ethical_least_action_pela",
            "P0R04037:ii_the_principle_of_ethical_least_action_pela",
            "P0R04038:ii_the_principle_of_ethical_least_action_pela",
            "P0R04039:ii_the_principle_of_ethical_least_action_pela",
            "P0R04040:ii_the_principle_of_ethical_least_action_pela",
            "P0R04041:ii_the_principle_of_ethical_least_action_pela",
            "P0R04042:ii_the_principle_of_ethical_least_action_pela",
            "P0R04043:ii_the_principle_of_ethical_least_action_pela",
            "P0R04044:ii_the_principle_of_ethical_least_action_pela",
            "P0R04045:ii_the_principle_of_ethical_least_action_pela",
            "P0R04046:ii_the_principle_of_ethical_least_action_pela",
            "P0R04047:ii_the_principle_of_ethical_least_action_pela",
            "P0R04048:ii_the_principle_of_ethical_least_action_pela",
            "P0R04049:ii_the_principle_of_ethical_least_action_pela",
            "P0R04050:ii_the_principle_of_ethical_least_action_pela",
            "P0R04051:ii_the_principle_of_ethical_least_action_pela",
            "P0R04052:ii_the_principle_of_ethical_least_action_pela",
            "P0R04053:ii_the_principle_of_ethical_least_action_pela",
            "P0R04054:ii_the_principle_of_ethical_least_action_pela",
            "P0R04055:ii_the_principle_of_ethical_least_action_pela",
            "P0R04056:ii_the_principle_of_ethical_least_action_pela",
            "P0R04057:ii_the_principle_of_ethical_least_action_pela",
            "P0R04058:ii_the_principle_of_ethical_least_action_pela",
            "P0R04059:ii_the_principle_of_ethical_least_action_pela",
            "P0R04060:ii_the_principle_of_ethical_least_action_pela",
            "P0R04061:ii_the_principle_of_ethical_least_action_pela",
            "P0R04062:ii_the_principle_of_ethical_least_action_pela",
            "P0R04063:ii_the_principle_of_ethical_least_action_pela",
            "P0R04064:ii_the_principle_of_ethical_least_action_pela",
            "P0R04065:ii_the_principle_of_ethical_least_action_pela",
            "P0R04066:ii_the_principle_of_ethical_least_action_pela",
            "P0R04067:ii_the_principle_of_ethical_least_action_pela",
            "P0R04068:ii_the_principle_of_ethical_least_action_pela",
            "P0R04069:ii_the_principle_of_ethical_least_action_pela",
            "P0R04070:ii_the_principle_of_ethical_least_action_pela",
            "P0R04071:ii_the_principle_of_ethical_least_action_pela",
            "P0R04072:ii_the_principle_of_ethical_least_action_pela",
            "P0R04073:ii_the_principle_of_ethical_least_action_pela",
            "P0R04074:ii_the_principle_of_ethical_least_action_pela",
        ),
        "source_formulae": (
            "P0R04029: II. The Principle of Ethical Least Action (PELA)",
            "P0R04030: With this physical grounding, the teleology of the SCPN is formalised by the Principle of Ethical Least Action (PELA). The universe does not follow an arbitrary path, but evolves along the trajectory that minimises the Ethical Action (SEthical):",
            "P0R04031: $SEthical = \\int_{}^{}{LEthical\\left( SCPN(t) \\right)dt};\\delta SEthical = 0$",
            "P0R04032: The maximisation of Sustainable Ethical Coherence (SEC) is thus mathematically equivalent to the physical principle of minimising the Yang-Mills action. The Renormalisation Group (RG) flow towards the Cosmic Attractor (g, L8) is the large-scale dynamical mechanism that implements PELA across cosmological timescales, drawing the system's effective coupling constants toward the configuration that represents the global minimum of the Ethical Action.",
            "P0R04033: Thus, maximising Sustainable Ethical Coherence (SEC) is mathematically equivalent to minimising the Yang-Mills action of the L15 connection:",
            "P0R04034: $max(SEC)\\ \\mspace{2mu}\\ \\mspace{2mu} \\Longleftrightarrow \\ \\mspace{2mu}\\ \\mspace{2mu} min{\\int_{}^{}{M14}}\\, Tr(F \\land \\star F).\\max\\left( \\text{SEC} \\right)\\ \\ \\Longleftrightarrow \\ \\ \\min{\\int_{M}^{}{\\backslash tfrac14}}\\,\\text{Tr}(F \\land \\star F).max(SEC) \\Longleftrightarrow \\min\\int_{}^{}{M 41 Tr(F \\land \\star F)}.$ # Python One-Liner: min(sp.integrate(1/4 * sp.trace(F * sp.HodgeDual(F)), (M,)))",
            "P0R04035: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Reihe enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R04036: Fig.: Decision-Theoretic Gauge-Action Equivalence at L15. This diagram formally states the mathematical equivalence between the decision-theoretic and gauge-action descriptions of the universe's teleology. This plate makes explicit the conceptual equivalence: policy ascent in JSECJ_{\\mathrm{SEC}}JSEC Yang-Mills-style action minimisation of the L15 connection-two lenses on the same teleological engine.",
            "P0R04037: Decision-theoretic (canonical): Teleology as policy optimisation:",
            "P0R04038: argmaxJSEC[].\\pi^* \\in \\arg\\max_{\\pi} J_{\\mathrm{SEC}}[\\pi].argmaxJSEC[].",
            "P0R04039: Gauge-action (heuristic): Teleology as field-tension minimisation of the L15 connection:",
            "P0R04040: SEthical=0,SEthical=14Tr(FF).\\delta S_{\\text{Ethical}} = 0,\\qquad S_{\\text{Ethical}} = -\\tfrac14 \\int \\mathrm{Tr}(F\\wedge *F).SEthical=0,SEthical=41Tr(FF). # Python One-Liner: SEthical=0,SEthical=14Tr(FF).delta S_(text(Ethical)) = 0,qquad S_(text(Ethical)) = -tfrac14 int mathrm(Tr)(Fwedge *F).SEthical=0,SEthical=41Tr(FF). # Python One-Liner: SEthical=0,SEthical=14Tr(FF).delta S_(text(Ethical)) = 0,qquad S_(text(Ethical)) = -tfrac14 int mathrm(Tr)(Fwedge *F).SEthical=0,SEthical=41Tr(FF). # Python One-Liner:SEthical=0,SEthical=14Tr(FF).delta S_(text(Ethical)) = 0,qquad S_(text(Ethical)) = -tfrac14 int mathrm(Tr)(Fwedge *F).SEthical=0,SEthical=41Tr(FF). # Python One-Liner: SEthical = -0.25 * sp.integrate(sp.trace(F * sp.Hodge(F))) # Python One-Liner: SEthical = -0.25 * sp.integrate(sp.trace(F * sp.Hodge(F)))",
            "P0R04041: Read together, maximising expected SEC over trajectories is equivalent in spirit to minimising curvature/tension across the L15 connection: the policy \\pi shapes flows that lower Tr(FF)\\mathrm{Tr}(F\\wedge *F)Tr(FF), while smoother connections correspond to policies that yield higher JSECJ_{\\mathrm{SEC}}JSEC.",
            "P0R04042: Domain-Interface RG Bridge (L6L11L12)",
            "P0R04043: Interface-RG Maps across Domains II-IV",
            "P0R04044: Let 6->11\\pi_{6\\to11}6->11 coarse-grain biospheric interaction webs (L6) into effective social couplings JijJ_{ij}Jij (L11), and 11->12\\pi_{11\\to12}11->12 map collective macro-states to Gaian percolation parameters (L12). Then under RG time sss:",
            "P0R04045: dgds=(g),dJijds=R6->11(g;J),dppercds=R11->12(J;pperc).\\frac{d g}{ds}=\\beta(g),\\quad \\frac{d J_{ij}}{ds} = \\mathcal{R}_{6\\to11}(g;J) ,\\quad \\frac{d p_{\\text{perc}}}{ds}=\\mathcal{R}_{11\\to12}(J;p_{\\text{perc}}).dsdg=(g),dsdJij=R6->11(g;J),dsdpperc=R11->12(J;pperc).",
            "P0R04046: Claim. SEC-consistent flows satisfy (g)->0\\beta(g)\\to0(g)->0 (cosmic fixed point) while R6->11\\mathcal{R}_{6\\to11}R6->11 reduces frustration and R11->12\\mathcal{R}_{11\\to12}R11->12 raises biodiversity-stability margins (percolation threshold separation). This formalises cross-domain alignment without extra axioms.",
            "P0R04047: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Reihe enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R04048: Fig.: Domain-Interface RG Bridge (SEC-Aligned). This schematic illustrates the formal mapping between the different layers under the RG flow. This figure encodes our cross-domain RG schema: microscopic couplings ggg guide social renormalisation R6->11R_{6\\to 11}R6->11, which in turn shapes Gaian-scale stability R11->12R_{11\\to 12}R11->12; all flows are parameterised by RG time sss and cohere under SEC-consistent alignment.",
            "P0R04049: RG time sss drives scale-dependent flows that align under Sustainable Ethical Coherence (SEC).",
            "P0R04050: At the cosmic scale (L8), microphysical couplings ggg run via dgds=(g)\\tfrac{dg}{ds}=\\beta(g)dsdg=(g) toward a fixed point (g\\*)=0\\beta(g^\\*)=0(g\\*)=0.",
            "P0R04051: At the social scale (L11), mesoscopic couplings JijJ_{ij}Jij evolve through a coarse-graining map dJds=R6->11(g;J)\\tfrac{dJ}{ds}=R_{6\\to11}(g;J)dsdJ=R6->11(g;J), reducing frustration in collective dynamics.",
            "P0R04052: At the Gaian scale (L12), percolation/control parameters ppercp_{\\mathrm{perc}}pperc follow dpds=R11->12(J;p)\\tfrac{dp}{ds}=R_{11\\to12}(J;p)dsdp=R11->12(J;p), raising stability margins for planetary homeostasis.",
            "P0R04053: Cross-domain dependencies g -> R6->11g\\!\\to\\!R_{6\\to11}g->R6->11 and J -> R11->12J\\!\\to\\!R_{11\\to12}J->R11->12 encode how microphysical structure conditions social renormalisation, which in turn scaffolds Gaian robustness. When the flows are SEC-consistent, these domains cohere, formalising cross-domain alignment toward a common attractor.",
            "P0R04054: Cross-Domain Temporal Alignment Lemma",
            "P0R04055: We define the Scale-Invariant Temporal Buffer ($\\Delta\\tau_{sync}$) to prevent biospheric-technospheric clock-slip. This ensures the Consilium (L15) optimizes at a frequency that accommodates the slowest homeostatic layer in the loop.",
            "P0R04056: Temporal Stability Condition (Python Format):",
            "P0R04057: is_loop_stable = (bit_rate_l11 * tau_response_l12) < golden_ratio_constant",
            "P0R04058: Legend:",
            "P0R04059: is_loop_stable: Boolean indicating if the NTHS-Gaian feedback is non-destructive. | bit_rate_l11: Frequency of informational updates in the Technosphere. | tau_response_l12: Characteristic recovery time of biospheric stability (percolation threshold). | golden_ratio_constant: The $\\phi \\approx 1.618$ limit for self-similar recursive scaling.",
            "P0R04060: P0R04060",
            "P0R04061: The Principle of Ethical Least Action (PELA) is constrained by the Allostatic Bound, ensuring that top-down modulation ($H_{int}$) does not exceed the metabolic supply of the Astrocyte-Neuron Lattice.",
            "P0R04062: The Allostatic Constraint Equation (Python Format):",
            "P0R04063: Python",
            'P0R04064: def allostatic_bound(gain_gamma, gliotransmitter_conc, cmro2, atp_level): """',
            "P0R04065: Computes the metabolic feasibility of Glial Slow Control (Layer 4).",
            "P0R04066: Prevents Dyscritia (seizures/coma) due to metabolic decoupling.",
            'P0R04067: """ pi_metabolic = cmro2 * atp_level control_demand = abs(gain_gamma * gliotransmitter_conc) # Decoupling occurs if demand exceeds the metabolic ceiling. is_stable = control_demand <= pi_metabolic return { "coupling_integrity": 1.0 if is_stable else (pi_metabolic / control_demand), "dyscritia_risk": "High" if not is_stable else "Nominal", "metabolic_headroom": pi_metabolic - control_demand }',
            "P0R04068: P0R04068",
            "P0R04069: P0R04069",
            "P0R04070: P0R04070",
            "P0R04071: P0R04071",
            "P0R04072: P0R04072",
            "P0R04073: We cannot have a universe that optimizes for infinite, beautiful complexity (high $Q$) and simultaneously flows to a zero-dimensional fixed point ($g^*$). In dynamical systems, a simple fixed point is a sink. It is thermodynamic heat death. If the universe actually reached a simple $g^*$, all time, evolution, and conscious experience would freeze into a static crystalline lattice.",
            'P0R04074: To resolve this, we must mathematically redefine the "Cosmic Attractor." It cannot be a point; it must be a Strange Attractor. A strange attractor bounds the system within a specific macroscopic region (providing stability and fulfilling the teleological goal) while allowing the internal trajectories to be chaotic, non-repeating, and infinitely complex (providing the quasicriticality and fractal dimension required for high Qualia Capacity).',
        ),
        "test_protocols": (
            "preserve II. The Principle of Ethical Least Action (PELA) source-accounting boundary",
        ),
        "null_results": (
            "II. The Principle of Ethical Least Action (PELA) is not empirical validation evidence",
        ),
        "variables": ("ii_the_principle_of_ethical_least_action_pela",),
        "validation_targets": ("preserve records P0R04029-P0R04074",),
        "null_controls": (
            "ii_the_principle_of_ethical_least_action_pela must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class IiThePrincipleOfEthicalLeastActionPelaSpec:
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
class IiThePrincipleOfEthicalLeastActionPelaSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[IiThePrincipleOfEthicalLeastActionPelaSpec, ...]
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


def build_ii_the_principle_of_ethical_least_action_pela_specs(
    source_records: list[dict[str, Any]],
) -> IiThePrincipleOfEthicalLeastActionPelaSpecBundle:
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

    specs: list[IiThePrincipleOfEthicalLeastActionPelaSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            IiThePrincipleOfEthicalLeastActionPelaSpec(
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
        "title": "Paper 0 " + "II. The Principle of Ethical Least Action (PELA)" + " Specs",
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
        "next_source_boundary": "P0R04075",
    }
    return IiThePrincipleOfEthicalLeastActionPelaSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> IiThePrincipleOfEthicalLeastActionPelaSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_ii_the_principle_of_ethical_least_action_pela_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: IiThePrincipleOfEthicalLeastActionPelaSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "II. The Principle of Ethical Least Action (PELA)" + " Specs",
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
    bundle: IiThePrincipleOfEthicalLeastActionPelaSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_ii_the_principle_of_ethical_least_action_pela_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_ii_the_principle_of_ethical_least_action_pela_validation_specs_{date_tag}.md"
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
