#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 L15 Reformulation: The SEC Objective Functional (Decision-Theoretic Form) (revision 11.00) spec builder
"""Promote Paper 0 L15 Reformulation: The SEC Objective Functional (Decision-Theoretic Form) (revision 11.00) records."""

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
    "P0R03981",
    "P0R03982",
    "P0R03983",
    "P0R03984",
    "P0R03985",
    "P0R03986",
    "P0R03987",
    "P0R03988",
    "P0R03989",
    "P0R03990",
    "P0R03991",
    "P0R03992",
    "P0R03993",
    "P0R03994",
    "P0R03995",
    "P0R03996",
    "P0R03997",
    "P0R03998",
    "P0R03999",
    "P0R04000",
)
CLAIM_BOUNDARY = "source-bounded l15 reformulation the sec objective functional decision theoretic form r source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r.l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r": {
        "context_id": "l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r",
        "validation_protocol": "paper0.l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r.l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r",
        "canonical_statement": "The source-bounded component 'L15 Reformulation: The SEC Objective Functional (Decision-Theoretic Form) (revision 11.00)' preserves Paper 0 records P0R03981-P0R03983 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03981:l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r",
            "P0R03982:l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r",
            "P0R03983:l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r",
        ),
        "source_formulae": (
            "P0R03981: L15 Reformulation: The SEC Objective Functional (Decision-Theoretic Form) (revision 11.00)",
            "P0R03982: This section marks a significant shift in the manuscript's formalism, moving from a physical analogy (Yang-Mills) to a more rigorous and testable decision-theoretic framework.",
            "P0R03983: Purpose. To avoid the category error of identifying a normative criterion with a specific gauge action, Layer 15 is henceforth formalised in decision-theoretic terms. We retain a variational flavour, but abandon Yang-Mills symbolism at this layer. Teleology remains explicit via Axiom 3 (the universe evolves to maximise Sustainable Ethical Coherence).",
        ),
        "test_protocols": (
            "preserve L15 Reformulation: The SEC Objective Functional (Decision-Theoretic Form) (revision 11.00) source-accounting boundary",
        ),
        "null_results": (
            "L15 Reformulation: The SEC Objective Functional (Decision-Theoretic Form) (revision 11.00) is not empirical validation evidence",
        ),
        "variables": ("l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r",),
        "validation_targets": ("preserve records P0R03981-P0R03983",),
        "null_controls": (
            "l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r must remain source-bounded accounting",
        ),
    },
    "l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r.definition_canonical_form": {
        "context_id": "definition_canonical_form",
        "validation_protocol": "paper0.l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r.definition_canonical_form",
        "canonical_statement": "The source-bounded component 'Definition (canonical form).' preserves Paper 0 records P0R03984-P0R04000 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03984:definition_canonical_form",
            "P0R03985:definition_canonical_form",
            "P0R03986:definition_canonical_form",
            "P0R03987:definition_canonical_form",
            "P0R03988:definition_canonical_form",
            "P0R03989:definition_canonical_form",
            "P0R03990:definition_canonical_form",
            "P0R03991:definition_canonical_form",
            "P0R03992:definition_canonical_form",
            "P0R03993:definition_canonical_form",
            "P0R03994:definition_canonical_form",
            "P0R03995:definition_canonical_form",
            "P0R03996:definition_canonical_form",
            "P0R03997:definition_canonical_form",
            "P0R03998:definition_canonical_form",
            "P0R03999:definition_canonical_form",
            "P0R04000:definition_canonical_form",
        ),
        "source_formulae": (
            "P0R03984: Definition (canonical form).",
            "P0R03985: Let S\\mathcal{S}S denote the joint state-space across Layers 1 141\\!-\\!14114 (including coarse-grained phase and informational coordinates), and A\\mathcal{A}A the admissible action set (interventions, boundary updates, policy controls) available to agents embedded within the architecture. A policy (as)\\pi(a\\mid s)(as) induces trajectories (st,at)t0(s_t,a_t)_{t\\ge 0}(st,at)t0.",
            "P0R03986: The SEC Objective Functional is",
            "P0R03987: $JSEC\\lbrack\\pi\\rbrack\\ \\mspace{2mu} = \\ \\mspace{2mu} E\\pi\\,\\left\\lbrack \\,\\sum_{}^{}t = 0TrSEC\\,(st,at)\\, \\right\\rbrack\\boxed{\\quad J_{\\text{SEC}}\\lbrack\\pi\\rbrack\\ = \\ E_{\\pi}\\text{!}\\left\\lbrack \\,\\sum_{t = 0}^{T}r_{\\text{SEC}}\\text{!}\\backslash big\\left( s_{t},a_{t\\backslash}big \\right)\\, \\right\\rbrack\\quad}JSEC\\lbrack\\pi\\rbrack = E\\pi\\left\\lbrack t = 0\\sum_{}^{}{T rSEC(st,at)} \\right\\rbrack$",
            "P0R03988: with horizon TTT finite or discounted infinite (common choice: discount 0<<10<\\gamma<10<<1, giving $E\\pi\\lbrack\\sum t \\geq 0\\gamma trSEC\\rbrack\\Finv o\\{ E\\}\\_\\{\\pi\\}\\lbrack\\sum\\_\\{ t \\geq 0\\}\\gamma\\hat{}\\{ t\\}\\ r\\_\\{\\Finv r\\{ SEC\\}\\}\\rbrack E\\pi\\lbrack\\sum t \\geq 0\\gamma trSEC\\rbrack).$",
            "P0R03989: The instantaneous SEC reward decomposes as",
            "P0R03990: $rSEC(s,a)\\ \\mspace{2mu} = \\ \\mspace{2mu} wC\\, C(s)\\ \\mspace{2mu} + \\ \\mspace{2mu} wK\\, K(s)\\ \\mspace{2mu} + \\ \\mspace{2mu} wQ\\, Q(s)\\ \\mspace{2mu} - \\ \\mspace{2mu}\\sum_{}^{}{i\\lambda i}\\,\\left\\lbrack \\, gi(s,a)\\, \\right\\rbrack + ,r_{\\text{SEC}}(s,a)\\ = \\ w_{C}\\, C(s)\\ + \\ w_{K}\\, K(s)\\ + \\ w_{Q}\\, Q(s)\\ - \\ \\sum_{i}^{}\\lambda_{i}\\,\\backslash big\\left\\lbrack \\, g_{i}(s,a)\\,\\backslash big \\right\\rbrack_{+},rSEC(s,a) = wC C(s) + wK K(s) + wQ Q(s) - i\\sum_{}^{}{\\lambda i\\left\\lbrack gi(s,a) \\right\\rbrack} + ,$",
            "P0R03991: where CCC (system-wide coherence), KKK (integrated complexity), and QQQ (qualia-capacity/richness) are the already-defined physical/informational observables, gi 0g_i\\!\\le 0gi0 are ethical/feasibility constraints (e.g., non-harm, stability, biospheric bounds), []+[\\cdot]_+[]+ denotes the positive part, and w,lambdaw_\\cdot,\\lambda_\\cdotw,lambda are layer-aware weights.",
            "P0R03992: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Reihe enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            'P0R03993: Fig.: Universe as a Policy Learner (SEC). This flowchart uses the analogy of a reinforcement learning loop to explain how the universe "learns" to evolve towards states of higher meaning and harmony. It avoids jargon and focuses on the cyclical process of observation, action, and optimization. This figure frames cosmic evolution as policy improvement under SEC, making the teleological thesis operational for simulations and manuscript exposition.',
            "P0R03994: The universe iteratively observes its state (C,K,Q)(C,K,Q)(C,K,Q), chooses an action via a policy \\pi, receives the SEC reward rSECr_{\\text{SEC}}rSEC, and updates \\pi to improve future outcomes. This loop converges toward an optimal policy \\pi^* that maximises total Sustainable Ethical Coherence JSECJ_{\\text{SEC}}JSEC, aligning cosmological evolution with ethical teleology.",
            "P0R03995: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Display enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R03996: Fig.: SEC as a Decision-Theoretic Program. This diagram provides a formal schematic of the L15 framework, mapping the components of a Partially Observable Markov Decision Process (POMDP) onto the cosmological scale. It is designed for readers familiar with control theory or reinforcement learning. This figure operationalises Axiom 3: SEC-guided evolution is formalised as policy optimisation over trajectories with ethical constraints and discounted rewards.",
            "P0R03997: A policy (a s)\\pi(a\\!\\mid\\!s)(as) maps states SSS (configs of L1-14) to actions AAA (dynamics/interventions), generating trajectories (st,at)(s_t,a_t)(st,at). Each step is scored by the instantaneous SEC reward rSECr_{\\mathrm{SEC}}rSEC, combining positive contributions from C (coherence), K (complexity), Q (qualia-capacity) and soft penalties ilambdai[gi]+\\sum_i \\lambda_i [g_i]_+ilambdai[gi]+ for constraint violations. The objective functional",
            "P0R03998: JSEC[] = E [tt rSEC(st,at)]J_{\\mathrm{SEC}}[\\pi] \\;=\\; \\mathbb{E}_{\\pi}\\!\\left[\\sum_t \\gamma^t\\, r_{\\mathrm{SEC}}(s_t,a_t)\\right]JSEC[]=E[ttrSEC(st,at)]",
            "P0R03999: # Python One-Liner: J_SEC = sp.E * sp.Sum(sp.Pow(gamma, t) * r_SEC(s_t, a_t), (t, 0, sp.oo))",
            "P0R04000: is maximised to obtain the operational teleological policy \\pi^*.",
        ),
        "test_protocols": ("preserve Definition (canonical form). source-accounting boundary",),
        "null_results": ("Definition (canonical form). is not empirical validation evidence",),
        "variables": ("definition_canonical_form",),
        "validation_targets": ("preserve records P0R03984-P0R04000",),
        "null_controls": ("definition_canonical_form must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class L15ReformulationTheSecObjectiveFunctionalDecisionTheoreticFormRSpec:
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
class L15ReformulationTheSecObjectiveFunctionalDecisionTheoreticFormRSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[L15ReformulationTheSecObjectiveFunctionalDecisionTheoreticFormRSpec, ...]
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


def build_l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_specs(
    source_records: list[dict[str, Any]],
) -> L15ReformulationTheSecObjectiveFunctionalDecisionTheoreticFormRSpecBundle:
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

    specs: list[L15ReformulationTheSecObjectiveFunctionalDecisionTheoreticFormRSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            L15ReformulationTheSecObjectiveFunctionalDecisionTheoreticFormRSpec(
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
        + "L15 Reformulation: The SEC Objective Functional (Decision-Theoretic Form) (revision 11.00)"
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
        "next_source_boundary": "P0R04001",
    }
    return L15ReformulationTheSecObjectiveFunctionalDecisionTheoreticFormRSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> L15ReformulationTheSecObjectiveFunctionalDecisionTheoreticFormRSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_specs(
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
    bundle: L15ReformulationTheSecObjectiveFunctionalDecisionTheoreticFormRSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "L15 Reformulation: The SEC Objective Functional (Decision-Theoretic Form) (revision 11.00)"
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
    bundle: L15ReformulationTheSecObjectiveFunctionalDecisionTheoreticFormRSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_validation_specs_{date_tag}.md"
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
