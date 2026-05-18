#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Notes on correspondence (non-obligatory analogues). spec builder
"""Promote Paper 0 Notes on correspondence (non-obligatory analogues). records."""

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
    "P0R04009",
    "P0R04010",
    "P0R04011",
    "P0R04012",
    "P0R04013",
    "P0R04014",
    "P0R04015",
    "P0R04016",
    "P0R04017",
    "P0R04018",
    "P0R04019",
    "P0R04020",
    "P0R04021",
    "P0R04022",
    "P0R04023",
    "P0R04024",
    "P0R04025",
    "P0R04026",
    "P0R04027",
    "P0R04028",
)
CLAIM_BOUNDARY = "source-bounded notes on correspondence non obligatory analogues source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "notes_on_correspondence_non_obligatory_analogues.notes_on_correspondence_non_obligatory_analogues": {
        "context_id": "notes_on_correspondence_non_obligatory_analogues",
        "validation_protocol": "paper0.notes_on_correspondence_non_obligatory_analogues.notes_on_correspondence_non_obligatory_analogues",
        "canonical_statement": "The source-bounded component 'Notes on correspondence (non-obligatory analogues).' preserves Paper 0 records P0R04009-P0R04010 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04009:notes_on_correspondence_non_obligatory_analogues",
            "P0R04010:notes_on_correspondence_non_obligatory_analogues",
        ),
        "source_formulae": (
            "P0R04009: Notes on correspondence (non-obligatory analogues).",
            "P0R04010: Earlier drafts employed a Yang-Mills-like action as an analogy-class regulariser over high-level connections. That construction may be retained as an optional surrogate loss or prior, but it is not the definition of L15 and carries no ontological load. | Links to information-geometric or free-energy formalisms persist via the choice of observables C,K,QC,K,QC,K,Q and penalties gig_igi, not via curvature terms in an L15 field strength. Any such auxiliary terms must be interpreted as regularisers for learning dynamics, not as fundamental ethical physics.",
        ),
        "test_protocols": (
            "preserve Notes on correspondence (non-obligatory analogues). source-accounting boundary",
        ),
        "null_results": (
            "Notes on correspondence (non-obligatory analogues). is not empirical validation evidence",
        ),
        "variables": ("notes_on_correspondence_non_obligatory_analogues",),
        "validation_targets": ("preserve records P0R04009-P0R04010",),
        "null_controls": (
            "notes_on_correspondence_non_obligatory_analogues must remain source-bounded accounting",
        ),
    },
    "notes_on_correspondence_non_obligatory_analogues.terminology_and_notation_effective_immediately_revision_11_00": {
        "context_id": "terminology_and_notation_effective_immediately_revision_11_00",
        "validation_protocol": "paper0.notes_on_correspondence_non_obligatory_analogues.terminology_and_notation_effective_immediately_revision_11_00",
        "canonical_statement": "The source-bounded component 'Terminology and notation (effective immediately - revision 11.00).' preserves Paper 0 records P0R04011-P0R04012 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04011:terminology_and_notation_effective_immediately_revision_11_00",
            "P0R04012:terminology_and_notation_effective_immediately_revision_11_00",
        ),
        "source_formulae": (
            "P0R04011: Terminology and notation (effective immediately - revision 11.00).",
            'P0R04012: "Ethical Lagrangian", "Ethical Action" and "PELA" are deprecated for L15. Use SEC Objective Functional, decision return, and SEC maximisation principle. | Symbols: JSECJ_{\\mathrm{SEC}}JSEC (objective), rSECr_{\\mathrm{SEC}}rSEC (instantaneous reward), \\pi (policy), (C,K,Q)(C,K,Q)(C,K,Q) (observables), gig_igi (constraints), lambdai\\lambda_ilambdai (penalties), \\gamma (discount). | Where legacy equations appear, supply a parenthetical: "decision-theoretic form is canonical; gauge-action form, if shown, is heuristic."',
        ),
        "test_protocols": (
            "preserve Terminology and notation (effective immediately - revision 11.00). source-accounting boundary",
        ),
        "null_results": (
            "Terminology and notation (effective immediately - revision 11.00). is not empirical validation evidence",
        ),
        "variables": ("terminology_and_notation_effective_immediately_revision_11_00",),
        "validation_targets": ("preserve records P0R04011-P0R04012",),
        "null_controls": (
            "terminology_and_notation_effective_immediately_revision_11_00 must remain source-bounded accounting",
        ),
    },
    "notes_on_correspondence_non_obligatory_analogues.consequences": {
        "context_id": "consequences",
        "validation_protocol": "paper0.notes_on_correspondence_non_obligatory_analogues.consequences",
        "canonical_statement": "The source-bounded component 'Consequences.' preserves Paper 0 records P0R04013-P0R04028 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04013:consequences",
            "P0R04014:consequences",
            "P0R04015:consequences",
            "P0R04016:consequences",
            "P0R04017:consequences",
            "P0R04018:consequences",
            "P0R04019:consequences",
            "P0R04020:consequences",
            "P0R04021:consequences",
            "P0R04022:consequences",
            "P0R04023:consequences",
            "P0R04024:consequences",
            "P0R04025:consequences",
            "P0R04026:consequences",
            "P0R04027:consequences",
            "P0R04028:consequences",
        ),
        "source_formulae": (
            "P0R04013: Consequences.",
            "P0R04014: (i) Category separation is explicit: ethics at L15 is a normative objective over physically grounded observables, not a curvature integral.",
            "P0R04015: (ii) Falsifiability is improved: JSECJ_{\\mathrm{SEC}}JSEC reduces to measurable components and constraints, enabling preregistered tests and ablations.",
            "P0R04016: (iii) Interfacing with agents (biological or artificial) is straightforward via control/learning algorithms, while preserving the SCPN teleological programme.",
            "P0R04017: [IMAGE:Ein Bild, das Text, Screenshot, Reihe, Diagramm enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R04018: Fig.: SEC instantaneous reward vs. coherence plot",
            "P0R04019: What the Plot Shows",
            "P0R04020: This plot visualizes the core tension in the SEC framework:",
            "P0R04021: The Green Dashed Line shows the reward from increasing coherence. In isolation, the system would be incentivized to increase coherence indefinitely. | The Red Dashed Line shows the penalty. It is zero until the coherence level crosses the critical threshold (at C=6). After this point, the penalty becomes increasingly negative. | The Solid Cyan Line shows the total reward (r_SEC). It initially follows the green line, increasing with coherence. However, as soon as the stability constraint is violated, the heavy penalty kicks in, causing the total reward to plummet.",
            "P0R04022: This illustrates that the optimal policy (*) would guide the system to evolve towards states of high coherence right up to the edge of the constraint, but would be strongly disincentivized from ever crossing it. This is how the SEC functional grounds teleology in a concrete, measurable, and constrained optimization process.",
            "P0R04023: Algorithmic Framoworks for finding the Optimal policy",
            "P0R04024: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Reihe enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            'P0R04025: Fig: This diagram illustrates how two prominent families of reinforcement learning algorithms, Policy-Gradient and Actor-Critic methods, serve as the operational machinery for realizing Axiom 3-the maximization of the SEC Objective Functional. It shows how an "agent" (representing the universe\'s dynamics) learns the optimal policy * through interaction with the environment (the state of SCPN Layers 1-14). This figure clearly contrasts direct policy ascent with bootstrapped value-guided learning, providing concrete algorithmic routes to compute the SEC-optimal policy \\pi^* for L15.',
            "P0R04026: How to Interpret This Diagram",
            'P0R04027: Policy-Gradient Methods are the most direct approach. The "agent" tries an action, sees the outcome, and directly adjusts its policy to make good outcomes more likely. It\'s like learning to play darts by simply reinforcing the throws that get closer to the bullseye. | Actor-Critic Methods are more sophisticated. The Actor (the policy) is like the dart player, making the throws. The Critic (the value function) is like a coach who, instead of just saying "good" or "bad," says, "That throw was better than I expected for this situation." This more nuanced feedback (the TD Error) helps the player (Actor) learn faster and more effectively, especially in complex environments.',
            'P0R04028: In the context of the SCPN, these algorithms provide a formal mechanism by which the universe can "learn" and refine its evolutionary trajectory to maximize Sustainable Ethical Coherence.',
        ),
        "test_protocols": ("preserve Consequences. source-accounting boundary",),
        "null_results": ("Consequences. is not empirical validation evidence",),
        "variables": ("consequences",),
        "validation_targets": ("preserve records P0R04013-P0R04028",),
        "null_controls": ("consequences must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class NotesOnCorrespondenceNonObligatoryAnaloguesSpec:
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
class NotesOnCorrespondenceNonObligatoryAnaloguesSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[NotesOnCorrespondenceNonObligatoryAnaloguesSpec, ...]
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


def build_notes_on_correspondence_non_obligatory_analogues_specs(
    source_records: list[dict[str, Any]],
) -> NotesOnCorrespondenceNonObligatoryAnaloguesSpecBundle:
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

    specs: list[NotesOnCorrespondenceNonObligatoryAnaloguesSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            NotesOnCorrespondenceNonObligatoryAnaloguesSpec(
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
        "title": "Paper 0 " + "Notes on correspondence (non-obligatory analogues)." + " Specs",
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
        "next_source_boundary": "P0R04029",
    }
    return NotesOnCorrespondenceNonObligatoryAnaloguesSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> NotesOnCorrespondenceNonObligatoryAnaloguesSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_notes_on_correspondence_non_obligatory_analogues_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: NotesOnCorrespondenceNonObligatoryAnaloguesSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Notes on correspondence (non-obligatory analogues)." + " Specs",
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
    bundle: NotesOnCorrespondenceNonObligatoryAnaloguesSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_notes_on_correspondence_non_obligatory_analogues_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_notes_on_correspondence_non_obligatory_analogues_validation_specs_{date_tag}.md"
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
