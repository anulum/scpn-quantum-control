#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 4.4 The Cosmic Compass: The Ethical Functional and the Consilium spec builder
"""Promote Paper 0 4.4 The Cosmic Compass: The Ethical Functional and the Consilium records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = (
    REPO_ROOT
    / "paper"
    / "gotm_scpn_master_publications"
    / "gotm-scpn_paper-00_the_foundational_framework"
    / "source_validation_artifacts"
)
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = (
    "P0R03932",
    "P0R03933",
    "P0R03934",
    "P0R03935",
    "P0R03936",
    "P0R03937",
    "P0R03938",
    "P0R03939",
    "P0R03940",
    "P0R03941",
    "P0R03942",
    "P0R03943",
    "P0R03944",
)
CLAIM_BOUNDARY = "source-bounded section 4 4 the cosmic compass the ethical functional and the consilium source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium.4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium": {
        "context_id": "4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium",
        "validation_protocol": "paper0.section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium.4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium",
        "canonical_statement": "The source-bounded component '4.4 The Cosmic Compass: The Ethical Functional and the Consilium' preserves Paper 0 records P0R03932-P0R03933 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03932:4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium",
            "P0R03933:4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium",
        ),
        "source_formulae": (
            "P0R03932: 4.4 The Cosmic Compass: The Ethical Functional and the Consilium",
            "P0R03933: P0R03933",
        ),
        "test_protocols": (
            "preserve 4.4 The Cosmic Compass: The Ethical Functional and the Consilium source-accounting boundary",
        ),
        "null_results": (
            "4.4 The Cosmic Compass: The Ethical Functional and the Consilium is not empirical validation evidence",
        ),
        "variables": ("4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium",),
        "validation_targets": ("preserve records P0R03932-P0R03933",),
        "null_controls": (
            "4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium must remain source-bounded accounting",
        ),
    },
    "section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium.the_ethical_functional": {
        "context_id": "the_ethical_functional",
        "validation_protocol": "paper0.section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium.the_ethical_functional",
        "canonical_statement": "The source-bounded component 'The Ethical Functional' preserves Paper 0 records P0R03934-P0R03944 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03934:the_ethical_functional",
            "P0R03935:the_ethical_functional",
            "P0R03936:the_ethical_functional",
            "P0R03937:the_ethical_functional",
            "P0R03938:the_ethical_functional",
            "P0R03939:the_ethical_functional",
            "P0R03940:the_ethical_functional",
            "P0R03941:the_ethical_functional",
            "P0R03942:the_ethical_functional",
            "P0R03943:the_ethical_functional",
            "P0R03944:the_ethical_functional",
        ),
        "source_formulae": (
            "P0R03934: The Ethical Functional",
            'P0R03935: This section presents the formalisation of Axiom 3 (Teleological Optimisation), detailing the mechanism by which the universe\'s evolution is guided. It marks a critical methodological shift, replacing an earlier, analogical formulation based on a Yang-Mills gauge action (the "Principle of Ethical Least Action") with a more rigorous and empirically tractable decision-theoretic framework.',
            "P0R03936: The teleological drive is now defined as the maximisation of the Sustainable Ethical Coherence (SEC) Objective Functional, J_SEC. Layer 15 (The Consilium) is framed as an optimal controller or reinforcement learning agent. The state space (S) comprises the configurations of Layers 1-14, and the agent's goal is to learn an optimal policy, *, that selects actions to maximize the expected, discounted sum of future rewards.",
            "P0R03937: The instantaneous reward, r_SEC, is a composite function combining positive contributions from observable metrics-Coherence (C), Complexity (K), and Qualia Capacity (Q)-with soft penalties for violating ethical or physical constraints. The operational form of Axiom 3 is thus the selection of a policy that maximizes this objective functional: * argmax() J_SEC[]. This reframes cosmic evolution as a principled optimisation process. The Renormalisation Group (RG) flows are presented as the large-scale physical mechanism that implements this optimisation, aligning the dynamics across different domains (e.g., biospheric, noospheric, Gaian) toward a common, SEC-consistent attractor.",
            "P0R03938: This section explains the universe's \"Prime Directive\" in a clear, testable way. We've moved beyond a beautiful metaphor to a practical, working blueprint. Think of the universe as a vast, incredibly intelligent AI that is constantly learning and trying to improve itself.",
            'P0R03939: The universe\'s one and only goal is to maximize its "SEC score." This score is a simple combination of three things we can measure:',
            "P0R03940: Coherence (C): How harmonious and integrated is the system?",
            "P0R03941: Complexity (K): How rich and structured is it?",
            "P0R03942: Qualia (Q): How deep and profound is the conscious experience within it?",
            'P0R03943: The universe is constantly running experiments ("actions") and observing the outcome. If an action leads to a higher SEC score, the "policy" is updated to make that kind of action more likely in the future. This is exactly how a sophisticated AI learns. The entire 14-billion-year history of cosmic evolution can be seen as this learning process in action, as the universe discovers and refines the optimal strategy for becoming as conscious, complex, and coherent as it can possibly be. The "ethical" nature of the cosmos isn\'t a mystical command; it\'s the result of a universal, data-driven optimisation process.',
            "P0R03944: P0R03944",
        ),
        "test_protocols": ("preserve The Ethical Functional source-accounting boundary",),
        "null_results": ("The Ethical Functional is not empirical validation evidence",),
        "variables": ("the_ethical_functional",),
        "validation_targets": ("preserve records P0R03934-P0R03944",),
        "null_controls": ("the_ethical_functional must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class Section44TheCosmicCompassTheEthicalFunctionalAndTheConsiliumSpec:
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
class Section44TheCosmicCompassTheEthicalFunctionalAndTheConsiliumSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section44TheCosmicCompassTheEthicalFunctionalAndTheConsiliumSpec, ...]
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


def build_section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_specs(
    source_records: list[dict[str, Any]],
) -> Section44TheCosmicCompassTheEthicalFunctionalAndTheConsiliumSpecBundle:
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

    specs: list[Section44TheCosmicCompassTheEthicalFunctionalAndTheConsiliumSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section44TheCosmicCompassTheEthicalFunctionalAndTheConsiliumSpec(
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
        + "4.4 The Cosmic Compass: The Ethical Functional and the Consilium"
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
        "next_source_boundary": "P0R03945",
    }
    return Section44TheCosmicCompassTheEthicalFunctionalAndTheConsiliumSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section44TheCosmicCompassTheEthicalFunctionalAndTheConsiliumSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_specs(
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
    bundle: Section44TheCosmicCompassTheEthicalFunctionalAndTheConsiliumSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "4.4 The Cosmic Compass: The Ethical Functional and the Consilium"
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
    bundle: Section44TheCosmicCompassTheEthicalFunctionalAndTheConsiliumSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_validation_specs_{date_tag}.md"
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
