#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Logos recursive closure spec builder
"""Promote Paper 0 Logos recursive-closure records."""

from __future__ import annotations

import argparse
import json
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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(545, 578))
BLANK_SEPARATOR_IDS = ("P0R00546", "P0R00560", "P0R00577")
CLAIM_BOUNDARY = "source-bounded Logos recursive closure; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "logos_recursive_closure.recursive_closure_boundary": {
        "context_id": "recursive_closure_boundary",
        "validation_protocol": "paper0.logos_recursive_closure.recursive_closure_boundary",
        "canonical_statement": (
            "The Logos opening frames the 15-layer hierarchy as recursively closed, "
            "not an infinite regress."
        ),
        "source_equation_ids": (
            "P0R00547:recursive_closure_header",
            "P0R00548:not_infinite_regress",
            "P0R00573:closure_quote",
            "P0R00574:recursive_closure_figure",
        ),
        "source_formulae": (
            "recursive closure",
            "15-layer hierarchy is not an infinite regress",
            "SCPN Hierarchy & Recursive Closure",
        ),
        "test_protocols": ("preserve recursive-closure architectural boundary",),
        "null_results": ("recursive-closure figure/caption is not validation evidence",),
        "variables": ("hierarchy", "closure", "figure"),
        "validation_targets": (
            "preserve 15-layer closure label",
            "preserve non-infinite-regress boundary",
            "reject figure-caption-as-validation promotion",
        ),
        "null_controls": (
            "figure-caption-as-evidence control must be rejected",
            "infinite-regress control must be rejected",
        ),
    },
    "logos_recursive_closure.three_axiom_status_boundary": {
        "context_id": "three_axiom_status_boundary",
        "validation_protocol": "paper0.logos_recursive_closure.three_axiom_status_boundary",
        "canonical_statement": (
            "The three Logos axioms are introduced with distinct status labels: "
            "metaphysical postulate, falsifiable physical hypothesis, and normative teleological postulate."
        ),
        "source_equation_ids": (
            "P0R00550:axiom1_consciousness_fundamentality",
            "P0R00551:axiom2_information_geometry",
            "P0R00552:axiom3_teleological_optimisation",
        ),
        "source_formulae": (
            "Axiom 1 (Consciousness Fundamentality)",
            "metaphysical postulate",
            "Axiom 2 (Information Geometry)",
            "falsifiable physical hypothesis",
            "Axiom 3 (Teleological Optimisation)",
            "normative or teleological postulate",
        ),
        "test_protocols": ("classify three axiom status labels",),
        "null_results": ("axioms are not established truths",),
        "variables": ("axiom_1", "axiom_2", "axiom_3"),
        "validation_targets": (
            "preserve Axiom 1 metaphysical-postulate status",
            "preserve Axiom 2 physical-hypothesis status",
            "preserve Axiom 3 teleological-postulate status",
        ),
        "null_controls": (
            "axioms-as-established-truths control must be rejected",
            "missing-axiom-status control must be rejected",
        ),
    },
    "logos_recursive_closure.informal_law_restatement": {
        "context_id": "informal_law_restatement",
        "validation_protocol": "paper0.logos_recursive_closure.informal_law_restatement",
        "canonical_statement": (
            "The accessible law-language restates the three axioms as source, math-language, "
            "and goal-directed evolution."
        ),
        "source_equation_ids": (
            "P0R00555:three_laws_header",
            "P0R00556:law1_consciousness_source",
            "P0R00557:law2_universe_speaks_math",
            "P0R00558:law3_universe_has_goal",
        ),
        "source_formulae": (
            "Law #1: Consciousness is the Source",
            "Law #2: The Universe Speaks Math",
            "Law #3: The Universe Has a Goal",
            "constitution for our model of reality",
        ),
        "test_protocols": ("preserve informal restatement without changing axiom status",),
        "null_results": ("law-language is explanatory text, not additional evidence",),
        "variables": ("law_1", "law_2", "law_3"),
        "validation_targets": (
            "preserve source/mathematics/goal mapping",
            "preserve explanatory-language boundary",
        ),
        "null_controls": (
            "law-language-as-extra-axiom control must be rejected",
            "goal-language-as-measured-teleology control must be rejected",
        ),
    },
    "logos_recursive_closure.deep_priors_predictive_coding": {
        "context_id": "deep_priors_predictive_coding",
        "validation_protocol": "paper0.logos_recursive_closure.deep_priors_predictive_coding",
        "canonical_statement": (
            "The meta-framework integration maps the three axioms to deepest priors "
            "of the cosmic generative model."
        ),
        "source_equation_ids": (
            "P0R00561:meta_framework_integrations",
            "P0R00563:deepest_priors",
            "P0R00564:axiom1_inference_engine_prior",
            "P0R00566:axiom3_sec_surprise_prior",
        ),
        "source_formulae": (
            "deepest priors",
            "cosmic generative model",
            "inference engine",
            "geometric",
            "minimise a specific kind of surprise defined by SEC",
        ),
        "test_protocols": ("preserve predictive-coding prior mapping",),
        "null_results": ("deep-prior mapping is not empirical confirmation",),
        "variables": ("deep_priors", "predictive_coding", "sec_surprise"),
        "validation_targets": (
            "preserve three-axiom prior role",
            "preserve SEC surprise-minimisation boundary",
        ),
        "null_controls": (
            "prior-map-as-confirmation control must be rejected",
            "missing-sec-prior control must be rejected",
        ),
    },
    "logos_recursive_closure.hint_axiom_role_mapping": {
        "context_id": "hint_axiom_role_mapping",
        "validation_protocol": "paper0.logos_recursive_closure.hint_axiom_role_mapping",
        "canonical_statement": (
            "The Psi-field coupling integration maps the three axioms onto the "
            "context, nature, and purpose of H_int."
        ),
        "source_equation_ids": (
            "P0R00568:H_int=-lambda*Psi_s*sigma",
            "P0R00569:axiom1_defines_psi_s",
            "P0R00570:axiom2_defines_lambda_sigma",
            "P0R00571:axiom3_defines_sec_purpose",
            "P0R00577:blank_separator",
        ),
        "source_formulae": (
            "H_int = -lambda * Psi_s * sigma",
            "Axiom 1 defines the Psi_s term",
            "Axiom 2 defines the nature of the interaction",
            "lambda and sigma",
            "Axiom 3 defines the purpose of the interaction",
            "SEC",
        ),
        "test_protocols": ("classify H_int axiom-role mapping",),
        "null_results": ("H_int role mapping is not experimental validation",),
        "variables": ("Psi_s", "lambda", "sigma", "SEC"),
        "validation_targets": (
            "preserve Psi_s ground role",
            "preserve lambda/sigma information-geometry role",
            "preserve SEC directional-bias role",
        ),
        "null_controls": (
            "h_int-map-as-validation control must be rejected",
            "missing-axiom-role control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class LogosRecursiveClosureSpec:
    """Logos recursive-closure spec promoted from Paper 0 records."""

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
class LogosRecursiveClosureSpecBundle:
    """Logos recursive-closure specs plus source coverage summary."""

    specs: tuple[LogosRecursiveClosureSpec, ...]
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


def build_logos_recursive_closure_specs(
    source_records: list[dict[str, Any]],
) -> LogosRecursiveClosureSpecBundle:
    """Build source-covered Logos recursive-closure specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[LogosRecursiveClosureSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            LogosRecursiveClosureSpec(
                key=key,
                context_id=str(metadata["context_id"]),
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=tuple(str(item) for item in metadata["source_equation_ids"]),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(anchor["source_record_id"]) for anchor in anchors),
                source_block_indices=tuple(
                    int(anchor["source_block_index"]) for anchor in anchors
                ),
                source_formulae=tuple(str(item) for item in metadata["source_formulae"]),
                test_protocols=tuple(str(item) for item in metadata["test_protocols"]),
                null_results=tuple(str(item) for item in metadata["null_results"]),
                variables=tuple(str(item) for item in metadata["variables"]),
                validation_targets=tuple(str(item) for item in metadata["validation_targets"]),
                executable_validation_targets=tuple(
                    str(item) for item in metadata["validation_targets"]
                ),
                null_controls=tuple(str(item) for item in metadata["null_controls"]),
                claim_boundary=CLAIM_BOUNDARY,
                implementation_status="implemented_source_fixture",
                domain_review_status="requires_domain_review_before_scientific_claim",
                hardware_status=HARDWARE_STATUS,
            )
        )

    summary = {
        "title": "Paper 0 Logos Recursive Closure Specs",
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(SOURCE_LEDGER_IDS),
        "coverage_match": True,
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "blank_separator_count": len(BLANK_SEPARATOR_IDS),
        "axiom_count": 3,
        "hint_role_count": 3,
        "next_source_boundary": "P0R00578",
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "all_specs_are_source_anchored": all(spec.source_ledger_ids for spec in specs),
        "all_specs_have_null_controls": all(spec.null_controls for spec in specs),
        "unconsumed_source_ledger_ids": [],
    }
    return LogosRecursiveClosureSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(ledger_path: Path = DEFAULT_LEDGER_PATH) -> LogosRecursiveClosureSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = [
        record
        for record in load_jsonl(ledger_path)
        if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS
    ]
    return build_logos_recursive_closure_specs(records)


def write_outputs(
    bundle: LogosRecursiveClosureSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown spec artefacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_logos_recursive_closure_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_logos_recursive_closure_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "summary": bundle.summary,
        "specs": [asdict(spec) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def render_report(bundle: LogosRecursiveClosureSpecBundle) -> str:
    """Render a compact Markdown report for promoted Logos specs."""
    lines = [
        "# Paper 0 Logos Recursive Closure Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - "
        f"{bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Specs: {bundle.summary['spec_count']}",
        f"- Axioms: {bundle.summary['axiom_count']}",
        f"- H_int roles: {bundle.summary['hint_role_count']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                f"- `{spec.key}`",
                f"  - Context: `{spec.context_id}`",
                f"  - Statement: {spec.canonical_statement}",
                f"  - Formulae: {', '.join(spec.source_formulae)}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Build Logos recursive-closure specs and write artefacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args()

    bundle = build_from_ledger(args.ledger)
    write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
