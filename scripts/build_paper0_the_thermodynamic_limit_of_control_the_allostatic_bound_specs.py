#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Thermodynamic Limit of Control: The Allostatic Bound spec builder
"""Promote Paper 0 The Thermodynamic Limit of Control: The Allostatic Bound records."""

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
    "P0R05408",
    "P0R05409",
    "P0R05410",
    "P0R05411",
    "P0R05412",
    "P0R05413",
    "P0R05414",
    "P0R05415",
    "P0R05416",
    "P0R05417",
    "P0R05418",
    "P0R05419",
)
CLAIM_BOUNDARY = "source-bounded the thermodynamic limit of control the allostatic bound source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_thermodynamic_limit_of_control_the_allostatic_bound.the_thermodynamic_limit_of_control_the_allostatic_bound": {
        "context_id": "the_thermodynamic_limit_of_control_the_allostatic_bound",
        "validation_protocol": "paper0.the_thermodynamic_limit_of_control_the_allostatic_bound.the_thermodynamic_limit_of_control_the_allostatic_bound",
        "canonical_statement": "The source-bounded component 'The Thermodynamic Limit of Control: The Allostatic Bound' preserves Paper 0 records P0R05408-P0R05419 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05408:the_thermodynamic_limit_of_control_the_allostatic_bound",
            "P0R05409:the_thermodynamic_limit_of_control_the_allostatic_bound",
            "P0R05410:the_thermodynamic_limit_of_control_the_allostatic_bound",
            "P0R05411:the_thermodynamic_limit_of_control_the_allostatic_bound",
            "P0R05412:the_thermodynamic_limit_of_control_the_allostatic_bound",
            "P0R05413:the_thermodynamic_limit_of_control_the_allostatic_bound",
            "P0R05414:the_thermodynamic_limit_of_control_the_allostatic_bound",
            "P0R05415:the_thermodynamic_limit_of_control_the_allostatic_bound",
            "P0R05416:the_thermodynamic_limit_of_control_the_allostatic_bound",
            "P0R05417:the_thermodynamic_limit_of_control_the_allostatic_bound",
            "P0R05418:the_thermodynamic_limit_of_control_the_allostatic_bound",
            "P0R05419:the_thermodynamic_limit_of_control_the_allostatic_bound",
        ),
        "source_formulae": (
            "P0R05408: The Thermodynamic Limit of Control: The Allostatic Bound",
            "P0R05409: P0R05409",
            'P0R05410: While the coupled differential equations for glial-neuronal interaction elegantly describe the mathematical maintenance of quasicriticality, they implicitly assume an infinite energy sink. In biophysical reality, the astrocyte network\'s ability to propagate calcium waves, clear excitatory neurotransmitters against steep concentration gradients, and maintain ionic homeostasis is massively energetically expensive. This "slow control loop" relies fundamentally on ATP derived from local oxidative metabolism.',
            "P0R05411: To rigorously ground this cybernetic control theory in thermodynamics, we must introduce an Allostatic Bound on the glial corrective force. The magnitude of the modulatory influence applied by the glia, $|\\gamma G(t)|$, is strictly constrained by the local Cerebral Metabolic Rate of Oxygen ($CMRO_2$) and glucose availability.",
            "P0R05412: We formalize this by defining a metabolic capacity function, $\\Pi_{metabolic}(t)$, which serves as an absolute, physical upper bound on the glial control term:",
            "P0R05413: $$|\\gamma G(t)| \\le \\Pi_{metabolic}(CMRO_2(t), [ATP](t))$$",
            "P0R05414: If the neuronal network is pushed severely away from criticality-for instance, by massive excitatory surges from Spike-Timing-Dependent Plasticity (STDP) during intense learning, or by external allostatic load-the energetic demand to drive the system back to $\\sigma \\approx 1$ increases proportionally. If this demand exceeds the local metabolic supply:",
            "P0R05415: $$E_{demand}(|\\gamma G(t)|) > E_{supply}(\\Pi_{metabolic})$$",
            "P0R05416: the system undergoes Metabolic Decoupling. The effective coupling parameter $\\gamma$ is mathematically forced to collapse toward zero, functionally disabling the slow control loop.",
            "P0R05417: This thermodynamic limit provides a seamless, first-principles explanation for Dyscritia (the pathological loss of criticality). During states of metabolic crisis-such as ischemia, hypoxia, or chronic allostatic overload-the astrocyte network exhausts its ATP reserves. Stripped of its homeostatic governor, the fast neuronal network is left unconstrained. It rapidly diverges from the optimal edge of chaos, collapsing into either rigid subcriticality ($\\sigma \\ll 1$, e.g., depression, coma) or runaway supercriticality ($\\sigma \\gg 1$, e.g., epileptiform seizures).",
            "P0R05418: Therefore, the stability of the SCPN's computational core (Layer 4) is inextricably bound to the First and Second Laws of Thermodynamics, proving that the maintenance of consciousness and optimal cognitive function is a continuous, metabolically paid transaction.",
            "P0R05419: P0R05419",
        ),
        "test_protocols": (
            "preserve The Thermodynamic Limit of Control: The Allostatic Bound source-accounting boundary",
        ),
        "null_results": (
            "The Thermodynamic Limit of Control: The Allostatic Bound is not empirical validation evidence",
        ),
        "variables": ("the_thermodynamic_limit_of_control_the_allostatic_bound",),
        "validation_targets": ("preserve records P0R05408-P0R05419",),
        "null_controls": (
            "the_thermodynamic_limit_of_control_the_allostatic_bound must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class TheThermodynamicLimitOfControlTheAllostaticBoundSpec:
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
class TheThermodynamicLimitOfControlTheAllostaticBoundSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheThermodynamicLimitOfControlTheAllostaticBoundSpec, ...]
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


def build_the_thermodynamic_limit_of_control_the_allostatic_bound_specs(
    source_records: list[dict[str, Any]],
) -> TheThermodynamicLimitOfControlTheAllostaticBoundSpecBundle:
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

    specs: list[TheThermodynamicLimitOfControlTheAllostaticBoundSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheThermodynamicLimitOfControlTheAllostaticBoundSpec(
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
        + "The Thermodynamic Limit of Control: The Allostatic Bound"
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
        "next_source_boundary": "P0R05420",
    }
    return TheThermodynamicLimitOfControlTheAllostaticBoundSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheThermodynamicLimitOfControlTheAllostaticBoundSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_thermodynamic_limit_of_control_the_allostatic_bound_specs(
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


def render_report(bundle: TheThermodynamicLimitOfControlTheAllostaticBoundSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "The Thermodynamic Limit of Control: The Allostatic Bound" + " Specs",
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
    bundle: TheThermodynamicLimitOfControlTheAllostaticBoundSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_thermodynamic_limit_of_control_the_allostatic_bound_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_thermodynamic_limit_of_control_the_allostatic_bound_validation_specs_{date_tag}.md"
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
