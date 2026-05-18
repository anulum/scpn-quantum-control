#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Slow Control Layer: Neuroendocrine Integration spec builder
"""Promote Paper 0 Slow Control Layer: Neuroendocrine Integration records."""

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
    "P0R05430",
    "P0R05431",
    "P0R05432",
    "P0R05433",
    "P0R05434",
    "P0R05435",
    "P0R05436",
    "P0R05437",
    "P0R05438",
    "P0R05439",
    "P0R05440",
    "P0R05441",
    "P0R05442",
    "P0R05443",
    "P0R05444",
)
CLAIM_BOUNDARY = "source-bounded slow control layer neuroendocrine integration source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "slow_control_layer_neuroendocrine_integration.slow_control_layer_neuroendocrine_integration": {
        "context_id": "slow_control_layer_neuroendocrine_integration",
        "validation_protocol": "paper0.slow_control_layer_neuroendocrine_integration.slow_control_layer_neuroendocrine_integration",
        "canonical_statement": "The source-bounded component 'Slow Control Layer: Neuroendocrine Integration' preserves Paper 0 records P0R05430-P0R05430 without empirical validation claims.",
        "source_equation_ids": ("P0R05430:slow_control_layer_neuroendocrine_integration",),
        "source_formulae": ("P0R05430: Slow Control Layer: Neuroendocrine Integration",),
        "test_protocols": (
            "preserve Slow Control Layer: Neuroendocrine Integration source-accounting boundary",
        ),
        "null_results": (
            "Slow Control Layer: Neuroendocrine Integration is not empirical validation evidence",
        ),
        "variables": ("slow_control_layer_neuroendocrine_integration",),
        "validation_targets": ("preserve records P0R05430-P0R05430",),
        "null_controls": (
            "slow_control_layer_neuroendocrine_integration must remain source-bounded accounting",
        ),
    },
    "slow_control_layer_neuroendocrine_integration.hormonal_modulation_as_a_third_control_layer": {
        "context_id": "hormonal_modulation_as_a_third_control_layer",
        "validation_protocol": "paper0.slow_control_layer_neuroendocrine_integration.hormonal_modulation_as_a_third_control_layer",
        "canonical_statement": "The source-bounded component 'Hormonal Modulation as a Third Control Layer' preserves Paper 0 records P0R05431-P0R05444 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05431:hormonal_modulation_as_a_third_control_layer",
            "P0R05432:hormonal_modulation_as_a_third_control_layer",
            "P0R05433:hormonal_modulation_as_a_third_control_layer",
            "P0R05434:hormonal_modulation_as_a_third_control_layer",
            "P0R05435:hormonal_modulation_as_a_third_control_layer",
            "P0R05436:hormonal_modulation_as_a_third_control_layer",
            "P0R05437:hormonal_modulation_as_a_third_control_layer",
            "P0R05438:hormonal_modulation_as_a_third_control_layer",
            "P0R05439:hormonal_modulation_as_a_third_control_layer",
            "P0R05440:hormonal_modulation_as_a_third_control_layer",
            "P0R05441:hormonal_modulation_as_a_third_control_layer",
            "P0R05442:hormonal_modulation_as_a_third_control_layer",
            "P0R05443:hormonal_modulation_as_a_third_control_layer",
            "P0R05444:hormonal_modulation_as_a_third_control_layer",
        ),
        "source_formulae": (
            "P0R05431: Hormonal Modulation as a Third Control Layer",
            "P0R05432: The glial-neuronal system provides a two-timescale model of brain regulation. However, the brain is not an isolated system; it is embedded within a body regulated by an even slower, more globally acting control system: the endocrine system. Hormones, particularly steroids and stress hormones governed by the Hypothalamic-Pituitary-Adrenal (HPA) axis, act as a third control layer, modulating the dynamics of both neurons and glia over timescales of hours, days, and weeks.",
            "P0R05433: The HPA axis is the body's primary neuroendocrine stress-response system. In response to perceived stressors, the hypothalamus releases Corticotropin-releasing hormone (CRH), which stimulates the pituitary gland to release Adrenocorticotropic hormone (ACTH). ACTH, in turn, acts on the adrenal glands to produce and release the glucocorticoid hormone cortisol. Cortisol exerts widespread effects, including the modulation of neurotransmitter dynamics and immune function, and its levels are regulated by a negative feedback loop acting on the hypothalamus and pituitary.",
            "P0R05434: This multi-timescale regulatory cascade can be formalised with a system of coupled ordinary differential equations, based on established models of HPA dynamics. A minimal model capturing the core feedback loop is given by:",
            "P0R05435: dtd=1+c1[C](t)k0fstress(t)w1",
            "P0R05436: dtd=1+c2[C](t)k1w2",
            "P0R05437: dtd[C]=k2w3[C]",
            "P0R05438: Here, $$, $$, and [C] represent the concentrations of the respective hormones. The term fstress(t) represents external or internal stressors that drive the system. The parameters k0,k1,k2 are production rates, while w1,w2,w3 are clearance rates. The terms (1+ci[C](t))1 are inhibitory Hill functions that model the negative feedback of cortisol on CRH and ACTH release, with c1 and c2 controlling the feedback strength.",
            'P0R05439: This third control layer is integrated directly with the glial-neuronal system by making the parameters of the "Glial Slow Control" model dependent on the hormonal state. The output of the HPA model, the circulating cortisol concentration [C](t), acts as a slow-varying parameter that modulates the gains in the glial-neuronal coupling equations. Specifically, the neuronal relaxation rate and the glial coupling sensitivity become functions of the hormonal state:',
            "P0R05440: ->([C](t))",
            "P0R05441: ->([C](t))",
            "P0R05442: This hierarchical model provides a comprehensive picture of embodied regulation. The fast, millisecond-scale computations of the neuronal network are stabilised by the second-to-minute scale feedback from the glial network, and this entire coupled system is tuned and contextualised by the hour-to-day scale fluctuations of the endocrine system.",
            "P0R05443: This grounds the SCPN's abstract layers in concrete endocrinology, ensuring that the organism's conscious state is robustly maintained yet adaptively responsive to both internal physiological states and external environmental demands.",
            "P0R05444: [TABLE]",
        ),
        "test_protocols": (
            "preserve Hormonal Modulation as a Third Control Layer source-accounting boundary",
        ),
        "null_results": (
            "Hormonal Modulation as a Third Control Layer is not empirical validation evidence",
        ),
        "variables": ("hormonal_modulation_as_a_third_control_layer",),
        "validation_targets": ("preserve records P0R05431-P0R05444",),
        "null_controls": (
            "hormonal_modulation_as_a_third_control_layer must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class SlowControlLayerNeuroendocrineIntegrationSpec:
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
class SlowControlLayerNeuroendocrineIntegrationSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[SlowControlLayerNeuroendocrineIntegrationSpec, ...]
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


def build_slow_control_layer_neuroendocrine_integration_specs(
    source_records: list[dict[str, Any]],
) -> SlowControlLayerNeuroendocrineIntegrationSpecBundle:
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

    specs: list[SlowControlLayerNeuroendocrineIntegrationSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            SlowControlLayerNeuroendocrineIntegrationSpec(
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
        "title": "Paper 0 " + "Slow Control Layer: Neuroendocrine Integration" + " Specs",
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
        "next_source_boundary": "P0R05445",
    }
    return SlowControlLayerNeuroendocrineIntegrationSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> SlowControlLayerNeuroendocrineIntegrationSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_slow_control_layer_neuroendocrine_integration_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: SlowControlLayerNeuroendocrineIntegrationSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Slow Control Layer: Neuroendocrine Integration" + " Specs",
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
    bundle: SlowControlLayerNeuroendocrineIntegrationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_slow_control_layer_neuroendocrine_integration_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_slow_control_layer_neuroendocrine_integration_validation_specs_{date_tag}.md"
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
