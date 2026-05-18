#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Glial-Neuronal Coupling Mechanism: Slow Control of Neuronal Criticality spec builder
"""Promote Paper 0 The Glial-Neuronal Coupling Mechanism: Slow Control of Neuronal Criticality records."""

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
    "P0R05390",
    "P0R05391",
    "P0R05392",
    "P0R05393",
    "P0R05394",
    "P0R05395",
    "P0R05396",
    "P0R05397",
    "P0R05398",
    "P0R05399",
    "P0R05400",
    "P0R05401",
    "P0R05402",
    "P0R05403",
    "P0R05404",
    "P0R05405",
    "P0R05406",
    "P0R05407",
)
CLAIM_BOUNDARY = "source-bounded the glial neuronal coupling mechanism slow control of neuronal criticali p0r05390 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_p0r05390.the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali": {
        "context_id": "the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali",
        "validation_protocol": "paper0.the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_p0r05390.the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali",
        "canonical_statement": "The source-bounded component 'The Glial-Neuronal Coupling Mechanism: Slow Control of Neuronal Criticality' preserves Paper 0 records P0R05390-P0R05392 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05390:the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali",
            "P0R05391:the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali",
            "P0R05392:the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali",
        ),
        "source_formulae": (
            "P0R05390: The Glial-Neuronal Coupling Mechanism: Slow Control of Neuronal Criticality",
            "P0R05391: The SCPN architecture is predicated on the principle of Quasicriticality, a universal dynamic regime poised between order and chaos that maximises information processing capacity. However, a system operating at a critical point is inherently unstable and susceptible to perturbations; it requires a robust homeostatic control mechanism to prevent it from collapsing into either a rigid, subcritical state or a chaotic, supercritical one.",
            'P0R05392: The SCPN identifies this "fine-tuning problem" and specifies the biological machinery that solves it. That machinery is the glial network, which functions as a "slow control layer" for the "fast" neuronal network.',
        ),
        "test_protocols": (
            "preserve The Glial-Neuronal Coupling Mechanism: Slow Control of Neuronal Criticality source-accounting boundary",
        ),
        "null_results": (
            "The Glial-Neuronal Coupling Mechanism: Slow Control of Neuronal Criticality is not empirical validation evidence",
        ),
        "variables": ("the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali",),
        "validation_targets": ("preserve records P0R05390-P0R05392",),
        "null_controls": (
            "the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali must remain source-bounded accounting",
        ),
    },
    "the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_p0r05390.the_astrocyte_network_as_the_slow_control_layer": {
        "context_id": "the_astrocyte_network_as_the_slow_control_layer",
        "validation_protocol": "paper0.the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_p0r05390.the_astrocyte_network_as_the_slow_control_layer",
        "canonical_statement": "The source-bounded component 'The Astrocyte Network as the Slow Control Layer' preserves Paper 0 records P0R05393-P0R05395 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05393:the_astrocyte_network_as_the_slow_control_layer",
            "P0R05394:the_astrocyte_network_as_the_slow_control_layer",
            "P0R05395:the_astrocyte_network_as_the_slow_control_layer",
        ),
        "source_formulae": (
            "P0R05393: The Astrocyte Network as the Slow Control Layer",
            "P0R05394: While neurons operate on a millisecond timescale, the brain's glial cells, particularly astrocytes, form a parallel, interconnected network that communicates on a much slower timescale of seconds to minutes. Astrocytes are not passive support cells; they actively regulate the neuronal environment. They integrate the activity of thousands of synapses within their domain, responding with propagating waves of intracellular calcium ([Ca2+]A).",
            'P0R05395: These calcium waves, in turn, trigger the release of "gliotransmitters" (such as glutamate, ATP, and D-serine) that modulate synaptic release probabilities and neuronal excitability over broad spatial domains. This establishes a slow-acting feedback loop: the fast neuronal network\'s activity is integrated by the slow glial network, which then applies a corrective, modulatory force back onto the neuronal network, gently and continuously steering it toward the optimal quasicritical state.',
        ),
        "test_protocols": (
            "preserve The Astrocyte Network as the Slow Control Layer source-accounting boundary",
        ),
        "null_results": (
            "The Astrocyte Network as the Slow Control Layer is not empirical validation evidence",
        ),
        "variables": ("the_astrocyte_network_as_the_slow_control_layer",),
        "validation_targets": ("preserve records P0R05393-P0R05395",),
        "null_controls": (
            "the_astrocyte_network_as_the_slow_control_layer must remain source-bounded accounting",
        ),
    },
    "the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_p0r05390.formal_model_of_glial_neuronal_coupling": {
        "context_id": "formal_model_of_glial_neuronal_coupling",
        "validation_protocol": "paper0.the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_p0r05390.formal_model_of_glial_neuronal_coupling",
        "canonical_statement": "The source-bounded component 'Formal Model of Glial-Neuronal Coupling' preserves Paper 0 records P0R05396-P0R05407 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05396:formal_model_of_glial_neuronal_coupling",
            "P0R05397:formal_model_of_glial_neuronal_coupling",
            "P0R05398:formal_model_of_glial_neuronal_coupling",
            "P0R05399:formal_model_of_glial_neuronal_coupling",
            "P0R05400:formal_model_of_glial_neuronal_coupling",
            "P0R05401:formal_model_of_glial_neuronal_coupling",
            "P0R05402:formal_model_of_glial_neuronal_coupling",
            "P0R05403:formal_model_of_glial_neuronal_coupling",
            "P0R05404:formal_model_of_glial_neuronal_coupling",
            "P0R05405:formal_model_of_glial_neuronal_coupling",
            "P0R05406:formal_model_of_glial_neuronal_coupling",
            "P0R05407:formal_model_of_glial_neuronal_coupling",
        ),
        "source_formulae": (
            "P0R05396: Formal Model of Glial-Neuronal Coupling",
            "P0R05397: This cybernetic relationship can be formalised with a set of coupled differential equations that explicitly model the two-timescale dynamics. The fast dynamics of the neuronal network are described by the evolution of its average branching parameter, sigma, which represents the average number of downstream neurons activated by a single firing neuron. For criticality, sigma must be maintained near unity. The slow modulatory influence of the astrocyte network is introduced as a dynamic shift in the homeostatic set-point of sigma.",
            "P0R05398: The evolution of the neuronal network's branching parameter is given by:",
            "P0R05399: $dtd\\sigma = - \\kappa\\left( \\sigma - \\left( 1 + \\gamma G(t) \\right) \\right) + \\eta(t)$",
            "P0R05400: where is the intrinsic homeostatic relaxation rate of the neuronal network and (t) represents stochastic fluctuations. The crucial addition is the term G(t), which represents the modulatory influence of the astrocyte network. Here, G(t) is the local concentration of a gliotransmitter, and the coupling parameter represents the sensitivity of the neuronal network's excitation-inhibition balance to that gliotransmitter.",
            "P0R05401: For an excitatory gliotransmitter, >0, meaning that sustained astrocyte activity pushes the neuronal network's stable operating point into a slightly supercritical regime (sigma>1).",
            "P0R05402: The dynamics of the gliotransmitter concentration are, in turn, driven by the astrocyte's intracellular calcium signalling:",
            "P0R05403: $dtdG = \\alpha\\lbrack Ca2 + \\rbrack A(t) - \\beta G(t)$",
            "P0R05404: where is the calcium-dependent release rate and is the clearance rate (due to uptake and degradation). The astrocyte calcium concentration, [Ca2+]A(t), is itself a complex function that integrates the activity of the surrounding neuronal synapses over a slower timescale.",
            'P0R05405: This formal model provides a direct mechanistic explanation for "Glial Slow Control." The slow timescale of astrocyte calcium waves induces a slow modulation of the parameter sigma, which in turn alters the statistics of the fast (millisecond-scale) neuronal avalanches.',
            "P0R05406: This generates a clear, falsifiable prediction: pharmacological blockade of gliotransmitter receptors should functionally decouple the two systems. In the presence of such antagonists, astrocyte calcium waves would persist, but the correlated shifts in the power-law exponent of neuronal avalanches would be significantly attenuated or abolished.",
            "P0R05407: P0R05407",
        ),
        "test_protocols": (
            "preserve Formal Model of Glial-Neuronal Coupling source-accounting boundary",
        ),
        "null_results": (
            "Formal Model of Glial-Neuronal Coupling is not empirical validation evidence",
        ),
        "variables": ("formal_model_of_glial_neuronal_coupling",),
        "validation_targets": ("preserve records P0R05396-P0R05407",),
        "null_controls": (
            "formal_model_of_glial_neuronal_coupling must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliP0r05390Spec:
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
class TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliP0r05390SpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliP0r05390Spec, ...]
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


def build_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_p0r05390_specs(
    source_records: list[dict[str, Any]],
) -> TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliP0r05390SpecBundle:
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

    specs: list[TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliP0r05390Spec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliP0r05390Spec(
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
        + "The Glial-Neuronal Coupling Mechanism: Slow Control of Neuronal Criticality"
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
        "next_source_boundary": "P0R05408",
    }
    return TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliP0r05390SpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliP0r05390SpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_p0r05390_specs(
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
    bundle: TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliP0r05390SpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "The Glial-Neuronal Coupling Mechanism: Slow Control of Neuronal Criticality"
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
    bundle: TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliP0r05390SpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_p0r05390_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_p0r05390_validation_specs_{date_tag}.md"
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
