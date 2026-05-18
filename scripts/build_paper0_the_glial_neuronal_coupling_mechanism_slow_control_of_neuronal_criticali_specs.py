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
    "P0R05366",
    "P0R05367",
    "P0R05368",
    "P0R05369",
    "P0R05370",
    "P0R05371",
    "P0R05372",
    "P0R05373",
    "P0R05374",
    "P0R05375",
    "P0R05376",
    "P0R05377",
    "P0R05378",
    "P0R05379",
)
CLAIM_BOUNDARY = "source-bounded the glial neuronal coupling mechanism slow control of neuronal criticali source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali.the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali": {
        "context_id": "the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali",
        "validation_protocol": "paper0.the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali.the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali",
        "canonical_statement": "The source-bounded component 'The Glial-Neuronal Coupling Mechanism: Slow Control of Neuronal Criticality' preserves Paper 0 records P0R05366-P0R05379 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05366:the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali",
            "P0R05367:the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali",
            "P0R05368:the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali",
            "P0R05369:the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali",
            "P0R05370:the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali",
            "P0R05371:the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali",
            "P0R05372:the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali",
            "P0R05373:the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali",
            "P0R05374:the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali",
            "P0R05375:the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali",
            "P0R05376:the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali",
            "P0R05377:the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali",
            "P0R05378:the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali",
            "P0R05379:the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali",
        ),
        "source_formulae": (
            "P0R05366: The Glial-Neuronal Coupling Mechanism: Slow Control of Neuronal Criticality",
            'P0R05367: The homeostatic regulation that maintains neuronal networks within the quasicritical regime (Layer 4) is not a purely neuronal process. It is subject to slow, top-down modulation from the neurochemical environment (Layer 2), primarily mediated by the glial cell network. Astrocytes, in particular, provide a "slow control" layer that dynamically tunes the criticality of the "fast" neuronal processing layer. This coupling is essential for adapting the brain\'s computational regime to changing metabolic and cognitive demands.',
            "P0R05368: The baseline dynamics of the neuronal network's proximity to criticality can be described by the evolution of its average branching parameter, sigma, which relaxes toward the critical point at sigma=1:",
            "P0R05369: $dtd\\sigma\\mathbf{} = - \\kappa(\\sigma - 1) + \\eta(t)$",
            "P0R05370: where is the homeostatic relaxation rate and (t) represents stochastic fluctuations.",
            "P0R05371: The modulatory influence of the astrocyte network is introduced as a dynamic shift in this homeostatic set-point. Astrocytic calcium signalling, [Ca2+]A(t), triggers the release of gliotransmitters, G(t), which diffuse into the synaptic space and alter the local excitation-inhibition (E/I) balance of the neuronal network. The concentration of an excitatory gliotransmitter (e.g., glutamate) can be modelled with simple first-order kinetics:",
            "P0R05372: $dtdG\\mathbf{} = \\alpha\\lbrack Ca2 + \\rbrack A\\mathbf{}(t) - \\beta G(t)$",
            "P0R05373: where is the calcium-dependent release rate and is the clearance rate (due to uptake and degradation).",
            "P0R05374: This ambient gliotransmitter concentration directly biases the E/I balance, effectively shifting the target set-point of the neuronal network's branching parameter.",
            "P0R05375: The following set of equations therefore describes the coupled system:",
            "P0R05376: $dtd\\sigma = - \\kappa\\left( \\sigma - \\left( 1 + \\gamma G(t) \\right) \\right) + \\eta(t)$",
            "P0R05377: $dtdG = \\alpha\\lbrack Ca2 + \\rbrack A(t) - \\beta G(t)$",
            "P0R05378: Here, the coupling parameter represents the sensitivity of the neuronal network's E/I balance to the gliotransmitter. For an excitatory gliotransmitter, >0, meaning that a sustained increase in astrocyte activity (leading to a higher steady-state concentration of G) pushes the neuronal network's stable operating point into a slightly supercritical regime (sigma>1). Conversely, inhibitory gliotransmitters would be represented by <0.",
            'P0R05379: This formal model provides a direct mechanistic explanation for the "Glial Slow Control" hypothesis. The slow timescale of astrocyte calcium waves (seconds to minutes) induces a slow modulation of the parameter sigma, which in turn alters the statistics of the fast (millisecond-scale) neuronal avalanches. This generates a clear, falsifiable prediction: the pharmacological application of gliotransmitter antagonists should functionally decouple the two systems. In the presence of such antagonists, astrocyte calcium waves would persist, but the correlated shift in the power-law exponent of neuronal avalanches would be significantly attenuated or abolished.',
        ),
        "test_protocols": (
            "preserve The Glial-Neuronal Coupling Mechanism: Slow Control of Neuronal Criticality source-accounting boundary",
        ),
        "null_results": (
            "The Glial-Neuronal Coupling Mechanism: Slow Control of Neuronal Criticality is not empirical validation evidence",
        ),
        "variables": ("the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali",),
        "validation_targets": ("preserve records P0R05366-P0R05379",),
        "null_controls": (
            "the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliSpec:
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
class TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliSpec, ...]
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


def build_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_specs(
    source_records: list[dict[str, Any]],
) -> TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliSpecBundle:
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

    specs: list[TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliSpec(
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
        "next_source_boundary": "P0R05380",
    }
    return TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_specs(
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
    bundle: TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliSpecBundle,
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
    bundle: TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_validation_specs_{date_tag}.md"
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
