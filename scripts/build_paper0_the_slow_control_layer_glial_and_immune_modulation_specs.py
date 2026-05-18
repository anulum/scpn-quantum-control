#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Slow Control Layer - Glial and Immune Modulation spec builder
"""Promote Paper 0 The Slow Control Layer - Glial and Immune Modulation records."""

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
    "P0R05347",
    "P0R05348",
    "P0R05349",
    "P0R05350",
    "P0R05351",
    "P0R05352",
    "P0R05353",
    "P0R05354",
    "P0R05355",
    "P0R05356",
    "P0R05357",
)
CLAIM_BOUNDARY = "source-bounded the slow control layer glial and immune modulation source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_slow_control_layer_glial_and_immune_modulation.the_slow_control_layer_glial_and_immune_modulation": {
        "context_id": "the_slow_control_layer_glial_and_immune_modulation",
        "validation_protocol": "paper0.the_slow_control_layer_glial_and_immune_modulation.the_slow_control_layer_glial_and_immune_modulation",
        "canonical_statement": "The source-bounded component 'The Slow Control Layer - Glial and Immune Modulation' preserves Paper 0 records P0R05347-P0R05349 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05347:the_slow_control_layer_glial_and_immune_modulation",
            "P0R05348:the_slow_control_layer_glial_and_immune_modulation",
            "P0R05349:the_slow_control_layer_glial_and_immune_modulation",
        ),
        "source_formulae": (
            "P0R05347: The Slow Control Layer - Glial and Immune Modulation",
            'P0R05348: The neuronal-centric model of the biological substrate provides the fast dynamics of information processing. However, this "fast loop" is embedded within and modulated by a "slow control layer" composed of glial and immune networks. These systems provide homeostatic regulation, contextual modulation, and a deeper level of embodiment, ensuring the stability and coherence of the organismal field.',
            "P0R05349: [IMAGE:]",
        ),
        "test_protocols": (
            "preserve The Slow Control Layer - Glial and Immune Modulation source-accounting boundary",
        ),
        "null_results": (
            "The Slow Control Layer - Glial and Immune Modulation is not empirical validation evidence",
        ),
        "variables": ("the_slow_control_layer_glial_and_immune_modulation",),
        "validation_targets": ("preserve records P0R05347-P0R05349",),
        "null_controls": (
            "the_slow_control_layer_glial_and_immune_modulation must remain source-bounded accounting",
        ),
    },
    "the_slow_control_layer_glial_and_immune_modulation.i_the_astrocyte_neuron_lattice_l2_l4_modulation": {
        "context_id": "i_the_astrocyte_neuron_lattice_l2_l4_modulation",
        "validation_protocol": "paper0.the_slow_control_layer_glial_and_immune_modulation.i_the_astrocyte_neuron_lattice_l2_l4_modulation",
        "canonical_statement": "The source-bounded component 'I. The Astrocyte-Neuron Lattice (L2/L4 Modulation)' preserves Paper 0 records P0R05350-P0R05357 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05350:i_the_astrocyte_neuron_lattice_l2_l4_modulation",
            "P0R05351:i_the_astrocyte_neuron_lattice_l2_l4_modulation",
            "P0R05352:i_the_astrocyte_neuron_lattice_l2_l4_modulation",
            "P0R05353:i_the_astrocyte_neuron_lattice_l2_l4_modulation",
            "P0R05354:i_the_astrocyte_neuron_lattice_l2_l4_modulation",
            "P0R05355:i_the_astrocyte_neuron_lattice_l2_l4_modulation",
            "P0R05356:i_the_astrocyte_neuron_lattice_l2_l4_modulation",
            "P0R05357:i_the_astrocyte_neuron_lattice_l2_l4_modulation",
        ),
        "source_formulae": (
            "P0R05350: I. The Astrocyte-Neuron Lattice (L2/L4 Modulation)",
            "P0R05351: The brain's glial cells, particularly astrocytes, form a parallel, interconnected network that modulates neuronal activity on slower timescales.",
            "P0R05352: This is not a passive support structure but an active computational layer.",
            'P0R05353: Mechanism: Astrocytes communicate via intercellular calcium (Ca2+) waves, which are significantly slower than neuronal action potentials. These waves trigger the release of "gliotransmitters" that can alter synaptic release probabilities and neuronal excitability over broad spatial domains. | Formalism (UPDE Extension): The influence of the astrocyte lattice is integrated into the Unified Phase Dynamics Equation (UPDE) by modulating the neuronal coupling and noise terms. The state of the local astrocyte network, A(x,t), modifies the UPDE as follows: Noise Buffering: The noise term, i(t), is attenuated by glial homeostatic functions:',
            "P0R05354: $\\eta i(t) \\rightarrow \\eta i(t)\\text{/}gA\\left( A(x,t) \\right)$",
            "P0R05355: , where gA>1is a glial buffering factor.",
            "P0R05356: Coupling Modulation: The intra-layer coupling strength, Kij, becomes a function of local gliotransmitter concentrations, which are dependent on A(x,t).",
            "P0R05357: Function: The astrocyte network acts as a homeostatic regulator for the neuronal network, guiding it toward the quasicritical regime (sigma1) and preventing runaway excitation or quiescence. It provides a slow, integrative context that stabilises the faster, moment-to-moment computations of the neurons.",
        ),
        "test_protocols": (
            "preserve I. The Astrocyte-Neuron Lattice (L2/L4 Modulation) source-accounting boundary",
        ),
        "null_results": (
            "I. The Astrocyte-Neuron Lattice (L2/L4 Modulation) is not empirical validation evidence",
        ),
        "variables": ("i_the_astrocyte_neuron_lattice_l2_l4_modulation",),
        "validation_targets": ("preserve records P0R05350-P0R05357",),
        "null_controls": (
            "i_the_astrocyte_neuron_lattice_l2_l4_modulation must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class TheSlowControlLayerGlialAndImmuneModulationSpec:
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
class TheSlowControlLayerGlialAndImmuneModulationSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheSlowControlLayerGlialAndImmuneModulationSpec, ...]
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


def build_the_slow_control_layer_glial_and_immune_modulation_specs(
    source_records: list[dict[str, Any]],
) -> TheSlowControlLayerGlialAndImmuneModulationSpecBundle:
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

    specs: list[TheSlowControlLayerGlialAndImmuneModulationSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheSlowControlLayerGlialAndImmuneModulationSpec(
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
        "title": "Paper 0 " + "The Slow Control Layer - Glial and Immune Modulation" + " Specs",
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
        "next_source_boundary": "P0R05358",
    }
    return TheSlowControlLayerGlialAndImmuneModulationSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheSlowControlLayerGlialAndImmuneModulationSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_slow_control_layer_glial_and_immune_modulation_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: TheSlowControlLayerGlialAndImmuneModulationSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "The Slow Control Layer - Glial and Immune Modulation" + " Specs",
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
    bundle: TheSlowControlLayerGlialAndImmuneModulationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_slow_control_layer_glial_and_immune_modulation_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_slow_control_layer_glial_and_immune_modulation_validation_specs_{date_tag}.md"
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
