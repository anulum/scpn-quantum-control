#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Microbiome as a Foundational Control Layer spec builder
"""Promote Paper 0 The Microbiome as a Foundational Control Layer records."""

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
    "P0R05479",
    "P0R05480",
    "P0R05481",
    "P0R05482",
    "P0R05483",
    "P0R05484",
    "P0R05485",
    "P0R05486",
    "P0R05487",
    "P0R05488",
    "P0R05489",
    "P0R05490",
    "P0R05491",
    "P0R05492",
)
CLAIM_BOUNDARY = "source-bounded the microbiome as a foundational control layer source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_microbiome_as_a_foundational_control_layer.the_microbiome_as_a_foundational_control_layer": {
        "context_id": "the_microbiome_as_a_foundational_control_layer",
        "validation_protocol": "paper0.the_microbiome_as_a_foundational_control_layer.the_microbiome_as_a_foundational_control_layer",
        "canonical_statement": "The source-bounded component 'The Microbiome as a Foundational Control Layer' preserves Paper 0 records P0R05479-P0R05484 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05479:the_microbiome_as_a_foundational_control_layer",
            "P0R05480:the_microbiome_as_a_foundational_control_layer",
            "P0R05481:the_microbiome_as_a_foundational_control_layer",
            "P0R05482:the_microbiome_as_a_foundational_control_layer",
            "P0R05483:the_microbiome_as_a_foundational_control_layer",
            "P0R05484:the_microbiome_as_a_foundational_control_layer",
        ),
        "source_formulae": (
            "P0R05479: The Microbiome as a Foundational Control Layer",
            "P0R05480: The Microbiome-Gut-Brain Axis: A Foundational Control Interface",
            "P0R05481: The embodied control hierarchy extends beyond the glial and endocrine systems to include the gut microbiome, which forms a foundational interface between the organism and its environment. The Microbiome-Gut-Brain (MGB) axis is a bidirectional communication network linking the central nervous system with the enteric nervous system, modulated by immune and endocrine pathways.",
            "P0R05482: The microbiome exerts a profound influence on the biological substrate of the SCPN:",
            'P0R05483: Neurochemical Modulation (L2): Gut microbes produce a vast array of neuroactive compounds, including precursors and metabolites of key neurotransmitters (e.g., serotonin, GABA). Dysbiosis directly alters the neurochemical milieu of the brain. | Immune Regulation (L1/L2 Interface): The microbiome is essential for the regulation of the immune system. It maintains the integrity of the gut barrier and modulates the balance of cytokines. A compromised gut barrier ("leaky gut") leads to systemic inflammation (increased C_cyto), which, as formalized above, directly degrades the topology of the qualia manifold (L5) by increasing noise ((t)) in the UPDE. | HPA Axis Modulation: The microbiome influences the sensitivity and set-points of the HPA axis, modulating the stress response.',
            "P0R05484: Formalism: The state of the microbiome (M_state) can be integrated into the SCPN by treating it as a slow-varying parameter that modulates the inputs to the neuro-immune and neuroendocrine control systems. Specifically, the systemic cytokine concentration C_cyto and the HPA axis parameters become functions of M_state. This integration highlights that the coherence of the organismal field (L5) is dependent on the ecological balance of the symbiotic microbial community it hosts.",
        ),
        "test_protocols": (
            "preserve The Microbiome as a Foundational Control Layer source-accounting boundary",
        ),
        "null_results": (
            "The Microbiome as a Foundational Control Layer is not empirical validation evidence",
        ),
        "variables": ("the_microbiome_as_a_foundational_control_layer",),
        "validation_targets": ("preserve records P0R05479-P0R05484",),
        "null_controls": (
            "the_microbiome_as_a_foundational_control_layer must remain source-bounded accounting",
        ),
    },
    "the_microbiome_as_a_foundational_control_layer.a_two_timescale_model_of_glial_neuronal_coupling": {
        "context_id": "a_two_timescale_model_of_glial_neuronal_coupling",
        "validation_protocol": "paper0.the_microbiome_as_a_foundational_control_layer.a_two_timescale_model_of_glial_neuronal_coupling",
        "canonical_statement": "The source-bounded component 'A Two-Timescale Model of Glial-Neuronal Coupling' preserves Paper 0 records P0R05485-P0R05492 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05485:a_two_timescale_model_of_glial_neuronal_coupling",
            "P0R05486:a_two_timescale_model_of_glial_neuronal_coupling",
            "P0R05487:a_two_timescale_model_of_glial_neuronal_coupling",
            "P0R05488:a_two_timescale_model_of_glial_neuronal_coupling",
            "P0R05489:a_two_timescale_model_of_glial_neuronal_coupling",
            "P0R05490:a_two_timescale_model_of_glial_neuronal_coupling",
            "P0R05491:a_two_timescale_model_of_glial_neuronal_coupling",
            "P0R05492:a_two_timescale_model_of_glial_neuronal_coupling",
        ),
        "source_formulae": (
            "P0R05485: A Two-Timescale Model of Glial-Neuronal Coupling",
            "P0R05486: The stability of the quasicritical dynamics in Layer 4 is not a product of fine-tuning but emerges from a two-timescale control system involving Glial-Neuronal coupling. The fast dynamics of neuronal avalanches are modulated by the slow dynamics of the astrocyte network (Layer 2).",
            "P0R05487: This can be formalised by making the effective coupling or excitability parameter in the neuronal network model (e.g., the branching parameter sigma in a critical branching process) a direct function of the local astrocyte calcium concentration, [Ca2+]A.",
            "P0R05488: For instance, the evolution of neuronal activity can be described by a fast-timescale equation. At the same time, the astrocyte calcium concentration evolves on a slow timescale, driven by neuronal activity and releasing gliotransmitters that modulate neuronal excitability.",
            "P0R05489: This creates a homeostatic feedback loop where hyperactivity in the neuronal network triggers a slow, integrative astrocytic response that downregulates excitability, and vice versa, gently steering the neuronal system back toward the critical point (sigma1).",
            'P0R05490: The proposed solution lies in the astrocyte network, which functions as a "slow control" system for the fast neuronal network. Astrocytes, the most abundant glial cells in the brain, are known to modulate neuronal signalling at multiple levels. They exhibit slow calcium waves, with characteristic frequencies in the 0.01-0.1 Hz range, that can propagate over long distances.',
            "P0R05491: These calcium elevations are stimulated by neuronal activity and, in turn, trigger the release of gliotransmitters that regulate synaptic function and neuronal excitability. This establishes a slow-acting feedback loop. The significant separation of timescales between fast neuronal dynamics (milliseconds) and slow glial modulation (seconds to minutes) is a classic feature of robust control systems.",
            "P0R05492: The astrocyte network acts as the governor of the neuronal engine, providing the homeostatic stability that allows the fast network to operate safely at its computationally optimal critical point, thus solving the fine-tuning problem for Layer 4",
        ),
        "test_protocols": (
            "preserve A Two-Timescale Model of Glial-Neuronal Coupling source-accounting boundary",
        ),
        "null_results": (
            "A Two-Timescale Model of Glial-Neuronal Coupling is not empirical validation evidence",
        ),
        "variables": ("a_two_timescale_model_of_glial_neuronal_coupling",),
        "validation_targets": ("preserve records P0R05485-P0R05492",),
        "null_controls": (
            "a_two_timescale_model_of_glial_neuronal_coupling must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class TheMicrobiomeAsAFoundationalControlLayerSpec:
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
class TheMicrobiomeAsAFoundationalControlLayerSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheMicrobiomeAsAFoundationalControlLayerSpec, ...]
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


def build_the_microbiome_as_a_foundational_control_layer_specs(
    source_records: list[dict[str, Any]],
) -> TheMicrobiomeAsAFoundationalControlLayerSpecBundle:
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

    specs: list[TheMicrobiomeAsAFoundationalControlLayerSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheMicrobiomeAsAFoundationalControlLayerSpec(
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
        "title": "Paper 0 " + "The Microbiome as a Foundational Control Layer" + " Specs",
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
        "next_source_boundary": "P0R05493",
    }
    return TheMicrobiomeAsAFoundationalControlLayerSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheMicrobiomeAsAFoundationalControlLayerSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_microbiome_as_a_foundational_control_layer_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: TheMicrobiomeAsAFoundationalControlLayerSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "The Microbiome as a Foundational Control Layer" + " Specs",
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
    bundle: TheMicrobiomeAsAFoundationalControlLayerSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_microbiome_as_a_foundational_control_layer_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_microbiome_as_a_foundational_control_layer_validation_specs_{date_tag}.md"
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
