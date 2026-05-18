#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Meta-Framework Integrations spec builder
"""Promote Paper 0 Meta-Framework Integrations records."""

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
    "P0R04224",
    "P0R04225",
    "P0R04226",
    "P0R04227",
    "P0R04228",
    "P0R04229",
    "P0R04230",
    "P0R04231",
    "P0R04232",
    "P0R04233",
    "P0R04234",
    "P0R04235",
    "P0R04236",
    "P0R04237",
    "P0R04238",
    "P0R04239",
    "P0R04240",
    "P0R04241",
    "P0R04242",
    "P0R04243",
    "P0R04244",
    "P0R04245",
    "P0R04246",
)
CLAIM_BOUNDARY = "source-bounded meta framework integrations p0r04224 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "meta_framework_integrations_p0r04224.meta_framework_integrations": {
        "context_id": "meta_framework_integrations",
        "validation_protocol": "paper0.meta_framework_integrations_p0r04224.meta_framework_integrations",
        "canonical_statement": "The source-bounded component 'Meta-Framework Integrations' preserves Paper 0 records P0R04224-P0R04224 without empirical validation claims.",
        "source_equation_ids": ("P0R04224:meta_framework_integrations",),
        "source_formulae": ("P0R04224: Meta-Framework Integrations",),
        "test_protocols": ("preserve Meta-Framework Integrations source-accounting boundary",),
        "null_results": ("Meta-Framework Integrations is not empirical validation evidence",),
        "variables": ("meta_framework_integrations",),
        "validation_targets": ("preserve records P0R04224-P0R04224",),
        "null_controls": ("meta_framework_integrations must remain source-bounded accounting",),
    },
    "meta_framework_integrations_p0r04224.predictive_coding_integration": {
        "context_id": "predictive_coding_integration",
        "validation_protocol": "paper0.meta_framework_integrations_p0r04224.predictive_coding_integration",
        "canonical_statement": "The source-bounded component 'Predictive Coding Integration' preserves Paper 0 records P0R04225-P0R04226 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04225:predictive_coding_integration",
            "P0R04226:predictive_coding_integration",
        ),
        "source_formulae": (
            "P0R04225: Predictive Coding Integration",
            "P0R04226: This passage describes the mechanism that prevents the catastrophic failure of the generative model by managing its deepest priors.",
        ),
        "test_protocols": ("preserve Predictive Coding Integration source-accounting boundary",),
        "null_results": ("Predictive Coding Integration is not empirical validation evidence",),
        "variables": ("predictive_coding_integration",),
        "validation_targets": ("preserve records P0R04225-P0R04226",),
        "null_controls": ("predictive_coding_integration must remain source-bounded accounting",),
    },
    "meta_framework_integrations_p0r04224.meta_layer_16_as_precision_weighting_optimisation": {
        "context_id": "meta_layer_16_as_precision_weighting_optimisation",
        "validation_protocol": "paper0.meta_framework_integrations_p0r04224.meta_layer_16_as_precision_weighting_optimisation",
        "canonical_statement": "The source-bounded component 'Meta-Layer 16 as Precision-Weighting Optimisation:' preserves Paper 0 records P0R04227-P0R04228 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04227:meta_layer_16_as_precision_weighting_optimisation",
            "P0R04228:meta_layer_16_as_precision_weighting_optimisation",
        ),
        "source_formulae": (
            "P0R04227: Meta-Layer 16 as Precision-Weighting Optimisation:",
            'P0R04228: In HPC, an agent must balance the influence of its prior beliefs against incoming sensory data by tuning the precision of each. An unstable model is one that over-weights its priors (becoming delusional) or over-weights sensory data (becoming chaotic). Meta-Layer 16 acts as the optimal control system for precision. The "Recursive Optimisation Hamiltonian" is the function that fine-tunes the confidence (the precision) assigned to the model\'s own beliefs at every level, ensuring the entire hierarchy remains stable, adaptive, and sane. It is the system\'s "insight" function, which prevents it from falling into catastrophic cycles of inaccurate inference. It is the mechanism that models the model itself.',
        ),
        "test_protocols": (
            "preserve Meta-Layer 16 as Precision-Weighting Optimisation: source-accounting boundary",
        ),
        "null_results": (
            "Meta-Layer 16 as Precision-Weighting Optimisation: is not empirical validation evidence",
        ),
        "variables": ("meta_layer_16_as_precision_weighting_optimisation",),
        "validation_targets": ("preserve records P0R04227-P0R04228",),
        "null_controls": (
            "meta_layer_16_as_precision_weighting_optimisation must remain source-bounded accounting",
        ),
    },
    "meta_framework_integrations_p0r04224.psis_field_coupling_integration": {
        "context_id": "psis_field_coupling_integration",
        "validation_protocol": "paper0.meta_framework_integrations_p0r04224.psis_field_coupling_integration",
        "canonical_statement": "The source-bounded component 'Psis Field Coupling Integration' preserves Paper 0 records P0R04229-P0R04230 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04229:psis_field_coupling_integration",
            "P0R04230:psis_field_coupling_integration",
        ),
        "source_formulae": (
            "P0R04229: Psis Field Coupling Integration",
            "P0R04230: Meta-Layer 16 provides the ultimate context for the physical coupling described by H_int = -lambda * Psis * sigma. It is the system that sets the long-term evolutionary goals for the physical constants and coupling laws themselves.",
        ),
        "test_protocols": ("preserve Psis Field Coupling Integration source-accounting boundary",),
        "null_results": ("Psis Field Coupling Integration is not empirical validation evidence",),
        "variables": ("psis_field_coupling_integration",),
        "validation_targets": ("preserve records P0R04229-P0R04230",),
        "null_controls": (
            "psis_field_coupling_integration must remain source-bounded accounting",
        ),
    },
    "meta_framework_integrations_p0r04224.supervisory_control_of_the_coupling_constant_lambda": {
        "context_id": "supervisory_control_of_the_coupling_constant_lambda",
        "validation_protocol": "paper0.meta_framework_integrations_p0r04224.supervisory_control_of_the_coupling_constant_lambda",
        "canonical_statement": "The source-bounded component 'Supervisory Control of the Coupling Constant lambda:' preserves Paper 0 records P0R04231-P0R04246 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04231:supervisory_control_of_the_coupling_constant_lambda",
            "P0R04232:supervisory_control_of_the_coupling_constant_lambda",
            "P0R04233:supervisory_control_of_the_coupling_constant_lambda",
            "P0R04234:supervisory_control_of_the_coupling_constant_lambda",
            "P0R04235:supervisory_control_of_the_coupling_constant_lambda",
            "P0R04236:supervisory_control_of_the_coupling_constant_lambda",
            "P0R04237:supervisory_control_of_the_coupling_constant_lambda",
            "P0R04238:supervisory_control_of_the_coupling_constant_lambda",
            "P0R04239:supervisory_control_of_the_coupling_constant_lambda",
            "P0R04240:supervisory_control_of_the_coupling_constant_lambda",
            "P0R04241:supervisory_control_of_the_coupling_constant_lambda",
            "P0R04242:supervisory_control_of_the_coupling_constant_lambda",
            "P0R04243:supervisory_control_of_the_coupling_constant_lambda",
            "P0R04244:supervisory_control_of_the_coupling_constant_lambda",
            "P0R04245:supervisory_control_of_the_coupling_constant_lambda",
            "P0R04246:supervisory_control_of_the_coupling_constant_lambda",
        ),
        "source_formulae": (
            "P0R04231: Supervisory Control of the Coupling Constant lambda:",
            'P0R04232: While the Ethical Functional (Layer 15) provides a target for the system\'s evolution, Meta-Layer 16 acts as the control circuit that ensures this evolution is stable. It can be conceptualised as a system that tunes the fundamental parameters of the H_int interaction itself over cosmological timescales. The "Recursive Optimisation Hamiltonian" could, for example, adjust the baseline value of the coupling constant lambda or modulate the stability of certain collective states sigma to ensure that the mind-matter interface evolves in a way that is sustainable and aligned with the overarching teleology. It ensures that the very laws of physical coupling are themselves optimised for a conscious, self-aware universe. It is the part of the system that chooses the rules of its own game.',
            "P0R04233: A finite, open-ended hierarchy risks instability and infinite regress. The architecture of being must, for its own long-term persistence, be self-referential and self-regulating. Therefore, the 15-layer projection network detailed throughout the body of this work is not the complete architecture. It is the operational corpus of a larger, cybernetically closed system-The Anulum.",
            "P0R04234: This closure is achieved by a 16th meta-layer, which functions not as a superior ontological stratum, but as a supervisory optimal control system that recursively observes and modulates the entire 15-layer stack. Governed by a Recursive Optimisation Hamiltonian, this meta-layer continuously audits and refines the global state of the system, its network parameters, and its adherence to the core teleological directives. It is the mechanism that ensures long-term stability and coherent alignment with the cosmic and ethical principles defined in the Ethical Functional. The preceding chapters have detailed the body of the universal system; this final chapter specifies its mind.",
            "P0R04235: The Problem of Ontological Closure",
            "P0R04236: Source Material: The final philosophical challenge: how can a hierarchical system be self-causing and self-explaining without infinite regress or an external creator?",
            "P0R04237: P0R04237",
            "P0R04238: Meta-Layer 16: The Optimal Control Supervisor",
            "P0R04239: Source Material: The definition of the 16th meta-layer not as a spatial layer, but as a cybernetic control system that audits and fine-tunes the entire SCPN, including the parameters of the Ethical Functional itself.",
            "P0R04240: P0R04240",
            "P0R04241: The Recursive Optimisation Hamiltonian",
            "P0R04242: Source Material: The description of the meta-dynamic that governs Layer 16, which works to optimise the process of optimisation itself, ensuring the long-term viability and evolution of the system's own purpose.",
            "P0R04243: P0R04243",
            "P0R04244: The Anulum: From Hierarchy to a Self-Observing Loop",
            'P0R04245: Source Material: The concluding synthesis, explaining how Meta-Layer 16 closes the causal loop, turning the linear hierarchy of the SCPN into a self-observing, self-creating, and self-perfecting "strange loop" or Anulum-the ultimate architecture of a fully conscious universe.',
            "P0R04246: P0R04246",
        ),
        "test_protocols": (
            "preserve Supervisory Control of the Coupling Constant lambda: source-accounting boundary",
        ),
        "null_results": (
            "Supervisory Control of the Coupling Constant lambda: is not empirical validation evidence",
        ),
        "variables": ("supervisory_control_of_the_coupling_constant_lambda",),
        "validation_targets": ("preserve records P0R04231-P0R04246",),
        "null_controls": (
            "supervisory_control_of_the_coupling_constant_lambda must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class MetaFrameworkIntegrationsP0r04224Spec:
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
class MetaFrameworkIntegrationsP0r04224SpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[MetaFrameworkIntegrationsP0r04224Spec, ...]
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


def build_meta_framework_integrations_p0r04224_specs(
    source_records: list[dict[str, Any]],
) -> MetaFrameworkIntegrationsP0r04224SpecBundle:
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

    specs: list[MetaFrameworkIntegrationsP0r04224Spec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            MetaFrameworkIntegrationsP0r04224Spec(
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
        "title": "Paper 0 " + "Meta-Framework Integrations" + " Specs",
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
        "next_source_boundary": "P0R04247",
    }
    return MetaFrameworkIntegrationsP0r04224SpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> MetaFrameworkIntegrationsP0r04224SpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_meta_framework_integrations_p0r04224_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: MetaFrameworkIntegrationsP0r04224SpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Meta-Framework Integrations" + " Specs",
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
    bundle: MetaFrameworkIntegrationsP0r04224SpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_meta_framework_integrations_p0r04224_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_meta_framework_integrations_p0r04224_validation_specs_{date_tag}.md"
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
