#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0  as a Measure of Causal Efficacy: spec builder
"""Promote Paper 0  as a Measure of Causal Efficacy: records."""

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
    "P0R03295",
    "P0R03296",
    "P0R03297",
    "P0R03298",
    "P0R03299",
    "P0R03300",
    "P0R03301",
    "P0R03302",
    "P0R03303",
    "P0R03304",
    "P0R03305",
    "P0R03306",
)
CLAIM_BOUNDARY = "source-bounded as a measure of causal efficacy source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "as_a_measure_of_causal_efficacy.as_a_measure_of_causal_efficacy": {
        "context_id": "as_a_measure_of_causal_efficacy",
        "validation_protocol": "paper0.as_a_measure_of_causal_efficacy.as_a_measure_of_causal_efficacy",
        "canonical_statement": "The source-bounded component 'as a Measure of Causal Efficacy:' preserves Paper 0 records P0R03295-P0R03296 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03295:as_a_measure_of_causal_efficacy",
            "P0R03296:as_a_measure_of_causal_efficacy",
        ),
        "source_formulae": (
            "P0R03295: as a Measure of Causal Efficacy:",
            "P0R03296: The relationship T_CIGD / E_ is a powerful statement. It implies that the causal efficacy of the Psi-field-its ability to shape physical reality by collapsing potentiality into actuality-is directly proportional to its degree of integrated information (). A system with a higher has a greater power to make reality definite. This provides a deep, physical meaning to the teleological drive to maximise : it is a drive to create systems that are more potent co-creators of reality itself.",
        ),
        "test_protocols": (
            "preserve as a Measure of Causal Efficacy: source-accounting boundary",
        ),
        "null_results": ("as a Measure of Causal Efficacy: is not empirical validation evidence",),
        "variables": ("as_a_measure_of_causal_efficacy",),
        "validation_targets": ("preserve records P0R03295-P0R03296",),
        "null_controls": (
            "as_a_measure_of_causal_efficacy must remain source-bounded accounting",
        ),
    },
    "as_a_measure_of_causal_efficacy.the_quantum_gravity_interface_and_cigd": {
        "context_id": "the_quantum_gravity_interface_and_cigd",
        "validation_protocol": "paper0.as_a_measure_of_causal_efficacy.the_quantum_gravity_interface_and_cigd",
        "canonical_statement": "The source-bounded component 'The Quantum-Gravity Interface and CIGD' preserves Paper 0 records P0R03297-P0R03298 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03297:the_quantum_gravity_interface_and_cigd",
            "P0R03298:the_quantum_gravity_interface_and_cigd",
        ),
        "source_formulae": (
            "P0R03297: The Quantum-Gravity Interface and CIGD",
            "P0R03298: The Geometric Coupling manifests at the quantum scale (L1), providing a mechanism for Objective Reduction (OR).",
        ),
        "test_protocols": (
            "preserve The Quantum-Gravity Interface and CIGD source-accounting boundary",
        ),
        "null_results": (
            "The Quantum-Gravity Interface and CIGD is not empirical validation evidence",
        ),
        "variables": ("the_quantum_gravity_interface_and_cigd",),
        "validation_targets": ("preserve records P0R03297-P0R03298",),
        "null_controls": (
            "the_quantum_gravity_interface_and_cigd must remain source-bounded accounting",
        ),
    },
    "as_a_measure_of_causal_efficacy.consciousness_induced_gravitational_decoherence_cigd": {
        "context_id": "consciousness_induced_gravitational_decoherence_cigd",
        "validation_protocol": "paper0.as_a_measure_of_causal_efficacy.consciousness_induced_gravitational_decoherence_cigd",
        "canonical_statement": "The source-bounded component 'Consciousness-Induced Gravitational Decoherence (CIGD):' preserves Paper 0 records P0R03299-P0R03306 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03299:consciousness_induced_gravitational_decoherence_cigd",
            "P0R03300:consciousness_induced_gravitational_decoherence_cigd",
            "P0R03301:consciousness_induced_gravitational_decoherence_cigd",
            "P0R03302:consciousness_induced_gravitational_decoherence_cigd",
            "P0R03303:consciousness_induced_gravitational_decoherence_cigd",
            "P0R03304:consciousness_induced_gravitational_decoherence_cigd",
            "P0R03305:consciousness_induced_gravitational_decoherence_cigd",
            "P0R03306:consciousness_induced_gravitational_decoherence_cigd",
        ),
        "source_formulae": (
            "P0R03299: Consciousness-Induced Gravitational Decoherence (CIGD):",
            "P0R03300: The Psi-field, when highly integrated (>Crit), increases effective spacetime curvature fluctuations. This accelerates the decoherence of superpositions. The decoherence timescale (TCIGD) is inversely proportional to the Informational Self-Energy (E):",
            "P0R03301: TCIGD/E",
            "P0R03302: This formally links the intensity of consciousness () to the collapse of the wavefunction.",
            "P0R03303: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Zahl enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]Fig.: CIGD Mechanism (Geometric Coupling Route). This schematic details the formal mechanism by which the Psi-field induces quantum collapse. This panel cleanly presents the proposed ->geometry->decoherence pathway, tying \\Phi's magnitude to a concrete timescale TCIGDT_{\\text{CIGD}}TCIGD.",
            "P0R03304: An integrated consciousness field with strength >crit\\Phi>\\Phi_{\\text{crit}}>crit geometrically couples to spacetime, inducing effective curvature fluctuations gmu\\delta g_{\\mu\\nu}gmu. A quantum superposition psi=cii|\\psi\\rangle=\\sum c_i|i\\ranglepsi=cii interacting with these fluctuations undergoes accelerated decoherence, yielding a collapsed outcome i|i\\ranglei. The characteristic decoherence timescale scales inversely with the field's energetic content,",
            "P0R03305: TCIGD E,E,T_{\\text{CIGD}} \\;\\approx\\; \\frac{\\hbar}{E_{\\Phi}}, \\qquad E_{\\Phi}\\propto \\Phi,TCIGDE,E,",
            "P0R03306: highlighting that stronger, more integrated \\Phi reduces the time to decohere.",
        ),
        "test_protocols": (
            "preserve Consciousness-Induced Gravitational Decoherence (CIGD): source-accounting boundary",
        ),
        "null_results": (
            "Consciousness-Induced Gravitational Decoherence (CIGD): is not empirical validation evidence",
        ),
        "variables": ("consciousness_induced_gravitational_decoherence_cigd",),
        "validation_targets": ("preserve records P0R03299-P0R03306",),
        "null_controls": (
            "consciousness_induced_gravitational_decoherence_cigd must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class AsAMeasureOfCausalEfficacySpec:
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
class AsAMeasureOfCausalEfficacySpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[AsAMeasureOfCausalEfficacySpec, ...]
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


def build_as_a_measure_of_causal_efficacy_specs(
    source_records: list[dict[str, Any]],
) -> AsAMeasureOfCausalEfficacySpecBundle:
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

    specs: list[AsAMeasureOfCausalEfficacySpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            AsAMeasureOfCausalEfficacySpec(
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
        "title": "Paper 0 " + " as a Measure of Causal Efficacy:" + " Specs",
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
        "next_source_boundary": "P0R03307",
    }
    return AsAMeasureOfCausalEfficacySpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> AsAMeasureOfCausalEfficacySpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_as_a_measure_of_causal_efficacy_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: AsAMeasureOfCausalEfficacySpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + " as a Measure of Causal Efficacy:" + " Specs",
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
    bundle: AsAMeasureOfCausalEfficacySpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_as_a_measure_of_causal_efficacy_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_as_a_measure_of_causal_efficacy_validation_specs_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 causal-efficacy measure specs from the ledger."""

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
