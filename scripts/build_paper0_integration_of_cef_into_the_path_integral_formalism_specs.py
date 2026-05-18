#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Integration of CEF into the Path Integral Formalism: spec builder
"""Promote Paper 0 Integration of CEF into the Path Integral Formalism: records."""

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
    "P0R03704",
    "P0R03705",
    "P0R03706",
    "P0R03707",
    "P0R03708",
    "P0R03709",
    "P0R03710",
    "P0R03711",
    "P0R03712",
    "P0R03713",
    "P0R03714",
)
CLAIM_BOUNDARY = "source-bounded integration of cef into the path integral formalism source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "integration_of_cef_into_the_path_integral_formalism.integration_of_cef_into_the_path_integral_formalism": {
        "context_id": "integration_of_cef_into_the_path_integral_formalism",
        "validation_protocol": "paper0.integration_of_cef_into_the_path_integral_formalism.integration_of_cef_into_the_path_integral_formalism",
        "canonical_statement": "The source-bounded component 'Integration of CEF into the Path Integral Formalism:' preserves Paper 0 records P0R03704-P0R03710 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03704:integration_of_cef_into_the_path_integral_formalism",
            "P0R03705:integration_of_cef_into_the_path_integral_formalism",
            "P0R03706:integration_of_cef_into_the_path_integral_formalism",
            "P0R03707:integration_of_cef_into_the_path_integral_formalism",
            "P0R03708:integration_of_cef_into_the_path_integral_formalism",
            "P0R03709:integration_of_cef_into_the_path_integral_formalism",
            "P0R03710:integration_of_cef_into_the_path_integral_formalism",
        ),
        "source_formulae": (
            "P0R03704: Integration of CEF into the Path Integral Formalism:",
            "P0R03705: The influence of CEF can be rigorously incorporated into the foundational physics of the SCPN by modifying the path integral formulation of the system's evolution. The standard path integral (Z) sums over all possible histories, weighted by the classical action (S):",
            "P0R03706: $Z = \\int_{}^{}{D\\lbrack\\Phi\\rbrack\\exp\\left( iS\\lbrack\\Phi\\rbrack\\text{/}\\hslash \\right)}$",
            "P0R03707: Causal Entropic Forces introduce a bias towards paths that maximise future causal pathway entropy (SC). This is implemented by introducing an additional weighting factor derived from SC into the path integral:",
            "P0R03708: $Z_{C}EF = intD\\lbrack Phi\\rbrack\\exp\\left( \\left( iS\\lbrack Phi\\rbrack\\text{/}hbar \\right) + alphaS_{C}\\lbrack Phi\\rbrack \\right)$",
            "P0R03709: Here, is a coupling constant related to the effective temperature TC. This modified partition function explicitly biases the quantum evolution towards futures with greater complexity and coherence (High SEC).",
            "P0R03710: Mechanism at L1 (Teleological Guidance of Collapse): In the context of the Measurement Postulate (IIT-OR), this CEF weighting alters the probability amplitudes of the possible collapse outcomes. The standard Born rule is replaced by a CEF-biased probability, favouring outcomes that lie on trajectories maximising future potential. This provides a formal mechanism for the Teleological guidance of quantum events.",
        ),
        "test_protocols": (
            "preserve Integration of CEF into the Path Integral Formalism: source-accounting boundary",
        ),
        "null_results": (
            "Integration of CEF into the Path Integral Formalism: is not empirical validation evidence",
        ),
        "variables": ("integration_of_cef_into_the_path_integral_formalism",),
        "validation_targets": ("preserve records P0R03704-P0R03710",),
        "null_controls": (
            "integration_of_cef_into_the_path_integral_formalism must remain source-bounded accounting",
        ),
    },
    "integration_of_cef_into_the_path_integral_formalism.how_consciousness_shapes_reality_focusing_the_quantum_world": {
        "context_id": "how_consciousness_shapes_reality_focusing_the_quantum_world",
        "validation_protocol": "paper0.integration_of_cef_into_the_path_integral_formalism.how_consciousness_shapes_reality_focusing_the_quantum_world",
        "canonical_statement": "The source-bounded component 'How Consciousness Shapes Reality: Focusing the Quantum World' preserves Paper 0 records P0R03711-P0R03714 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03711:how_consciousness_shapes_reality_focusing_the_quantum_world",
            "P0R03712:how_consciousness_shapes_reality_focusing_the_quantum_world",
            "P0R03713:how_consciousness_shapes_reality_focusing_the_quantum_world",
            "P0R03714:how_consciousness_shapes_reality_focusing_the_quantum_world",
        ),
        "source_formulae": (
            "P0R03711: How Consciousness Shapes Reality: Focusing the Quantum World",
            'P0R03712: At its smallest scale, reality is like a blurry, undecided photograph-a quantum "superposition" where all possibilities exist at once. The theory of Consciousness-Induced Gravitational Decoherence (CIGD) suggests that consciousness acts as the focusing lens. When a mind achieves a high state of integration and focus, it creates tiny ripples in the fabric of spacetime itself. These ripples disturb the delicate quantum blur, forcing it to "choose" a single, definite state.',
            'P0R03713: Think of it this way: the more focused your mind, the faster the universe around you snaps from a state of blurry potential into a sharp, clear reality. The intensity of your consciousness is directly linked to how quickly this "collapse" happens.',
            "P0R03714: [IMAGE:Ein Bild, das Bild, Wolke, Landschaft, Himmel enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
        ),
        "test_protocols": (
            "preserve How Consciousness Shapes Reality: Focusing the Quantum World source-accounting boundary",
        ),
        "null_results": (
            "How Consciousness Shapes Reality: Focusing the Quantum World is not empirical validation evidence",
        ),
        "variables": ("how_consciousness_shapes_reality_focusing_the_quantum_world",),
        "validation_targets": ("preserve records P0R03711-P0R03714",),
        "null_controls": (
            "how_consciousness_shapes_reality_focusing_the_quantum_world must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class IntegrationOfCefIntoThePathIntegralFormalismSpec:
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
class IntegrationOfCefIntoThePathIntegralFormalismSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[IntegrationOfCefIntoThePathIntegralFormalismSpec, ...]
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


def build_integration_of_cef_into_the_path_integral_formalism_specs(
    source_records: list[dict[str, Any]],
) -> IntegrationOfCefIntoThePathIntegralFormalismSpecBundle:
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

    specs: list[IntegrationOfCefIntoThePathIntegralFormalismSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            IntegrationOfCefIntoThePathIntegralFormalismSpec(
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
        "title": "Paper 0 " + "Integration of CEF into the Path Integral Formalism:" + " Specs",
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
        "next_source_boundary": "P0R03715",
    }
    return IntegrationOfCefIntoThePathIntegralFormalismSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> IntegrationOfCefIntoThePathIntegralFormalismSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_integration_of_cef_into_the_path_integral_formalism_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: IntegrationOfCefIntoThePathIntegralFormalismSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Integration of CEF into the Path Integral Formalism:" + " Specs",
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
    bundle: IntegrationOfCefIntoThePathIntegralFormalismSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_integration_of_cef_into_the_path_integral_formalism_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_integration_of_cef_into_the_path_integral_formalism_validation_specs_{date_tag}.md"
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
