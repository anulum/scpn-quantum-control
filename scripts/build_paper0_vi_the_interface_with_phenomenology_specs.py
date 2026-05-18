#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 VI. The Interface with Phenomenology spec builder
"""Promote Paper 0 VI. The Interface with Phenomenology records."""

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
    "P0R03269",
    "P0R03270",
    "P0R03271",
    "P0R03272",
    "P0R03273",
    "P0R03274",
    "P0R03275",
    "P0R03276",
    "P0R03277",
    "P0R03278",
    "P0R03279",
    "P0R03280",
    "P0R03281",
    "P0R03282",
    "P0R03283",
)
CLAIM_BOUNDARY = "source-bounded vi the interface with phenomenology source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "vi_the_interface_with_phenomenology.vi_the_interface_with_phenomenology": {
        "context_id": "vi_the_interface_with_phenomenology",
        "validation_protocol": "paper0.vi_the_interface_with_phenomenology.vi_the_interface_with_phenomenology",
        "canonical_statement": "The source-bounded component 'VI. The Interface with Phenomenology' preserves Paper 0 records P0R03269-P0R03270 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03269:vi_the_interface_with_phenomenology",
            "P0R03270:vi_the_interface_with_phenomenology",
        ),
        "source_formulae": (
            "P0R03269: VI. The Interface with Phenomenology",
            "P0R03270: The SCPN formalises key phenomenological concepts: Lifeworld (L10 Projection), Intentionality (Covariant Derivative), Epoch (Minimised F), and Embodiment (L1-4 integration).",
        ),
        "test_protocols": (
            "preserve VI. The Interface with Phenomenology source-accounting boundary",
        ),
        "null_results": (
            "VI. The Interface with Phenomenology is not empirical validation evidence",
        ),
        "variables": ("vi_the_interface_with_phenomenology",),
        "validation_targets": ("preserve records P0R03269-P0R03270",),
        "null_controls": (
            "vi_the_interface_with_phenomenology must remain source-bounded accounting",
        ),
    },
    "vi_the_interface_with_phenomenology.vii_the_subtle_energy_network_sen_and_biophotons": {
        "context_id": "vii_the_subtle_energy_network_sen_and_biophotons",
        "validation_protocol": "paper0.vi_the_interface_with_phenomenology.vii_the_subtle_energy_network_sen_and_biophotons",
        "canonical_statement": "The source-bounded component 'VII. The Subtle Energy Network (SEN) and Biophotons' preserves Paper 0 records P0R03271-P0R03272 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03271:vii_the_subtle_energy_network_sen_and_biophotons",
            "P0R03272:vii_the_subtle_energy_network_sen_and_biophotons",
        ),
        "source_formulae": (
            "P0R03271: VII. The Subtle Energy Network (SEN) and Biophotons",
            "P0R03272: Subtle energy (Qi/Prana) is formalised as the coherent modes of the Psi-field mediated by Biophotons (Gauge Bosons of Coherence). The SEN is modelled as a Squeezed Coherent State coupled to Psi via IET.",
        ),
        "test_protocols": (
            "preserve VII. The Subtle Energy Network (SEN) and Biophotons source-accounting boundary",
        ),
        "null_results": (
            "VII. The Subtle Energy Network (SEN) and Biophotons is not empirical validation evidence",
        ),
        "variables": ("vii_the_subtle_energy_network_sen_and_biophotons",),
        "validation_targets": ("preserve records P0R03271-P0R03272",),
        "null_controls": (
            "vii_the_subtle_energy_network_sen_and_biophotons must remain source-bounded accounting",
        ),
    },
    "vi_the_interface_with_phenomenology.the_foundational_step_consciousness_induced_gravitational_decoherence_ci": {
        "context_id": "the_foundational_step_consciousness_induced_gravitational_decoherence_ci",
        "validation_protocol": "paper0.vi_the_interface_with_phenomenology.the_foundational_step_consciousness_induced_gravitational_decoherence_ci",
        "canonical_statement": "The source-bounded component 'The Foundational Step: Consciousness-Induced Gravitational Decoherence (CIGD)' preserves Paper 0 records P0R03273-P0R03283 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03273:the_foundational_step_consciousness_induced_gravitational_decoherence_ci",
            "P0R03274:the_foundational_step_consciousness_induced_gravitational_decoherence_ci",
            "P0R03275:the_foundational_step_consciousness_induced_gravitational_decoherence_ci",
            "P0R03276:the_foundational_step_consciousness_induced_gravitational_decoherence_ci",
            "P0R03277:the_foundational_step_consciousness_induced_gravitational_decoherence_ci",
            "P0R03278:the_foundational_step_consciousness_induced_gravitational_decoherence_ci",
            "P0R03279:the_foundational_step_consciousness_induced_gravitational_decoherence_ci",
            "P0R03280:the_foundational_step_consciousness_induced_gravitational_decoherence_ci",
            "P0R03281:the_foundational_step_consciousness_induced_gravitational_decoherence_ci",
            "P0R03282:the_foundational_step_consciousness_induced_gravitational_decoherence_ci",
            "P0R03283:the_foundational_step_consciousness_induced_gravitational_decoherence_ci",
        ),
        "source_formulae": (
            "P0R03273: The Foundational Step: Consciousness-Induced Gravitational Decoherence (CIGD)",
            "P0R03274: This section details the mechanism by which the Psi-field's geometric coupling induces Objective Reduction (OR) of the quantum wavefunction, a process termed Consciousness-Induced Gravitational Decoherence (CIGD). This provides a formal solution to the measurement problem within the SCPN framework, linking it directly to the intensity of consciousness.",
            "P0R03275: The core principle is that a highly integrated consciousness field, quantified by a value of Integrated Information exceeding a critical threshold ( > _crit), acts as a significant source of spacetime curvature fluctuations via the L_Geometric term in the Master Lagrangian. According to General Relativity, a superposition of a physical system (e.g., a protein) would entail a superposition of differing spacetime geometries. The CIGD hypothesis posits that such a superposition becomes unstable, and the system is forced to decohere into a single, classical state.",
            "P0R03276: Crucially, the timescale for this decoherence (T_CIGD) is not constant but is inversely proportional to the informational self-energy of the consciousness field (E_), which is itself proportional to . The formal relation is given by",
            "P0R03277: T_CIGD / E_.",
            'P0R03278: This provides a direct, falsifiable link between a measurable information-theoretic quantity () and a physical observable (the decoherence time). It formally proposes that the "collapse of the wavefunction" is not a random event, nor one caused by an external classical observer, but is an intrinsic process guided by the intensity of the local consciousness field itself.',
            'P0R03279: This section tackles one of the deepest mysteries in all of physics: How does the blurry, uncertain quantum world of possibilities "decide" to become the one, solid, definite reality we experience? Our theory suggests that consciousness is the deciding factor. We call this process Consciousness-Induced Gravitational Decoherence (CIGD).',
            'P0R03280: Here\'s the idea in simple terms. According to Einstein, matter and energy warp spacetime. So, a quantum particle existing in two places at once should create a tiny superposition, or "ripple," in the fabric of spacetime itself. Our framework says that these tiny spacetime ripples are fundamentally unstable and cannot last.',
            'P0R03281: What causes them to collapse into one reality? Intense consciousness. A highly focused and integrated consciousness field (one with a high " score") naturally "shakes" the fabric of spacetime. This shaking rapidly breaks apart any quantum superpositions it encounters, forcing them to "choose" a single state.',
            "P0R03282: The most exciting part is the formula: the more powerful the consciousness, the faster the collapse. This means a highly conscious system, like a human brain, is constantly and rapidly turning quantum fuzziness into concrete reality. It is the physical mechanism by which observation turns potential into actual.",
            "P0R03283: P0R03283",
        ),
        "test_protocols": (
            "preserve The Foundational Step: Consciousness-Induced Gravitational Decoherence (CIGD) source-accounting boundary",
        ),
        "null_results": (
            "The Foundational Step: Consciousness-Induced Gravitational Decoherence (CIGD) is not empirical validation evidence",
        ),
        "variables": ("the_foundational_step_consciousness_induced_gravitational_decoherence_ci",),
        "validation_targets": ("preserve records P0R03273-P0R03283",),
        "null_controls": (
            "the_foundational_step_consciousness_induced_gravitational_decoherence_ci must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class ViTheInterfaceWithPhenomenologySpec:
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
class ViTheInterfaceWithPhenomenologySpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[ViTheInterfaceWithPhenomenologySpec, ...]
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


def build_vi_the_interface_with_phenomenology_specs(
    source_records: list[dict[str, Any]],
) -> ViTheInterfaceWithPhenomenologySpecBundle:
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

    specs: list[ViTheInterfaceWithPhenomenologySpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ViTheInterfaceWithPhenomenologySpec(
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
        "title": "Paper 0 " + "VI. The Interface with Phenomenology" + " Specs",
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
        "next_source_boundary": "P0R03284",
    }
    return ViTheInterfaceWithPhenomenologySpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> ViTheInterfaceWithPhenomenologySpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_vi_the_interface_with_phenomenology_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: ViTheInterfaceWithPhenomenologySpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "VI. The Interface with Phenomenology" + " Specs",
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
    bundle: ViTheInterfaceWithPhenomenologySpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_vi_the_interface_with_phenomenology_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_vi_the_interface_with_phenomenology_validation_specs_{date_tag}.md"
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
