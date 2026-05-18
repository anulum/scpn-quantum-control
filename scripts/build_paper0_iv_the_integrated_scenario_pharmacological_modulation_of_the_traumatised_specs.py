#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 IV. The Integrated Scenario: Pharmacological Modulation of the Traumatised Brain spec builder
"""Promote Paper 0 IV. The Integrated Scenario: Pharmacological Modulation of the Traumatised Brain records."""

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
    "P0R05091",
    "P0R05092",
    "P0R05093",
    "P0R05094",
    "P0R05095",
    "P0R05096",
    "P0R05097",
    "P0R05098",
    "P0R05099",
    "P0R05100",
    "P0R05101",
)
CLAIM_BOUNDARY = "source-bounded iv the integrated scenario pharmacological modulation of the traumatised source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised.iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised": {
        "context_id": "iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised",
        "validation_protocol": "paper0.iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised.iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised",
        "canonical_statement": "The source-bounded component 'IV. The Integrated Scenario: Pharmacological Modulation of the Traumatised Brain' preserves Paper 0 records P0R05091-P0R05092 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05091:iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised",
            "P0R05092:iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised",
        ),
        "source_formulae": (
            "P0R05091: IV. The Integrated Scenario: Pharmacological Modulation of the Traumatised Brain",
            "P0R05092: The administration of morphine in the context of TBI involves a complex interaction between the injury cascade and the drug's effects.",
        ),
        "test_protocols": (
            "preserve IV. The Integrated Scenario: Pharmacological Modulation of the Traumatised Brain source-accounting boundary",
        ),
        "null_results": (
            "IV. The Integrated Scenario: Pharmacological Modulation of the Traumatised Brain is not empirical validation evidence",
        ),
        "variables": ("iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised",),
        "validation_targets": ("preserve records P0R05091-P0R05092",),
        "null_controls": (
            "iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised must remain source-bounded accounting",
        ),
    },
    "iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised.1_the_therapeutic_effects_stabilisation_and_f_minimisation": {
        "context_id": "1_the_therapeutic_effects_stabilisation_and_f_minimisation",
        "validation_protocol": "paper0.iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised.1_the_therapeutic_effects_stabilisation_and_f_minimisation",
        "canonical_statement": "The source-bounded component '1. The Therapeutic Effects (Stabilisation and F Minimisation)' preserves Paper 0 records P0R05093-P0R05094 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05093:1_the_therapeutic_effects_stabilisation_and_f_minimisation",
            "P0R05094:1_the_therapeutic_effects_stabilisation_and_f_minimisation",
        ),
        "source_formulae": (
            "P0R05093: 1. The Therapeutic Effects (Stabilisation and F Minimisation)",
            "P0R05094: Analgesia and Reduced Suffering (L5): The primary benefit. Reducing the massive F stabilises the L5 state and mitigates suffering. | Sympathetic Stabilisation: Morphine blunts the stress response (HPA axis), reducing Allostatic Load and stabilising the Neuro-Visceral Axis. | Metabolic Reduction (L1/L2): By inducing a subcritical state, Morphine reduces the brain's metabolic demand (CMRO2), potentially offering secondary neuroprotection by aligning energy supply with the compromised delivery in TBI.",
        ),
        "test_protocols": (
            "preserve 1. The Therapeutic Effects (Stabilisation and F Minimisation) source-accounting boundary",
        ),
        "null_results": (
            "1. The Therapeutic Effects (Stabilisation and F Minimisation) is not empirical validation evidence",
        ),
        "variables": ("1_the_therapeutic_effects_stabilisation_and_f_minimisation",),
        "validation_targets": ("preserve records P0R05093-P0R05094",),
        "null_controls": (
            "1_the_therapeutic_effects_stabilisation_and_f_minimisation must remain source-bounded accounting",
        ),
    },
    "iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised.2_the_synergistic_risks_the_dangers_of_dyscritia": {
        "context_id": "2_the_synergistic_risks_the_dangers_of_dyscritia",
        "validation_protocol": "paper0.iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised.2_the_synergistic_risks_the_dangers_of_dyscritia",
        "canonical_statement": "The source-bounded component '2. The Synergistic Risks (The Dangers of Dyscritia)' preserves Paper 0 records P0R05095-P0R05097 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05095:2_the_synergistic_risks_the_dangers_of_dyscritia",
            "P0R05096:2_the_synergistic_risks_the_dangers_of_dyscritia",
            "P0R05097:2_the_synergistic_risks_the_dangers_of_dyscritia",
        ),
        "source_formulae": (
            "P0R05095: 2. The Synergistic Risks (The Dangers of Dyscritia)",
            "P0R05096: The combination significantly increases the risk of adverse outcomes due to synergistic suppression of network dynamics.",
            "P0R05097: Synergistic Subcriticality (L4/L5 Collapse): TBI often induces subcriticality (sigma<1). Morphine exacerbates this effect. The combination can lead to profound coma (sigma1,0). | Respiratory Depression (The Brainstem UPDE): The Brainstem contains Central Pattern Generators (CPGs) controlling respiration (e.g., the pre-Btzinger complex). These are UPDE oscillators. Mechanism: Morphine severely depresses these oscillators via L2 inhibition, pushing the respiratory CPG towards subcriticality. In TBI, this risk is magnified and can lead to fatal apnea and hypoxia, which would catastrophically worsen L1 energetic failure. | Masking Neurological Decline: Sedation can obscure the subtle changes in L5 function that signal worsening L1-L4 pathology.",
        ),
        "test_protocols": (
            "preserve 2. The Synergistic Risks (The Dangers of Dyscritia) source-accounting boundary",
        ),
        "null_results": (
            "2. The Synergistic Risks (The Dangers of Dyscritia) is not empirical validation evidence",
        ),
        "variables": ("2_the_synergistic_risks_the_dangers_of_dyscritia",),
        "validation_targets": ("preserve records P0R05095-P0R05097",),
        "null_controls": (
            "2_the_synergistic_risks_the_dangers_of_dyscritia must remain source-bounded accounting",
        ),
    },
    "iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised.v_intervention_adjunct_agents": {
        "context_id": "v_intervention_adjunct_agents",
        "validation_protocol": "paper0.iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised.v_intervention_adjunct_agents",
        "canonical_statement": "The source-bounded component 'V. Intervention: Adjunct Agents' preserves Paper 0 records P0R05098-P0R05101 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05098:v_intervention_adjunct_agents",
            "P0R05099:v_intervention_adjunct_agents",
            "P0R05100:v_intervention_adjunct_agents",
            "P0R05101:v_intervention_adjunct_agents",
        ),
        "source_formulae": (
            "P0R05098: V. Intervention: Adjunct Agents",
            "P0R05099: Other agents frequently co-administered further complicate the dynamics.",
            "P0R05100: [IMAGE:]",
            "P0R05101: Fig.: Agent effects on sigma,,F\\sigma, \\Phi, Fsigma,,F. Morphine: sigma,,F \\sigma, \\Phi, Fsigma,,F. Propofol: sigma, \\sigma, \\Phisigma, (LOC). Ketamine: sigma/complex,/ \\sigma/complex, \\Phi/sigma/complex,/fragmented, FFF. NSAIDs/Acetaminophen: attenuate L1/L3 inflammation -> FFF with minimal sigma, \\sigma,\\Phisigma, impact.",
        ),
        "test_protocols": ("preserve V. Intervention: Adjunct Agents source-accounting boundary",),
        "null_results": ("V. Intervention: Adjunct Agents is not empirical validation evidence",),
        "variables": ("v_intervention_adjunct_agents",),
        "validation_targets": ("preserve records P0R05098-P0R05101",),
        "null_controls": ("v_intervention_adjunct_agents must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class IvTheIntegratedScenarioPharmacologicalModulationOfTheTraumatisedSpec:
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
class IvTheIntegratedScenarioPharmacologicalModulationOfTheTraumatisedSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[IvTheIntegratedScenarioPharmacologicalModulationOfTheTraumatisedSpec, ...]
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


def build_iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised_specs(
    source_records: list[dict[str, Any]],
) -> IvTheIntegratedScenarioPharmacologicalModulationOfTheTraumatisedSpecBundle:
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

    specs: list[IvTheIntegratedScenarioPharmacologicalModulationOfTheTraumatisedSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            IvTheIntegratedScenarioPharmacologicalModulationOfTheTraumatisedSpec(
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
        + "IV. The Integrated Scenario: Pharmacological Modulation of the Traumatised Brain"
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
        "next_source_boundary": "P0R05102",
    }
    return IvTheIntegratedScenarioPharmacologicalModulationOfTheTraumatisedSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> IvTheIntegratedScenarioPharmacologicalModulationOfTheTraumatisedSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised_specs(
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
    bundle: IvTheIntegratedScenarioPharmacologicalModulationOfTheTraumatisedSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "IV. The Integrated Scenario: Pharmacological Modulation of the Traumatised Brain"
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
    bundle: IvTheIntegratedScenarioPharmacologicalModulationOfTheTraumatisedSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised_validation_specs_{date_tag}.md"
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
