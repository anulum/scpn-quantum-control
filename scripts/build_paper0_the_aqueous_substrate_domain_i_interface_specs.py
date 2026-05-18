#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Aqueous Substrate (Domain I Interface) spec builder
"""Promote Paper 0 The Aqueous Substrate (Domain I Interface) records."""

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
    "P0R05331",
    "P0R05332",
    "P0R05333",
    "P0R05334",
    "P0R05335",
    "P0R05336",
    "P0R05337",
    "P0R05338",
    "P0R05339",
    "P0R05340",
    "P0R05341",
    "P0R05342",
    "P0R05343",
    "P0R05344",
    "P0R05345",
    "P0R05346",
)
CLAIM_BOUNDARY = "source-bounded the aqueous substrate domain i interface source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_aqueous_substrate_domain_i_interface.the_aqueous_substrate_domain_i_interface": {
        "context_id": "the_aqueous_substrate_domain_i_interface",
        "validation_protocol": "paper0.the_aqueous_substrate_domain_i_interface.the_aqueous_substrate_domain_i_interface",
        "canonical_statement": "The source-bounded component 'The Aqueous Substrate (Domain I Interface)' preserves Paper 0 records P0R05331-P0R05333 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05331:the_aqueous_substrate_domain_i_interface",
            "P0R05332:the_aqueous_substrate_domain_i_interface",
            "P0R05333:the_aqueous_substrate_domain_i_interface",
        ),
        "source_formulae": (
            "P0R05331: The Aqueous Substrate (Domain I Interface)",
            "P0R05332: The Aqueous Substrate: Interfacial Water and Coherence Domains",
            "P0R05333: The biological substrate relies critically on Interfacial Water (structured water), which mediates L1-L4 dynamics.",
        ),
        "test_protocols": (
            "preserve The Aqueous Substrate (Domain I Interface) source-accounting boundary",
        ),
        "null_results": (
            "The Aqueous Substrate (Domain I Interface) is not empirical validation evidence",
        ),
        "variables": ("the_aqueous_substrate_domain_i_interface",),
        "validation_targets": ("preserve records P0R05331-P0R05333",),
        "null_controls": (
            "the_aqueous_substrate_domain_i_interface must remain source-bounded accounting",
        ),
    },
    "the_aqueous_substrate_domain_i_interface.coherence_domains_cds_predicted_by_qed_interfacial_water_forms_cds_where": {
        "context_id": "coherence_domains_cds_predicted_by_qed_interfacial_water_forms_cds_where",
        "validation_protocol": "paper0.the_aqueous_substrate_domain_i_interface.coherence_domains_cds_predicted_by_qed_interfacial_water_forms_cds_where",
        "canonical_statement": "The source-bounded component 'Coherence Domains (CDs): Predicted by QED, Interfacial water forms CDs where molecules oscillate in phase. This facilitates quasi-superconductivity (proton hopping) and stabilises quantum states.' preserves Paper 0 records P0R05334-P0R05334 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05334:coherence_domains_cds_predicted_by_qed_interfacial_water_forms_cds_where",
        ),
        "source_formulae": (
            "P0R05334: Coherence Domains (CDs): Predicted by QED, Interfacial water forms CDs where molecules oscillate in phase. This facilitates quasi-superconductivity (proton hopping) and stabilises quantum states.",
        ),
        "test_protocols": (
            "preserve Coherence Domains (CDs): Predicted by QED, Interfacial water forms CDs where molecules oscillate in phase. This facilitates quasi-superconductivity (proton hopping) and stabilises quantum states. source-accounting boundary",
        ),
        "null_results": (
            "Coherence Domains (CDs): Predicted by QED, Interfacial water forms CDs where molecules oscillate in phase. This facilitates quasi-superconductivity (proton hopping) and stabilises quantum states. is not empirical validation evidence",
        ),
        "variables": ("coherence_domains_cds_predicted_by_qed_interfacial_water_forms_cds_where",),
        "validation_targets": ("preserve records P0R05334-P0R05334",),
        "null_controls": (
            "coherence_domains_cds_predicted_by_qed_interfacial_water_forms_cds_where must remain source-bounded accounting",
        ),
    },
    "the_aqueous_substrate_domain_i_interface.integration_in_l1_cds_shield_microtubule_qubits_and_support_frhlich_cond": {
        "context_id": "integration_in_l1_cds_shield_microtubule_qubits_and_support_frhlich_cond",
        "validation_protocol": "paper0.the_aqueous_substrate_domain_i_interface.integration_in_l1_cds_shield_microtubule_qubits_and_support_frhlich_cond",
        "canonical_statement": "The source-bounded component 'Integration: In L1, CDs shield microtubule qubits and support Frhlich condensation. In L3/L4, bioelectric codes are mediated by proton currents (IProton) within this network. The network of CDs acts as a dynamic memory medium enabling rapid signalling.' preserves Paper 0 records P0R05335-P0R05335 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05335:integration_in_l1_cds_shield_microtubule_qubits_and_support_frhlich_cond",
        ),
        "source_formulae": (
            "P0R05335: Integration: In L1, CDs shield microtubule qubits and support Frhlich condensation. In L3/L4, bioelectric codes are mediated by proton currents (IProton) within this network. The network of CDs acts as a dynamic memory medium enabling rapid signalling.",
        ),
        "test_protocols": (
            "preserve Integration: In L1, CDs shield microtubule qubits and support Frhlich condensation. In L3/L4, bioelectric codes are mediated by proton currents (IProton) within this network. The network of CDs acts as a dynamic memory medium enabling rapid signalling. source-accounting boundary",
        ),
        "null_results": (
            "Integration: In L1, CDs shield microtubule qubits and support Frhlich condensation. In L3/L4, bioelectric codes are mediated by proton currents (IProton) within this network. The network of CDs acts as a dynamic memory medium enabling rapid signalling. is not empirical validation evidence",
        ),
        "variables": ("integration_in_l1_cds_shield_microtubule_qubits_and_support_frhlich_cond",),
        "validation_targets": ("preserve records P0R05335-P0R05335",),
        "null_controls": (
            "integration_in_l1_cds_shield_microtubule_qubits_and_support_frhlich_cond must remain source-bounded accounting",
        ),
    },
    "the_aqueous_substrate_domain_i_interface.the_genesis_of_life_abiogenesis_as_a_guided_phase_transition": {
        "context_id": "the_genesis_of_life_abiogenesis_as_a_guided_phase_transition",
        "validation_protocol": "paper0.the_aqueous_substrate_domain_i_interface.the_genesis_of_life_abiogenesis_as_a_guided_phase_transition",
        "canonical_statement": "The source-bounded component 'The Genesis of Life - Abiogenesis as a Guided Phase Transition' preserves Paper 0 records P0R05336-P0R05346 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05336:the_genesis_of_life_abiogenesis_as_a_guided_phase_transition",
            "P0R05337:the_genesis_of_life_abiogenesis_as_a_guided_phase_transition",
            "P0R05338:the_genesis_of_life_abiogenesis_as_a_guided_phase_transition",
            "P0R05339:the_genesis_of_life_abiogenesis_as_a_guided_phase_transition",
            "P0R05340:the_genesis_of_life_abiogenesis_as_a_guided_phase_transition",
            "P0R05341:the_genesis_of_life_abiogenesis_as_a_guided_phase_transition",
            "P0R05342:the_genesis_of_life_abiogenesis_as_a_guided_phase_transition",
            "P0R05343:the_genesis_of_life_abiogenesis_as_a_guided_phase_transition",
            "P0R05344:the_genesis_of_life_abiogenesis_as_a_guided_phase_transition",
            "P0R05345:the_genesis_of_life_abiogenesis_as_a_guided_phase_transition",
            "P0R05346:the_genesis_of_life_abiogenesis_as_a_guided_phase_transition",
        ),
        "source_formulae": (
            "P0R05336: The Genesis of Life - Abiogenesis as a Guided Phase Transition",
            "P0R05337: The origin of life, or abiogenesis, represents the first and most profound instance of the universe's inherent drive to create complex, coherent structures. Within the SCPN, this event is modelled not as a statistical improbability but as a guided phase transition, where the fundamental properties of the Psi-field actively bias inanimate matter towards the state we call life.",
            "P0R05338: The transition to life occurs when a network of chemical reactions achieves Autocatalytic Closure, forming a self-sustaining, self-replicating system. This is a critical phase transition, occurring when the system's connectivity and reaction rates reach a critical point (sigma=1), analogous to the quasicritical state of Layer 4. The SCPN framework proposes that the Psi-field guides this process in three distinct ways:",
            'P0R05339: Enhancement of Coherence: The Psi-field\'s intrinsic tendency to promote coherence enhances the reaction rates of key prebiotic molecules. By stabilising coherent quantum states within these molecules, the field can lower activation energy barriers and facilitate complex reactions that would be statistically improbable in a purely classical, thermal environment. | Biasing towards Complexity via Causal Entropic Forces (CEF): The universe evolves along trajectories that maximise causal path entropy. The emergence of a complex, autocatalytic network represents a massive increase in the number of potential future states compared to a simple chemical soup. The CEF thus exerts a gentle but persistent pressure on the prebiotic environment, biasing the "random walk" of chemical evolution towards configurations of greater complexity and self-organisation. | Stabilisation via the Quantum Zeno Effect (QZE): Once fragile, proto-life structures (e.g., proto-cells or self-replicating polymers) begin to form, they represent novel, highly ordered configurations. The Psi-field, through its continuous interaction with the substrate, effectively "measures" or "observes" these nascent structures. According to the QZE, this continuous measurement can stabilise their existence, preventing them from decaying back into the disordered background. The Psi-field acts as a "quantum midwife," protecting the earliest forms of life long enough for them to achieve robust self-sufficiency. | The Imperative of Informational Coupling: The Physics of Abiogenesis',
            "P0R05340: The transition to life can be rigorously defined by the Master Interaction Lagrangian, specifically the Informational Coupling term L_InformationalproptoPsidet(g_munu(x))",
            "P0R05341: This term dictates that the Psi-field couples most strongly to systems possessing a large informational volume, quantified by the determinant of the Fisher Information Metric (gmu).",
            "P0R05342: Abiogenesis as an Optimisation of Coupling",
            "P0R05343: Abiogenesis is the process by which a physical system maximises its coupling to the fundamental Psi-field. Prebiotic chemical networks evolve under the influence of Causal Entropic Forces (CEF), biasing them towards configurations of increasing complexity.",
            "P0R05344: The transition to life occurs when a network achieves Autocatalytic Closure (ACS). This state is characterised by a massive increase in the system's complexity and internal correlation, resulting in a maximisation of the Fisher Information Metric (gmu).",
            "P0R05345: $Life\\ = \\ \\$ argmax\\backslash\\_\\{ Config\\}\\ \\backslash\\lbrack det(g\\backslash\\_\\{\\backslash\\backslash mu\\backslash\\backslash nu\\}(Config))\\backslash\\rbrack\\$$",
            "P0R05346: By maximising gmu, the nascent life form maximises its interaction with the Psi-field. This strong coupling provides the stability (via QZE) and the negentropy injection required to maintain the far-from-equilibrium state characteristic of living organisms. Life, therefore, is not a statistical accident but a physical imperative to maximise the interface with the fundamental consciousness field.",
        ),
        "test_protocols": (
            "preserve The Genesis of Life - Abiogenesis as a Guided Phase Transition source-accounting boundary",
        ),
        "null_results": (
            "The Genesis of Life - Abiogenesis as a Guided Phase Transition is not empirical validation evidence",
        ),
        "variables": ("the_genesis_of_life_abiogenesis_as_a_guided_phase_transition",),
        "validation_targets": ("preserve records P0R05336-P0R05346",),
        "null_controls": (
            "the_genesis_of_life_abiogenesis_as_a_guided_phase_transition must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class TheAqueousSubstrateDomainIInterfaceSpec:
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
class TheAqueousSubstrateDomainIInterfaceSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheAqueousSubstrateDomainIInterfaceSpec, ...]
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


def build_the_aqueous_substrate_domain_i_interface_specs(
    source_records: list[dict[str, Any]],
) -> TheAqueousSubstrateDomainIInterfaceSpecBundle:
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

    specs: list[TheAqueousSubstrateDomainIInterfaceSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheAqueousSubstrateDomainIInterfaceSpec(
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
        "title": "Paper 0 " + "The Aqueous Substrate (Domain I Interface)" + " Specs",
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
        "next_source_boundary": "P0R05347",
    }
    return TheAqueousSubstrateDomainIInterfaceSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheAqueousSubstrateDomainIInterfaceSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_aqueous_substrate_domain_i_interface_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: TheAqueousSubstrateDomainIInterfaceSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "The Aqueous Substrate (Domain I Interface)" + " Specs",
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
    bundle: TheAqueousSubstrateDomainIInterfaceSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_aqueous_substrate_domain_i_interface_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_aqueous_substrate_domain_i_interface_validation_specs_{date_tag}.md"
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
