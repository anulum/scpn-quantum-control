#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 II. The Molecular Machinery of Signalling: Ion Channels and Receptors (L1/L2) spec builder
"""Promote Paper 0 II. The Molecular Machinery of Signalling: Ion Channels and Receptors (L1/L2) records."""

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
    "P0R04769",
    "P0R04770",
    "P0R04771",
    "P0R04772",
    "P0R04773",
    "P0R04774",
    "P0R04775",
    "P0R04776",
    "P0R04777",
)
CLAIM_BOUNDARY = "source-bounded ii the molecular machinery of signalling ion channels and receptors l1 l source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l.ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l": {
        "context_id": "ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l",
        "validation_protocol": "paper0.ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l.ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l",
        "canonical_statement": "The source-bounded component 'II. The Molecular Machinery of Signalling: Ion Channels and Receptors (L1/L2)' preserves Paper 0 records P0R04769-P0R04770 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04769:ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l",
            "P0R04770:ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l",
        ),
        "source_formulae": (
            "P0R04769: II. The Molecular Machinery of Signalling: Ion Channels and Receptors (L1/L2)",
            "P0R04770: Ion channels and receptors are the molecular machines controlling information flow. Their operation involves precise structural dynamics and non-trivial quantum effects.",
        ),
        "test_protocols": (
            "preserve II. The Molecular Machinery of Signalling: Ion Channels and Receptors (L1/L2) source-accounting boundary",
        ),
        "null_results": (
            "II. The Molecular Machinery of Signalling: Ion Channels and Receptors (L1/L2) is not empirical validation evidence",
        ),
        "variables": ("ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l",),
        "validation_targets": ("preserve records P0R04769-P0R04770",),
        "null_controls": (
            "ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l must remain source-bounded accounting",
        ),
    },
    "ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l.1_the_architecture_of_gating_and_iet": {
        "context_id": "1_the_architecture_of_gating_and_iet",
        "validation_protocol": "paper0.ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l.1_the_architecture_of_gating_and_iet",
        "canonical_statement": "The source-bounded component '1. The Architecture of Gating and IET' preserves Paper 0 records P0R04771-P0R04773 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04771:1_the_architecture_of_gating_and_iet",
            "P0R04772:1_the_architecture_of_gating_and_iet",
            "P0R04773:1_the_architecture_of_gating_and_iet",
        ),
        "source_formulae": (
            "P0R04771: 1. The Architecture of Gating and IET",
            "P0R04772: Gating involves conformational changes, such as the movement of the Voltage Sensor Domain (VSD) in VGICs. This transition involves overcoming an energy barrier (DeltaGGate), providing a critical leverage point for the Psi-field.",
            "P0R04773: IET at the Gate (Downward Causation): The Psi-field modulates the Quantum Potential (Q) associated with the VSD or the gating hinge. This alters DeltaGGate, changing the probability of channel opening without classical energy exchange. This provides a precise mechanism for modulating neuronal excitability (L4 UPDE).",
        ),
        "test_protocols": (
            "preserve 1. The Architecture of Gating and IET source-accounting boundary",
        ),
        "null_results": (
            "1. The Architecture of Gating and IET is not empirical validation evidence",
        ),
        "variables": ("1_the_architecture_of_gating_and_iet",),
        "validation_targets": ("preserve records P0R04771-P0R04773",),
        "null_controls": (
            "1_the_architecture_of_gating_and_iet must remain source-bounded accounting",
        ),
    },
    "ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l.2_quantum_effects_in_selectivity_and_binding_l1_l2": {
        "context_id": "2_quantum_effects_in_selectivity_and_binding_l1_l2",
        "validation_protocol": "paper0.ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l.2_quantum_effects_in_selectivity_and_binding_l1_l2",
        "canonical_statement": "The source-bounded component '2. Quantum Effects in Selectivity and Binding (L1/L2)' preserves Paper 0 records P0R04774-P0R04775 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04774:2_quantum_effects_in_selectivity_and_binding_l1_l2",
            "P0R04775:2_quantum_effects_in_selectivity_and_binding_l1_l2",
        ),
        "source_formulae": (
            "P0R04774: 2. Quantum Effects in Selectivity and Binding (L1/L2)",
            'P0R04775: The Selectivity Filter (Quantum Knock-On): The rapid and selective transport of ions (e.g., K$^{+}$) through the selectivity filter utilises quantum coherence. The "Knock-on" mechanism involves coherent delocalisation of ions and water molecules within the pore. | Neurotransmitter Binding (VAT): The binding of a ligand to its receptor may be enhanced by Vibrational Assisted Tunnelling (VAT), where vibrational energy matches the energy difference between the initial and final states.',
        ),
        "test_protocols": (
            "preserve 2. Quantum Effects in Selectivity and Binding (L1/L2) source-accounting boundary",
        ),
        "null_results": (
            "2. Quantum Effects in Selectivity and Binding (L1/L2) is not empirical validation evidence",
        ),
        "variables": ("2_quantum_effects_in_selectivity_and_binding_l1_l2",),
        "validation_targets": ("preserve records P0R04774-P0R04775",),
        "null_controls": (
            "2_quantum_effects_in_selectivity_and_binding_l1_l2 must remain source-bounded accounting",
        ),
    },
    "ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l.3_qze_and_attentional_stabilisation": {
        "context_id": "3_qze_and_attentional_stabilisation",
        "validation_protocol": "paper0.ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l.3_qze_and_attentional_stabilisation",
        "canonical_statement": "The source-bounded component '3. QZE and Attentional Stabilisation:' preserves Paper 0 records P0R04776-P0R04777 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04776:3_qze_and_attentional_stabilisation",
            "P0R04777:3_qze_and_attentional_stabilisation",
        ),
        "source_formulae": (
            "P0R04776: 3. QZE and Attentional Stabilisation:",
            "P0R04777: The Quantum Zeno Effect (QZE) stabilises the state of ion channels. Focused attention (L5) acts as a continuous measurement operator on the channel population, implementing attentional selection by stabilising channels in a specific state.",
        ),
        "test_protocols": (
            "preserve 3. QZE and Attentional Stabilisation: source-accounting boundary",
        ),
        "null_results": (
            "3. QZE and Attentional Stabilisation: is not empirical validation evidence",
        ),
        "variables": ("3_qze_and_attentional_stabilisation",),
        "validation_targets": ("preserve records P0R04776-P0R04777",),
        "null_controls": (
            "3_qze_and_attentional_stabilisation must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class IiTheMolecularMachineryOfSignallingIonChannelsAndReceptorsL1LSpec:
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
class IiTheMolecularMachineryOfSignallingIonChannelsAndReceptorsL1LSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[IiTheMolecularMachineryOfSignallingIonChannelsAndReceptorsL1LSpec, ...]
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


def build_ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_specs(
    source_records: list[dict[str, Any]],
) -> IiTheMolecularMachineryOfSignallingIonChannelsAndReceptorsL1LSpecBundle:
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

    specs: list[IiTheMolecularMachineryOfSignallingIonChannelsAndReceptorsL1LSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            IiTheMolecularMachineryOfSignallingIonChannelsAndReceptorsL1LSpec(
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
        + "II. The Molecular Machinery of Signalling: Ion Channels and Receptors (L1/L2)"
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
        "next_source_boundary": "P0R04778",
    }
    return IiTheMolecularMachineryOfSignallingIonChannelsAndReceptorsL1LSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> IiTheMolecularMachineryOfSignallingIonChannelsAndReceptorsL1LSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_specs(
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
    bundle: IiTheMolecularMachineryOfSignallingIonChannelsAndReceptorsL1LSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "II. The Molecular Machinery of Signalling: Ion Channels and Receptors (L1/L2)"
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
    bundle: IiTheMolecularMachineryOfSignallingIonChannelsAndReceptorsL1LSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_validation_specs_{date_tag}.md"
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
