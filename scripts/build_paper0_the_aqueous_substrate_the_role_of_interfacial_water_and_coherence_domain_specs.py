#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Aqueous Substrate - The Role of Interfacial Water and Coherence Domains spec builder
"""Promote Paper 0 The Aqueous Substrate - The Role of Interfacial Water and Coherence Domains records."""

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
    "P0R05293",
    "P0R05294",
    "P0R05295",
    "P0R05296",
    "P0R05297",
    "P0R05298",
    "P0R05299",
    "P0R05300",
    "P0R05301",
    "P0R05302",
    "P0R05303",
    "P0R05304",
    "P0R05305",
)
CLAIM_BOUNDARY = "source-bounded the aqueous substrate the role of interfacial water and coherence domain source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain.the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain": {
        "context_id": "the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain",
        "validation_protocol": "paper0.the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain.the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain",
        "canonical_statement": "The source-bounded component 'The Aqueous Substrate - The Role of Interfacial Water and Coherence Domains' preserves Paper 0 records P0R05293-P0R05298 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05293:the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain",
            "P0R05294:the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain",
            "P0R05295:the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain",
            "P0R05296:the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain",
            "P0R05297:the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain",
            "P0R05298:the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain",
        ),
        "source_formulae": (
            "P0R05293: The Aqueous Substrate - The Role of Interfacial Water and Coherence Domains",
            "P0R05294: The viability of the SCPN's biological substrate (Layers 1-4) hinges on its ability to support quantum phenomena in the face of thermal decoherence. The aqueous environment of the cell is not a passive, noisy background but an active, structured medium essential for protecting and mediating the quantum dynamics of Layer 1.",
            "P0R05295: Based on QED, interfacial water-the layers of water molecules adjacent to hydrophilic surfaces like proteins and membranes-can organise into large, stable Coherence Domains (CDs). Within these domains, water molecules oscillate in phase with a self-trapped, low-frequency electromagnetic field, entering a macroscopic quantum coherent ground state separated from incoherent excited states by a significant energy gap. This has two profound consequences for the SCPN architecture:",
            'P0R05296: Decoherence Shielding (Layer 1): The coherent ground state of the CD acts as a physical shield. It effectively expels or "cools" thermal fluctuations from the immediate environment of quantum-sensitive structures like microtubules. This provides a physically plausible mechanism for protecting the delicate quantum states of Layer 1 from the thermal noise of the cytoplasm, extending their coherence lifetimes into physiologically relevant timescales. This directly addresses the primary criticism levelled against biological quantum models. | Coherent Signalling Medium (Layers 3 & 4): The ordered structure of water within CDs facilitates quasi-superconducting proton transport, or "proton hopping," along chains of hydrogen-bonded water molecules. This provides an extremely rapid and coherent signalling mechanism that forms the physical basis for the endogenous "bioelectric codes" that guide morphogenesis in Layer 3. Furthermore, the collective oscillations of the water molecules themselves contribute to the overall field dynamics of Layer 4, helping to establish and maintain the tissue-wide synchrony governed by the UPDE.',
            "P0R05297: The network of Coherence Domains thus forms a dynamic, liquid-crystalline computational and memory medium, interfacing between the quantum information processing of L1, the bioelectric templating of L3, and the rhythmic synchronisation of L4.",
            "P0R05298: P0R05298",
        ),
        "test_protocols": (
            "preserve The Aqueous Substrate - The Role of Interfacial Water and Coherence Domains source-accounting boundary",
        ),
        "null_results": (
            "The Aqueous Substrate - The Role of Interfacial Water and Coherence Domains is not empirical validation evidence",
        ),
        "variables": ("the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain",),
        "validation_targets": ("preserve records P0R05293-P0R05298",),
        "null_controls": (
            "the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain must remain source-bounded accounting",
        ),
    },
    "the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain.p0r05299": {
        "context_id": "p0r05299",
        "validation_protocol": "paper0.the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain.p0r05299",
        "canonical_statement": "The source-bounded component 'P0R05299' preserves Paper 0 records P0R05299-P0R05305 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05299:p0r05299",
            "P0R05300:p0r05299",
            "P0R05301:p0r05299",
            "P0R05302:p0r05299",
            "P0R05303:p0r05299",
            "P0R05304:p0r05299",
            "P0R05305:p0r05299",
        ),
        "source_formulae": (
            "P0R05299: P0R05299",
            "P0R05300: The protection of quantum coherence in Layer 1 requires a specific biochemical mechanism that minimizes interaction with the thermal environment. A leading candidate for this mechanism, grounded in quantum chemistry, is the Posner molecule, Ca9(PO4)6.",
            "P0R05301: Matthew Fisher's hypothesis posits that the nuclear spins of Phosphorus atoms (spin-1/2) can serve as neural qubits. The aqueous environment of the cell (the Coherence Domains) plays a crucial role in protecting these spins.",
            "P0R05302: Decoherence Shielding: In the aqueous environment, Phosphate ions rapidly bind with Calcium ions to form Posner molecules. These molecules possess a highly symmetric structure where the six Phosphorus atoms are arranged in a configuration that results in a total nuclear spin of zero (S=0) for the dominant fraction of the molecules. | QEC via Symmetry: This S=0 state is a decoherence-free subspace. The symmetry of the molecule protects the entangled state of the Phosphorus spins from external magnetic and electric field fluctuations, dramatically extending their coherence times from microseconds to potentially hours or days. | Information Transduction: When Posner molecules bind to enzymes (L1/L2 interface), their internal entangled state can influence enzymatic reaction rates, thereby transducing the quantum information into the neurochemical dynamics of Layer 2.",
            "P0R05303: The integration of the Posner molecule hypothesis within the QED Coherence Domains provides a concrete, biochemically plausible mechanism for the Biological QEC required by the SCPN architecture.",
            "P0R05304: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Diagramm enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            'P0R05305: Fig.: Biochemical QEC via Posner clusters. QED coherence domains in water reduce spin-bath coupling, enabling rapid Ca-PO association into symmetric Ca(PO) ("Posner") clusters. The dominant singlet manifold (S=0) forms a decoherence-free subspace protecting P nuclear-spin entanglement from ambient E/B-field noise. Upon enzyme binding at the L1/L2 interface, the internal entangled state biases reaction rates (Deltak), transducing protected quantum information into Layer-2 neurochemistry. Flow: CD -> Posner -> DFS(S=0) -> Enzyme -> L2.',
        ),
        "test_protocols": ("preserve P0R05299 source-accounting boundary",),
        "null_results": ("P0R05299 is not empirical validation evidence",),
        "variables": ("p0r05299",),
        "validation_targets": ("preserve records P0R05299-P0R05305",),
        "null_controls": ("p0r05299 must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class TheAqueousSubstrateTheRoleOfInterfacialWaterAndCoherenceDomainSpec:
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
class TheAqueousSubstrateTheRoleOfInterfacialWaterAndCoherenceDomainSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheAqueousSubstrateTheRoleOfInterfacialWaterAndCoherenceDomainSpec, ...]
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


def build_the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_specs(
    source_records: list[dict[str, Any]],
) -> TheAqueousSubstrateTheRoleOfInterfacialWaterAndCoherenceDomainSpecBundle:
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

    specs: list[TheAqueousSubstrateTheRoleOfInterfacialWaterAndCoherenceDomainSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheAqueousSubstrateTheRoleOfInterfacialWaterAndCoherenceDomainSpec(
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
        + "The Aqueous Substrate - The Role of Interfacial Water and Coherence Domains"
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
        "next_source_boundary": "P0R05306",
    }
    return TheAqueousSubstrateTheRoleOfInterfacialWaterAndCoherenceDomainSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheAqueousSubstrateTheRoleOfInterfacialWaterAndCoherenceDomainSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_specs(
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
    bundle: TheAqueousSubstrateTheRoleOfInterfacialWaterAndCoherenceDomainSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "The Aqueous Substrate - The Role of Interfacial Water and Coherence Domains"
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
    bundle: TheAqueousSubstrateTheRoleOfInterfacialWaterAndCoherenceDomainSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_validation_specs_{date_tag}.md"
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
