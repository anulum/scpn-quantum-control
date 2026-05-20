#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Quantum Enzymology of the Immune Response spec builder
"""Promote Paper 0 Quantum Enzymology of the Immune Response records."""

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
    "P0R05508",
    "P0R05509",
    "P0R05510",
    "P0R05511",
    "P0R05512",
    "P0R05513",
    "P0R05514",
    "P0R05515",
    "P0R05516",
)
CLAIM_BOUNDARY = "source-bounded quantum enzymology of the immune response source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "quantum_enzymology_of_the_immune_response.quantum_enzymology_of_the_immune_response": {
        "context_id": "quantum_enzymology_of_the_immune_response",
        "validation_protocol": "paper0.quantum_enzymology_of_the_immune_response.quantum_enzymology_of_the_immune_response",
        "canonical_statement": "The source-bounded component 'Quantum Enzymology of the Immune Response' preserves Paper 0 records P0R05508-P0R05509 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05508:quantum_enzymology_of_the_immune_response",
            "P0R05509:quantum_enzymology_of_the_immune_response",
        ),
        "source_formulae": (
            "P0R05508: Quantum Enzymology of the Immune Response",
            "P0R05509: The SCPN framework posits that the universal Consciousness Field (Psi) acts as a stabilising agent that universally minimises entropy across living systems, enabling processes like photosynthesis and enzyme action to achieve efficiencies that defy classical explanation. Quantum tunnelling in enzyme catalysis is a prime example of biological systems leveraging non-trivial quantum mechanics for functional advantage. This section extends this principle specifically to the immune system, formalising the quantum underpinnings of its dynamic and responsive nature.",
        ),
        "test_protocols": (
            "preserve Quantum Enzymology of the Immune Response source-accounting boundary",
        ),
        "null_results": (
            "Quantum Enzymology of the Immune Response is not empirical validation evidence",
        ),
        "variables": ("quantum_enzymology_of_the_immune_response",),
        "validation_targets": ("preserve records P0R05508-P0R05509",),
        "null_controls": (
            "quantum_enzymology_of_the_immune_response must remain source-bounded accounting",
        ),
    },
    "quantum_enzymology_of_the_immune_response.mechanism_of_nuclear_quantum_tunnelling": {
        "context_id": "mechanism_of_nuclear_quantum_tunnelling",
        "validation_protocol": "paper0.quantum_enzymology_of_the_immune_response.mechanism_of_nuclear_quantum_tunnelling",
        "canonical_statement": "The source-bounded component 'Mechanism of Nuclear Quantum Tunnelling' preserves Paper 0 records P0R05510-P0R05512 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05510:mechanism_of_nuclear_quantum_tunnelling",
            "P0R05511:mechanism_of_nuclear_quantum_tunnelling",
            "P0R05512:mechanism_of_nuclear_quantum_tunnelling",
        ),
        "source_formulae": (
            "P0R05510: Mechanism of Nuclear Quantum Tunnelling",
            'P0R05511: In classical chemistry, a reaction proceeds only if the reactants possess sufficient thermal energy to overcome an activation energy barrier. Quantum mechanics, however, offers an alternative pathway. Due to the wave-like properties inherent to all particles, a particle\'s position is described by a diffuse wave function that does not abruptly end at the boundary of a potential barrier but instead decays exponentially into it. If the barrier is sufficiently narrow, the wavefunction can have a non-zero amplitude on the other side, implying a finite probability that the particle can "tunnel" through the barrier, appearing on the product side without ever having had the classical energy to surmount the peak. This phenomenon is particularly relevant for light particles like electrons and protons (hydrogen nuclei), whose de Broglie wavelengths are significant on the scale of molecular bonds.',
            "P0R05512: The probability of tunnelling, and thus the reaction rate, is exquisitely sensitive to the mass of the tunnelling particle and the width and height of the energy barrier. Protein dynamics-the subtle, rapid vibrations and conformational fluctuations of the enzyme structure-are now understood to be critical in promoting tunnelling. These motions can transiently compress the reaction distance, narrowing the barrier and creating configurations where the reactant and product quantum states are degenerate in energy, thereby dramatically increasing the probability of a tunnelling event.",
        ),
        "test_protocols": (
            "preserve Mechanism of Nuclear Quantum Tunnelling source-accounting boundary",
        ),
        "null_results": (
            "Mechanism of Nuclear Quantum Tunnelling is not empirical validation evidence",
        ),
        "variables": ("mechanism_of_nuclear_quantum_tunnelling",),
        "validation_targets": ("preserve records P0R05510-P0R05512",),
        "null_controls": (
            "mechanism_of_nuclear_quantum_tunnelling must remain source-bounded accounting",
        ),
    },
    "quantum_enzymology_of_the_immune_response.formalism_for_tunnelling_enhanced_reaction_rates": {
        "context_id": "formalism_for_tunnelling_enhanced_reaction_rates",
        "validation_protocol": "paper0.quantum_enzymology_of_the_immune_response.formalism_for_tunnelling_enhanced_reaction_rates",
        "canonical_statement": "The source-bounded component 'Formalism for Tunnelling-Enhanced Reaction Rates' preserves Paper 0 records P0R05513-P0R05516 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05513:formalism_for_tunnelling_enhanced_reaction_rates",
            "P0R05514:formalism_for_tunnelling_enhanced_reaction_rates",
            "P0R05515:formalism_for_tunnelling_enhanced_reaction_rates",
            "P0R05516:formalism_for_tunnelling_enhanced_reaction_rates",
        ),
        "source_formulae": (
            "P0R05513: Formalism for Tunnelling-Enhanced Reaction Rates",
            "P0R05514: The rate of a tunnelling-dependent reaction can be expressed as ktunnel=(/2)Ptunnel, where is the attempt frequency and Ptunnel is the tunnelling probability. Using the Wentzel-Kramers-Brillouin (WKB) approximation, this probability is given by:",
            "P0R05515: Ptunnelexp(2ab2m(V(x)E)dx)",
            "P0R05516: Here, m is the mass of the tunnelling particle, E is its energy, and the integral is taken over the classically forbidden region of the potential energy barrier V(x). Within the SCPN framework, the Psi-field's role in stabilising coherence directly impacts this process. By coupling to the enzyme-substrate complex, the Psi-field can reduce the effective environmental decoherence, leading to a more defined and narrower effective potential barrier, Veff(x). This Psi-field-mediated barrier narrowing serves as a primary mechanism for enhancing Ptunnel and, consequently, accelerating the reaction rate beyond classical limits.",
        ),
        "test_protocols": (
            "preserve Formalism for Tunnelling-Enhanced Reaction Rates source-accounting boundary",
        ),
        "null_results": (
            "Formalism for Tunnelling-Enhanced Reaction Rates is not empirical validation evidence",
        ),
        "variables": ("formalism_for_tunnelling_enhanced_reaction_rates",),
        "validation_targets": ("preserve records P0R05513-P0R05516",),
        "null_controls": (
            "formalism_for_tunnelling_enhanced_reaction_rates must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class QuantumEnzymologyOfTheImmuneResponseSpec:
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
class QuantumEnzymologyOfTheImmuneResponseSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[QuantumEnzymologyOfTheImmuneResponseSpec, ...]
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


def build_quantum_enzymology_of_the_immune_response_specs(
    source_records: list[dict[str, Any]],
) -> QuantumEnzymologyOfTheImmuneResponseSpecBundle:
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

    specs: list[QuantumEnzymologyOfTheImmuneResponseSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            QuantumEnzymologyOfTheImmuneResponseSpec(
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
        "title": "Paper 0 " + "Quantum Enzymology of the Immune Response" + " Specs",
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
        "next_source_boundary": "P0R05517",
    }
    return QuantumEnzymologyOfTheImmuneResponseSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> QuantumEnzymologyOfTheImmuneResponseSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_quantum_enzymology_of_the_immune_response_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: QuantumEnzymologyOfTheImmuneResponseSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Quantum Enzymology of the Immune Response" + " Specs",
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
    bundle: QuantumEnzymologyOfTheImmuneResponseSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_quantum_enzymology_of_the_immune_response_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_quantum_enzymology_of_the_immune_response_validation_specs_{date_tag}.md"
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
