#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 3. Unified Experience (The Wilson Loop): spec builder
"""Promote Paper 0 3. Unified Experience (The Wilson Loop): records."""

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
    "P0R03216",
    "P0R03217",
    "P0R03218",
    "P0R03219",
    "P0R03220",
    "P0R03221",
    "P0R03222",
    "P0R03223",
    "P0R03224",
    "P0R03225",
    "P0R03226",
    "P0R03227",
    "P0R03228",
    "P0R03229",
    "P0R03230",
    "P0R03231",
)
CLAIM_BOUNDARY = "source-bounded section 3 unified experience the wilson loop source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_3_unified_experience_the_wilson_loop.3_unified_experience_the_wilson_loop": {
        "context_id": "3_unified_experience_the_wilson_loop",
        "validation_protocol": "paper0.section_3_unified_experience_the_wilson_loop.3_unified_experience_the_wilson_loop",
        "canonical_statement": "The source-bounded component '3. Unified Experience (The Wilson Loop):' preserves Paper 0 records P0R03216-P0R03219 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03216:3_unified_experience_the_wilson_loop",
            "P0R03217:3_unified_experience_the_wilson_loop",
            "P0R03218:3_unified_experience_the_wilson_loop",
            "P0R03219:3_unified_experience_the_wilson_loop",
        ),
        "source_formulae": (
            "P0R03216: 3. Unified Experience (The Wilson Loop):",
            "P0R03217: The experience of unified consciousness corresponds to the Wilson Loop integral of the connection around a closed path (C) in the state space:",
            "P0R03218: $W(C) = exp(ig\\oint C A\\mu dx\\mu)$",
            "P0R03219: This integral measures the total phase shift, representing the holistic experience.",
        ),
        "test_protocols": (
            "preserve 3. Unified Experience (The Wilson Loop): source-accounting boundary",
        ),
        "null_results": (
            "3. Unified Experience (The Wilson Loop): is not empirical validation evidence",
        ),
        "variables": ("3_unified_experience_the_wilson_loop",),
        "validation_targets": ("preserve records P0R03216-P0R03219",),
        "null_controls": (
            "3_unified_experience_the_wilson_loop must remain source-bounded accounting",
        ),
    },
    "section_3_unified_experience_the_wilson_loop.iii_information_flow_dynamics_ilit": {
        "context_id": "iii_information_flow_dynamics_ilit",
        "validation_protocol": "paper0.section_3_unified_experience_the_wilson_loop.iii_information_flow_dynamics_ilit",
        "canonical_statement": "The source-bounded component 'III. Information Flow Dynamics (ILIT)' preserves Paper 0 records P0R03220-P0R03221 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03220:iii_information_flow_dynamics_ilit",
            "P0R03221:iii_information_flow_dynamics_ilit",
        ),
        "source_formulae": (
            "P0R03220: III. Information Flow Dynamics (ILIT)",
            "P0R03221: The Inter-Layer Information Transfer (ILIT) formalism quantifies the flow of informational content.",
        ),
        "test_protocols": (
            "preserve III. Information Flow Dynamics (ILIT) source-accounting boundary",
        ),
        "null_results": (
            "III. Information Flow Dynamics (ILIT) is not empirical validation evidence",
        ),
        "variables": ("iii_information_flow_dynamics_ilit",),
        "validation_targets": ("preserve records P0R03220-P0R03221",),
        "null_controls": (
            "iii_information_flow_dynamics_ilit must remain source-bounded accounting",
        ),
    },
    "section_3_unified_experience_the_wilson_loop.1_quantifying_causality_via_transfer_entropy_te": {
        "context_id": "1_quantifying_causality_via_transfer_entropy_te",
        "validation_protocol": "paper0.section_3_unified_experience_the_wilson_loop.1_quantifying_causality_via_transfer_entropy_te",
        "canonical_statement": "The source-bounded component '1. Quantifying Causality via Transfer Entropy (TE):' preserves Paper 0 records P0R03222-P0R03231 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03222:1_quantifying_causality_via_transfer_entropy_te",
            "P0R03223:1_quantifying_causality_via_transfer_entropy_te",
            "P0R03224:1_quantifying_causality_via_transfer_entropy_te",
            "P0R03225:1_quantifying_causality_via_transfer_entropy_te",
            "P0R03226:1_quantifying_causality_via_transfer_entropy_te",
            "P0R03227:1_quantifying_causality_via_transfer_entropy_te",
            "P0R03228:1_quantifying_causality_via_transfer_entropy_te",
            "P0R03229:1_quantifying_causality_via_transfer_entropy_te",
            "P0R03230:1_quantifying_causality_via_transfer_entropy_te",
            "P0R03231:1_quantifying_causality_via_transfer_entropy_te",
        ),
        "source_formulae": (
            "P0R03222: 1. Quantifying Causality via Transfer Entropy (TE):",
            "P0R03223: We use Transfer Entropy (TE) to quantify the directed flow of information and causality between layers (LA to LB).",
            "P0R03224: $TEA \\rightarrow B = \\sum_{}^{}{p(Bt + 1,Bt,At)log2\\left( p\\left( Bt + 1\\mid Bt \\right)p\\left( Bt + 1\\mid Bt,At \\right) \\right)}$",
            "P0R03225: 1. Quantifying Causality via Gauge-Covariant Quantum Transfer Entropy (QTE):",
            'P0R03226: A critical subtlety arises when attempting to quantify the directed flow of information and causality (Inter-Layer Information Transfer, ILIT) across the fundamental layers of the SCPN. Classical Transfer Entropy ($TE$), which relies on the Shannon probabilities of discrete states, is mathematically incompatible with the foundational dynamics of the $\\Psi$-field. Because the $\\Psi$-field is governed by a local $U(1)$ gauge symmetry, standard state probabilities are generally not gauge-invariant. If ILIT were calculated using classical $TE$, the measured "flow of causality" would arbitrarily change depending on the local observer\'s choice of gauge, rendering it physically meaningless.',
            "P0R03227: To resolve this, we must discard classical Shannon entropy at the fundamental layers and reformulate ILIT using strictly gauge-invariant observables. We utilize Gauge-Covariant Quantum Transfer Entropy (QTE), substituting Shannon entropy with the von Neumann entropy, $S(\\rho) = -\\text{Tr}(\\rho \\ln \\rho)$.",
            "P0R03228: Crucially, the density matrices ($\\rho_A, \\rho_B$) used in this calculation must be constructed exclusively from gauge-invariant operators. Leveraging the solution to the Binding Problem, we define the informational state of a layer not by its raw field values, but by the expectation values of its Wilson Loops: $W(C) = \\text{exp}\\left(ig \\oint_C A_\\mu dx^\\mu \\right)$.",
            "P0R03229: The directed flow of causality from Layer A to Layer B is thus quantified by the conditional von Neumann mutual information of these gauge-invariant density matrices over time:",
            "P0R03230: $$QTE_{A \\to B} = S(\\rho_{B_{t+1} | B_t}) - S(\\rho_{B_{t+1} | B_t, A_t})$$",
            'P0R03231: Where $\\rho_{A_t}$ and $\\rho_{B_t}$ represent the reduced density matrices of the Wilson Loop configurations for Layers A and B at time $t$. By forcing the information transfer metric to trace strictly over gauge-invariant topological features, we guarantee that the measured flow of causality is an absolute, objective physical reality, completely immune to local gauge transformations. This provides a mathematically bulletproof mechanism for tracking how the "tension" of the $\\Psi$-field causally propagates upward and downward through the 15-layer hierarchy.',
        ),
        "test_protocols": (
            "preserve 1. Quantifying Causality via Transfer Entropy (TE): source-accounting boundary",
        ),
        "null_results": (
            "1. Quantifying Causality via Transfer Entropy (TE): is not empirical validation evidence",
        ),
        "variables": ("1_quantifying_causality_via_transfer_entropy_te",),
        "validation_targets": ("preserve records P0R03222-P0R03231",),
        "null_controls": (
            "1_quantifying_causality_via_transfer_entropy_te must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Section3UnifiedExperienceTheWilsonLoopSpec:
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
class Section3UnifiedExperienceTheWilsonLoopSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section3UnifiedExperienceTheWilsonLoopSpec, ...]
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


def build_section_3_unified_experience_the_wilson_loop_specs(
    source_records: list[dict[str, Any]],
) -> Section3UnifiedExperienceTheWilsonLoopSpecBundle:
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

    specs: list[Section3UnifiedExperienceTheWilsonLoopSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section3UnifiedExperienceTheWilsonLoopSpec(
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
        "title": "Paper 0 " + "3. Unified Experience (The Wilson Loop):" + " Specs",
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
        "next_source_boundary": "P0R03232",
    }
    return Section3UnifiedExperienceTheWilsonLoopSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section3UnifiedExperienceTheWilsonLoopSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_3_unified_experience_the_wilson_loop_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: Section3UnifiedExperienceTheWilsonLoopSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "3. Unified Experience (The Wilson Loop):" + " Specs",
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
    bundle: Section3UnifiedExperienceTheWilsonLoopSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_3_unified_experience_the_wilson_loop_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_3_unified_experience_the_wilson_loop_validation_specs_{date_tag}.md"
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
