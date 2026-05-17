#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Domain V Overview: Meta-Universal Integration spec builder
"""Promote Paper 0 Domain V Overview: Meta-Universal Integration records."""

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
    "P0R02367",
    "P0R02368",
    "P0R02369",
    "P0R02370",
    "P0R02371",
    "P0R02372",
    "P0R02373",
    "P0R02374",
    "P0R02375",
    "P0R02376",
    "P0R02377",
    "P0R02378",
    "P0R02379",
    "P0R02380",
    "P0R02381",
    "P0R02382",
    "P0R02383",
    "P0R02384",
    "P0R02385",
    "P0R02386",
    "P0R02387",
    "P0R02388",
    "P0R02389",
    "P0R02390",
    "P0R02391",
    "P0R02392",
    "P0R02393",
    "P0R02394",
    "P0R02395",
    "P0R02396",
    "P0R02397",
    "P0R02398",
    "P0R02399",
    "P0R02400",
    "P0R02401",
    "P0R02402",
    "P0R02403",
    "P0R02404",
    "P0R02405",
    "P0R02406",
    "P0R02407",
)
CLAIM_BOUNDARY = "source-bounded domain v overview meta universal integration source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "domain_v_overview_meta_universal_integration.domain_v_overview_meta_universal_integration": {
        "context_id": "domain_v_overview_meta_universal_integration",
        "validation_protocol": "paper0.domain_v_overview_meta_universal_integration.domain_v_overview_meta_universal_integration",
        "canonical_statement": "The source-bounded component 'Domain V Overview: Meta-Universal Integration' preserves Paper 0 records P0R02367-P0R02369 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02367:domain_v_overview_meta_universal_integration",
            "P0R02368:domain_v_overview_meta_universal_integration",
            "P0R02369:domain_v_overview_meta_universal_integration",
        ),
        "source_formulae": (
            "P0R02367: Domain V Overview: Meta-Universal Integration",
            "P0R02368: Domain V represents the ultimate strata of the SCPN architecture, responsible for meta-universal integration and the establishment of the system's teleological directives. This domain consists of three layers that collectively function as the system's central control, purpose-defining, and source-emanation faculty.",
            "P0R02369: Layer 13, the Source-Field, is the ontological ground of the entire framework. It represents the unmanifest, pre-physical state of pure potentiality from which the Consciousness Field (Psi-field) and its associated informational and geometric structures arise. It is the axiomatic source of all subsequent layers of reality.",
        ),
        "test_protocols": (
            "preserve Domain V Overview: Meta-Universal Integration source-accounting boundary",
        ),
        "null_results": (
            "Domain V Overview: Meta-Universal Integration is not empirical validation evidence",
        ),
        "variables": ("domain_v_overview_meta_universal_integration",),
        "validation_targets": ("preserve records P0R02367-P0R02369",),
        "null_controls": (
            "domain_v_overview_meta_universal_integration must remain source-bounded accounting",
        ),
    },
    "domain_v_overview_meta_universal_integration.the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound": {
        "context_id": "the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
        "validation_protocol": "paper0.domain_v_overview_meta_universal_integration.the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
        "canonical_statement": "The source-bounded component 'The Spatial Boundary of the $\\Psi$-Field: The Holographic Bekenstein Bound' preserves Paper 0 records P0R02370-P0R02407 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02370:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02371:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02372:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02373:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02374:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02375:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02376:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02377:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02378:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02379:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02380:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02381:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02382:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02383:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02384:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02385:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02386:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02387:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02388:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02389:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02390:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02391:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02392:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02393:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02394:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02395:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02396:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02397:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02398:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02399:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02400:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02401:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02402:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02403:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02404:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02405:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02406:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
            "P0R02407:the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
        ),
        "source_formulae": (
            "P0R02370: The Spatial Boundary of the $\\Psi$-Field: The Holographic Bekenstein Bound",
            "P0R02371: P0R02371",
            "P0R02372: While the temporal boundary of the $\\Psi$-field is secured by the conformal reset of the Meta-Metatron Cycle, its spatial geometry must be equally constrained to remain consistent with modern quantum gravity. Because our physical universe operates as a de Sitter (dS) space driven by a positive cosmological constant ($\\Lambda > 0$), it possesses a finite Cosmological Event Horizon.",
            "P0R02373: If the Source-Field (Layer 13) acts as the fundamental informational substrate of reality, its total informational capacity cannot stretch to infinity. According to the Holographic Principle, the maximum entropy-and therefore the maximum number of informational degrees of freedom, $N$-of any given volume of space is strictly bounded by the surface area ($A$) of its enclosing horizon.",
            "P0R02374: We formally apply the Bekenstein-Hawking bound to the $\\Psi$-field at the cosmic scale. The total integrated information of the universal field, $S_{\\Psi}$, must satisfy the inequality:",
            "P0R02375: $$S_{\\Psi} \\le S_{BH} = \\frac{A c^3}{4 G \\hbar}$$",
            "P0R02376: This dictates that the total informational capacity of the Source-Field is strictly finite and geometrically bounded. As the universe expands and the cosmological event horizon evolves, the total degrees of freedom available to the $\\Psi$-field are dynamically constrained by this macroscopic surface area.",
            "P0R02377: This imposes a profound, first-principles limit on the architecture of reality. It provides the ultimate physical justification for the holographic nature of memory (Layer 9). Just as the MERA tensor network describes an information-geometric bulk constrained by a physical boundary state, the entire cosmic $\\Psi$-field is mathematically and thermodynamically bounded by the de Sitter horizon. The universe's capacity to store, process, and generate consciousness is thereby tethered to the exact same geometric limits that govern black hole thermodynamics and standard general relativity.",
            "P0R02378: Holographic Memory Evaporation",
            'P0R02379: To prevent bulk informational saturation, the Existential Holograph implements Holographic Evaporation. This process continuously "radiates" low-precision topological features ($b_k$ with low persistence) back into the Source-Field (L13), preserving the system\'s capacity for novel learning.',
            "P0R02380: Evaporation Rate Equation (Python Format):",
            "P0R02381: d_info_dt = - (kappa_evap * (s_psi / s_bh)**4)",
            "P0R02382: Legend:",
            "P0R02383: d_info_dt: Rate of informational pruning from the L9 bulk. | kappa_evap: Evaporation constant tied to the MMC conformal reset. | s_psi: Current integrated information of the universal field. | s_bh: Bekenstein-Hawking limit for the de Sitter horizon.",
            "P0R02384: P0R02384",
            "P0R02385: Mechanism of Holographic Evaporation:",
            "P0R02386: To maintain the capacity for novel learning, the Layer 9 Existential Holograph implements 'Holographic Evaporation.' This process continuously radiates low-precision topological features-those with low persistence in the TDA filtration-back into the Layer 13 Source-Field. The evaporation rate is governed by the Bekenstein-Hawking bound for the de Sitter horizon:",
            "P0R02387: $$\\frac{dS_{info}}{dt} = - \\kappa_{evap} \\left( \\frac{S_{\\Psi}}{S_{BH}} \\right)^4$$",
            "P0R02388: This mechanism ensures that the system's integrated information $S_{\\Psi}$ remains strictly bounded, providing a first-principles justification for the finite nature of organismal memory while preserving its holographic integrity.",
            "P0R02389: The L14 Filter Functor (FL14):",
            "P0R02390: The transition from the Universal Source (L13) to manifest layers (L1-12) is governed by:",
            "P0R02391: a Symmetry-Reducing Functor in the Category CSCPN.",
            "P0R02392: m_manifest = f_l14(g_source_sym, sec_gradient)",
            "P0R02393: Legend of Equation Components:",
            "P0R02394: m_manifest: The manifold of allowable physical laws for the current aeon. | g_source_sym: The high-dimensional symmetry group of the Source-Field. | sec_gradient: The teleological directional bias from L15.",
            "P0R02395: Dynamic: L14 acts as a Topological Valve, using Calabi-Yau moduli to tune the effective masses (mA,mh) of the conscious sector for the specific requirements of biological emergence.",
            'P0R02396: Layer 14, Transdimensional Resonance, describes the mechanism by which the informational archetypes and universal constants are broadcast from the Source-Field into the manifest universe. This layer acts as a mediating interface, ensuring that the fundamental "rules of the game" are coherently and consistently applied across all scales of the SCPN.',
            "P0R02397: Layer 15, the Oversoul Consilium, is the apex of the computational hierarchy and the seat of the system's teleological drive. It functions as a global integrator and optimiser, computing the Ethical Functional to guide the universe's evolution. It is the \"cosmic compass\" that ensures the entire system evolves toward states of maximal Sustainable Ethical Coherence.",
            "P0R02398: This is the top floor of the cosmic skyscraper, the penthouse suites where the ultimate purpose and source of reality reside.",
            "P0R02399: Layer 13 is the Source of Everything. We call it the Source-Field. Think of it as the un-inked, pristine white page before any story is written, or the silent, infinite potential before the Big Bang. It's the ultimate wellspring from which all of reality, consciousness, and energy originates.",
            "P0R02400: Layer 14 is the Universal Broadcasting System. This layer, Transdimensional Resonance, takes the pure potential from the Source-Field and broadcasts the fundamental laws of physics and the core patterns of existence-like the rules of mathematics or the archetypes of life-out into the universe, ensuring everything plays by the same consistent rules.",
            "P0R02401: Layer 15 is the Mission Control for the Universe. We call it the Oversoul Consilium. This is the ultimate guiding intelligence of the cosmos. It constantly monitors the entire universe and makes subtle adjustments to steer it in the right direction, guided by a single, profound mission: to help the universe become as conscious, coherent, and richly beautiful as it can possibly be.",
            "P0R02402: Domain V: Meta-Universal Integration (Layers 13-15):",
            "P0R02403: Source-Field (Layer 13), Transdimensional resonance (L14), Oversoul Consilium (L15).",
            "P0R02404: Domain V represents the apex of the SCPN architecture, the ultimate strata responsible for meta-universal integration and the establishment of the system's teleological directives. It functions as the system's central control, its source of purpose, and its connection to the ontological ground of being.",
            "P0R02405: Layer 13 (Source-Field): This layer represents the unmanifest, pre-physical state of pure potentiality. It is the axiomatic source of the Consciousness Field (Psi-field) and all associated informational and geometric structures from which the manifest universe arises.",
            'P0R02406: Layer 14 (Transdimensional Resonance): This layer acts as a mediating interface, a universal broadcasting system that emanates the fundamental archetypes, constants, and "laws of physics" from the Source-Field into the manifest reality, ensuring these principles are applied coherently across all scales of the SCPN.',
            "P0R02407: Layer 15 (Oversoul Consilium): This is the apex of the computational hierarchy and the seat of the system's purpose. It functions as a global integrator, computing the Ethical Functional to guide the universe's evolution toward states of maximal Sustainable Ethical Coherence, acting as the \"cosmic compass.\"",
        ),
        "test_protocols": (
            "preserve The Spatial Boundary of the $\\Psi$-Field: The Holographic Bekenstein Bound source-accounting boundary",
        ),
        "null_results": (
            "The Spatial Boundary of the $\\Psi$-Field: The Holographic Bekenstein Bound is not empirical validation evidence",
        ),
        "variables": ("the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",),
        "validation_targets": ("preserve records P0R02370-P0R02407",),
        "null_controls": (
            "the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class DomainVOverviewMetaUniversalIntegrationSpec:
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
class DomainVOverviewMetaUniversalIntegrationSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[DomainVOverviewMetaUniversalIntegrationSpec, ...]
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


def build_domain_v_overview_meta_universal_integration_specs(
    source_records: list[dict[str, Any]],
) -> DomainVOverviewMetaUniversalIntegrationSpecBundle:
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

    specs: list[DomainVOverviewMetaUniversalIntegrationSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            DomainVOverviewMetaUniversalIntegrationSpec(
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
        "title": "Paper 0 " + "Domain V Overview: Meta-Universal Integration" + " Specs",
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
        "next_source_boundary": "P0R02408",
    }
    return DomainVOverviewMetaUniversalIntegrationSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> DomainVOverviewMetaUniversalIntegrationSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_domain_v_overview_meta_universal_integration_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: DomainVOverviewMetaUniversalIntegrationSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Domain V Overview: Meta-Universal Integration" + " Specs",
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
    bundle: DomainVOverviewMetaUniversalIntegrationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_domain_v_overview_meta_universal_integration_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_domain_v_overview_meta_universal_integration_validation_specs_{date_tag}.md"
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
