#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0  spec builder
"""Promote Paper 0  records."""

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
    "P0R01959",
    "P0R01960",
    "P0R01961",
    "P0R01962",
    "P0R01963",
    "P0R01964",
    "P0R01965",
    "P0R01966",
    "P0R01967",
    "P0R01968",
    "P0R01969",
    "P0R01970",
    "P0R01971",
    "P0R01972",
    "P0R01973",
    "P0R01974",
    "P0R01975",
    "P0R01976",
    "P0R01977",
    "P0R01978",
    "P0R01979",
    "P0R01980",
    "P0R01981",
    "P0R01982",
    "P0R01983",
    "P0R01984",
    "P0R01985",
    "P0R01986",
    "P0R01987",
    "P0R01988",
    "P0R01989",
    "P0R01990",
    "P0R01991",
    "P0R01992",
)
CLAIM_BOUNDARY = "source-bounded paper0 slice source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "paper0_slice.source_component": {
        "context_id": "source_component",
        "validation_protocol": "paper0.paper0_slice.source_component",
        "canonical_statement": "The source-bounded component '' preserves Paper 0 records P0R01959-P0R01964 without empirical validation claims.",
        "source_equation_ids": (
            "P0R01959:source_component",
            "P0R01960:source_component",
            "P0R01961:source_component",
            "P0R01962:source_component",
            "P0R01963:source_component",
            "P0R01964:source_component",
        ),
        "source_formulae": (
            "P0R01959: The hierarchy ratio R 8 x 10 between EM and gravity couplings is enormous. Option 2 (seed with calculable hierarchy) is the most physically motivated path forward: a single fundamental coupling in a higher-dimensional theory that produces the observed ratios via geometric overlap factors.",
            "P0R01960: The H_int = lambdaPsi_ssigma notation in the SCPN papers implicitly adopts a variant of Option 2: the single lambda is modulated by the layer-specific sigma, which absorbs the hierarchy. This is conceptually clean but the explicit computation - showing how sigma at Layer 1 (microtubules) vs. sigma at Layer 6 (planetary Schumann resonances) vs. sigma at Layer 11 (noospheric spin states) produces the observed coupling strengths - remains to be done.",
            "P0R01961: 2.6.7 Resolution of the Coupling Hierarchy via Warp-Factor Localization (Revision 12.00):",
            "P0R01962: We resolve the $10^{120}$ discrepancy between $\\lambda_{\\psi,G}$ and $\\lambda_{\\psi,EM}$ by deriving the effective 4D couplings from the 5D action. The $\\Psi$-field is localized on the 'Consciousness Brane' at $y = \\pi r_c$, while gravity propagates in the bulk.",
            "P0R01963: Electromagnetic Sector: Since $A_\\mu$ and $\\Psi$ are co-localized, the overlap integral is $O(1)$, yielding $\\lambda_{\\psi,EM} = \\lambda_0 \\approx 0.092$. | Gravitational Sector: The gravitational coupling is suppressed by the exponential warp factor $e^{-2k\\pi r_c}$ of the RS1 geometry. By setting the compactification radius such that $k\\pi r_c \\approx 140$, we obtain:",
            "P0R01964: $$\\lambda_{\\psi,G} = \\lambda_0 \\int_0^{\\pi r_c} e^{-2ky} \\delta(y - \\pi r_c) dy = \\lambda_0 e^{-2k\\pi r_c}$$ This provides a first-principles derivation of $\\lambda_{\\psi,G} \\approx 10^{-122}$, matching the Dark Energy density without fine-tuning. The immense hierarchy is revealed to be a geometric consequence of the $\\Psi$-field's proximity to the Standard Model brane and its distance from the Planckian source of curvature.",
        ),
        "test_protocols": ("preserve  source-accounting boundary",),
        "null_results": (" is not empirical validation evidence",),
        "variables": ("source_component",),
        "validation_targets": ("preserve records P0R01959-P0R01964",),
        "null_controls": ("source_component must remain source-bounded accounting",),
    },
    "paper0_slice.p0r01965": {
        "context_id": "p0r01965",
        "validation_protocol": "paper0.paper0_slice.p0r01965",
        "canonical_statement": "The source-bounded component 'P0R01965' preserves Paper 0 records P0R01965-P0R01965 without empirical validation claims.",
        "source_equation_ids": ("P0R01965:p0r01965",),
        "source_formulae": ("P0R01965: P0R01965",),
        "test_protocols": ("preserve P0R01965 source-accounting boundary",),
        "null_results": ("P0R01965 is not empirical validation evidence",),
        "variables": ("p0r01965",),
        "validation_targets": ("preserve records P0R01965-P0R01965",),
        "null_controls": ("p0r01965 must remain source-bounded accounting",),
    },
    "paper0_slice.2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral": {
        "context_id": "2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
        "validation_protocol": "paper0.paper0_slice.2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
        "canonical_statement": "The source-bounded component '2.6.8 Formal Derivation of the Hierarchy via Bulk-Brane Overlap Integrals' preserves Paper 0 records P0R01966-P0R01992 without empirical validation claims.",
        "source_equation_ids": (
            "P0R01966:2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
            "P0R01967:2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
            "P0R01968:2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
            "P0R01969:2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
            "P0R01970:2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
            "P0R01971:2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
            "P0R01972:2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
            "P0R01973:2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
            "P0R01974:2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
            "P0R01975:2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
            "P0R01976:2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
            "P0R01977:2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
            "P0R01978:2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
            "P0R01979:2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
            "P0R01980:2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
            "P0R01981:2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
            "P0R01982:2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
            "P0R01983:2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
            "P0R01984:2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
            "P0R01985:2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
            "P0R01986:2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
            "P0R01987:2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
            "P0R01988:2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
            "P0R01989:2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
            "P0R01990:2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
            "P0R01991:2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
            "P0R01992:2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
        ),
        "source_formulae": (
            "P0R01966: 2.6.8 Formal Derivation of the Hierarchy via Bulk-Brane Overlap Integrals",
            "P0R01967: P0R01967",
            "P0R01968: To resolve the hierarchy between $\\lambda_{\\psi, EM}$ and $\\lambda_{\\psi, G}$ without resorting to arbitrary parameter tuning, we must move beyond a purely 4D effective field theory. We formalize Option 2 by embedding the SCPN within a 5D Randall-Sundrum (RS1) warped geometry.",
            "P0R01969: In this framework, the universe possesses a fifth dimension $y$ bounded by two 3-branes: the Planck Brane at $y=0$ (where gravity is strongly localized) and the Standard Model/Consciousness Brane at $y = \\pi r_c$. The metric for this bulk spacetime is given by the non-factorizable line element:",
            "P0R01970: $$ds^2 = e^{-2ky} \\eta_{\\mu\\nu} dx^\\mu dx^\\nu - dy^2$$",
            "P0R01971: where $k$ is the AdS curvature scale and $e^{-2ky}$ is the exponential warp factor that distorts spacetime geometry along the extra dimension.",
            "P0R01972: We posit a single, fundamental 5D interaction coupling constant, $\\lambda_0$. The phenomenological 4D couplings we observe are the result of integrating the 5D action over the extra dimension $y$. The resulting effective coupling strengths are determined strictly by the geometric overlap of the fields' wavefunctions in the bulk.",
            "P0R01973: 2.6.8.1 The Electromagnetic-Informational Coupling ($\\lambda_{\\psi, EM}$)",
            "P0R01974: The Standard Model electromagnetic field ($A_\\mu$) and the complex scalar $\\Psi$-field are both confined to the Consciousness Brane at $y = \\pi r_c$. Because both fields share the exact same spatial localization in the bulk, their overlap integral is $O(1)$ and is unsuppressed by the bulk warp factor. Integrating out the fifth dimension yields:",
            "P0R01975: $$\\lambda_{\\psi, EM} = \\int_0^{\\pi r_c} dy \\ \\lambda_0 \\delta(y - \\pi r_c) = \\lambda_0$$",
            "P0R01976: 2.6.8.2 The Gravitational-Geometric Coupling ($\\lambda_{\\psi, G}$)",
            "P0R01977: Conversely, gravity propagates throughout the entire 5D bulk. However, the graviton zero-mode wave function is heavily peaked at the Planck Brane ($y=0$) and decays exponentially across the bulk. The effective 4D gravitational coupling is determined by the overlap of the $\\Psi$-field (localized at $y = \\pi r_c$) with the tail of the graviton zero-mode profile ($\\sim e^{-2ky}$):",
            "P0R01978: $$\\lambda_{\\psi, G} = \\lambda_0 \\int_0^{\\pi r_c} dy \\ \\delta(y - \\pi r_c) e^{-2ky} = \\lambda_0 e^{-2k \\pi r_c}$$",
            "P0R01979: 2.6.8.3 Deriving the $10^{120}$ Discrepancy",
            "P0R01980: The ratio $R$ between the electromagnetic and gravitational coupling strengths of the $\\Psi$-field is therefore completely independent of the bare coupling $\\lambda_0$. It is purely a function of the spacetime geometry:",
            "P0R01981: $$R = \\frac{\\lambda_{\\psi, EM}}{\\lambda_{\\psi, G}} = \\frac{\\lambda_0}{\\lambda_0 e^{-2k \\pi r_c}} = e^{2k \\pi r_c}$$",
            "P0R01982: As derived in Chapter 20, observational constraints require a hierarchy ratio of $R \\approx 8 \\times 10^{120}$. Equating this phenomenological requirement to our geometric derivation:",
            "P0R01983: $$e^{2k \\pi r_c} = 8 \\times 10^{120}$$",
            "P0R01984: Taking the natural logarithm of both sides allows us to solve for the necessary size of the extra dimension:",
            "P0R01985: $$2k \\pi r_c \\approx \\ln(8 \\times 10^{120}) \\approx 278.3$$",
            "P0R01986: $$k \\pi r_c \\approx 139.15$$",
            "P0R01987: This provides a mathematically rigorous, first-principles derivation for the $8 \\times 10^{120}$ discrepancy. The immense difference in interaction strength is not an ad hoc postulate; it is the inescapable geometric consequence of the $\\Psi$-field being spatially separated from the center of gravity's mass across a warped fifth dimension. The coupling hierarchy is thus completely resolved, leaving zero free parameters to be tuned by hand.",
            "P0R01988: P0R01988",
            "P0R01989: 2.6.9 What This Means for Falsifiability",
            "P0R01990: No wiggle room: each coupling is either a derived number or lives on a 1-D fixed curve.",
            "P0R01991: Distinct predictions: astronomers, gravitational-wave analysts, and qubit laboratories now have precise targets. If DeltaT/T ever exceeds 4 x 10 at Psi-score = 5sigma, the framework is in trouble. Cross-checks become mandatory: the huge ratio R must manifest as detectable EM anomalies but zero gravity anomalies. If the opposite is observed, the framework is refuted.",
            "P0R01992: P0R01992",
        ),
        "test_protocols": (
            "preserve 2.6.8 Formal Derivation of the Hierarchy via Bulk-Brane Overlap Integrals source-accounting boundary",
        ),
        "null_results": (
            "2.6.8 Formal Derivation of the Hierarchy via Bulk-Brane Overlap Integrals is not empirical validation evidence",
        ),
        "variables": ("2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",),
        "validation_targets": ("preserve records P0R01966-P0R01992",),
        "null_controls": (
            "2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Paper0SliceSpec:
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
class Paper0SliceSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Paper0SliceSpec, ...]
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


def build_paper0_slice_specs(source_records: list[dict[str, Any]]) -> Paper0SliceSpecBundle:
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

    specs: list[Paper0SliceSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Paper0SliceSpec(
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
        "title": "Paper 0 " + "" + " Specs",
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
        "next_source_boundary": "P0R01993",
    }
    return Paper0SliceSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(ledger_path: Path = DEFAULT_LEDGER_PATH) -> Paper0SliceSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_paper0_slice_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: Paper0SliceSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "" + " Specs",
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
    bundle: Paper0SliceSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_paper0_slice_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_paper0_slice_validation_specs_{date_tag}.md"
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
