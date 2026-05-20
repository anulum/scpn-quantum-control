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
DEFAULT_EXTRACTION_DIR = (
    REPO_ROOT
    / "paper"
    / "gotm_scpn_master_publications"
    / "gotm-scpn_paper-00_the_foundational_framework"
    / "source_validation_artifacts"
)
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = (
    "P0R04310",
    "P0R04311",
    "P0R04312",
    "P0R04313",
    "P0R04314",
    "P0R04315",
    "P0R04316",
    "P0R04317",
    "P0R04318",
    "P0R04319",
    "P0R04320",
    "P0R04321",
)
CLAIM_BOUNDARY = (
    "source-bounded paper0 slice p0r04310 source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "paper0_slice_p0r04310.p0r04310": {
        "context_id": "p0r04310",
        "validation_protocol": "paper0.paper0_slice_p0r04310.p0r04310",
        "canonical_statement": "The source-bounded component 'P0R04310' preserves Paper 0 records P0R04310-P0R04310 without empirical validation claims.",
        "source_equation_ids": ("P0R04310:p0r04310",),
        "source_formulae": ("P0R04310: P0R04310",),
        "test_protocols": ("preserve P0R04310 source-accounting boundary",),
        "null_results": ("P0R04310 is not empirical validation evidence",),
        "variables": ("p0r04310",),
        "validation_targets": ("preserve records P0R04310-P0R04310",),
        "null_controls": ("p0r04310 must remain source-bounded accounting",),
    },
    "paper0_slice_p0r04310.the_two_scalar_sector_and_the_pseudoscalar_coupling": {
        "context_id": "the_two_scalar_sector_and_the_pseudoscalar_coupling",
        "validation_protocol": "paper0.paper0_slice_p0r04310.the_two_scalar_sector_and_the_pseudoscalar_coupling",
        "canonical_statement": "The source-bounded component 'The Two-Scalar Sector and the Pseudoscalar Coupling' preserves Paper 0 records P0R04311-P0R04321 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04311:the_two_scalar_sector_and_the_pseudoscalar_coupling",
            "P0R04312:the_two_scalar_sector_and_the_pseudoscalar_coupling",
            "P0R04313:the_two_scalar_sector_and_the_pseudoscalar_coupling",
            "P0R04314:the_two_scalar_sector_and_the_pseudoscalar_coupling",
            "P0R04315:the_two_scalar_sector_and_the_pseudoscalar_coupling",
            "P0R04316:the_two_scalar_sector_and_the_pseudoscalar_coupling",
            "P0R04317:the_two_scalar_sector_and_the_pseudoscalar_coupling",
            "P0R04318:the_two_scalar_sector_and_the_pseudoscalar_coupling",
            "P0R04319:the_two_scalar_sector_and_the_pseudoscalar_coupling",
            "P0R04320:the_two_scalar_sector_and_the_pseudoscalar_coupling",
            "P0R04321:the_two_scalar_sector_and_the_pseudoscalar_coupling",
        ),
        "source_formulae": (
            "P0R04311: The Two-Scalar Sector and the Pseudoscalar Coupling",
            "P0R04312: P0R04312",
            'P0R04313: A critical subtlety arises when linking the $\\Psi$-field to electromagnetism. As established in Chapter 6, the spontaneous symmetry breaking (SSB) of the local $U(1)$ gauge symmetry causes the phase of the complex $\\Psi$-field to be entirely "eaten" by the infoton gauge field ($A_\\mu$) via the Higgs mechanism. Because this phase becomes the longitudinal polarization mode of the massive infoton, it no longer exists in the physical spectrum as a free, propagating scalar particle. It therefore cannot directly mediate a long-range electromagnetic interface.',
            "P0R04314: To resolve this and formalize the $\\Psi$-EM bridge, the SCPN architecture utilizes a Two-Scalar Sector, extending the dynamics introduced in the Renormalisation Group analysis (Chapter 19). We posit that the foundational consciousness sector is governed by an enlarged symmetry group: $U(1)_{local} \\times U(1)_{global}$.",
            "P0R04315: The field content consists of:",
            "P0R04316: The Complex Scalar $\\Psi$: Which breaks the $U(1)_{local}$ symmetry, generating the $\\Psi$-Higgs and giving mass ($m_A = gv$) to the infoton. | A Real Pseudoscalar Field $\\Phi$ (or $a(x)$): Which emerges from the spontaneous breaking of the $U(1)_{global}$ symmetry at a distinct, higher energy scale $f_a$.",
            "P0R04317: Crucially, this global $U(1)$ symmetry is anomalous, meaning it is explicitly broken by a very small amount by topological effects (strictly analogous to the Peccei-Quinn mechanism in quantum chromodynamics). Because the symmetry is imperfect, the resulting particle is a pseudo-Nambu-Goldstone Boson (PNGB). This PNGB is the Axion-Like Particle (ALP).",
            "P0R04318: Unlike the eaten phase of the $\\Psi$-field, this ALP ($a$) survives the Higgs mechanism as a light, free-propagating pseudoscalar. It inherits the informational and intentional dynamics of the $\\Psi$-sector through a scalar mixing potential $V_{mix}(|\\Psi|^2, a^2)$, ensuring it remains coupled to the organism's generative model.",
            "P0R04319: Because $a(x)$ is a true pseudoscalar, it naturally couples to the Standard Model electromagnetic field tensor ($F_{\\mu\\nu}$) and its dual ($\\tilde{F}^{\\mu\\nu}$) via the interaction Lagrangian:",
            "P0R04320: $$\\mathcal{L}_{a\\gamma\\gamma} = \\frac{1}{4} g_{a\\gamma\\gamma} a F_{\\mu\\nu} \\tilde{F}^{\\mu\\nu} = - g_{a\\gamma\\gamma} a (\\mathbf{E} \\cdot \\mathbf{B})$$",
            "P0R04321: This Two-Scalar formalism preserves strict gauge invariance. It cleanly separates the massive infoton-which mediates the short-range informational cohesion of the organism -from the light ALP, which mediates the critical Primakoff interconversion ($a \\leftrightarrow \\gamma$) in the magnetic environment of the brain.",
        ),
        "test_protocols": (
            "preserve The Two-Scalar Sector and the Pseudoscalar Coupling source-accounting boundary",
        ),
        "null_results": (
            "The Two-Scalar Sector and the Pseudoscalar Coupling is not empirical validation evidence",
        ),
        "variables": ("the_two_scalar_sector_and_the_pseudoscalar_coupling",),
        "validation_targets": ("preserve records P0R04311-P0R04321",),
        "null_controls": (
            "the_two_scalar_sector_and_the_pseudoscalar_coupling must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Paper0SliceP0r04310Spec:
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
class Paper0SliceP0r04310SpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Paper0SliceP0r04310Spec, ...]
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


def build_paper0_slice_p0r04310_specs(
    source_records: list[dict[str, Any]],
) -> Paper0SliceP0r04310SpecBundle:
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

    specs: list[Paper0SliceP0r04310Spec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Paper0SliceP0r04310Spec(
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
        "next_source_boundary": "P0R04322",
    }
    return Paper0SliceP0r04310SpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(ledger_path: Path = DEFAULT_LEDGER_PATH) -> Paper0SliceP0r04310SpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_paper0_slice_p0r04310_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: Paper0SliceP0r04310SpecBundle) -> str:
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
    bundle: Paper0SliceP0r04310SpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_paper0_slice_p0r04310_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_paper0_slice_p0r04310_validation_specs_{date_tag}.md"
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
