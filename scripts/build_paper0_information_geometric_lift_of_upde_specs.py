#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Information-Geometric Lift of UPDE spec builder
"""Promote Paper 0 Information-Geometric Lift of UPDE records."""

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
    "P0R02640",
    "P0R02641",
    "P0R02642",
    "P0R02643",
    "P0R02644",
    "P0R02645",
    "P0R02646",
    "P0R02647",
    "P0R02648",
    "P0R02649",
    "P0R02650",
    "P0R02651",
    "P0R02652",
    "P0R02653",
    "P0R02654",
)
CLAIM_BOUNDARY = "source-bounded information geometric lift of upde source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "information_geometric_lift_of_upde.information_geometric_lift_of_upde": {
        "context_id": "information_geometric_lift_of_upde",
        "validation_protocol": "paper0.information_geometric_lift_of_upde.information_geometric_lift_of_upde",
        "canonical_statement": "The source-bounded component 'Information-Geometric Lift of UPDE' preserves Paper 0 records P0R02640-P0R02654 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02640:information_geometric_lift_of_upde",
            "P0R02641:information_geometric_lift_of_upde",
            "P0R02642:information_geometric_lift_of_upde",
            "P0R02643:information_geometric_lift_of_upde",
            "P0R02644:information_geometric_lift_of_upde",
            "P0R02645:information_geometric_lift_of_upde",
            "P0R02646:information_geometric_lift_of_upde",
            "P0R02647:information_geometric_lift_of_upde",
            "P0R02648:information_geometric_lift_of_upde",
            "P0R02649:information_geometric_lift_of_upde",
            "P0R02650:information_geometric_lift_of_upde",
            "P0R02651:information_geometric_lift_of_upde",
            "P0R02652:information_geometric_lift_of_upde",
            "P0R02653:information_geometric_lift_of_upde",
            "P0R02654:information_geometric_lift_of_upde",
        ),
        "source_formulae": (
            "P0R02640: Information-Geometric Lift of UPDE",
            "P0R02641: We treat the phase distribution at layer LLL as pL(theta)p_L(\\theta)pL(theta) on the statistical manifold MLDelta(S1)NL\\mathcal{M}_L\\cong\\Delta(\\mathbb{S}^1)^{N_L}MLDelta(S1)NL with Fisher metric gFg_FgF. Writing a phase potential FL(theta)\\mathcal{F}_L(\\theta)FL(theta) (e.g., effective Kuramoto-type Lyapunov), the natural-gradient flow gives:",
            "P0R02642: $\\theta iL = - \\eta L\\,(gF - 1\\nabla\\theta FL)i\\ \\mspace{2mu} + \\ \\mspace{2mu} CInterLayer + CField + \\eta iL(t).\\dot{\\theta_{i}^{L}} = - \\eta_{L}\\,\\backslash big\\left( g_{F}^{- 1}\\nabla_{\\theta\\mathcal{F}_{\\mathcal{L\\backslash}}}big \\right)_{i}\\ + \\ C_{\\text{InterLayer}} + C_{\\text{Field}} + \\eta_{i}^{L}(t).\\theta iL\\mathbf{=} - \\eta L(gF - 1\\nabla\\theta FL)i + CInterLayer + CField + \\eta iL(t).$",
            "P0R02643: Thus UPDE is interpreted as gradient flow on (ML,gF)(\\mathcal{M}_L,g_F)(ML,gF) with extra drives; CFieldC_{\\text{Field}}CField acts as a gauge-connection 1-form AL\\mathcal{A}_LAL on S1\\mathbb{S}^1S1 (phase fibre) with phase bias:",
            "P0R02644: $CField = \\zeta L\\,\\Psi Globalcos\\,(\\theta iL - \\Theta\\Psi)\\ \\mspace{2mu} \\equiv \\ \\mspace{2mu}\\left\\langle AL,\\partial t\\theta iL \\right\\rangle.C_{\\text{Field}} = \\zeta_{L}\\,\\Psi_{\\text{Global}}\\cos{\\text{!}\\backslash}big\\left( \\theta_{i}^{L} - \\Theta_{\\Psi\\backslash}big \\right)\\ \\equiv \\ \\left\\langle \\mathcal{A}_{\\mathcal{L}},\\partial_{t}\\theta_{i}^{L} \\right\\rangle.CField\\mathbf{=}\\zeta L\\Psi Global cos(\\theta iL - \\Theta\\Psi)\\mathbf{\\equiv}\\left\\langle AL,\\partial t\\theta iL \\right\\rangle.$",
            'P0R02645: Implications. (i) harmonises Axiom-2 "interactions are informational and geometric" with UPDE; (ii) supplies a metric basis for stability proofs and controller design; (iii) sets up holonomy results used by the SCPN Torus.',
            "P0R02646: Geometric UPDE.",
            "P0R02647: On MMM with metric ggg and effective connection A(L)\\mathcal A^{(L)}A(L), the phase dynamics become",
            "P0R02648: thetai(L)=i(L)+ L,j Kij(LL)sin (thetaj(L)(ttauij(LL))thetai(L)ij(LL))L (gF1thetaFL)i+i(L).\\dot\\theta_i^{(L)} = \\omega_i^{(L)} +\\!\\!\\sum_{L',j}\\! K_{ij}^{(LL')}\\sin\\!\\Big(\\theta_j^{(L')}(t-\\tau_{ij}^{(LL')})-\\theta_i^{(L)}-\\alpha_{ij}^{(LL')}\\Big) -\\eta_L\\, (g_F^{-1}\\nabla_\\theta \\mathcal F_L)_i +\\xi_i^{(L)}.thetai(L)=i(L)+L,jKij(LL)sin(thetaj(L)(ttauij(LL))thetai(L)ij(LL))L(gF1thetaFL)i+i(L).",
            "P0R02649: [IMAGE:]",
            "P0R02650: Here ij(LL) = ij(\\*A(L)\\*A(L))dx\\alpha_{ij}^{(LL')}\\!=\\!\\int_{\\gamma_{ij}} (\\pi^\\*\\mathcal A^{(L')}-\\pi^\\*\\mathcal A^{(L)})\\cdot dxij(LL)=ij(\\*A(L)\\*A(L))dx encodes Berryphase holonomy around toroidal cycles (memory loops), aligning with the SCPN torus picture.",
            "P0R02651: P0R02651",
            "P0R02652: P0R02652",
            "P0R02653: P0R02653",
            "P0R02654: P0R02654",
        ),
        "test_protocols": (
            "preserve Information-Geometric Lift of UPDE source-accounting boundary",
        ),
        "null_results": (
            "Information-Geometric Lift of UPDE is not empirical validation evidence",
        ),
        "variables": ("information_geometric_lift_of_upde",),
        "validation_targets": ("preserve records P0R02640-P0R02654",),
        "null_controls": (
            "information_geometric_lift_of_upde must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class InformationGeometricLiftOfUpdeSpec:
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
class InformationGeometricLiftOfUpdeSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[InformationGeometricLiftOfUpdeSpec, ...]
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


def build_information_geometric_lift_of_upde_specs(
    source_records: list[dict[str, Any]],
) -> InformationGeometricLiftOfUpdeSpecBundle:
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

    specs: list[InformationGeometricLiftOfUpdeSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            InformationGeometricLiftOfUpdeSpec(
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
        "title": "Paper 0 " + "Information-Geometric Lift of UPDE" + " Specs",
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
        "next_source_boundary": "P0R02655",
    }
    return InformationGeometricLiftOfUpdeSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> InformationGeometricLiftOfUpdeSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_information_geometric_lift_of_upde_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: InformationGeometricLiftOfUpdeSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Information-Geometric Lift of UPDE" + " Specs",
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
    bundle: InformationGeometricLiftOfUpdeSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_information_geometric_lift_of_upde_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_information_geometric_lift_of_upde_validation_specs_{date_tag}.md"
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
