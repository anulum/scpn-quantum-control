#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Components: spec builder
"""Promote Paper 0 Components: records."""

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
    "P0R01779",
    "P0R01780",
    "P0R01781",
    "P0R01782",
    "P0R01783",
    "P0R01784",
    "P0R01785",
    "P0R01786",
    "P0R01787",
    "P0R01788",
    "P0R01789",
    "P0R01790",
    "P0R01791",
)
CLAIM_BOUNDARY = "source-bounded components source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "components.components": {
        "context_id": "components",
        "validation_protocol": "paper0.components.components",
        "canonical_statement": "The source-bounded component 'Components:' preserves Paper 0 records P0R01779-P0R01791 without empirical validation claims.",
        "source_equation_ids": (
            "P0R01779:components",
            "P0R01780:components",
            "P0R01781:components",
            "P0R01782:components",
            "P0R01783:components",
            "P0R01784:components",
            "P0R01785:components",
            "P0R01786:components",
            "P0R01787:components",
            "P0R01788:components",
            "P0R01789:components",
            "P0R01790:components",
            "P0R01791:components",
        ),
        "source_formulae": (
            "P0R01779: Components:",
            "P0R01780: Kinetic Term: Describes field propagation. | Potential Term (V(Psi)): A non-linear potential enabling Spontaneous Symmetry Breaking (SSB) and the formation of localised structures (solitons, L5 Selves). We utilize the 6 potential, which allows for hierarchical structure formation: V(Psi)=m2Psi2+lambdaPsi4Psi6 | Topological Term (LTopological): Ensures the conservation of topological charge, related to the integrity of conscious experience (e.g., a Theta term thetaFF).",
            "P0R01781: Stability and EFT structure of V(Psi)V(|\\Psi|)V(Psi).",
            "P0R01782: In 3+1D, the sextic interaction must stabilise the vacuum at large field. We adopt the EFTconsistent form",
            "P0R01783: $V\\left( \\mid\\Psi\\mid \\right)\\ \\mspace{2mu} = \\ \\mspace{2mu} - \\mu 2 \\mid \\Psi \\mid 2 + \\lambda \\mid \\Psi \\mid 4 + \\gamma\\Lambda 2 \\mid \\Psi \\mid 6,\\lambda > 0,\\ \\mspace{2mu}\\gamma > 0,V\\left( |\\Psi| \\right)\\ = \\ - \\mu^{2}|\\Psi|^{2} + \\lambda|\\Psi|^{4} + \\frac{\\gamma}{\\Lambda^{2}}|\\Psi|^{6},\\quad\\quad\\lambda > 0,\\ \\gamma > 0,V\\left( \\mid\\Psi\\mid \\right) = - \\mu 2 \\mid \\Psi \\mid 2 + \\lambda \\mid \\Psi \\mid 4 + \\Lambda 2\\gamma \\mid \\Psi \\mid 6,\\lambda > 0,\\gamma > 0$,",
            "P0R01784: where \\Lambda is a UV scale suppressing the dimension6 operator. Writing xPsi2x\\equiv|\\Psi|^2xPsi2, stationary points satisfy",
            "P0R01785: $dVdx = - \\mu 2 + 2\\lambda x + 3\\gamma\\Lambda 2x2 = 0\\ \\mspace{2mu} \\Rightarrow \\ \\mspace{2mu} x \\star = \\Lambda 26\\gamma\\,( - 2\\lambda + 4\\lambda 2 + 12\\gamma\\mu 2\\Lambda 2)\\,.\\frac{dV}{dx} = - \\mu^{2} + 2\\lambda x + \\frac{3\\gamma}{\\Lambda^{2}}x^{2} = 0\\ \\Rightarrow \\ x_{\\star} = \\frac{\\Lambda^{2}}{6\\gamma}\\text{!}\\left( - 2\\lambda + \\sqrt[]{4\\lambda^{2} + \\frac{12\\gamma\\mu^{2}}{\\Lambda^{2}}} \\right)\\text{!}.dxdV = - \\mu 2 + 2\\lambda x + \\Lambda 23\\gamma x2 = 0 \\Rightarrow x \\star = 6\\gamma\\Lambda 2( - 2\\lambda + 4\\lambda 2 + \\Lambda 212\\gamma\\mu 2).$",
            "P0R01786: Thus, v=xv=\\sqrt{x_\\star}v=x and the radial (PsiHiggs) mass is",
            "P0R01787: $mh2\\ \\mspace{2mu} = \\ \\mspace{2mu} d2Vd\\rho 2 \\mid \\rho = v = - 2\\mu 2 + 12\\lambda v2 + 30\\gamma\\Lambda 2v4,\\rho \\equiv \\mid \\Psi \\mid .m\\_ h\\hat{}2\\ \\backslash; = \\backslash;\\ \\backslash left.\\backslash frac\\{ d\\hat{}2V\\}\\{ d\\backslash rho\\hat{}2\\}\\backslash right|\\_\\{\\backslash rho = v\\}\\ = \\ - 2\\backslash mu\\hat{}2\\ + \\ 12\\backslash lambda\\ v\\hat{}2\\ + \\ 30\\backslash frac\\{\\backslash gamma\\}\\{\\backslash Lambda\\hat{}2\\} v\\hat{}4,\\ \\backslash quad\\ \\backslash rho\\backslash equiv|\\backslash Psi|.mh2 = d\\rho 2d2V\\rho = v = - 2\\mu 2 + 12\\lambda v2 + 30\\Lambda 2\\gamma v4,\\rho \\equiv \\mid \\Psi \\mid .\\ $",
            'P0R01788: Consistency note. Expressions elsewhere that assume mh2=2lambdav2m_h^2=2\\lambda v^2mh2=2lambdav2 are the -> 0\\gamma\\!\\to\\!0->0 limit and should be interpreted as specialcase approximations. This correction preserves the "PsiHiggs" phenomenology while ensuring vacuum stability.',
            "P0R01789: [IMAGE:Ein Bild, das Text, Screenshot, Diagramm, Reihe enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R01790: Fig.: Psi-Field Potential and Vacuum Stability. This plot illustrates the necessity of the sextic (Psi6) term in the potential V(Psi) to ensure a stable universe. It contrasts the stable, EFT-consistent potential with unstable alternatives.",
            "P0R01791: We plot V(Psi)=mu2Psi2+lambdaPsi4+(/2)Psi6V(|\\Psi|) = -\\mu^2 |\\Psi|^2 + \\lambda |\\Psi|^4 + (\\gamma/\\Lambda^2) |\\Psi|^6V(Psi)=mu2Psi2+lambdaPsi4+(/2)Psi6 with mu=1\\mu=1mu=1. The stable quartic case (lambda>0\\lambda>0lambda>0) yields a Mexican-hat profile with a bounded minimum. The unstable quartic case (lambda<0\\lambda<0lambda<0) runs away (unbounded from below). Adding a positive sextic term (>0\\gamma>0>0) stabilises the potential at large Psi|\\Psi|Psi, restoring a bounded vacuum consistent with EFT expectations.",
        ),
        "test_protocols": ("preserve Components: source-accounting boundary",),
        "null_results": ("Components: is not empirical validation evidence",),
        "variables": ("components",),
        "validation_targets": ("preserve records P0R01779-P0R01791",),
        "null_controls": ("components must remain source-bounded accounting",),
    }
}


@dataclass(frozen=True, slots=True)
class ComponentsSpec:
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
class ComponentsSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[ComponentsSpec, ...]
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


def build_components_specs(source_records: list[dict[str, Any]]) -> ComponentsSpecBundle:
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

    specs: list[ComponentsSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ComponentsSpec(
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
        "title": "Paper 0 Components: Specs",
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
        "next_source_boundary": "P0R01792",
    }
    return ComponentsSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(ledger_path: Path = DEFAULT_LEDGER_PATH) -> ComponentsSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_components_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: ComponentsSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 Components: Specs",
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
    bundle: ComponentsSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_components_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_components_validation_specs_{date_tag}.md"
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 component specs from the ledger."""

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
