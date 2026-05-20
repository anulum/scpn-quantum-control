#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 III. Euler-Lagrange for the Ethical Connection spec builder
"""Promote Paper 0 III. Euler-Lagrange for the Ethical Connection records."""

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
    "P0R06099",
    "P0R06100",
    "P0R06101",
    "P0R06102",
    "P0R06103",
    "P0R06104",
    "P0R06105",
    "P0R06106",
)
CLAIM_BOUNDARY = "source-bounded iii euler lagrange for the ethical connection source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "iii_euler_lagrange_for_the_ethical_connection.iii_euler_lagrange_for_the_ethical_connection": {
        "context_id": "iii_euler_lagrange_for_the_ethical_connection",
        "validation_protocol": "paper0.iii_euler_lagrange_for_the_ethical_connection.iii_euler_lagrange_for_the_ethical_connection",
        "canonical_statement": "The source-bounded component 'III. Euler-Lagrange for the Ethical Connection' preserves Paper 0 records P0R06099-P0R06106 without empirical validation claims.",
        "source_equation_ids": (
            "P0R06099:iii_euler_lagrange_for_the_ethical_connection",
            "P0R06100:iii_euler_lagrange_for_the_ethical_connection",
            "P0R06101:iii_euler_lagrange_for_the_ethical_connection",
            "P0R06102:iii_euler_lagrange_for_the_ethical_connection",
            "P0R06103:iii_euler_lagrange_for_the_ethical_connection",
            "P0R06104:iii_euler_lagrange_for_the_ethical_connection",
            "P0R06105:iii_euler_lagrange_for_the_ethical_connection",
            "P0R06106:iii_euler_lagrange_for_the_ethical_connection",
        ),
        "source_formulae": (
            "P0R06099: III. Euler-Lagrange for the Ethical Connection",
            "P0R06100: Varying SEthical=14MTr(FF)S_{\\text{Ethical}}=\\tfrac14\\int_M \\mathrm{Tr}(F\\wedge\\star F)SEthical=41MTr(FF) yields",
            "P0R06101: $D \\dagger F = JCEF,JCEF \\propto \\nabla XSC(X,\\tau),D^{\\backslash dagger}F = J_{\\text{CEF}},\\quad\\quad J_{\\text{CEF}} \\propto \\nabla_{X}S_{C}(X,\\tau),D \\dagger F = JCEF,JCEF \\propto \\nabla X SC(X,\\tau),$",
            "P0R06102: linking Yang-Mills stationarity to the Causal-Entropic gradient that biases accessible futures. The boundary term from the variation is",
            "P0R06103: $\\delta SEthical \\mid \\partial M = \\int_{}^{}{\\partial M}\\, Tr\\,(\\delta A \\land \\star F).\\delta S_{\\text{Ethical}\\backslash}big\\left. \\ \\right|_{\\partial M} = \\int_{\\partial M}^{}{\\text{!}\\text{Tr}}\\text{!}(\\delta A \\land \\star F).\\delta SEthical\\partial M = \\int_{}^{}{\\partial M Tr(\\delta A \\land \\star F)}.$",
            "P0R06104: Firewall correspondence. Interpreting the L10 firewall as a boundary controller that reduces rendering complexity implies a Complexity-Flux Inequality",
            "P0R06105: $dCL10dt\\ \\mspace{2mu} \\leq \\ \\mspace{2mu} - \\kappa Eth\\,\\Phi\\partial M\\ \\mspace{2mu} + \\ \\mspace{2mu}\\xi(t),\\Phi\\partial M = \\int_{}^{}{\\partial M}\\, Tr( \\star F),\\frac{d\\mathcal{C}_{\\text{L0}}}{dt}\\ \\leq \\ - \\kappa_{\\text{Eth}}\\,\\Phi_{\\partial M}\\ + \\ \\xi(t),\\quad\\Phi_{\\partial M} = \\int_{\\partial M}^{}{\\text{!}\\text{Tr}( \\star F)},dtdCL10 \\leq - \\kappa Eth\\Phi\\partial M + \\xi(t),\\Phi\\partial M = \\int_{}^{}{\\partial M Tr( \\star F)},$",
            "P0R06106: tying lower boundary complexity to higher SEC consistency.",
        ),
        "test_protocols": (
            "preserve III. Euler-Lagrange for the Ethical Connection source-accounting boundary",
        ),
        "null_results": (
            "III. Euler-Lagrange for the Ethical Connection is not empirical validation evidence",
        ),
        "variables": ("iii_euler_lagrange_for_the_ethical_connection",),
        "validation_targets": ("preserve records P0R06099-P0R06106",),
        "null_controls": (
            "iii_euler_lagrange_for_the_ethical_connection must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class IiiEulerLagrangeForTheEthicalConnectionSpec:
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
class IiiEulerLagrangeForTheEthicalConnectionSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[IiiEulerLagrangeForTheEthicalConnectionSpec, ...]
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


def build_iii_euler_lagrange_for_the_ethical_connection_specs(
    source_records: list[dict[str, Any]],
) -> IiiEulerLagrangeForTheEthicalConnectionSpecBundle:
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

    specs: list[IiiEulerLagrangeForTheEthicalConnectionSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            IiiEulerLagrangeForTheEthicalConnectionSpec(
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
        "title": "Paper 0 " + "III. Euler-Lagrange for the Ethical Connection" + " Specs",
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
        "next_source_boundary": "P0R06107",
    }
    return IiiEulerLagrangeForTheEthicalConnectionSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> IiiEulerLagrangeForTheEthicalConnectionSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_iii_euler_lagrange_for_the_ethical_connection_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: IiiEulerLagrangeForTheEthicalConnectionSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "III. Euler-Lagrange for the Ethical Connection" + " Specs",
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
    bundle: IiiEulerLagrangeForTheEthicalConnectionSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_iii_euler_lagrange_for_the_ethical_connection_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_iii_euler_lagrange_for_the_ethical_connection_validation_specs_{date_tag}.md"
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
