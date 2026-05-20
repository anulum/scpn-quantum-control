#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Psi-Higgs LHC phenomenology spec builder
"""Promote Paper 0 Psi-Higgs LHC phenomenology records."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(1655, 1669))
CLAIM_BOUNDARY = "source-bounded Psi-Higgs LHC phenomenology bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "psi_higgs_lhc_phenomenology.phenomenology_bridge": {
        "context_id": "phenomenology_bridge",
        "validation_protocol": "paper0.psi_higgs_lhc_phenomenology.phenomenology_bridge",
        "canonical_statement": (
            "The source frames the Psi-Higgs as a massive scalar excitation that bridges SCPN to high-energy collider phenomenology."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:phenomenology_bridge" for n in range(1655, 1658)
        ),
        "source_formulae": (
            "The Psi-Higgs Boson: Phenomenology and Experimental Signatures at the LHC",
            "new fundamental scalar field Psi acquires a non-zero VEV through a Mexican-hat potential",
            "quantum excitation of the Psi-field is termed the Psi-Higgs boson h_Psi",
            "the prediction is framed as a falsifiable bridge to experimental high-energy physics",
        ),
        "test_protocols": ("preserve Psi-Higgs LHC phenomenology bridge boundary",),
        "null_results": ("phenomenology framing is not collider evidence",),
        "variables": ("Psi", "VEV", "h_Psi", "LHC"),
        "validation_targets": (
            "preserve scalar-excitation bridge",
            "preserve falsifiability framing",
        ),
        "null_controls": (
            "falsifiable bridge language must not be promoted to observed LHC signal",
        ),
    },
    "psi_higgs_lhc_phenomenology.scalar_mixing_mechanism": {
        "context_id": "scalar_mixing_mechanism",
        "validation_protocol": "paper0.psi_higgs_lhc_phenomenology.scalar_mixing_mechanism",
        "canonical_statement": (
            "The source states that radial Psi-field excitation yields h_Psi and that scalar mixing with the Standard Model Higgs is the collider portal."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:scalar_mixing_mechanism" for n in range(1658, 1662)
        ),
        "source_formulae": (
            "The Psi-Higgs Mechanism and Mixing",
            "V(|Psi|) = -m^2 |Psi|^2 + lambda |Psi|^4 facilitates SSB",
            "radial excitation around the Psi-field VEV manifests as massive h_Psi",
            "two scalar fields with identical quantum numbers generically mix",
            "Psi-field scalar can mix with the Standard Model Higgs h_SM",
            "h_Psi production and decay through SM particles is source-framed collider accessibility",
        ),
        "test_protocols": ("preserve scalar-mixing mechanism boundary",),
        "null_results": ("mixing mechanism claim is not a measured mixing angle",),
        "variables": ("Psi", "h_Psi", "h_SM", "V", "lambda", "VEV"),
        "validation_targets": (
            "preserve radial-excitation claim",
            "preserve Higgs-mixing portal claim",
        ),
        "null_controls": (
            "generic-mixing language must not be reported as measured Higgs admixture",
        ),
    },
    "psi_higgs_lhc_phenomenology.scalar_potential_and_cross_term": {
        "context_id": "scalar_potential_and_cross_term",
        "validation_protocol": "paper0.psi_higgs_lhc_phenomenology.scalar_potential_and_cross_term",
        "canonical_statement": (
            "The source formalizes the Higgs portal scalar potential and the post-SSB mass-matrix cross-term."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:scalar_potential_and_cross_term" for n in range(1662, 1669)
        ),
        "source_formulae": (
            "Formalizing the Scalar Mixing Potential and the theta Parameter",
            "Higgs Portal is the only renormalizable operator connecting a hidden scalar sector to the Standard Model",
            "V(H, Psi) = V_SM(H) + V_Psi(Psi) + V_mix(H, Psi)",
            "V_mix = lambda_mix (H^dagger H) |Psi|^2",
            "v_h approximately 246 GeV for the SM Higgs and v_psi for the Psi-field",
            "H -> v_h + h_bare and |Psi| -> v_psi + h_Psi,bare",
            "mixing term generates lambda_mix v_h v_psi h_bare h_Psi,bare in the mass matrix",
        ),
        "test_protocols": ("preserve Higgs-portal potential and cross-term boundary",),
        "null_results": ("portal potential is not an experimentally fitted parameter set",),
        "variables": ("H", "Psi", "V_SM", "V_Psi", "V_mix", "lambda_mix", "v_h", "v_psi"),
        "validation_targets": ("preserve total scalar potential", "preserve post-SSB cross-term"),
        "null_controls": (
            "portal formalism must remain source equation, not fitted LHC model evidence",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class PsiHiggsLHCPhenomenologySpec:
    """Psi-Higgs LHC phenomenology spec promoted from Paper 0 records."""

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
class PsiHiggsLHCPhenomenologySpecBundle:
    """Psi-Higgs LHC phenomenology specs plus source coverage summary."""

    specs: tuple[PsiHiggsLHCPhenomenologySpec, ...]
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


def build_psi_higgs_lhc_phenomenology_specs(
    source_records: list[dict[str, Any]],
) -> PsiHiggsLHCPhenomenologySpecBundle:
    """Build source-covered Psi-Higgs LHC phenomenology specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    category_counts = Counter(
        str(record.get("canonical_category", "unknown")) for record in anchors
    )
    block_counts = Counter(str(record.get("block_type", "unknown")) for record in anchors)

    specs: list[PsiHiggsLHCPhenomenologySpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            PsiHiggsLHCPhenomenologySpec(
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
        "title": "Paper 0 Psi-Higgs LHC Phenomenology Specs",
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
        "next_source_boundary": "P0R01669",
    }
    return PsiHiggsLHCPhenomenologySpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> PsiHiggsLHCPhenomenologySpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_psi_higgs_lhc_phenomenology_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: PsiHiggsLHCPhenomenologySpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 Psi-Higgs LHC Phenomenology Specs",
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
    bundle: PsiHiggsLHCPhenomenologySpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_psi_higgs_lhc_phenomenology_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_psi_higgs_lhc_phenomenology_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build this Paper 0 generated spec bundle from the ledger."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-17")
    args = parser.parse_args()
    bundle = build_from_ledger(args.ledger)
    outputs = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    print(outputs["json"])
    print(outputs["report"])


if __name__ == "__main__":
    main()
