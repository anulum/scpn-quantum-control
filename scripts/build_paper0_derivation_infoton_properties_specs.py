#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 infoton-properties derivation spec builder
"""Promote Paper 0 infoton-properties derivation records."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(1623, 1638))
CLAIM_BOUNDARY = "source-bounded infoton-properties derivation bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "derivation_infoton_properties.lagrangian_and_potential": {
        "context_id": "lagrangian_and_potential",
        "validation_protocol": "paper0.derivation_infoton_properties.lagrangian_and_potential",
        "canonical_statement": (
            "The source starts the infoton-property derivation from a complex Psi-field coupled to a local U(1) gauge field and a Mexican-hat potential."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:lagrangian_and_potential" for n in range(1623, 1629)
        ),
        "source_formulae": (
            "Derivation of the Infoton's Properties",
            "spontaneous breaking of local U(1) gauge symmetry gives a massive infoton A_mu",
            "L = (D_mu Psi)^*(D^mu Psi) - V(|Psi|) - 1/4 F_mu_nu F^mu_nu",
            "D_mu = partial_mu - i g A_mu",
            "V(|Psi|) = -mu^2 |Psi|^2 + lambda |Psi|^4",
        ),
        "test_protocols": ("preserve Lagrangian and potential derivation boundary",),
        "null_results": ("formal source equation is not measured particle evidence",),
        "variables": ("Psi", "A_mu", "D_mu", "F_mu_nu", "g", "mu", "lambda"),
        "validation_targets": (
            "preserve gauge-covariant derivative",
            "preserve Mexican-hat potential",
        ),
        "null_controls": (
            "equation transcription must not be promoted to experimental confirmation",
        ),
    },
    "derivation_infoton_properties.vev_and_goldstone_absorption": {
        "context_id": "vev_and_goldstone_absorption",
        "validation_protocol": "paper0.derivation_infoton_properties.vev_and_goldstone_absorption",
        "canonical_statement": (
            "The source describes non-zero VEV expansion, radial and phase fluctuations, and Goldstone absorption by the gauge field."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:vev_and_goldstone_absorption" for n in range(1629, 1632)
        ),
        "source_formulae": (
            "Psi-field acquires a non-zero vacuum expectation value v",
            "|Psi| = mu / sqrt(2 lambda) = v / sqrt(2)",
            "Psi(x) = 1/sqrt(2) (v + h(x)) exp(i xi(x)/v)",
            "|D_mu Psi|^2 = 1/2 (partial_mu h)^2 + 1/2 (v+h)^2 (g A_mu - 1/v partial_mu xi)^2",
            "Goldstone boson xi(x) is absorbed by A_mu and removed from the physical spectrum",
        ),
        "test_protocols": ("preserve VEV expansion and Goldstone absorption boundary",),
        "null_results": ("Goldstone absorption description is not direct detector evidence",),
        "variables": ("v", "h_x", "xi_x", "A_mu", "Goldstone"),
        "validation_targets": (
            "preserve VEV expansion",
            "preserve Goldstone absorption mechanism",
        ),
        "null_controls": (
            "Goldstone wording must remain a source derivation, not a measured event",
        ),
    },
    "derivation_infoton_properties.mass_identification": {
        "context_id": "mass_identification",
        "validation_protocol": "paper0.derivation_infoton_properties.mass_identification",
        "canonical_statement": (
            "The source identifies the infoton vector-boson mass from the post-SSB mass term."
        ),
        "source_equation_ids": tuple(f"P0R{n:05d}:mass_identification" for n in range(1632, 1635)),
        "source_formulae": (
            "L_mass ~= 1/2 g^2 v^2 A_mu A^mu",
            "standard vector-boson mass term",
            "m_A = g v",
        ),
        "test_protocols": ("preserve infoton mass-identification boundary",),
        "null_results": ("mass relation is not an observed mass measurement",),
        "variables": ("L_mass", "g", "v", "A_mu", "m_A"),
        "validation_targets": ("preserve mass term", "preserve m_A relation"),
        "null_controls": ("derived mass relation must not be treated as measured mass",),
    },
    "derivation_infoton_properties.range_consequence": {
        "context_id": "range_consequence",
        "validation_protocol": "paper0.derivation_infoton_properties.range_consequence",
        "canonical_statement": (
            "The source links the derived infoton mass to a short-range informational force with a characteristic scale fixed by VEV and coupling."
        ),
        "source_equation_ids": tuple(f"P0R{n:05d}:range_consequence" for n in range(1635, 1638)),
        "source_formulae": (
            "range of a force mediated by mass m_A is approximately lambda_range ~= hbar / (m_A c)",
            "spontaneous breaking predicts a short-range informational force",
            "characteristic length scale is determined by Psi-field VEV and gauge coupling",
            "short range distinguishes the SCPN informational force from infinite-range forces",
        ),
        "test_protocols": ("preserve range-consequence boundary",),
        "null_results": ("range estimate is not a measured force profile",),
        "variables": ("lambda_range", "hbar", "m_A", "c", "v", "g"),
        "validation_targets": (
            "preserve force-range relation",
            "preserve short-range consequence",
        ),
        "null_controls": ("biological and cosmological implications must remain source claims",),
    },
}


@dataclass(frozen=True, slots=True)
class DerivationInfotonPropertiesSpec:
    """Infoton-properties derivation spec promoted from Paper 0 records."""

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
class DerivationInfotonPropertiesSpecBundle:
    """Infoton-properties derivation specs plus source coverage summary."""

    specs: tuple[DerivationInfotonPropertiesSpec, ...]
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


def build_derivation_infoton_properties_specs(
    source_records: list[dict[str, Any]],
) -> DerivationInfotonPropertiesSpecBundle:
    """Build source-covered infoton-properties derivation specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    category_counts = Counter(
        str(record.get("canonical_category", "unknown")) for record in anchors
    )
    block_counts = Counter(str(record.get("block_type", "unknown")) for record in anchors)

    specs: list[DerivationInfotonPropertiesSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            DerivationInfotonPropertiesSpec(
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
        "title": "Paper 0 Derivation Infoton Properties Specs",
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
        "next_source_boundary": "P0R01638",
    }
    return DerivationInfotonPropertiesSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> DerivationInfotonPropertiesSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_derivation_infoton_properties_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: DerivationInfotonPropertiesSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 Derivation Infoton Properties Specs",
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
    bundle: DerivationInfotonPropertiesSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_derivation_infoton_properties_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_derivation_infoton_properties_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
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
