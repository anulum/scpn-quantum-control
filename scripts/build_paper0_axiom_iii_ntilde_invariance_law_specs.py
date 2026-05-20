#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Axiom III Ntilde invariance-law spec builder
"""Promote Paper 0 Axiom III Ntilde-invariance-law records."""

from __future__ import annotations

import argparse
import json
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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(800, 811))
CLAIM_BOUNDARY = "source-bounded Axiom III Ntilde-invariance-law map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "axiom_iii_ntilde_invariance_law.physical_law_identification": {
        "context_id": "physical_law_identification",
        "validation_protocol": "paper0.axiom_iii_ntilde_invariance_law.physical_law_identification",
        "canonical_statement": (
            "The source identifies the Axiom III teleological drive with the "
            "Ntilde invariance law, framed as a fundamental falsifiable physical "
            "law and a dimensionless measurable invariant linking energy, "
            "information, and time."
        ),
        "source_equation_ids": (
            "P0R00800:ntilde_invariance_law_heading",
            "P0R00801:fundamental_falsifiable_physical_law",
            "P0R00801:dimensionless_measurable_energy_information_time_invariant",
        ),
        "source_formulae": (
            "Formal Physical Definition: The tilde_N_t Invariance Law",
            "Axiom III is a fundamental falsifiable physical law",
            "teleological drive identified with the tilde_N_t Invariance Law",
            "dimensionless measurable invariant linking energy information and time",
            "Petrasek 2025a 2025b",
        ),
        "test_protocols": ("preserve Ntilde physical-law identification",),
        "null_results": ("Ntilde invariance law is source claim, not empirical evidence",),
        "variables": ("tilde_N_t", "energy", "information", "time"),
        "validation_targets": (
            "preserve formal Ntilde law heading",
            "preserve fundamental falsifiable physical-law framing",
            "preserve dimensionless invariant linkage",
        ),
        "null_controls": (
            "physical-law-as-observed-result control must be rejected",
            "missing-energy-information-time-link control must be rejected",
        ),
    },
    "axiom_iii_ntilde_invariance_law.invariant_ratio_equation": {
        "context_id": "invariant_ratio_equation",
        "validation_protocol": "paper0.axiom_iii_ntilde_invariance_law.invariant_ratio_equation",
        "canonical_statement": (
            "The source defines Ntilde as actual power divided by the minimum "
            "reversible free-energy cost required to process information flow."
        ),
        "source_equation_ids": (
            "P0R00802:actual_power_over_reversible_information_cost",
            "P0R00803:ntilde_ratio_equation",
        ),
        "source_formulae": (
            "actual power energy flux over minimum reversible free-energy cost for information flow",
            "tilde_N_t = P / (epsilon_b dot_I) = (E/t) / ((Delta F_rev / Delta I) dot_I)",
        ),
        "test_protocols": ("preserve Ntilde ratio equation",),
        "null_results": ("ratio equation requires downstream measurement protocol",),
        "variables": ("tilde_N_t", "P", "epsilon_b", "dot_I", "E", "t", "Delta_F_rev", "Delta_I"),
        "validation_targets": (
            "preserve actual-power numerator",
            "preserve reversible information-processing cost denominator",
            "preserve expanded E/t and Delta_F_rev/Delta_I equation",
        ),
        "null_controls": (
            "ratio-equation-as-validated-invariant control must be rejected",
            "missing-expanded-equation control must be rejected",
        ),
    },
    "axiom_iii_ntilde_invariance_law.variable_definitions": {
        "context_id": "variable_definitions",
        "validation_protocol": "paper0.axiom_iii_ntilde_invariance_law.variable_definitions",
        "canonical_statement": (
            "The source defines the equation variables P, dot-I, and epsilon-b "
            "as power, reliably processed information rate, and reversible "
            "free-energy cost per bit."
        ),
        "source_equation_ids": (
            "P0R00804:where_clause",
            "P0R00805:power_energy_flux_definition",
            "P0R00806:information_rate_definition",
            "P0R00807:reversible_cost_per_bit_definition",
        ),
        "source_formulae": (
            "Where:",
            "P = E/t is actual power or energy flux",
            "dot_I is rate of reliably processed information bit/s",
            "epsilon_b = Delta F_rev / Delta I is reversible free-energy cost per bit",
            "non-arbitrary reversible free-energy cost per bit for that specific process",
        ),
        "test_protocols": ("preserve Ntilde variable definitions",),
        "null_results": ("variable definitions do not validate measurement availability",),
        "variables": ("P", "E", "t", "dot_I", "epsilon_b", "Delta_F_rev", "Delta_I"),
        "validation_targets": (
            "preserve P equals E over t definition",
            "preserve dot-I reliable information-rate definition",
            "preserve epsilon-b reversible free-energy cost definition",
        ),
        "null_controls": (
            "undefined-variable control must be rejected",
            "arbitrary-cost-per-bit control must be rejected",
        ),
    },
    "axiom_iii_ntilde_invariance_law.unity_threshold_limit": {
        "context_id": "unity_threshold_limit",
        "validation_protocol": "paper0.axiom_iii_ntilde_invariance_law.unity_threshold_limit",
        "canonical_statement": (
            "The source states that teleological optimisation tends towards the "
            "resonant unity threshold Ntilde equals one, identified as the "
            "reversible limit and maximum thermodynamic efficiency."
        ),
        "source_equation_ids": (
            "P0R00808:unity_threshold_tendency",
            "P0R00809:ntilde_tends_to_one",
            "P0R00810:reversible_limit_maximum_efficiency",
        ),
        "source_formulae": (
            "teleological optimisation seeks resonant threshold where ratio approaches unity",
            "tilde_N_t -> 1",
            "tilde_N_t = 1 is the reversible limit",
            "maximum thermodynamic efficiency minimum irreversibility or entropy production",
        ),
        "test_protocols": ("preserve Ntilde unity-threshold limit",),
        "null_results": ("unity threshold requires downstream operational validation",),
        "variables": ("tilde_N_t", "irreversibility", "entropy_production", "efficiency"),
        "validation_targets": (
            "preserve tendency towards unity",
            "preserve Ntilde tends to one threshold equation",
            "preserve reversible-limit and thermodynamic-efficiency statement",
        ),
        "null_controls": (
            "unity-threshold-as-measured-result control must be rejected",
            "missing-reversible-limit control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class AxiomIIINtildeInvarianceLawSpec:
    """Axiom III Ntilde-invariance-law spec promoted from Paper 0 records."""

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
class AxiomIIINtildeInvarianceLawSpecBundle:
    """Axiom III Ntilde-invariance-law specs plus source coverage summary."""

    specs: tuple[AxiomIIINtildeInvarianceLawSpec, ...]
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


def build_axiom_iii_ntilde_invariance_law_specs(
    source_records: list[dict[str, Any]],
) -> AxiomIIINtildeInvarianceLawSpecBundle:
    """Build source-covered Axiom III Ntilde-invariance-law specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[AxiomIIINtildeInvarianceLawSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            AxiomIIINtildeInvarianceLawSpec(
                key=key,
                context_id=str(metadata["context_id"]),
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0].get("section_path", "")),
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
                implementation_status="implemented_source_accounting_fixture",
                domain_review_status="requires_domain_review_before_public_claim",
                hardware_status=HARDWARE_STATUS,
            )
        )

    consumed = sorted({ledger_id for spec in specs for ledger_id in spec.source_ledger_ids})
    summary = {
        "title": "Paper 0 Axiom III Ntilde Invariance Law Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed
        ],
        "spec_count": len(specs),
        "invariant_definition_count": 3,
        "variable_definition_count": 3,
        "threshold_equation_count": 1,
        "reversible_limit_count": 1,
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R00811",
        "spec_keys": [spec.key for spec in specs],
    }
    return AxiomIIINtildeInvarianceLawSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> AxiomIIINtildeInvarianceLawSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = load_jsonl(ledger_path)
    selected = [record for record in records if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS]
    return build_axiom_iii_ntilde_invariance_law_specs(selected)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: AxiomIIINtildeInvarianceLawSpecBundle) -> str:
    """Render a Markdown report for the promoted specs."""
    lines = [
        "# Paper 0 Axiom III Ntilde Invariance Law Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Invariant-definition records: {bundle.summary['invariant_definition_count']}",
        f"- Variable-definition records: {bundle.summary['variable_definition_count']}",
        f"- Threshold-equation records: {bundle.summary['threshold_equation_count']}",
        f"- Reversible-limit records: {bundle.summary['reversible_limit_count']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        f"- Next source boundary: {bundle.summary['next_source_boundary']}",
        "",
        "## Promoted Specs",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                f"### {spec.key}",
                "",
                spec.canonical_statement,
                "",
                "Formulae / source labels:",
            ]
        )
        for formula in spec.source_formulae:
            lines.append(f"- {formula}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    bundle: AxiomIIINtildeInvarianceLawSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write spec JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_axiom_iii_ntilde_invariance_law_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_axiom_iii_ntilde_invariance_law_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "summary": _json_ready(bundle.summary),
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 Axiom III n-tilde invariance-law specs from the ledger."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args()

    bundle = build_from_ledger(args.ledger)
    outputs = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    print(outputs["json"])
    print(outputs["report"])


if __name__ == "__main__":
    main()
