#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 SSB Psi-field spec builder
"""Promote Paper 0 spontaneous-symmetry-breaking Psi-field records."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(1272, 1333))
CLAIM_BOUNDARY = "source-bounded SSB Psi-field mechanism claims; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "ssb_psi_field.section_overview_and_three_implications": {
        "context_id": "section_overview_and_three_implications",
        "validation_protocol": "paper0.ssb_psi_field.section_overview_and_three_implications",
        "canonical_statement": (
            "The source frames spontaneous symmetry breaking as the mechanism by which the Psi-field "
            "moves from symmetric laws to a structured vacuum with three source-claimed implications."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:ssb_overview_and_implications" for number in range(1272, 1276)
        ),
        "source_formulae": (
            "The Physics of Form: Spontaneous Symmetry Breaking and the Psi-Field",
            "V(|Psi|) = -mu^2 |Psi|^2 + lambda |Psi|^4",
            "sextic EFT stabilisation term +(gamma/Lambda^2)|Psi|^6",
            "local U(1) gauge symmetry breaking gives infoton mass m_A proportional to g v",
            "three implications: short-range informational force, Psi-Higgs prediction, L5 solitons",
        ),
        "test_protocols": ("preserve section overview and implication boundary",),
        "null_results": ("overview claims are not empirical validation evidence",),
        "variables": ("Psi", "mu", "lambda", "gamma", "Lambda", "m_A", "g", "v"),
        "validation_targets": (
            "preserve Mexican-hat potential source form",
            "preserve EFT-stabilised potential caveat",
            "preserve prediction-vs-evidence boundary for Psi-Higgs and solitons",
        ),
        "null_controls": (
            "implication language must not be promoted to observed particles or solitons",
        ),
    },
    "ssb_psi_field.popular_context_short_range_particle_self": {
        "context_id": "popular_context_short_range_particle_self",
        "validation_protocol": "paper0.ssb_psi_field.popular_context_short_range_particle_self",
        "canonical_statement": (
            "The source repeats the SSB mechanism in popular explanatory language: field roll-down, "
            "short-range force, new particle prediction, and soliton Self language."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:popular_ssb_explanation" for number in range(1276, 1282)
        ),
        "source_formulae": (
            "balanced pen / Mexican Hat analogy for spontaneous symmetry breaking",
            "infoton becomes heavy and informational force becomes short-range",
            "Psi-Higgs is a predicted new particle, not observed evidence",
            "solitons are described as stable knots or standing waves of the consciousness field",
        ),
        "test_protocols": ("preserve explanatory analogy without treating it as measurement",),
        "null_results": ("popular wording is source context only",),
        "variables": ("infoton", "Psi-Higgs", "soliton", "Self"),
        "validation_targets": (
            "preserve short-range force explanatory role",
            "preserve predicted-particle status",
            "preserve soliton Self source claim boundary",
        ),
        "null_controls": ("popular analogy must not satisfy mechanism derivation checks",),
    },
    "ssb_psi_field.predictive_coding_core_belief": {
        "context_id": "predictive_coding_core_belief",
        "validation_protocol": "paper0.ssb_psi_field.predictive_coding_core_belief",
        "canonical_statement": (
            "The source maps SSB into predictive-coding language: a symmetric maximum-entropy prior, "
            "a non-zero VEV as core belief, and soliton as embodied active-inference agent."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:predictive_coding_ssb" for number in range(1282, 1288)
        ),
        "source_formulae": (
            "Psi = 0 represents a flat maximum-entropy prior",
            "VEV v is framed as a core foundational belief",
            "soliton is a stable embodied active-inference agent",
        ),
        "test_protocols": ("preserve predictive-coding mapping as source mapping",),
        "null_results": ("predictive-coding analogy is not an independent derivation",),
        "variables": ("Psi", "v", "VEV", "soliton", "prior"),
        "validation_targets": (
            "preserve maximum-entropy prior mapping",
            "preserve VEV core-belief mapping",
            "preserve soliton active-inference mapping",
        ),
        "null_controls": (
            "metaphor-only predictive coding must not count as physical validation",
        ),
    },
    "ssb_psi_field.psi_s_coupling_integration": {
        "context_id": "psi_s_coupling_integration",
        "validation_protocol": "paper0.ssb_psi_field.psi_s_coupling_integration",
        "canonical_statement": (
            "The source states that SSB gives the H_int components their physical properties: potent "
            "Psi_s background, short-range interaction, and stable sigma_soliton handles."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:psi_s_coupling_integration" for number in range(1288, 1293)
        ),
        "source_formulae": (
            "H_int = -lambda * Psi_s * sigma",
            "before SSB background Psi-field VEV is zero",
            "after SSB Psi_s acquires non-zero VEV v",
            "Higgs mechanism gives infoton mass m_A and makes H_int short-range",
            "sigma_soliton is the stable collective state variable for an individual Self",
        ),
        "test_protocols": ("preserve Psi_s coupling integration boundaries",),
        "null_results": ("H_int integration language is not a measured coupling constant",),
        "variables": ("H_int", "lambda", "Psi_s", "sigma", "m_A", "sigma_soliton"),
        "validation_targets": (
            "preserve H_int source equation",
            "preserve VEV transition role",
            "preserve sigma_soliton claim boundary",
        ),
        "null_controls": ("missing sigma_soliton role must fail source accounting",),
    },
    "ssb_psi_field.mexican_hat_vacuum_selection": {
        "context_id": "mexican_hat_vacuum_selection",
        "validation_protocol": "paper0.ssb_psi_field.mexican_hat_vacuum_selection",
        "canonical_statement": (
            "The source introduces SSB from symmetric laws to specific realities and defines the Mexican-hat "
            "potential, degenerate vacua, VEV radius, and spontaneous vacuum choice."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:mexican_hat_vacuum_selection" for number in range(1293, 1302)
        ),
        "source_formulae": (
            "free-field potential V(|Psi|) = mu^2 |Psi|^2 lacks emergent structure",
            "Mexican-hat potential V(|Psi|) = -mu^2 |Psi|^2 + lambda |Psi|^4",
            "Psi = 0 is an unstable symmetric state",
            "true vacua form a circle with |Psi| = sqrt(mu^2/lambda)",
            "chosen vacuum example <Psi> = v breaks initial U(1) rotational symmetry",
        ),
        "test_protocols": ("preserve Mexican-hat vacuum selection equations",),
        "null_results": ("vacuum-selection formulae are not parameter measurements",),
        "variables": ("Psi", "mu", "lambda", "v", "U1"),
        "validation_targets": (
            "preserve unstable symmetric state",
            "preserve circle of true vacua",
            "preserve spontaneous choice of vacuum",
        ),
        "null_controls": (
            "single-minimum free-field potential must not satisfy SSB source boundary",
        ),
    },
    "ssb_psi_field.eft_sextic_stability_and_mass": {
        "context_id": "eft_sextic_stability_and_mass",
        "validation_protocol": "paper0.ssb_psi_field.eft_sextic_stability_and_mass",
        "canonical_statement": (
            "The source adds an EFT sextic term for vacuum stability, gives the stationary-point equation, "
            "positive-root VEV, and radial Psi-Higgs mass expression."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:eft_sextic_stability" for number in range(1302, 1317)
        ),
        "source_formulae": (
            "EFT in 3+1 dimensions requires potential bounded from below",
            "V(|Psi|) = -mu^2 |Psi|^2 + lambda |Psi|^4 + (gamma/Lambda^2)|Psi|^6",
            "x = |Psi|^2 and dV/dx = -mu^2 + 2 lambda x + (3 gamma/Lambda^2) x^2 = 0",
            "v = sqrt(x_*) from the positive quadratic root",
            "x_* = Lambda^2/(6 gamma) * (-2 lambda + sqrt(4 lambda^2 + 12 gamma mu^2/Lambda^2))",
            "m_h^2 = -2 mu^2 + 12 lambda v^2 + 30 gamma v^4/Lambda^2",
            "positive sextic term gamma > 0 stabilises the vacuum when lambda is driven negative",
        ),
        "test_protocols": ("preserve EFT sextic stability and mass-expression accounting",),
        "null_results": ("stability equation is not a measured Psi-Higgs mass",),
        "variables": ("Psi", "mu", "lambda", "gamma", "Lambda", "x", "v", "m_h"),
        "validation_targets": (
            "preserve bounded-from-below requirement",
            "preserve stationary-point equation",
            "preserve VEV root and radial mass expression",
        ),
        "null_controls": ("quartic-only potential must not satisfy EFT-stability boundary",),
    },
    "ssb_psi_field.global_goldstone_boundary": {
        "context_id": "global_goldstone_boundary",
        "validation_protocol": "paper0.ssb_psi_field.global_goldstone_boundary",
        "canonical_statement": (
            "The source states the global-U(1) counterfactual boundary: spontaneous breaking of a global "
            "symmetry yields one massive radial scalar and one massless Goldstone boson."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:global_goldstone_boundary" for number in range(1317, 1322)
        ),
        "source_formulae": (
            "global U(1): Psi -> exp(i alpha) Psi",
            "Goldstone theorem: broken continuous global symmetry yields massless spin-0 boson",
            "Psi(x) = v + eta(x) radial excitation",
            "m_eta^2 = 2 mu^2",
            "global U(1) breaking gives one massive scalar and one massless Goldstone boson",
        ),
        "test_protocols": ("preserve global-symmetry counterfactual boundary",),
        "null_results": ("global Goldstone result must not be confused with local gauge case",),
        "variables": ("Psi", "alpha", "eta", "m_eta", "Goldstone"),
        "validation_targets": (
            "preserve global-U1 condition",
            "preserve radial and angular excitation distinction",
            "preserve massless Goldstone boundary",
        ),
        "null_controls": (
            "local gauge Higgs mechanism must not satisfy global-Goldstone classifier",
        ),
    },
    "ssb_psi_field.local_higgs_architecture_implications": {
        "context_id": "local_higgs_architecture_implications",
        "validation_protocol": "paper0.ssb_psi_field.local_higgs_architecture_implications",
        "canonical_statement": (
            "The source switches to local U(1), where the Goldstone mode is absorbed by the infoton, "
            "yielding a massive vector, massive scalar, short-range force, Psi-Higgs prediction, and soliton Self claims."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:local_higgs_architecture_implications"
            for number in range(1322, 1333)
        ),
        "source_formulae": (
            "local gauge symmetry breaking absorbs the Goldstone boson into the gauge field",
            "A_mu is the infoton mediator of the informational force",
            "|D_mu Psi|^2 = |(partial_mu - i g A_mu) Psi|^2",
            "g^2 v^2 A_mu A^mu gives the gauge-field mass term",
            "m_A = sqrt(2) g v",
            "massive vector boson plus massive scalar eta(x) final particle spectrum",
            "range approximately hbar/(m_A c)",
            "Psi-Higgs remains a falsifiable prediction, not observed evidence",
            "L5 Selves are source-claimed solitons of the Psi-field",
        ),
        "test_protocols": ("preserve local-Higgs and architecture implication boundaries",),
        "null_results": ("architecture implications are not detection evidence",),
        "variables": ("A_mu", "D_mu", "Psi", "g", "v", "m_A", "eta", "L5"),
        "validation_targets": (
            "preserve local-vs-global distinction",
            "preserve infoton mass expression",
            "preserve short-range force and soliton implication boundaries",
        ),
        "null_controls": (
            "Psi-Higgs prediction must not be marked as observed particle evidence",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class SSBPsiFieldSpec:
    """SSB Psi-field spec promoted from Paper 0 records."""

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
class SSBPsiFieldSpecBundle:
    """SSB Psi-field specs plus source coverage summary."""

    specs: tuple[SSBPsiFieldSpec, ...]
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


def build_ssb_psi_field_specs(source_records: list[dict[str, Any]]) -> SSBPsiFieldSpecBundle:
    """Build source-covered SSB Psi-field specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    category_counts = Counter(
        str(record.get("canonical_category", "unknown")) for record in anchors
    )
    block_counts = Counter(str(record.get("block_type", "unknown")) for record in anchors)

    specs: list[SSBPsiFieldSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            SSBPsiFieldSpec(
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
        "title": "Paper 0 SSB Psi-Field Specs",
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
        "next_source_boundary": "P0R01333",
    }
    return SSBPsiFieldSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(ledger_path: Path = DEFAULT_LEDGER_PATH) -> SSBPsiFieldSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_ssb_psi_field_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: SSBPsiFieldSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 SSB Psi-Field Specs",
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
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    bundle: SSBPsiFieldSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artifacts for promoted specs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_ssb_psi_field_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_ssb_psi_field_validation_specs_report_{date_tag}.md"
    payload = {
        "summary": _json_ready(bundle.summary),
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
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
