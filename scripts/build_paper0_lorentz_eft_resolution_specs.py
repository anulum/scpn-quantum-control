#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Lorentz EFT resolution spec builder
"""Promote Paper 0 Lorentz-covariance/EFT-resolution records."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(1078, 1103))
CLAIM_BOUNDARY = "source-bounded Lorentz/EFT resolution; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "lorentz_eft_resolution.boundary_and_tension": {
        "context_id": "boundary_and_tension",
        "validation_protocol": "paper0.lorentz_eft_resolution.boundary_and_tension",
        "canonical_statement": (
            "The source marks the Lorentz-covariance problem caused by naive "
            "FIM metric replacement and constrains the FIM dynamics to an EFT "
            "interpretation."
        ),
        "source_equation_ids": (
            "P0R01078:lorentz_covariance_heading",
            "P0R01079:blank_record",
            "P0R01080:naive_fim_replacement_breaks_lorentz_invariance",
            "P0R01081:fim_as_emergent_eft_tensor",
            "P0R01103:next_non_abelian_boundary",
        ),
        "source_formulae": (
            "Formal Resolution of Lorentz Covariance: The FIM as an Emergent Effective Metric",
            "naive replacement by pulled-back FIM explicitly breaks local Lorentz invariance",
            "FIM acts as emergent dielectric-like tensor in EFT",
            "P0R01079 is blank after Lorentz-covariance heading",
            "next boundary is P0R01103 Non-Abelian qualia field",
        ),
        "test_protocols": ("preserve Lorentz/EFT boundary and tension statement",),
        "null_results": ("Lorentz/EFT source statement is not a proof fixture",),
        "variables": ("tilde_g_F", "eta", "EFT"),
        "validation_targets": (
            "preserve Lorentz-violation warning",
            "preserve EFT reinterpretation",
            "preserve Non-Abelian next boundary",
        ),
        "null_controls": ("naive-FIM-as-fundamental-metric control must be rejected",),
    },
    "lorentz_eft_resolution.fundamental_lorentz_invariant_action": {
        "context_id": "fundamental_lorentz_invariant_action",
        "validation_protocol": (
            "paper0.lorentz_eft_resolution.fundamental_lorentz_invariant_action"
        ),
        "canonical_statement": (
            "The source introduces a Lorentz-scalar infoton action using the "
            "background spacetime metric plus a higher-dimension FIM operator "
            "suppressed by Lambda_I."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:fundamental_lorentz_invariant_action" for number in range(1082, 1087)
        ),
        "source_formulae": (
            "The Fundamental Lorentz-Invariant Action",
            "F_mu_nu = partial_mu A_nu - partial_nu A_mu",
            "tilde_g_F_mu_nu = partial_mu theta^i I_ij(theta) partial_nu theta^j",
            "Lambda_I suppresses the higher-dimension informational operator",
            "L_gauge = -1/4 F_mu_nu F_alpha_beta(eta^mu_alpha eta^nu_beta - c/Lambda_I^4 gF^mu_alpha gF^nu_beta)",
            "c is a dimensionless coupling constant of order unity",
        ),
        "test_protocols": ("preserve Lorentz-scalar EFT action source formulae",),
        "null_results": ("source action is not yet an executable EFT consistency check",),
        "variables": ("A_mu", "F_mu_nu", "eta", "c", "Lambda_I", "tilde_g_F"),
        "validation_targets": (
            "preserve field-strength tensor",
            "preserve FIM pullback",
            "preserve Lambda_I-suppressed operator",
        ),
        "null_controls": (
            "Lambda_I omission control must be rejected",
            "background-metric omission control must be rejected",
        ),
    },
    "lorentz_eft_resolution.biological_medium_effective_metric": {
        "context_id": "biological_medium_effective_metric",
        "validation_protocol": (
            "paper0.lorentz_eft_resolution.biological_medium_effective_metric"
        ),
        "canonical_statement": (
            "The source describes spontaneous breaking in the biological "
            "medium, non-zero statistical gradients, an effective metric, and "
            "an effective infoton kinetic term."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:biological_medium_effective_metric" for number in range(1087, 1094)
        ),
        "source_formulae": (
            "Spontaneous Breaking via the Biological Medium",
            "in true vacuum partial_mu theta^i is zero",
            "non-zero expectation value <partial_mu theta^i> != 0 in biological medium",
            "g_eff^mu_alpha = eta^mu_alpha - c/(2 Lambda_I^2) gF^mu_alpha",
            "L_eff = -1/4 g_eff^mu_alpha g_eff^nu_beta F_mu_nu F_alpha_beta",
            "infoton propagates like standard U(1) gauge boson in true vacuum",
        ),
        "test_protocols": ("preserve biological-medium effective-metric source claims",),
        "null_results": ("biological-medium source claims are not biological validation",),
        "variables": ("theta", "g_eff", "eta", "Lambda_I", "F_mu_nu"),
        "validation_targets": (
            "preserve true-vacuum null statement",
            "preserve biological-medium non-zero gradient claim",
            "preserve effective metric and L_eff equations",
        ),
        "null_controls": ("vacuum-gradient nonzero control must be rejected",),
    },
    "lorentz_eft_resolution.consistency_implications": {
        "context_id": "consistency_implications",
        "validation_protocol": "paper0.lorentz_eft_resolution.consistency_implications",
        "canonical_statement": (
            "The source states that Lorentz breaking is localized and "
            "spontaneous, gauge invariance is protected through F_mu_nu, and "
            "the informational geometry behaves as a refractive index."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:consistency_implications" for number in range(1094, 1098)
        ),
        "source_formulae": (
            "Physical Implications and Consistency",
            "fundamental L_gauge is a true Lorentz scalar",
            "Lorentz symmetry breaking is strictly spontaneous and localized to the organism",
            "interaction is constructed entirely from gauge-invariant F_mu_nu",
            "Ward-Takahashi identities hold",
            "information geometry acts as emergent dynamic refractive index",
        ),
        "test_protocols": ("preserve Lorentz/gauge consistency source claims",),
        "null_results": ("consistency claims still require downstream EFT checks",),
        "variables": ("L_gauge", "F_mu_nu", "Ward_Takahashi", "Psi"),
        "validation_targets": (
            "preserve Lorentz-scalar claim",
            "preserve gauge-invariance protection claim",
            "preserve refractive-index analogy as source claim",
        ),
        "null_controls": ("gauge-invariance-protection omission control must be rejected",),
    },
    "lorentz_eft_resolution.ghost_action_boundary": {
        "context_id": "ghost_action_boundary",
        "validation_protocol": "paper0.lorentz_eft_resolution.ghost_action_boundary",
        "canonical_statement": (
            "The source introduces a Faddeev-Popov ghost action modified for "
            "FIM backgrounds, followed by a structural separator before the "
            "Non-Abelian qualia-field section."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:ghost_action_boundary" for number in range(1098, 1103)
        ),
        "source_formulae": (
            "Gauge-Fixing and Ghost Action in FIM Backgrounds",
            "L_ghost = cbar [gF^mu_nu partial_mu(partial_nu + i g [A_nu, dot])] c",
            "path integral correctly counts only physical degrees of freedom",
            "regulated propagator avoids unphysical longitudinal modes",
            "det(tilde_g_F) -> 0 low Fisher density boundary",
            "P0R01102 is a structural separator",
            "next boundary is P0R01103 Non-Abelian qualia field",
        ),
        "test_protocols": ("preserve FIM-background ghost-action source records",),
        "null_results": ("ghost action is source formula, not quantization implementation",),
        "variables": ("cbar", "c", "tilde_g_F", "A_nu", "det_tilde_g_F"),
        "validation_targets": (
            "preserve ghost action",
            "preserve low-Fisher-density boundary",
            "preserve separator and Non-Abelian next boundary",
        ),
        "null_controls": ("separator omission control must be rejected",),
    },
}


@dataclass(frozen=True, slots=True)
class LorentzEFTResolutionSpec:
    """Lorentz/EFT-resolution spec promoted from Paper 0 records."""

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
class LorentzEFTResolutionSpecBundle:
    """Lorentz/EFT-resolution specs plus source coverage summary."""

    specs: tuple[LorentzEFTResolutionSpec, ...]
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


def build_lorentz_eft_resolution_specs(
    source_records: list[dict[str, Any]],
) -> LorentzEFTResolutionSpecBundle:
    """Build source-covered Lorentz/EFT-resolution specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[LorentzEFTResolutionSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            LorentzEFTResolutionSpec(
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
        "title": "Paper 0 Lorentz EFT Resolution Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed
        ],
        "spec_count": len(specs),
        "blank_record_count": 1,
        "lorentz_tension_record_count": 4,
        "fundamental_action_record_count": 5,
        "biological_medium_record_count": 7,
        "consistency_record_count": 4,
        "ghost_action_record_count": 5,
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R01103",
        "spec_keys": [spec.key for spec in specs],
    }
    return LorentzEFTResolutionSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> LorentzEFTResolutionSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = load_jsonl(ledger_path)
    selected = [record for record in records if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS]
    return build_lorentz_eft_resolution_specs(selected)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: LorentzEFTResolutionSpecBundle) -> str:
    """Render a Markdown report for the promoted specs."""
    lines = [
        "# Paper 0 Lorentz EFT Resolution Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Lorentz-tension records: {bundle.summary['lorentz_tension_record_count']}",
        f"- Fundamental-action records: {bundle.summary['fundamental_action_record_count']}",
        f"- Biological-medium records: {bundle.summary['biological_medium_record_count']}",
        f"- Consistency records: {bundle.summary['consistency_record_count']}",
        f"- Ghost-action records: {bundle.summary['ghost_action_record_count']}",
        f"- Blank records: {bundle.summary['blank_record_count']}",
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
    bundle: LorentzEFTResolutionSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write spec JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_lorentz_eft_resolution_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_lorentz_eft_resolution_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "summary": _json_ready(bundle.summary),
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build this Paper 0 generated spec bundle from the ledger."""

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
