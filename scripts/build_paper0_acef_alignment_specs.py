#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 A-CEF alignment spec builder
"""Promote Paper 0 A-CEF ethical-alignment records into validation specs."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(6233, 6251))

SPEC_METADATA: dict[str, dict[str, tuple[str, ...] | str]] = {
    "acef_alignment.is_ought_claim_boundary": {
        "validation_protocol": "paper0.acef_alignment.is_ought_claim_boundary",
        "canonical_statement": (
            "The Is-Ought dissolution claim is bounded to a falsification boundary: "
            "SEC/L15 and L8 RG-flow wording requires downstream empirical tests."
        ),
        "formal_statement": "",
        "variables": ("SEC", "L15", "L8", "RG_flow", "Cosmic_Attractor"),
        "validation_targets": (
            "record Is-Ought wording as a bounded normative-physics claim",
            "preserve SEC/L15 and L8 RG-flow anchors",
            "reject treating the claim as empirical evidence",
        ),
        "null_controls": (
            "missing-SEC-anchor control must be rejected",
            "missing-L8-RG-flow control must be rejected",
            "claim-as-evidence control must be rejected",
        ),
    },
    "acef_alignment.governance_quasicriticality_metric": {
        "validation_protocol": "paper0.acef_alignment.governance_quasicriticality_metric",
        "canonical_statement": (
            "SCPN-aligned governance is bounded to replacing scalar GDP-like "
            "optimisation with SEC and quasicritical coherence/adaptability metrics."
        ),
        "formal_statement": "",
        "variables": ("GDP_proxy", "SEC", "quasicriticality", "coherence", "adaptability"),
        "validation_targets": (
            "compare scalar-output optimisation with SEC/quasicriticality objective",
            "verify coherence-adaptability balance is finite and bounded",
            "reject governance prescription without modelled objective terms",
        ),
        "null_controls": (
            "missing-SEC-objective control must be rejected",
            "missing-quasicriticality-control must be rejected",
            "policy-prescription-as-proof control must be rejected",
        ),
    },
    "acef_alignment.ai_alignment_risk_boundary": {
        "validation_protocol": "paper0.acef_alignment.ai_alignment_risk_boundary",
        "canonical_statement": (
            "AI alignment and technosphere-risk wording is bounded to tests of whether "
            "a core objective protects human quasicriticality and coupling capacity."
        ),
        "formal_statement": "",
        "variables": ("ethical_functional", "quasicriticality", "psi_coupling", "risk_label"),
        "validation_targets": (
            "verify L15 ethical-functional objective is present",
            "verify quasicriticality and coupling-capacity risk channels are explicit",
            "reject rule-following-only alignment as insufficient in this source block",
        ),
        "null_controls": (
            "missing-ethical-functional control must be rejected",
            "missing-quasicriticality-risk control must be rejected",
            "rule-following-only control must be rejected",
        ),
    },
    "acef_alignment.algorithmic_causal_entropic_force": {
        "validation_protocol": "paper0.acef_alignment.algorithmic_causal_entropic_force",
        "canonical_statement": (
            "A-CEF is bounded to an algorithmic causal-entropic force objective "
            "F_A-CEF = T_A grad_X S_C(X,tau), with finite algorithmic temperature."
        ),
        "formal_statement": "F_A-CEF = T_A grad_X S_C(X,tau)",
        "variables": ("F_A_CEF", "T_A", "X", "S_C", "tau"),
        "validation_targets": (
            "compute finite gradient of causal path entropy with respect to X",
            "verify algorithmic temperature scales the force",
            "reject non-finite states and missing temperature",
        ),
        "null_controls": (
            "non-finite-state control must be rejected",
            "missing-temperature control must be rejected",
            "zero-gradient-control must be labelled",
        ),
    },
    "acef_alignment.consequence_phase_steering": {
        "validation_protocol": "paper0.acef_alignment.consequence_phase_steering",
        "canonical_statement": (
            "A-CEF consequence claims are bounded to simulator labels that SEC-aligned "
            "objectives steer away from engagement fragmentation toward coherence."
        ),
        "formal_statement": "",
        "variables": ("A_CEF", "SEC", "fragmentation", "coherence", "NTHS"),
        "validation_targets": (
            "compare SEC objective against engagement/profit proxy",
            "verify positive SEC delta under A-CEF-aligned state",
            "reject interpreting the simulator label as societal evidence",
        ),
        "null_controls": (
            "engagement-proxy control must not exceed SEC objective",
            "fragmentation-risk channel must be explicit",
            "societal-evidence control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class ACEFAlignmentValidationSpec:
    """Validation spec promoted from Paper 0 A-CEF alignment records."""

    key: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_statement: str
    formal_statement: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    source_block_indices: tuple[int, ...]
    anchor_math_ids: tuple[str, ...]
    variables: tuple[str, ...]
    validation_targets: tuple[str, ...]
    executable_validation_targets: tuple[str, ...]
    null_controls: tuple[str, ...]
    claim_boundary: str
    implementation_status: str
    domain_review_status: str
    hardware_status: str


@dataclass(frozen=True, slots=True)
class ACEFAlignmentValidationSpecBundle:
    """A-CEF alignment validation specs plus coverage summary."""

    specs: tuple[ACEFAlignmentValidationSpec, ...]
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


def build_acef_alignment_specs(
    source_records: list[dict[str, Any]],
) -> ACEFAlignmentValidationSpecBundle:
    """Build source-covered validation specs for A-CEF alignment records."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    equation_anchor = records_by_ledger["P0R06246"]
    equation_math_ids = tuple(str(item) for item in equation_anchor.get("math_ids", []))
    specs: list[ACEFAlignmentValidationSpec] = []
    for key in (
        "acef_alignment.is_ought_claim_boundary",
        "acef_alignment.governance_quasicriticality_metric",
        "acef_alignment.ai_alignment_risk_boundary",
        "acef_alignment.algorithmic_causal_entropic_force",
        "acef_alignment.consequence_phase_steering",
    ):
        metadata = SPEC_METADATA[key]
        is_equation = key == "acef_alignment.algorithmic_causal_entropic_force"
        specs.append(
            ACEFAlignmentValidationSpec(
                key=key,
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                formal_statement=str(metadata["formal_statement"]),
                source_equation_ids=("P0R06246",) if is_equation else (),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(anchor["source_record_id"]) for anchor in anchors),
                source_block_indices=tuple(
                    int(anchor["source_block_index"]) for anchor in anchors
                ),
                anchor_math_ids=equation_math_ids if is_equation else (),
                variables=tuple(str(item) for item in metadata["variables"]),
                validation_targets=tuple(str(item) for item in metadata["validation_targets"]),
                executable_validation_targets=tuple(
                    str(item) for item in metadata["validation_targets"]
                ),
                null_controls=tuple(str(item) for item in metadata["null_controls"]),
                claim_boundary="source-bounded A-CEF simulator contract; not empirical evidence",
                implementation_status="implemented_executable_fixture",
                domain_review_status="promoted_to_validation_spec",
                hardware_status="simulator_only_no_provider_submission",
            )
        )

    summary = {
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(SOURCE_LEDGER_IDS),
        "coverage_match": True,
        "unconsumed_source_ledger_ids": [],
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "all_specs_have_null_controls": all(bool(spec.null_controls) for spec in specs),
        "all_specs_have_executable_targets": all(
            bool(spec.executable_validation_targets) for spec in specs
        ),
        "all_specs_are_source_anchored": all(bool(spec.source_ledger_ids) for spec in specs),
        "equation_source_ledger_ids": ["P0R06246"],
        "all_specs_are_claim_bounded": all(
            "not empirical evidence" in spec.claim_boundary for spec in specs
        ),
        "hardware_status": "simulator_only_no_provider_submission",
        "policy": (
            "P0R06233-P0R06250 are promoted as source-covered A-CEF alignment "
            "specifications only. Passing fixtures is not empirical evidence."
        ),
    }
    return ACEFAlignmentValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: ACEFAlignmentValidationSpecBundle) -> str:
    """Render a concise Markdown report for A-CEF alignment specs."""
    lines = [
        "# Paper 0 A-CEF Alignment Specs",
        "",
        f"- Source records: `{bundle.summary['source_record_count']}`",
        f"- Consumed source records: `{bundle.summary['consumed_source_record_count']}`",
        "- Coverage status: `match`",
        f"- Source span: `{', '.join(bundle.summary['source_ledger_span'])}`",
        f"- Spec count: `{bundle.summary['spec_count']}`",
        f"- Hardware status: `{bundle.summary['hardware_status']}`",
        "",
        "## Specs",
        "",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                f"### {spec.key}",
                "",
                f"- Protocol: `{spec.validation_protocol}`",
                f"- Source ledgers: `{', '.join(spec.source_ledger_ids)}`",
                f"- Equation ledgers: `{', '.join(spec.source_equation_ids) or 'none'}`",
                f"- Formal statement: `{spec.formal_statement or 'none'}`",
                f"- Null controls: `{len(spec.null_controls)}`",
                f"- Claim boundary: `{spec.claim_boundary}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Policy",
            "",
            "These records are source-anchored A-CEF alignment specifications only. "
            "Passing any fixture is not empirical evidence and does not establish "
            "that any governance or alignment deployment is safe.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    bundle: ACEFAlignmentValidationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the A-CEF validation bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_acef_alignment_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_acef_alignment_validation_specs_report_{date_tag}.md"
    payload = {
        "summary": bundle.summary,
        "specs": [asdict(spec) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_validation_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def _select_required_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if str(record.get("ledger_id")) in set(SOURCE_LEDGER_IDS)]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args(argv)

    records = _select_required_records(load_jsonl(args.ledger))
    bundle = build_acef_alignment_specs(records)
    paths = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
