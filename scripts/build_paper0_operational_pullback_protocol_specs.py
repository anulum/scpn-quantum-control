#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 operational pullback protocol spec builder
"""Promote Paper 0 operational pullback protocol records."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(1242, 1272))
CLAIM_BOUNDARY = "source-bounded operational pullback protocol; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "operational_pullback_protocol.section_and_protocol_boundary": {
        "context_id": "section_and_protocol_boundary",
        "validation_protocol": "paper0.operational_pullback_protocol.section_and_protocol_boundary",
        "canonical_statement": (
            "The source opens the SSB section and declares an operational pullback protocol "
            "for relating the abstract Fisher Information Metric to measurable quantities."
        ),
        "source_equation_ids": (
            "P0R01242:ssb_section_heading",
            "P0R01243:complete_operational_pullback_protocol",
            "P0R01244:operational_pullback_protocol_revision_11",
            "P0R01245:fim_to_measurable_bridge_statement",
        ),
        "source_formulae": (
            "2.3 The Physics of Form: Spontaneous Symmetry Breaking",
            "Complete Operational Pullback Protocol",
            "Operational Pullback Protocol Revision 11.00",
            "formal bridge between abstract FIM and measurable physical quantities",
        ),
        "test_protocols": ("preserve SSB/pullback section boundary",),
        "null_results": ("protocol declaration is not a completed measurement",),
        "variables": ("FIM", "SSB", "infoton", "Psi"),
        "validation_targets": (
            "preserve protocol title",
            "preserve FIM-to-measurable bridge role",
            "preserve source-only status",
        ),
        "null_controls": ("protocol heading alone must not count as empirical validation",),
    },
    "operational_pullback_protocol.statistical_bundle_and_fim": {
        "context_id": "statistical_bundle_and_fim",
        "validation_protocol": "paper0.operational_pullback_protocol.statistical_bundle_and_fim",
        "canonical_statement": (
            "The source defines a statistical bundle, local model section, and Fisher "
            "Information Metric on the fibre."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:statistical_bundle_and_fim" for number in range(1246, 1251)
        ),
        "source_formulae": (
            "pi: Theta -> M statistical fibre bundle over spacetime M",
            "Section theta: M -> Theta indexes probability models p(y|x, theta(x))",
            "I_ij(theta) = E_p(y|x,theta)[partial_i log p dot partial_j log p]",
        ),
        "test_protocols": ("preserve statistical bundle and FIM definitions",),
        "null_results": ("source FIM definition is not an estimated dataset",),
        "variables": ("Theta", "M", "theta", "p", "I_ij"),
        "validation_targets": (
            "preserve statistical fibre bundle",
            "preserve model-indexing section",
            "preserve FIM expectation formula",
        ),
        "null_controls": ("FIM without model section must not satisfy pullback protocol",),
    },
    "operational_pullback_protocol.spacetime_pullback_and_normalisation": {
        "context_id": "spacetime_pullback_and_normalisation",
        "validation_protocol": "paper0.operational_pullback_protocol.spacetime_pullback_and_normalisation",
        "canonical_statement": (
            "The source pulls the FIM to spacetime and normalises its inverse for gauge kinetics "
            "with an information energy scale."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:spacetime_pullback_and_normalisation" for number in range(1251, 1256)
        ),
        "source_formulae": (
            "g_F_mu_nu(x) = (partial_mu theta^i(x)) I_ij(theta(x)) (partial_nu theta^j(x))",
            "g_tilde_F^mu_nu(x) = Lambda_I^(-2) (g_F^(-1))^mu_nu",
            "Lambda_I is the characteristic information energy scale",
        ),
        "test_protocols": ("preserve FIM pullback and normalisation equations",),
        "null_results": ("pullback formula is not a calibrated spacetime metric measurement",),
        "variables": ("g_F", "g_tilde_F", "theta", "I_ij", "Lambda_I"),
        "validation_targets": (
            "preserve spacetime pullback equation",
            "preserve inverse-metric normalisation",
            "preserve Lambda_I scaling boundary",
        ),
        "null_controls": ("normalisation omitted control must fail source accounting",),
    },
    "operational_pullback_protocol.observable_sections_and_l4_l5_case": {
        "context_id": "observable_sections_and_l4_l5_case",
        "validation_protocol": "paper0.operational_pullback_protocol.observable_sections_and_l4_l5_case",
        "canonical_statement": (
            "The source lists observable sections for L5 and L11 and gives an L4-to-L5 neural "
            "coding-efficiency case with NV-centre prediction language."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:observable_sections_and_l4_l5_case" for number in range(1256, 1264)
        ),
        "source_formulae": (
            "Observable Sections examples",
            "L5 organismal theta parameterises ensemble posteriors for neural latents",
            "L11 noosphere theta parameterises population-level phase densities",
            "L4 to L5 neural coding efficiency case study",
            "maximise coding efficiency equivalent to maximise det(I(theta))",
            "system adapts synaptic weights to minimise prediction error and embody the FIM",
            "Psi-field coupling strength is strongest where information density is maximised",
            "NV-centre probes show signal modulation correlated with local coding-efficiency increases",
        ),
        "test_protocols": ("preserve observable-section and L4-L5 validation-target boundaries",),
        "null_results": ("NV-centre prediction is not experimental evidence",),
        "variables": ("L5", "L11", "theta", "det_I", "NV_center", "Psi"),
        "validation_targets": (
            "preserve L5 and L11 observable sections",
            "preserve coding-efficiency objective",
            "preserve NV-centre prediction as target only",
        ),
        "null_controls": ("prediction wording must not be promoted to observed modulation",),
    },
    "operational_pullback_protocol.full_covariance_fim_strategy": {
        "context_id": "full_covariance_fim_strategy",
        "validation_protocol": "paper0.operational_pullback_protocol.full_covariance_fim_strategy",
        "canonical_statement": (
            "The source states that FIM computation must use the full covariance matrix, including "
            "mean-gradient and covariance-gradient contributions."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:full_covariance_fim_strategy" for number in range(1264, 1268)
        ),
        "source_formulae": (
            "Computational Strategy for FIM",
            "must use full covariance matrix Sigma(theta)",
            "I(theta) = (nabla_theta mu(theta))^T Sigma(theta)^(-1) (nabla_theta mu(theta)) + 0.5 Tr[(nabla_theta Sigma(theta)) Sigma(theta)^(-1) (nabla_theta Sigma(theta)) Sigma(theta)^(-1)]",
            "Constraints",
        ),
        "test_protocols": ("preserve full-covariance FIM computation requirement",),
        "null_results": ("diagonal or mean-only FIM is insufficient for this source protocol",),
        "variables": ("I", "theta", "mu", "Sigma", "Tr"),
        "validation_targets": (
            "preserve full covariance matrix requirement",
            "preserve mean-gradient term",
            "preserve covariance-gradient trace term",
        ),
        "null_controls": ("diagonal covariance shortcut must be rejected for this protocol",),
    },
    "operational_pullback_protocol.eft_lorentz_locality_constraints": {
        "context_id": "eft_lorentz_locality_constraints",
        "validation_protocol": "paper0.operational_pullback_protocol.eft_lorentz_locality_constraints",
        "canonical_statement": (
            "The source constrains pullback dynamics as EFT-level, Lorentz-invariance-preserving, "
            "and locally/causally dependent on measurable observables."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:eft_lorentz_locality_constraints" for number in range(1268, 1272)
        ),
        "source_formulae": (
            "EFT Interpretation: FIM-based dynamics are effective field theory",
            "Lorentz Invariance: Fundamental Lorentz invariance preserved",
            "Locality/Causality: pullback map pi depends only on locally measurable observables",
        ),
        "test_protocols": ("preserve EFT, Lorentz, and locality constraints",),
        "null_results": ("constraint list is not a Lorentz-invariance proof",),
        "variables": ("EFT", "Lorentz", "pi", "local_observables"),
        "validation_targets": (
            "preserve EFT interpretation",
            "preserve Lorentz-invariance caveat",
            "preserve local-observable dependency",
        ),
        "null_controls": ("nonlocal pullback dependency must fail protocol boundary",),
    },
}


@dataclass(frozen=True, slots=True)
class OperationalPullbackProtocolSpec:
    """Operational pullback protocol spec promoted from Paper 0 records."""

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
class OperationalPullbackProtocolSpecBundle:
    """Operational pullback protocol specs plus source coverage summary."""

    specs: tuple[OperationalPullbackProtocolSpec, ...]
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


def build_operational_pullback_protocol_specs(
    source_records: list[dict[str, Any]],
) -> OperationalPullbackProtocolSpecBundle:
    """Build source-covered operational pullback protocol specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    category_counts = Counter(
        str(record.get("canonical_category", "unknown")) for record in anchors
    )
    block_counts = Counter(str(record.get("block_type", "unknown")) for record in anchors)

    specs: list[OperationalPullbackProtocolSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            OperationalPullbackProtocolSpec(
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
        "title": "Paper 0 Operational Pullback Protocol Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed
        ],
        "spec_count": len(specs),
        "category_counts": dict(sorted(category_counts.items())),
        "block_type_counts": dict(sorted(block_counts.items())),
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R01272",
        "spec_keys": [spec.key for spec in specs],
    }
    return OperationalPullbackProtocolSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> OperationalPullbackProtocolSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = load_jsonl(ledger_path)
    selected = [record for record in records if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS]
    return build_operational_pullback_protocol_specs(selected)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: OperationalPullbackProtocolSpecBundle) -> str:
    """Render a Markdown report for the promoted specs."""
    lines = [
        "# Paper 0 Operational Pullback Protocol Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Category counts: {bundle.summary['category_counts']}",
        f"- Block-type counts: {bundle.summary['block_type_counts']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        f"- Next source boundary: {bundle.summary['next_source_boundary']}",
        "",
        "## Promoted Specs",
    ]
    for spec in bundle.specs:
        lines.extend(
            [f"### {spec.key}", "", spec.canonical_statement, "", "Formulae / source labels:"]
        )
        for formula in spec.source_formulae:
            lines.append(f"- {formula}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    bundle: OperationalPullbackProtocolSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write spec JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_operational_pullback_protocol_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_operational_pullback_protocol_validation_specs_report_{date_tag}.md"
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
    parser.add_argument("--date-tag", default="2026-05-17")
    args = parser.parse_args()

    bundle = build_from_ledger(args.ledger)
    outputs = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    print(outputs["json"])
    print(outputs["report"])


if __name__ == "__main__":
    main()
