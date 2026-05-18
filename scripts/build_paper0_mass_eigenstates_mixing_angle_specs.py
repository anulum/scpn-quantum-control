#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 mass eigenstates mixing-angle spec builder
"""Promote Paper 0 mass-eigenstates and mixing-angle records."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(1669, 1684))
CLAIM_BOUNDARY = "source-bounded mass-eigenstates mixing-angle bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "mass_eigenstates_mixing_angle.mass_eigenstate_rotation": {
        "context_id": "mass_eigenstate_rotation",
        "validation_protocol": "paper0.mass_eigenstates_mixing_angle.mass_eigenstate_rotation",
        "canonical_statement": (
            "The source defines physical h_SM and h_Psi mass eigenstates by rotating the bare scalar fields with a mixing angle theta."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:mass_eigenstate_rotation" for n in range(1669, 1675)
        ),
        "source_formulae": (
            "Mass Eigenstates and the Mixing Angle (theta)",
            "bare fields h_bare and h_Psi,bare are not the physical particles",
            "physical mass eigenstates are h_SM and h_Psi",
            "[h_SM, h_Psi]^T = [[cos theta, sin theta], [-sin theta, cos theta]] [h_bare, h_Psi,bare]^T",
            "tan(2 theta) = 2 lambda_mix v_h v_psi / (m_h_bare^2 - m_Psi_bare^2)",
        ),
        "test_protocols": ("preserve mass-eigenstate rotation boundary",),
        "null_results": ("mixing-angle formula is not measured scalar-sector evidence",),
        "variables": (
            "h_SM",
            "h_Psi",
            "h_bare",
            "h_Psi_bare",
            "theta",
            "lambda_mix",
            "v_h",
            "v_psi",
        ),
        "validation_targets": ("preserve orthogonal rotation", "preserve tan-two-theta relation"),
        "null_controls": ("rotation formalism must not be treated as measured Higgs mixing",),
    },
    "mass_eigenstates_mixing_angle.lhc_invisible_decay_bound": {
        "context_id": "lhc_invisible_decay_bound",
        "validation_protocol": "paper0.mass_eigenstates_mixing_angle.lhc_invisible_decay_bound",
        "canonical_statement": (
            "The source maps sin(theta) to Standard Model interaction suppression and cites an invisible-Higgs branching-ratio limit as a working bound."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:lhc_invisible_decay_bound" for n in range(1675, 1681)
        ),
        "source_formulae": (
            "LHC Constraints and Perturbative Bounds",
            "sin theta dictates universal suppression of Psi-Higgs interactions with SM fermions and gauge bosons",
            "h_SM hidden-sector fraction enables h_SM -> h_Psi h_Psi or h_SM -> infotons if kinematically allowed",
            "these decays would appear as invisible or undetected decays at the LHC",
            "ATLAS and CMS constrain BR_inv < 0.11 at 95 percent confidence level",
            "sin^2 theta lesssim 0.1 implies sin theta lesssim 0.31",
        ),
        "test_protocols": ("preserve LHC invisible-decay bound boundary",),
        "null_results": (
            "current invisible-branching bound is a constraint, not a Psi-sector detection",
        ),
        "variables": ("sin_theta", "BR_inv", "h_SM", "h_Psi", "infoton", "ATLAS", "CMS"),
        "validation_targets": (
            "preserve invisible-decay channels",
            "preserve working sin-theta bound",
        ),
        "null_controls": (
            "empirical upper limit must not be reported as observed Psi-sector signal",
        ),
    },
    "mass_eigenstates_mixing_angle.perturbative_target_boundary": {
        "context_id": "perturbative_target_boundary",
        "validation_protocol": "paper0.mass_eigenstates_mixing_angle.perturbative_target_boundary",
        "canonical_statement": (
            "The source frames perturbatively small lambda_mix and sin(theta) <= 0.31 as a working search target for falsification or discovery."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:perturbative_target_boundary" for n in range(1681, 1684)
        ),
        "source_formulae": (
            "lambda_mix must be perturbatively small, lambda_mix << 1",
            "smallness is source-framed as compatible with subtle shielded matter coupling",
            "specialized quasicritical biological substrates are named as macroscopic amplification contexts",
            "sin theta lesssim 0.31 is established as a working bound",
            "subsequent search strategies are anchored in a falsification-or-discovery parameter target",
        ),
        "test_protocols": ("preserve perturbative target boundary",),
        "null_results": ("working target is not a completed discovery or falsification",),
        "variables": ("lambda_mix", "sin_theta", "quasicritical_substrate", "parameter_space"),
        "validation_targets": (
            "preserve perturbative-smallness constraint",
            "preserve search-target framing",
        ),
        "null_controls": ("working-bound language must not imply model confirmation",),
    },
}


@dataclass(frozen=True, slots=True)
class MassEigenstatesMixingAngleSpec:
    """Mass-eigenstates mixing-angle spec promoted from Paper 0 records."""

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
class MassEigenstatesMixingAngleSpecBundle:
    """Mass-eigenstates mixing-angle specs plus source coverage summary."""

    specs: tuple[MassEigenstatesMixingAngleSpec, ...]
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


def build_mass_eigenstates_mixing_angle_specs(
    source_records: list[dict[str, Any]],
) -> MassEigenstatesMixingAngleSpecBundle:
    """Build source-covered mass-eigenstates mixing-angle specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    category_counts = Counter(
        str(record.get("canonical_category", "unknown")) for record in anchors
    )
    block_counts = Counter(str(record.get("block_type", "unknown")) for record in anchors)

    specs: list[MassEigenstatesMixingAngleSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            MassEigenstatesMixingAngleSpec(
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
        "title": "Paper 0 Mass Eigenstates Mixing Angle Specs",
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
        "next_source_boundary": "P0R01684",
    }
    return MassEigenstatesMixingAngleSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> MassEigenstatesMixingAngleSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_mass_eigenstates_mixing_angle_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: MassEigenstatesMixingAngleSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 Mass Eigenstates Mixing Angle Specs",
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
    bundle: MassEigenstatesMixingAngleSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_mass_eigenstates_mixing_angle_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_mass_eigenstates_mixing_angle_validation_specs_report_{date_tag}.md"
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
