#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 grand-synthesis spec builder
"""Promote Paper 0 Grand Synthesis records into validation specs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(6223, 6233))

SPEC_METADATA: dict[str, dict[str, tuple[str, ...] | str]] = {
    "grand_synthesis.anulum_claim_boundary": {
        "validation_protocol": "paper0.grand_synthesis.anulum_claim_boundary",
        "canonical_statement": (
            "The Anulum synthesis is bounded to a cross-domain claim map linking "
            "physics, biology, metaphysics, consciousness, geometry, and SEC."
        ),
        "variables": ("physics", "biology", "metaphysics", "consciousness", "geometry", "SEC"),
        "validation_targets": (
            "record synthesis as a bounded cross-domain claim rather than empirical proof",
            "preserve Consciousness, Geometry, and Teleology/SEC as source axioms",
            "require downstream validation targets before any empirical conclusion",
        ),
        "null_controls": (
            "missing-axiom-domain control must be rejected",
            "unsupported-empirical-evidence control must be rejected",
            "source-span truncation control must be rejected",
        ),
    },
    "grand_synthesis.architecture_mechanism_map": {
        "validation_protocol": "paper0.grand_synthesis.architecture_mechanism_map",
        "canonical_statement": (
            "The structure, dynamics, experience, guidance, metaphysical stance, "
            "and free-will entries are bounded to a mechanism map with explicit inputs and outputs."
        ),
        "variables": ("torus", "L13", "SSB", "UPDE", "HPC", "L5", "L15", "PELA", "CEF"),
        "validation_targets": (
            "map torus emergence to L13 and SSB source terms",
            "map UPDE/HPC dynamics to quasicritical information processing",
            "map L15 guidance to PELA and CEF without treating free-will wording as evidence",
        ),
        "null_controls": (
            "missing-structure-channel control must be rejected",
            "missing-dynamics-channel control must be rejected",
            "unsupported-free-will-evidence control must be rejected",
        ),
    },
    "grand_synthesis.nths_phase_test": {
        "validation_protocol": "paper0.grand_synthesis.nths_phase_test",
        "canonical_statement": (
            "The NTHS phase-test claim is bounded to a simulator contrast between "
            "collective free-energy minimisation and engagement/surprise maximisation."
        ),
        "variables": ("agent_policy", "adaptive_Jij", "free_energy", "engagement", "SEC"),
        "validation_targets": (
            "compare adaptive Jij under coherence and engagement/surprise policy regimes",
            "label spin-glass risk from finite signed-coupling frustration",
            "label ferromagnetic-like consensus from positive finite coupling and SEC proxy",
        ),
        "null_controls": (
            "missing-policy-regime control must be rejected",
            "missing-adaptive-coupling control must be rejected",
            "unsupported-societal-evidence control must be rejected",
        ),
    },
    "grand_synthesis.figure_caption_boundary": {
        "validation_protocol": "paper0.grand_synthesis.figure_caption_boundary",
        "canonical_statement": (
            "The figure-caption claims are bounded to media context for the NTHS "
            "phase-test contrast: rugged spin-glass landscape versus smooth consensus basin."
        ),
        "variables": ("IMG0150", "spin_glass_landscape", "ferromagnetic_basin", "SEC_policy"),
        "validation_targets": (
            "link IMG0150 to NTHS phase-test context only",
            "record rugged landscape and smooth basin as caption-level expected outcomes",
            "prevent caption text from being treated as independent evidence",
        ),
        "null_controls": (
            "missing-image-link control must be rejected",
            "caption-as-evidence control must be rejected",
            "missing-right-panel-SEC-policy control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class GrandSynthesisValidationSpec:
    """Validation spec promoted from Paper 0 Grand Synthesis records."""

    key: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_statement: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    source_block_indices: tuple[int, ...]
    source_image_ids: tuple[str, ...]
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
class GrandSynthesisValidationSpecBundle:
    """Grand Synthesis validation specs plus coverage summary."""

    specs: tuple[GrandSynthesisValidationSpec, ...]
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


def build_grand_synthesis_specs(
    source_records: list[dict[str, Any]],
) -> GrandSynthesisValidationSpecBundle:
    """Build source-covered validation specs for Grand Synthesis records."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    required_ids = set(SOURCE_LEDGER_IDS)
    missing = sorted(required_ids - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    anchor_math_ids = tuple(
        sorted({str(math_id) for anchor in anchors for math_id in anchor.get("math_ids", [])})
    )
    source_image_ids = tuple(
        sorted({str(image_id) for anchor in anchors for image_id in anchor.get("image_ids", [])})
    )
    specs: list[GrandSynthesisValidationSpec] = []
    for key in (
        "grand_synthesis.anulum_claim_boundary",
        "grand_synthesis.architecture_mechanism_map",
        "grand_synthesis.nths_phase_test",
        "grand_synthesis.figure_caption_boundary",
    ):
        metadata = SPEC_METADATA[key]
        specs.append(
            GrandSynthesisValidationSpec(
                key=key,
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=(),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(anchor["source_record_id"]) for anchor in anchors),
                source_block_indices=tuple(
                    int(anchor["source_block_index"]) for anchor in anchors
                ),
                source_image_ids=source_image_ids
                if key.endswith("figure_caption_boundary")
                else (),
                anchor_math_ids=anchor_math_ids,
                variables=tuple(str(item) for item in metadata["variables"]),
                validation_targets=tuple(str(item) for item in metadata["validation_targets"]),
                executable_validation_targets=tuple(
                    str(item) for item in metadata["validation_targets"]
                ),
                null_controls=tuple(str(item) for item in metadata["null_controls"]),
                claim_boundary="source-bounded simulator contract; not empirical evidence",
                implementation_status="implemented_executable_fixture",
                domain_review_status="promoted_to_validation_spec",
                hardware_status="simulator_only_no_provider_submission",
            )
        )

    summary = {
        "source_record_count": len(required_ids),
        "consumed_source_record_count": len(required_ids),
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
        "all_specs_reference_complete_source_span": all(
            spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in specs
        ),
        "all_specs_avoid_invented_equation_ids": all(
            not spec.source_equation_ids and not spec.anchor_math_ids for spec in specs
        ),
        "all_specs_are_claim_bounded": all(
            "not empirical evidence" in spec.claim_boundary for spec in specs
        ),
        "hardware_status": "simulator_only_no_provider_submission",
        "policy": (
            "P0R06223-P0R06232 are promoted as source-covered Grand Synthesis "
            "and NTHS phase-test specifications only. Passing fixtures is not empirical evidence."
        ),
    }
    return GrandSynthesisValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: GrandSynthesisValidationSpecBundle) -> str:
    """Render a concise Markdown report for Grand Synthesis specs."""
    lines = [
        "# Paper 0 Grand Synthesis Specs",
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
                f"- Source images: `{', '.join(spec.source_image_ids) or 'none'}`",
                f"- Null controls: `{len(spec.null_controls)}`",
                f"- Executable targets: `{len(spec.executable_validation_targets)}`",
                f"- Claim boundary: `{spec.claim_boundary}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Policy",
            "",
            "These records are source-anchored Grand Synthesis and NTHS phase-test "
            "specifications only. Passing any fixture is not empirical evidence and "
            "does not establish that any societal phase transition has occurred.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    bundle: GrandSynthesisValidationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the validation-spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_grand_synthesis_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_grand_synthesis_validation_specs_report_{date_tag}.md"
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
    bundle = build_grand_synthesis_specs(records)
    paths = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
