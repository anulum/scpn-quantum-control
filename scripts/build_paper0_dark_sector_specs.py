#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 dark-sector spec builder
"""Promote Paper 0 dark-energy and psi-DM records into specs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(6311, 6324))
SOURCE_FORMULA_ID = "P0R06319:L_geometric"
SOURCE_FORMULA = "L_Geometric proportional to -xi R Psi* Psi"

SPEC_METADATA: dict[str, dict[str, tuple[str, ...] | str]] = {
    "dark_sector.mmc_operator_information_preservation": {
        "validation_protocol": "paper0.dark_sector.mmc_operator_information_preservation",
        "canonical_statement": (
            "MMC operator wording is bounded to conformal rescaling between aeons, "
            "Ethical Functional conservation, information preservation, and entropy reset."
        ),
        "variables": ("O_MMC", "conformal_rescaling", "E", "L15", "entropy_reset"),
        "validation_targets": (
            "score conformal-rescaling channel",
            "require Ethical Functional conservation across the boundary",
            "require entropy-reset and information-preservation labels",
        ),
        "null_controls": (
            "missing-conformal-rescaling control must be rejected",
            "missing-L15-conserved-quantity control must be rejected",
            "missing-entropy-reset control must be rejected",
        ),
    },
    "dark_sector.dark_energy_teleological_potential_boundary": {
        "validation_protocol": "paper0.dark_sector.dark_energy_teleological_potential_boundary",
        "canonical_statement": (
            "Dark-energy wording is bounded to a Lambda-as-teleological-potential "
            "context score over RG-flow pressure and Cosmic Attractor drive."
        ),
        "variables": ("DE", "Λ", "RG_flow", "E", "L8"),
        "validation_targets": (
            "score Lambda-potential context",
            "require RG-flow pressure and Cosmic Attractor channels",
            "reject treating dark-energy interpretation as observation",
        ),
        "null_controls": (
            "missing-Lambda-potential control must be rejected",
            "missing-RG-flow-pressure control must be rejected",
            "dark-energy-observation control must be rejected",
        ),
    },
    "dark_sector.psi_dark_matter_hypothesis_boundary": {
        "validation_protocol": "paper0.dark_sector.psi_dark_matter_hypothesis_boundary",
        "canonical_statement": (
            "Psi-DM wording is bounded to a hypothesis that dark matter is coherent "
            "Psi-field structure, with ALP/BEC and Q-ball candidates."
        ),
        "variables": ("DM", "Psi_field", "SSB", "ALP", "BEC", "Q_ball", "V_abs_Psi"),
        "validation_targets": (
            "classify complete and incomplete Psi-DM candidate descriptions",
            "require SSB and non-linear potential labels",
            "reject treating the hypothesis as dark-matter evidence",
        ),
        "null_controls": (
            "missing-SSB control must be rejected",
            "missing-candidate-structure control must be rejected",
            "dark-matter-evidence control must be rejected",
        ),
    },
    "dark_sector.psi_dm_interaction_mechanisms": {
        "validation_protocol": "paper0.dark_sector.psi_dm_interaction_mechanisms",
        "canonical_statement": (
            "Psi-DM interaction wording is bounded to stress-energy tensor, geometric "
            "curvature coupling, and weak informational coupling mechanisms."
        ),
        "variables": ("T_mu_nu_Psi", "R", "xi", "A_mu", "Psi_Higgs", "ordinary_matter"),
        "validation_targets": (
            "score stress-energy and curvature-coupling channels",
            "require weak ordinary-matter coupling boundary",
            "preserve the source geometric-coupling formula text",
        ),
        "null_controls": (
            "missing-geometric-coupling control must be rejected",
            "missing-weak-coupling-boundary control must be rejected",
            "dark-matter-evidence control must be rejected",
        ),
    },
    "dark_sector.cosmic_coherence_reservoir_boundary": {
        "validation_protocol": "paper0.dark_sector.cosmic_coherence_reservoir_boundary",
        "canonical_statement": (
            "Cosmic-coherence-reservoir wording is bounded to source labels for "
            "structure scaffolding, halo coherence, L8 phase-locking, and L12 Gaian synchrony."
        ),
        "variables": ("structure_formation", "halo_coherence", "negentropy", "L8", "L12"),
        "validation_targets": (
            "score structure-scaffolding and halo-coherence labels",
            "require L8 and L12 integration channels",
            "reject biological negentropy coupling as empirical evidence",
        ),
        "null_controls": (
            "missing-halo-coherence control must be rejected",
            "missing-L8-L12 control must be rejected",
            "biological-negentropy-evidence control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class DarkSectorValidationSpec:
    """Validation spec promoted from Paper 0 dark-sector records."""

    key: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_statement: str
    source_equation_ids: tuple[str, ...]
    source_formulae: tuple[str, ...]
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
class DarkSectorValidationSpecBundle:
    """Dark-sector validation specs plus coverage summary."""

    specs: tuple[DarkSectorValidationSpec, ...]
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


def build_dark_sector_specs(
    source_records: list[dict[str, Any]],
) -> DarkSectorValidationSpecBundle:
    """Build source-covered validation specs for dark-sector records."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[DarkSectorValidationSpec] = []
    for key in (
        "dark_sector.mmc_operator_information_preservation",
        "dark_sector.dark_energy_teleological_potential_boundary",
        "dark_sector.psi_dark_matter_hypothesis_boundary",
        "dark_sector.psi_dm_interaction_mechanisms",
        "dark_sector.cosmic_coherence_reservoir_boundary",
    ):
        metadata = SPEC_METADATA[key]
        equation_ids = (
            (SOURCE_FORMULA_ID,) if key == "dark_sector.psi_dm_interaction_mechanisms" else ()
        )
        formulae = (SOURCE_FORMULA,) if key == "dark_sector.psi_dm_interaction_mechanisms" else ()
        specs.append(
            DarkSectorValidationSpec(
                key=key,
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=equation_ids,
                source_formulae=formulae,
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(anchor["source_record_id"]) for anchor in anchors),
                source_block_indices=tuple(
                    int(anchor["source_block_index"]) for anchor in anchors
                ),
                anchor_math_ids=(),
                variables=tuple(str(item) for item in metadata["variables"]),
                validation_targets=tuple(str(item) for item in metadata["validation_targets"]),
                executable_validation_targets=tuple(
                    str(item) for item in metadata["validation_targets"]
                ),
                null_controls=tuple(str(item) for item in metadata["null_controls"]),
                claim_boundary="source-bounded dark-sector simulator contract; not empirical evidence",
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
        "source_formula_ids": [SOURCE_FORMULA_ID],
        "all_specs_have_null_controls": all(bool(spec.null_controls) for spec in specs),
        "all_specs_have_executable_targets": all(
            bool(spec.executable_validation_targets) for spec in specs
        ),
        "all_specs_are_source_anchored": all(bool(spec.source_ledger_ids) for spec in specs),
        "all_specs_are_claim_bounded": all(
            "not empirical evidence" in spec.claim_boundary for spec in specs
        ),
        "hardware_status": "simulator_only_no_provider_submission",
        "policy": (
            "P0R06311-P0R06323 are promoted as source-covered dark-sector "
            "specifications only. Passing fixtures is not empirical evidence."
        ),
    }
    return DarkSectorValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: DarkSectorValidationSpecBundle) -> str:
    """Render a concise Markdown report for dark-sector specs."""
    lines = [
        "# Paper 0 Dark Sector Specs",
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
                f"- Source formulae: `{', '.join(spec.source_formulae) if spec.source_formulae else 'none'}`",
                f"- Null controls: `{len(spec.null_controls)}`",
                f"- Claim boundary: `{spec.claim_boundary}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Policy",
            "",
            "These records are source-anchored dark-sector specifications only. "
            "Passing any fixture is not empirical evidence and does not validate "
            "dark energy, dark matter, psi-DM, halo coherence, L8, or L12 claims.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    bundle: DarkSectorValidationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the dark-sector bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_dark_sector_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_dark_sector_validation_specs_report_{date_tag}.md"
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
    bundle = build_dark_sector_specs(records)
    paths = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
