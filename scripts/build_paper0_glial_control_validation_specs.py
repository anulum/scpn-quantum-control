#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 glial-control validation spec builder
"""Promote Paper 0 abiogenesis/cellular sigma anchors into validation specs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scpn_quantum_control.paper0.equation_register import get_paper0_equation_record

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SPEC_SOURCE_LEDGER_IDS: dict[str, tuple[str, ...]] = {
    "embodied.quantum_immune_interface": (
        "P0R05360",
        "P0R05361",
        "P0R05363",
    ),
    "embodied.glial_sigma_control": (
        "P0R05366",
        "P0R05367",
        "P0R05368",
        "P0R05369",
        "P0R05370",
        "P0R05371",
        "P0R05372",
        "P0R05376",
        "P0R05377",
        "P0R05385",
        "P0R05388",
        "P0R05390",
        "P0R05391",
        "P0R05392",
        "P0R05395",
        "P0R05396",
        "P0R05397",
        "P0R05399",
        "P0R05400",
        "P0R05403",
        "P0R05404",
        "P0R05405",
        "P0R05406",
    ),
}

SPEC_METADATA: dict[str, dict[str, tuple[str, ...] | str]] = {
    "embodied.quantum_immune_interface": {
        "validation_protocol": "paper0.embodied.quantum_immune.hamiltonian_parameter_scan",
        "implementation_status": "validation_spec_pending_executable_fixture",
        "implementation_links": (
            "src/scpn_quantum_control/paper0/equation_register.py",
            "docs/internal/paper0_foundational_extraction/"
            "paper0_canonical_review_ledger_2026-05-13.jsonl",
        ),
        "null_controls": (
            "zero lambda immune-decoupled Hamiltonian control",
            "fixed C_cyto parameter control with no cytokine sensitivity",
            "operator-sign reversal control for the sigma_x coupling convention",
            "non-Hermitian perturbation rejection guard",
        ),
        "executable_validation_targets": (
            "construct the finite-dimensional H_int operator and verify Hermiticity",
            "scan cytokine-state inputs through lambda(Psi_s, C_cyto)",
            "measure spectral shift and operator norm under bounded cytokine drive",
            "label all quantum-biological interpretation as simulator-only mechanism evidence",
        ),
    },
    "embodied.glial_sigma_control": {
        "validation_protocol": "paper0.embodied.glial_sigma.two_timescale_control",
        "implementation_status": "validation_spec_pending_executable_fixture",
        "implementation_links": (
            "src/scpn_quantum_control/paper0/equation_register.py",
            "docs/internal/paper0_foundational_extraction/"
            "paper0_canonical_review_ledger_2026-05-13.jsonl",
        ),
        "null_controls": (
            "gamma = 0 gliotransmitter blockade control",
            "zero calcium-drive control for G(t) clearance to baseline",
            "negative beta rejection guard for non-physical gliotransmitter growth",
            "matched-noise sigma-only control without astrocyte set-point shift",
        ),
        "executable_validation_targets": (
            "integrate coupled sigma and G dynamics under measured or generated calcium drive",
            "verify sigma relaxes toward one without glial drive",
            "quantify glial set-point shift under finite gamma and bounded G(t)",
            "test gliotransmitter blockade falsifier as attenuation of sigma response",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class GlialControlValidationSpec:
    """Validation spec promoted from Paper 0 glial and immune source anchors."""

    key: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_latex: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    source_block_indices: tuple[int, ...]
    anchor_math_ids: tuple[str, ...]
    variables: dict[str, str]
    assumptions: tuple[str, ...]
    validation_targets: tuple[str, ...]
    executable_validation_targets: tuple[str, ...]
    null_controls: tuple[str, ...]
    implementation_links: tuple[str, ...]
    implementation_status: str
    domain_review_status: str
    hardware_status: str


@dataclass(frozen=True, slots=True)
class GlialControlValidationSpecBundle:
    """Glial-control validation specs plus coverage summary."""

    specs: tuple[GlialControlValidationSpec, ...]
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


def build_glial_control_validation_specs(
    source_records: list[dict[str, Any]],
) -> GlialControlValidationSpecBundle:
    """Build source-covered validation specs for the Paper 0 glial-control family."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    required_ids = {
        ledger_id for ledger_ids in SPEC_SOURCE_LEDGER_IDS.values() for ledger_id in ledger_ids
    }
    missing = sorted(required_ids - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    specs: list[GlialControlValidationSpec] = []
    consumed_ids: set[str] = set()
    for key in ("embodied.quantum_immune_interface", "embodied.glial_sigma_control"):
        equation_record = get_paper0_equation_record(key)
        ledger_ids = SPEC_SOURCE_LEDGER_IDS[key]
        anchors = [records_by_ledger[ledger_id] for ledger_id in ledger_ids]
        anchor_math_ids = tuple(
            sorted({str(math_id) for anchor in anchors for math_id in anchor.get("math_ids", [])})
        )
        metadata = SPEC_METADATA[key]
        consumed_ids.update(ledger_ids)
        specs.append(
            GlialControlValidationSpec(
                key=key,
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript=equation_record.manuscript,
                section_path=equation_record.section_path,
                canonical_latex=equation_record.canonical_latex,
                source_equation_ids=equation_record.source_equation_ids,
                source_ledger_ids=ledger_ids,
                source_record_ids=tuple(str(anchor["source_record_id"]) for anchor in anchors),
                source_block_indices=tuple(
                    int(anchor["source_block_index"]) for anchor in anchors
                ),
                anchor_math_ids=anchor_math_ids,
                variables=dict(equation_record.variables),
                assumptions=equation_record.assumptions,
                validation_targets=equation_record.validation_targets,
                executable_validation_targets=tuple(metadata["executable_validation_targets"]),
                null_controls=tuple(metadata["null_controls"]),
                implementation_links=tuple(metadata["implementation_links"]),
                implementation_status=str(metadata["implementation_status"]),
                domain_review_status="promoted_to_validation_spec",
                hardware_status="simulator_only_no_provider_submission",
            )
        )

    summary = {
        "source_record_count": len(required_ids),
        "consumed_source_record_count": len(consumed_ids),
        "coverage_match": consumed_ids == required_ids,
        "unconsumed_source_ledger_ids": sorted(required_ids - consumed_ids),
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "all_specs_have_null_controls": all(bool(spec.null_controls) for spec in specs),
        "all_specs_have_executable_targets": all(
            bool(spec.executable_validation_targets) for spec in specs
        ),
        "all_specs_are_source_anchored": all(bool(spec.source_ledger_ids) for spec in specs),
        "hardware_status": "simulator_only_no_provider_submission",
        "policy": (
            "EQ0105-EQ0112 anchors are promoted only into validation specifications. "
            "Abiogenesis, immune-interface, and glial-control claims remain bounded "
            "mechanism hypotheses until executable fixtures, null controls, source-to-result "
            "provenance, and empirical boundary labels pass."
        ),
    }
    return GlialControlValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: GlialControlValidationSpecBundle) -> str:
    """Render a concise Markdown report for glial-control validation specs."""
    status = "match" if bundle.summary["coverage_match"] else "mismatch"
    lines = [
        "# Paper 0 Glial-Control Validation Specs",
        "",
        f"- Source records: `{bundle.summary['source_record_count']}`",
        f"- Consumed source records: `{bundle.summary['consumed_source_record_count']}`",
        f"- Coverage status: `{status}`",
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
                f"- Source equations: `{', '.join(spec.source_equation_ids)}`",
                f"- Source ledgers: `{', '.join(spec.source_ledger_ids)}`",
                f"- Null controls: `{len(spec.null_controls)}`",
                f"- Executable targets: `{len(spec.executable_validation_targets)}`",
                f"- Status: `{spec.implementation_status}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Policy",
            "",
            "Provider submission remains out of scope. These records are "
            "source-anchored validation specifications only; executable simulator "
            "fixtures, biomedical boundary labels, and null controls must pass "
            "before any paper claim or hardware plan is promoted.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    bundle: GlialControlValidationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the glial-control spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_glial_control_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_glial_control_validation_specs_report_{date_tag}.md"
    payload = {
        "summary": bundle.summary,
        "specs": [asdict(spec) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_validation_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def _select_required_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required_ids = {
        ledger_id for ledger_ids in SPEC_SOURCE_LEDGER_IDS.values() for ledger_id in ledger_ids
    }
    return [record for record in records if str(record.get("ledger_id")) in required_ids]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args(argv)

    records = _select_required_records(load_jsonl(args.ledger))
    bundle = build_glial_control_validation_specs(records)
    paths = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
