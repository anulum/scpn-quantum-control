#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Torus surface flow: Lyapunov-style certificate. spec builder
"""Promote Paper 0 Torus surface flow: Lyapunov-style certificate. records."""

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

SOURCE_LEDGER_IDS = (
    "P0R02975",
    "P0R02976",
    "P0R02977",
    "P0R02978",
    "P0R02979",
    "P0R02980",
    "P0R02981",
    "P0R02982",
)
CLAIM_BOUNDARY = "source-bounded torus surface flow lyapunov style certificate source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "torus_surface_flow_lyapunov_style_certificate.torus_surface_flow_lyapunov_style_certificate": {
        "context_id": "torus_surface_flow_lyapunov_style_certificate",
        "validation_protocol": "paper0.torus_surface_flow_lyapunov_style_certificate.torus_surface_flow_lyapunov_style_certificate",
        "canonical_statement": "The source-bounded component 'Torus surface flow: Lyapunov-style certificate.' preserves Paper 0 records P0R02975-P0R02977 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02975:torus_surface_flow_lyapunov_style_certificate",
            "P0R02976:torus_surface_flow_lyapunov_style_certificate",
            "P0R02977:torus_surface_flow_lyapunov_style_certificate",
        ),
        "source_formulae": (
            "P0R02975: Torus surface flow: Lyapunov-style certificate.",
            "P0R02976: We constrain the phase flow on the SCPN torus to remain on a controlled invariant surface while allowing neutral motion along its cycles. Let denote the wrapped phase coordinates on the torus (SxS). Define a surface error energy W_torus as a wrapped-distance quadratic to the target invariant torus, plus a small curvature regulariser.",
            "P0R02977: The fast channel applies G_f to drive W_torus c_t W_torus in the transverse directions (no escape from the surface), while the slow channel allocates G_s to bias motion along the neutral directions (helical traversal) without changing W_torus. This is compatible with the UPDE holonomy construction and the toroidal flow primitives already present.",
        ),
        "test_protocols": (
            "preserve Torus surface flow: Lyapunov-style certificate. source-accounting boundary",
        ),
        "null_results": (
            "Torus surface flow: Lyapunov-style certificate. is not empirical validation evidence",
        ),
        "variables": ("torus_surface_flow_lyapunov_style_certificate",),
        "validation_targets": ("preserve records P0R02975-P0R02977",),
        "null_controls": (
            "torus_surface_flow_lyapunov_style_certificate must remain source-bounded accounting",
        ),
    },
    "torus_surface_flow_lyapunov_style_certificate.ms_qec_integration_fast_channel_realiser": {
        "context_id": "ms_qec_integration_fast_channel_realiser",
        "validation_protocol": "paper0.torus_surface_flow_lyapunov_style_certificate.ms_qec_integration_fast_channel_realiser",
        "canonical_statement": "The source-bounded component 'MS-QEC integration (fast channel realiser).' preserves Paper 0 records P0R02978-P0R02979 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02978:ms_qec_integration_fast_channel_realiser",
            "P0R02979:ms_qec_integration_fast_channel_realiser",
        ),
        "source_formulae": (
            "P0R02978: MS-QEC integration (fast channel realiser).",
            'P0R02979: MS-QEC provides the actuated stabilisers that realise G_f at substrate and network scales; its role in the composite Lyapunov is to guarantee the negative-definite fast term even under bounded disturbances, matching your present "V = H_rec descent under bounded noise" lines and QECC error-suppression clauses.',
        ),
        "test_protocols": (
            "preserve MS-QEC integration (fast channel realiser). source-accounting boundary",
        ),
        "null_results": (
            "MS-QEC integration (fast channel realiser). is not empirical validation evidence",
        ),
        "variables": ("ms_qec_integration_fast_channel_realiser",),
        "validation_targets": ("preserve records P0R02978-P0R02979",),
        "null_controls": (
            "ms_qec_integration_fast_channel_realiser must remain source-bounded accounting",
        ),
    },
    "torus_surface_flow_lyapunov_style_certificate.implementation_notes": {
        "context_id": "implementation_notes",
        "validation_protocol": "paper0.torus_surface_flow_lyapunov_style_certificate.implementation_notes",
        "canonical_statement": "The source-bounded component 'Implementation notes.' preserves Paper 0 records P0R02980-P0R02982 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02980:implementation_notes",
            "P0R02981:implementation_notes",
            "P0R02982:implementation_notes",
        ),
        "source_formulae": (
            "P0R02980: Implementation notes.",
            "P0R02981: The gains {G_f, G_s} are evaluated per layer using local A, sigma, and phase observables; cross-layer feed-forward terms enter as already defined in C_InterLayer.",
            "P0R02982: If |A/sigma| spikes (steep affective landscape), exploration is throttled (G_s down) and stabilisation is prioritised (G_f up). If |A/sigma| relaxes within the sigma1 window, exploration is gently increased (G_s up) while keeping the fast channel active. All claims are operational: parameters are to be preregistered and audited via your existing stability metrics and noise bounds.",
        ),
        "test_protocols": ("preserve Implementation notes. source-accounting boundary",),
        "null_results": ("Implementation notes. is not empirical validation evidence",),
        "variables": ("implementation_notes",),
        "validation_targets": ("preserve records P0R02980-P0R02982",),
        "null_controls": ("implementation_notes must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class TorusSurfaceFlowLyapunovStyleCertificateSpec:
    """Spec promoted from Paper 0 source records."""

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
class TorusSurfaceFlowLyapunovStyleCertificateSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TorusSurfaceFlowLyapunovStyleCertificateSpec, ...]
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


def build_torus_surface_flow_lyapunov_style_certificate_specs(
    source_records: list[dict[str, Any]],
) -> TorusSurfaceFlowLyapunovStyleCertificateSpecBundle:
    """Build source-covered specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    category_counts = Counter(
        str(record.get("canonical_category", "unknown")) for record in anchors
    )
    block_counts = Counter(str(record.get("block_type", "unknown")) for record in anchors)

    specs: list[TorusSurfaceFlowLyapunovStyleCertificateSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TorusSurfaceFlowLyapunovStyleCertificateSpec(
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
        "title": "Paper 0 " + "Torus surface flow: Lyapunov-style certificate." + " Specs",
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
        "next_source_boundary": "P0R02983",
    }
    return TorusSurfaceFlowLyapunovStyleCertificateSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TorusSurfaceFlowLyapunovStyleCertificateSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_torus_surface_flow_lyapunov_style_certificate_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: TorusSurfaceFlowLyapunovStyleCertificateSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Torus surface flow: Lyapunov-style certificate." + " Specs",
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
    bundle: TorusSurfaceFlowLyapunovStyleCertificateSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_torus_surface_flow_lyapunov_style_certificate_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_torus_surface_flow_lyapunov_style_certificate_validation_specs_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-17")
    args = parser.parse_args()
    outputs = write_outputs(
        build_from_ledger(args.ledger), output_dir=args.output_dir, date_tag=args.date_tag
    )
    print(outputs["json"])
    print(outputs["report"])


if __name__ == "__main__":
    main()
