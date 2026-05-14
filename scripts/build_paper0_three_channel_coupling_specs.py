#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 three-channel coupling spec builder
"""Promote Paper 0 unified coupling parameter-scan records into specs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(7081, 7130))
CLAIM_BOUNDARY = "source-bounded parameter scan; not empirical support"
HARDWARE_STATUS = "parameter_scan_protocol_no_execution"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "three_channel_coupling.section_boundary": {
        "scan_id": "chapter25",
        "validation_protocol": "paper0.three_channel_coupling.section_boundary",
        "canonical_statement": "Chapter 25 defines a unified coupling parameter scan across three constrained channels.",
        "source_equation_ids": (
            "P0R07081:chapter_boundary",
            "P0R07082:setup_geometry_factors",
            "P0R07094:experimental_constraints",
        ),
        "source_formulae": ("three-channel correlation test",),
        "test_protocols": ("preserve scan boundary and protocol-only status",),
        "null_results": ("scan definition is not empirical support",),
        "variables": ("lambda0", "channel", "constraint"),
        "validation_targets": (
            "preserve Chapter 25 boundary",
            "preserve three-channel constraint structure",
            "reject treating the scan as executed data",
        ),
        "null_controls": (
            "missing-channel control must be rejected",
            "execution-overclaim control must be rejected",
            "missing-constraint control must be rejected",
        ),
    },
    "three_channel_coupling.geometry_factors": {
        "scan_id": "geometry",
        "validation_protocol": "paper0.three_channel_coupling.geometry_factors",
        "canonical_statement": "All Psi-sector couplings derive from one bare lambda0 through canonical warp geometry factors.",
        "source_equation_ids": (
            "P0R07083:canonical_warp_parameters",
            "P0R07084:single_lambda0_mapping",
            "P0R07086:c_g",
            "P0R07087:c_em",
            "P0R07088:c_q",
            "P0R07089:c_s",
        ),
        "source_formulae": (
            "lambda_psi_i = c_i lambda0",
            "sigma approximately 28.3, n = 6, R = 5 l_P",
            "c_G approximately 1.1e-122",
            "c_EM approximately 9.3e-2",
            "c_Q approximately 8.0e-2",
            "c_S approximately 1.3e-6",
        ),
        "test_protocols": (
            "compute fixed channel couplings from lambda0",
            "preserve canonical geometry parameters",
            "reject independent sector fitting of channel couplings",
        ),
        "null_results": (
            "independently tuned channel couplings break the single-lambda0 hypothesis",
        ),
        "variables": ("lambda0", "c_G", "c_EM", "c_Q", "c_S"),
        "validation_targets": (
            "preserve single-lambda0 mapping",
            "preserve geometry factors",
            "reject free per-channel coupling fits",
        ),
        "null_controls": (
            "free-channel-coupling control must be rejected",
            "missing-geometry-factor control must be rejected",
            "wrong-warp-parameter control must be rejected",
        ),
    },
    "three_channel_coupling.fixed_ratios": {
        "scan_id": "ratios",
        "validation_protocol": "paper0.three_channel_coupling.fixed_ratios",
        "canonical_statement": "The source states fixed ratios between gravitational, electromagnetic, quantum, and scalar channels.",
        "source_equation_ids": (
            "P0R07090:fixed_ratios",
            "P0R07091:em_over_g",
            "P0R07092:q_over_em",
            "P0R07093:s_over_em",
        ),
        "source_formulae": (
            "lambda_psi_EM / lambda_psi_G approximately 8.5e120",
            "lambda_psi_Q / lambda_psi_EM approximately 0.86",
            "lambda_psi_S / lambda_psi_EM approximately 1.4e-5",
        ),
        "test_protocols": ("check ratios from geometry factors within source precision",),
        "null_results": ("ratio mismatch invalidates the claimed three-way fingerprint",),
        "variables": ("EM_over_G", "Q_over_EM", "S_over_EM"),
        "validation_targets": (
            "preserve fixed ratios",
            "preserve ratio precision as approximate",
            "reject ratio-free anomaly interpretation",
        ),
        "null_controls": (
            "ratio-mismatch control must be rejected",
            "missing-ratio control must be rejected",
            "single-sector-anomaly control must be rejected",
        ),
    },
    "three_channel_coupling.experimental_constraints": {
        "scan_id": "constraints",
        "validation_protocol": "paper0.three_channel_coupling.experimental_constraints",
        "canonical_statement": "Gravitational, EM clock, and quantum coherence limits jointly constrain lambda0.",
        "source_equation_ids": (
            "P0R07095:three_independent_channels",
            "P0R07099:gravitational_constraint",
            "P0R07101:em_clock_constraint",
            "P0R07103:quantum_coherence_constraint",
            "P0R07105:constraint_ranking",
        ),
        "source_formulae": (
            "lambda0 at gravitational limit approximately 1.6e-4",
            "lambda0 at EM clock limit approximately 1.8e-5",
            "lambda0 at quantum coherence limit approximately 2.0e-5",
        ),
        "test_protocols": (
            "rank current constraints by lambda0 upper limit",
            "preserve atomic-clock and matter-wave comparability",
            "preserve gravitational constraint as looser",
        ),
        "null_results": ("wrong constraint ordering changes the viable window",),
        "variables": ("G_limit", "EM_limit", "Q_limit", "lambda0"),
        "validation_targets": (
            "preserve three independent channel constraints",
            "preserve EM clock as current most stringent constraint",
            "preserve matter-wave comparability",
        ),
        "null_controls": (
            "missing-constraint-channel control must be rejected",
            "wrong-ranking control must be rejected",
            "constraint-as-detection control must be rejected",
        ),
    },
    "three_channel_coupling.cross_channel_propagation": {
        "scan_id": "sweet_spot",
        "validation_protocol": "paper0.three_channel_coupling.cross_channel_propagation",
        "canonical_statement": "The source defines a narrow lambda0 window and cross-channel propagation of bounds.",
        "source_equation_ids": (
            "P0R07107:sweet_spot_window",
            "P0R07108:extra_acceleration",
            "P0R07109:alpha_drift",
            "P0R07110:decoherence_fraction",
            "P0R07114:cross_channel_propagation",
            "P0R07115:cosmology_bound",
            "P0R07116:em_propagated_bound",
            "P0R07117:q_propagated_bound",
            "P0R07119:em_anomaly_propagation",
        ),
        "source_formulae": (
            "lambda0 approximately 1e-6 to 1e-5",
            "lambda0 = 1e-5 predicts 1e-9 m/s^2 acceleration, 5e-18/year alpha drift, 5e-6 decoherence",
            "lambda_psi_G < 1e-126 propagates to lambda_psi_EM < 1.2e-6 and lambda_psi_Q < 9.4e-7",
        ),
        "test_protocols": (
            "compute sweet-spot observables",
            "propagate single-channel bounds to other sectors",
            "preserve next-generation instrumentation targets",
        ),
        "null_results": ("missing propagation makes the single-lambda0 scan non-falsifiable",),
        "variables": ("lambda0", "delta_g", "alpha_drift", "decoherence", "bound"),
        "validation_targets": (
            "preserve sweet-spot window",
            "preserve source-stated observable magnitudes",
            "preserve cross-channel bound propagation",
        ),
        "null_controls": (
            "missing-propagation control must be rejected",
            "wrong-sweet-spot-window control must be rejected",
            "instrumentation-overclaim control must be rejected",
        ),
    },
    "three_channel_coupling.falsification_fingerprint": {
        "scan_id": "fingerprint",
        "validation_protocol": "paper0.three_channel_coupling.falsification_fingerprint",
        "canonical_statement": "The three-channel pattern is falsified by isolated single-channel signals and supported only by predicted-ratio coincidence.",
        "source_equation_ids": (
            "P0R07121:falsifiable_fingerprint",
            "P0R07127:three_way_correlation",
            "P0R07128:falsification_boundary",
        ),
        "source_formulae": (
            "micro-Gal gravimetric blips + 1e-16 clock drift + 1e-6 decoherence anomaly",
            "single channel signal with other channels null falsifies unified coupling",
        ),
        "test_protocols": (
            "classify all-null, single-channel, and three-channel outcomes",
            "require predicted ratios for supportive three-channel pattern",
            "reject single-sector confirmation language",
        ),
        "null_results": (
            "single-channel signal with other channels null falsifies the unified coupling framework",
        ),
        "variables": ("G_signal", "EM_signal", "Q_signal", "ratio_match"),
        "validation_targets": (
            "preserve three-channel fingerprint",
            "preserve falsification boundary",
            "reject isolated-channel overclaim",
        ),
        "null_controls": (
            "single-channel-overclaim control must be rejected",
            "missing-ratio-match control must be rejected",
            "all-null-as-confirmation control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class ThreeChannelCouplingSpec:
    """Three-channel coupling spec promoted from Paper 0 records."""

    key: str
    scan_id: str
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
class ThreeChannelCouplingSpecBundle:
    """Three-channel coupling specs plus coverage summary."""

    specs: tuple[ThreeChannelCouplingSpec, ...]
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


def build_three_channel_coupling_specs(
    source_records: list[dict[str, Any]],
) -> ThreeChannelCouplingSpecBundle:
    """Build source-covered three-channel coupling specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[ThreeChannelCouplingSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ThreeChannelCouplingSpec(
                key=key,
                scan_id=str(metadata["scan_id"]),
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=tuple(str(item) for item in metadata["source_equation_ids"]),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(anchor["source_record_id"]) for anchor in anchors),
                source_block_indices=tuple(
                    int(anchor["source_block_index"]) for anchor in anchors
                ),
                source_formulae=tuple(str(item) for item in metadata["source_formulae"]),
                test_protocols=tuple(str(item) for item in metadata["test_protocols"]),
                null_results=tuple(str(item) for item in metadata["null_results"]),
                variables=tuple(str(item) for item in metadata["variables"]),
                validation_targets=tuple(str(item) for item in metadata["validation_targets"]),
                executable_validation_targets=tuple(
                    str(item) for item in metadata["validation_targets"]
                ),
                null_controls=tuple(str(item) for item in metadata["null_controls"]),
                claim_boundary=CLAIM_BOUNDARY,
                implementation_status="implemented_executable_fixture",
                domain_review_status="promoted_to_validation_spec",
                hardware_status=HARDWARE_STATUS,
            )
        )

    summary = {
        "title": "Paper 0 Three-Channel Coupling Specs",
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(SOURCE_LEDGER_IDS),
        "coverage_match": True,
        "unconsumed_source_ledger_ids": [],
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "spec_count": len(specs),
        "channel_count": 3,
        "sweet_spot_window": [1.0e-6, 1.0e-5],
        "spec_keys": [spec.key for spec in specs],
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "all_specs_have_null_controls": all(bool(spec.null_controls) for spec in specs),
        "all_specs_are_source_anchored": all(
            spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in specs
        ),
    }
    return ThreeChannelCouplingSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: ThreeChannelCouplingSpecBundle) -> str:
    """Render a compact Markdown report for internal review."""
    lines = [
        "# Paper 0 Three-Channel Coupling Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records consumed: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Channel count: {bundle.summary['channel_count']}",
        f"- Sweet-spot window: {bundle.summary['sweet_spot_window']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                "",
                f"### {spec.key}",
                f"- Scan: {spec.scan_id}",
                f"- Protocol: {spec.validation_protocol}",
                f"- Statement: {spec.canonical_statement}",
                f"- Null controls: {len(spec.null_controls)}",
            ]
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    bundle: ThreeChannelCouplingSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for three-channel coupling specs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_three_channel_coupling_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_three_channel_coupling_validation_specs_report_{date_tag}.md"
    )
    payload = {"specs": [asdict(spec) for spec in bundle.specs], "summary": bundle.summary}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_validation_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> int:
    """Build three-channel coupling specs from the canonical review ledger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args()

    bundle = build_three_channel_coupling_specs(load_jsonl(args.ledger))
    write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
