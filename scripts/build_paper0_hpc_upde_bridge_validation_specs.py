#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 HPC/UPDE bridge spec builder
"""Promote Paper 0 HPC/UPDE bridge anchors into validation specs."""

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

SPEC_SOURCE_LEDGER_IDS: dict[str, tuple[str, ...]] = {
    "computational.hpc_bidirectional_flow": (
        "P0R06156",
        "P0R06157",
        "P0R06158",
    ),
    "computational.upde_phase_prediction_error": (
        "P0R06159",
        "P0R06160",
        "P0R06161",
        "P0R06162",
        "P0R06163",
    ),
    "computational.upde_free_energy_gradient_bridge": (
        "P0R06164",
        "P0R06165",
        "P0R06166",
        "P0R06167",
        "P0R06168",
        "P0R06169",
        "P0R06170",
        "P0R06171",
        "P0R06172",
        "P0R06173",
        "P0R06174",
        "P0R06175",
        "P0R06176",
        "P0R06177",
        "P0R06178",
    ),
}

SPEC_METADATA: dict[str, dict[str, tuple[str, ...] | str]] = {
    "computational.hpc_bidirectional_flow": {
        "validation_protocol": "paper0.computational.hpc.bidirectional_flow",
        "canonical_statement": (
            "Hierarchical predictive coding is treated as a bounded computational "
            "mapping: downward generative predictions and upward prediction-error "
            "filtering must form a directed hierarchy with error-only upward flow."
        ),
        "variables": (
            "layer_state",
            "top_down_prediction",
            "prediction_error",
            "hierarchy_adjacency",
        ),
        "assumptions": (
            "hierarchy is directed from higher to lower layers for prediction",
            "upward messages carry residual error rather than duplicated state",
            "biological-cell correspondences are treated as interpretation boundaries",
        ),
        "validation_targets": (
            "construct finite layered graph with downward prediction edges",
            "verify upward message equals observed lower state minus prediction",
            "verify generative model refinement reduces held-out residual error",
        ),
        "null_controls": (
            "disconnected-hierarchy control must block residual propagation",
            "state-copy control must reject upward flow that is not prediction error",
            "layer-order reversal control must fail directional consistency",
        ),
    },
    "computational.upde_phase_prediction_error": {
        "validation_protocol": "paper0.computational.upde.phase_prediction_error",
        "canonical_statement": (
            "The UPDE phase-coupling term is promoted as a falsifiable prediction-error "
            "surrogate only where inter-layer phase differences and precision weights "
            "are explicitly represented."
        ),
        "variables": (
            "theta_lower",
            "theta_upper",
            "sin_delta_theta",
            "precision_weight",
            "effective_coupling",
        ),
        "assumptions": (
            "phase error is local to explicitly coupled layers",
            "small-step coupling is dissipative for attractive non-negative weights",
            "DMN/CEN and salience labels remain biological hypotheses outside fixture scope",
        ),
        "validation_targets": (
            "verify sin(theta_upper - theta_lower) is the signed phase residual",
            "verify attractive phase update reduces squared phase residual",
            "verify precision weights scale inter-layer residual contribution",
        ),
        "null_controls": (
            "zero-coupling control must not reduce residual through hidden dynamics",
            "negative-precision control must be rejected",
            "uncoupled-layer control must not receive precision-weighted residual",
        ),
    },
    "computational.upde_free_energy_gradient_bridge": {
        "validation_protocol": "paper0.computational.upde.free_energy_gradient_bridge",
        "canonical_statement": (
            "The UPDE-to-free-energy bridge is accepted only as an XY-potential "
            "gradient identity: F(theta) = -sum K_ij cos(theta_j - theta_i), "
            "with Kuramoto drift equal to intrinsic drive plus negative gradient "
            "and explicit noise."
        ),
        "variables": (
            "theta_i",
            "theta_j",
            "K_ij",
            "omega_i",
            "eta_i",
            "free_energy_gradient",
        ),
        "assumptions": (
            "couplings are finite and sign-explicit",
            "gradient claims are tested on finite phase vectors",
            "cosmic-scale interpretation is not promoted beyond the mathematical identity",
        ),
        "validation_targets": (
            "verify XY potential is minimised at zero coupled phase differences",
            "verify analytic gradient against central finite-difference derivative",
            "verify Kuramoto drift equals omega minus gradient plus eta",
        ),
        "null_controls": (
            "wrong-sign coupling control must increase rather than decrease potential",
            "asymmetric-coupling control must declare directed-gradient convention",
            "non-finite phase control must be rejected before gradient evaluation",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class HpcUpdeBridgeValidationSpec:
    """Validation spec promoted from Paper 0 HPC/UPDE bridge records."""

    key: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_statement: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    source_block_indices: tuple[int, ...]
    anchor_math_ids: tuple[str, ...]
    variables: tuple[str, ...]
    assumptions: tuple[str, ...]
    validation_targets: tuple[str, ...]
    executable_validation_targets: tuple[str, ...]
    null_controls: tuple[str, ...]
    implementation_status: str
    domain_review_status: str
    hardware_status: str


@dataclass(frozen=True, slots=True)
class HpcUpdeBridgeValidationSpecBundle:
    """HPC/UPDE bridge validation specs plus coverage summary."""

    specs: tuple[HpcUpdeBridgeValidationSpec, ...]
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


def build_hpc_upde_bridge_validation_specs(
    source_records: list[dict[str, Any]],
) -> HpcUpdeBridgeValidationSpecBundle:
    """Build source-covered validation specs for the HPC/UPDE bridge."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    required_ids = {
        ledger_id for ledger_ids in SPEC_SOURCE_LEDGER_IDS.values() for ledger_id in ledger_ids
    }
    missing = sorted(required_ids - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    specs: list[HpcUpdeBridgeValidationSpec] = []
    consumed_ids: set[str] = set()
    for key in (
        "computational.hpc_bidirectional_flow",
        "computational.upde_phase_prediction_error",
        "computational.upde_free_energy_gradient_bridge",
    ):
        ledger_ids = SPEC_SOURCE_LEDGER_IDS[key]
        anchors = [records_by_ledger[ledger_id] for ledger_id in ledger_ids]
        anchor_math_ids = tuple(
            sorted({str(math_id) for anchor in anchors for math_id in anchor.get("math_ids", [])})
        )
        metadata = SPEC_METADATA[key]
        consumed_ids.update(ledger_ids)
        specs.append(
            HpcUpdeBridgeValidationSpec(
                key=key,
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=(),
                source_ledger_ids=ledger_ids,
                source_record_ids=tuple(str(anchor["source_record_id"]) for anchor in anchors),
                source_block_indices=tuple(
                    int(anchor["source_block_index"]) for anchor in anchors
                ),
                anchor_math_ids=anchor_math_ids,
                variables=tuple(str(item) for item in metadata["variables"]),
                assumptions=tuple(str(item) for item in metadata["assumptions"]),
                validation_targets=tuple(str(item) for item in metadata["validation_targets"]),
                executable_validation_targets=tuple(
                    str(item) for item in metadata["validation_targets"]
                ),
                null_controls=tuple(str(item) for item in metadata["null_controls"]),
                implementation_status="validation_spec_pending_executable_fixture",
                domain_review_status="promoted_to_validation_spec",
                hardware_status="simulator_only_no_provider_submission",
            )
        )

    sorted_required = sorted(required_ids)
    summary = {
        "source_record_count": len(required_ids),
        "consumed_source_record_count": len(consumed_ids),
        "coverage_match": consumed_ids == required_ids,
        "unconsumed_source_ledger_ids": sorted(required_ids - consumed_ids),
        "source_ledger_span": [sorted_required[0], sorted_required[-1]],
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "all_specs_have_null_controls": all(bool(spec.null_controls) for spec in specs),
        "all_specs_have_executable_targets": all(
            bool(spec.executable_validation_targets) for spec in specs
        ),
        "all_specs_are_source_anchored": all(bool(spec.source_ledger_ids) for spec in specs),
        "all_specs_avoid_invented_equation_ids": all(
            not spec.source_equation_ids and not spec.anchor_math_ids for spec in specs
        ),
        "hardware_status": "simulator_only_no_provider_submission",
        "policy": (
            "P0R06156-P0R06178 are promoted as source-covered validation "
            "specifications. No standalone equation IDs are invented for formula "
            "text lacking canonical EQ anchors; all claims remain bounded to "
            "finite mathematical and computational fixtures before domain promotion."
        ),
    }
    return HpcUpdeBridgeValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: HpcUpdeBridgeValidationSpecBundle) -> str:
    """Render a concise Markdown report for HPC/UPDE bridge validation specs."""
    status = "match" if bundle.summary["coverage_match"] else "mismatch"
    lines = [
        "# Paper 0 HPC/UPDE Bridge Validation Specs",
        "",
        f"- Source records: `{bundle.summary['source_record_count']}`",
        f"- Consumed source records: `{bundle.summary['consumed_source_record_count']}`",
        f"- Coverage status: `{status}`",
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
            "No standalone equation IDs are invented for formula text that was not "
            "assigned canonical equation anchors during Paper 0 extraction. These "
            "records are source-anchored validation specifications only; the HPC, "
            "UPDE, and free-energy bridge interpretations require executable "
            "fixtures and falsification controls before claim promotion.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    bundle: HpcUpdeBridgeValidationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the HPC/UPDE bridge spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_hpc_upde_bridge_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_hpc_upde_bridge_validation_specs_report_{date_tag}.md"
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
    bundle = build_hpc_upde_bridge_validation_specs(records)
    paths = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
