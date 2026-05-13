#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Paper 0 UPDE validation spec builder
"""Promote source-anchored Paper 0 UPDE equations into validation specs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scpn_quantum_control.paper0.equation_register import paper0_upde_records

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_ANCHOR_PATH = DEFAULT_EXTRACTION_DIR / "paper0_upde_anchor_review_queue_2026-05-13.jsonl"

_SPEC_METADATA: dict[str, dict[str, Any]] = {
    "upde.base_phase": {
        "validation_protocol": "paper0.upde.base_phase.xy_gradient_and_locking",
        "implementation_status": "partially_implemented_quantum_xy_mapping",
        "implementation_links": (
            "src/scpn_quantum_control/phase/xy_kuramoto.py",
            "src/scpn_quantum_control/phase/trotter_upde.py",
            "src/scpn_quantum_control/kuramoto_core.py",
            "scripts/generate_s19_resource_signature_scan.py",
        ),
        "null_controls": (
            "set K_ij^L to zero and require no topology-induced phase locking",
            "shuffle coupling topology at fixed spectrum and require witness degradation",
            "flip coupling signs and require a distinguishable anti-locking response",
            "run an off-onset coupling grid to separate trivial synchrony from transition behaviour",
        ),
        "executable_validation_targets": (
            "compare finite-difference UPDE phase gradients against XY compiled generator terms",
            "scan Kuramoto order parameter R(t) over K_nm and omega disorder",
            "verify dense-budget guards before exact simulator paths are allowed",
            "record simulator provenance before any provider submission is considered",
        ),
    },
    "upde.interlayer_coupling": {
        "validation_protocol": "paper0.upde.interlayer.directional_coupling",
        "implementation_status": "validation_spec_pending_direct_implementation_audit",
        "implementation_links": (
            "src/scpn_quantum_control/fep/predictive_coding.py",
            "src/scpn_quantum_control/bridge/phase_artifact.py",
            "src/scpn_quantum_control/hardware/feedback_loop.py",
        ),
        "null_controls": (
            "set epsilon_{L-1} and epsilon_{L+1} to zero and require layer decoupling",
            "perturb lower and upper layers separately and require directional response separation",
            "randomise layer averages while preserving marginal phase distributions",
        ),
        "executable_validation_targets": (
            "construct two-layer and three-layer phase fixtures with controlled adjacent coupling",
            "measure response asymmetry under downward and upward perturbations",
            "reject non-adjacent or shape-inconsistent layer coupling tensors",
        ),
    },
    "upde.field_coupling": {
        "validation_protocol": "paper0.upde.field.global_phase_coupling",
        "implementation_status": "validation_spec_pending_direct_implementation_audit",
        "implementation_links": (
            "src/scpn_quantum_control/phase/pulse_shaping.py",
            "src/scpn_quantum_control/bridge/phase_artifact.py",
            "src/scpn_quantum_control/hardware/feedback_dryrun.py",
        ),
        "null_controls": (
            "set zeta_L to zero and require recovery of the no-field baseline",
            "randomise Theta_Psi across trials and require loss of coherent field alignment",
            "bound Psi_Global and reject non-finite field amplitudes",
        ),
        "executable_validation_targets": (
            "evaluate field-on versus field-off phase alignment at fixed K_nm",
            "track global-phase sensitivity with reproducible seeded perturbations",
            "verify field metadata is recorded before any hardware-facing plan is emitted",
        ),
    },
    "upde.natural_gradient": {
        "validation_protocol": "paper0.upde.natural_gradient.fim_free_energy",
        "implementation_status": "validation_spec_pending_direct_implementation_audit",
        "implementation_links": (
            "src/scpn_quantum_control/analysis/fim_hamiltonian.py",
            "src/scpn_quantum_control/analysis/adaptive_fim_feedback.py",
            "src/scpn_quantum_control/fep/variational_free_energy.py",
        ),
        "null_controls": (
            "replace g_F^{-1} by the identity and report Euclidean-gradient divergence",
            "provide a singular FIM and require explicit regularisation or fail-fast rejection",
            "provide a non-positive-definite metric and require validation failure",
        ),
        "executable_validation_targets": (
            "compare natural-gradient and Euclidean-gradient descent on the same free-energy fixture",
            "verify metric conditioning and regularisation metadata are preserved",
            "check finite-difference gradients against analytic free-energy gradients",
        ),
    },
    "upde.adaptive_coupling": {
        "validation_protocol": "paper0.upde.adaptive_coupling.quasicritical_controller",
        "implementation_status": "validation_spec_pending_direct_implementation_audit",
        "implementation_links": (
            "src/scpn_quantum_control/analysis/adaptive_fim_feedback.py",
            "src/scpn_quantum_control/control/realtime_feedback.py",
            "src/scpn_quantum_control/hardware/feedback_hardware_scheduler.py",
        ),
        "null_controls": (
            "set gamma_L and alpha_L to zero and require no adaptive movement",
            "invert feedback signs and require divergence or controlled failure",
            "reject unbounded gains, non-finite noise, and negative decay where unsupported",
        ),
        "executable_validation_targets": (
            "simulate bounded K_ij^L adaptation toward R_L^* under seeded perturbations",
            "track sigma_L convergence toward the quasicritical target",
            "verify scheduler outputs remain dry-run until hardware eligibility gates pass",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class UPDEValidationSpec:
    """Validation spec promoted from Paper 0 source anchors."""

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
    unmapped_anchor_math_ids: tuple[str, ...]
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
class UPDEValidationSpecBundle:
    """UPDE validation specs plus coverage summary."""

    specs: tuple[UPDEValidationSpec, ...]
    summary: dict[str, Any]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into dictionaries."""
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


def build_upde_validation_specs(anchor_records: list[dict[str, Any]]) -> UPDEValidationSpecBundle:
    """Build source-covered UPDE validation specs from the anchor queue."""
    records_by_key = {record.key: record for record in paper0_upde_records()}
    anchors_by_key: dict[str, list[dict[str, Any]]] = {key: [] for key in records_by_key}
    for anchor in anchor_records:
        for key in anchor.get("paper0_equation_record_keys", []):
            if key in anchors_by_key:
                anchors_by_key[key].append(anchor)

    specs: list[UPDEValidationSpec] = []
    consumed_ledger_ids: set[str] = set()
    for key, equation_record in records_by_key.items():
        metadata = _SPEC_METADATA[key]
        anchors = anchors_by_key[key]
        if not anchors:
            raise ValueError(f"missing Paper 0 UPDE anchor records for {key}")
        ledger_ids = tuple(sorted(str(anchor["ledger_id"]) for anchor in anchors))
        source_record_ids = tuple(sorted(str(anchor["source_record_id"]) for anchor in anchors))
        block_indices = tuple(sorted(int(anchor["source_block_index"]) for anchor in anchors))
        anchor_math_ids = tuple(
            sorted({str(math_id) for anchor in anchors for math_id in anchor.get("math_ids", [])})
        )
        mapped = set(equation_record.source_equation_ids)
        unmapped = tuple(math_id for math_id in anchor_math_ids if math_id not in mapped)
        consumed_ledger_ids.update(ledger_ids)
        specs.append(
            UPDEValidationSpec(
                key=key,
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript=equation_record.manuscript,
                section_path=equation_record.section_path,
                canonical_latex=equation_record.canonical_latex,
                source_equation_ids=equation_record.source_equation_ids,
                source_ledger_ids=ledger_ids,
                source_record_ids=source_record_ids,
                source_block_indices=block_indices,
                anchor_math_ids=anchor_math_ids,
                unmapped_anchor_math_ids=unmapped,
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

    anchor_ledger_ids = {str(anchor["ledger_id"]) for anchor in anchor_records}
    unconsumed = sorted(anchor_ledger_ids - consumed_ledger_ids)
    summary = {
        "anchor_record_count": len(anchor_records),
        "consumed_anchor_record_count": len(consumed_ledger_ids),
        "coverage_match": not unconsumed and len(consumed_ledger_ids) == len(anchor_ledger_ids),
        "unconsumed_anchor_ledger_ids": unconsumed,
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "all_specs_have_null_controls": all(bool(spec.null_controls) for spec in specs),
        "all_specs_have_executable_targets": all(
            bool(spec.executable_validation_targets) for spec in specs
        ),
        "all_specs_are_source_anchored": all(bool(spec.source_ledger_ids) for spec in specs),
        "hardware_status": "simulator_only_no_provider_submission",
        "policy": (
            "UPDE anchors are promoted only into validation specifications. "
            "Provider submission remains out of scope until simulator fixtures, "
            "controls, and implementation audits pass."
        ),
    }
    if not summary["coverage_match"]:
        raise ValueError(f"unconsumed UPDE anchor records: {unconsumed}")
    return UPDEValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: UPDEValidationSpecBundle) -> str:
    """Render a concise internal report for the UPDE validation specs."""
    status = "match" if bundle.summary["coverage_match"] else "mismatch"
    lines = [
        "# Paper 0 UPDE Validation Specs",
        "",
        f"- Anchor records: `{bundle.summary['anchor_record_count']}`",
        f"- Consumed anchor records: `{bundle.summary['consumed_anchor_record_count']}`",
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
                f"- Source ledgers: `{', '.join(spec.source_ledger_ids)}`",
                f"- Source equations: `{', '.join(spec.source_equation_ids)}`",
                f"- Implementation status: `{spec.implementation_status}`",
                f"- Null controls: `{len(spec.null_controls)}`",
                f"- Executable targets: `{len(spec.executable_validation_targets)}`",
                "",
            ]
        )
    lines.extend(["## Policy", "", bundle.summary["policy"], ""])
    return "\n".join(lines)


def write_outputs(
    bundle: UPDEValidationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write validation specs, summary, and report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_dir / f"paper0_upde_validation_specs_{date_tag}.json"
    summary_path = output_dir / f"paper0_upde_validation_specs_summary_{date_tag}.json"
    report_path = output_dir / f"paper0_upde_validation_specs_report_{date_tag}.md"

    serialised = {
        "summary": bundle.summary,
        "specs": [asdict(spec) for spec in bundle.specs],
    }
    spec_path.write_text(json.dumps(serialised, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(bundle.summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(build_validation_report(bundle), encoding="utf-8")
    return {"specs": spec_path, "summary": summary_path, "report": report_path}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, default=DEFAULT_ANCHOR_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args(argv)

    bundle = build_upde_validation_specs(load_jsonl(args.anchors))
    paths = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
