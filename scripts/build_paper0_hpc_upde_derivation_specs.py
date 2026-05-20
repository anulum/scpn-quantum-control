#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 HPC-UPDE derivation spec builder
"""Promote Paper 0 HPC-UPDE derivation records into validation specs."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(6615, 6646))
STRUCTURAL_SOURCE_LEDGER_IDS = (
    "P0R06615",
    "P0R06616",
    "P0R06617",
    "P0R06618",
    "P0R06621",
    "P0R06622",
    "P0R06627",
    "P0R06629",
    "P0R06632",
    "P0R06636",
    "P0R06645",
)
EQUATION_SOURCE_LEDGER_IDS = (
    "P0R06624",
    "P0R06628",
    "P0R06630",
    "P0R06631",
    "P0R06633",
    "P0R06634",
    "P0R06642",
    "P0R06644",
)

CLAIM_BOUNDARY = (
    "source-bounded HPC-UPDE mathematical bridge simulator contract; not empirical evidence"
)

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "hpc_upde_derivation.block_framing": {
        "validation_protocol": "paper0.hpc_upde_derivation.block_framing",
        "canonical_statement": (
            "The insertion block frames UPDE as a mathematical bridge from "
            "hierarchical predictive coding to free-energy gradient descent."
        ),
        "source_equation_ids": (),
        "source_formulae": (),
        "source_mechanisms": (
            "advanced integration mechanisms block",
            "HPC-UPDE mathematical bridge full derivation",
            "formal proof of UPDE as gradient descent on Free Energy",
        ),
        "variables": ("UPDE", "HPC", "Free_Energy", "theta_i"),
        "validation_targets": (
            "preserve source-bounded theorem framing",
            "preserve proof-step ordering",
            "reject simulator output as empirical active-inference evidence",
        ),
        "null_controls": (
            "missing-free-energy-functional control must be rejected",
            "missing-gradient-step control must be rejected",
            "unsupported-empirical-active-inference control must be rejected",
        ),
    },
    "hpc_upde_derivation.free_energy_functional": {
        "validation_protocol": "paper0.hpc_upde_derivation.free_energy_functional",
        "canonical_statement": (
            "For a hierarchical oscillatory system, Paper 0 defines a cosine "
            "phase-coherence free-energy functional whose minima occur at zero "
            "phase differences."
        ),
        "source_equation_ids": ("P0R06624:F",),
        "source_formulae": (
            "F(theta_1,...,theta_N) = -sum_{i,j} K_ij cos(theta_j - theta_i)",
            "minima occur at zero phase differences; perfect synchrony = zero prediction error",
        ),
        "source_mechanisms": (
            "free energy functional is defined over hierarchical oscillatory phases",
            "zero phase differences minimise the source functional under positive couplings",
            "perfect synchrony is mapped to zero prediction error",
        ),
        "variables": ("theta_i", "theta_j", "K_ij", "F"),
        "validation_targets": (
            "compute finite phase free energy",
            "confirm synchrony gives lower energy than spread phases for positive couplings",
            "reject non-square coupling matrices",
        ),
        "null_controls": (
            "non-square-K control must be rejected",
            "non-finite-theta control must be rejected",
            "negative-evidence-overclaim control must be rejected",
        ),
    },
    "hpc_upde_derivation.gradient_descent": {
        "validation_protocol": "paper0.hpc_upde_derivation.gradient_descent",
        "canonical_statement": (
            "Paper 0 derives UPDE drift from negative free-energy gradient flow "
            "using the sine phase-error term."
        ),
        "source_equation_ids": (
            "P0R06628:gradient_flow",
            "P0R06630:gradient",
            "P0R06631:sine_error",
        ),
        "source_formulae": (
            "d theta_i / dt = -partial F / partial theta_i",
            "partial F / partial theta_i = -partial/partial theta_i[-sum_j K_ij cos(theta_j - theta_i)]",
            "partial F / partial theta_i = -sum_j K_ij sin(theta_j - theta_i)",
        ),
        "source_mechanisms": (
            "system evolves to minimise F",
            "sine phase difference is the local phase-error term",
            "negative gradient flow produces the coupling drift term",
        ),
        "variables": ("theta_i", "K_ij", "partial_F_partial_theta_i"),
        "validation_targets": (
            "match analytic gradient to finite-difference gradient",
            "preserve sine-error sign convention",
            "reject invalid finite-difference step",
        ),
        "null_controls": (
            "invalid-step control must be rejected",
            "shape-mismatch control must be rejected",
            "sign-flipped-gradient control must be rejected",
        ),
    },
    "hpc_upde_derivation.upde_core_equation": {
        "validation_protocol": "paper0.hpc_upde_derivation.upde_core_equation",
        "canonical_statement": (
            "Adding intrinsic frequency and stochastic exploration yields the "
            "source UPDE core equation."
        ),
        "source_equation_ids": ("P0R06633:stochastic_gradient_flow", "P0R06634:upde_core"),
        "source_formulae": (
            "d theta_i / dt = omega_i - partial F / partial theta_i + eta_i(t)",
            "d theta_i / dt = omega_i + sum_j K_ij sin(theta_j - theta_i) + eta_i(t)",
        ),
        "source_mechanisms": (
            "intrinsic dynamics omega_i are added to gradient flow",
            "eta_i(t) is added as stochastic exploration",
            "the resulting expression is the UPDE core equation",
        ),
        "variables": ("theta_i", "omega_i", "eta_i", "K_ij"),
        "validation_targets": (
            "compute UPDE derivative from omega minus gradient plus eta",
            "match source-equivalent sine-sum equation",
            "reject omega/theta shape mismatches",
        ),
        "null_controls": (
            "theta-omega-shape-mismatch control must be rejected",
            "eta-shape-mismatch control must be rejected",
            "non-finite-coupling control must be rejected",
        ),
    },
    "hpc_upde_derivation.hpc_interpretation": {
        "validation_protocol": "paper0.hpc_upde_derivation.hpc_interpretation",
        "canonical_statement": (
            "The source maps free energy, cosine coherence, sine phase error, "
            "coupling weights, and noise to HPC active-inference roles."
        ),
        "source_equation_ids": (),
        "source_formulae": (),
        "source_mechanisms": (
            "F is variational Free Energy / surprise",
            "cos(theta_j - theta_i) is phase coherence / prediction accuracy",
            "sin(theta_j - theta_i) is phase error / prediction error epsilon",
            "K_ij is precision weighting / inverse variance of prediction",
            "eta_i is stochastic exploration / sampling",
        ),
        "variables": ("F", "cos_delta_theta", "sin_delta_theta", "K_ij", "eta_i"),
        "validation_targets": (
            "preserve all five HPC interpretation mappings",
            "label K_ij as precision weighting without treating it as measured variance",
            "label eta_i as stochastic exploration without sampling-data overclaim",
        ),
        "null_controls": (
            "missing-precision-mapping control must be rejected",
            "missing-stochastic-exploration control must be rejected",
            "empirical-HPC-claim control must be rejected",
        ),
    },
    "hpc_upde_derivation.active_inference_boundary": {
        "validation_protocol": "paper0.hpc_upde_derivation.active_inference_boundary",
        "canonical_statement": (
            "The source corollary equates phase locking with prediction-error and "
            "free-energy minimisation, then frames UPDE dynamics as a physical "
            "substrate for active inference."
        ),
        "source_equation_ids": (
            "P0R06642:phase_locking_corollary",
            "P0R06644:active_inference_boundary",
        ),
        "source_formulae": (
            "phase-locking sin Delta theta -> 0 equals prediction error minimization and Free Energy minimization",
            "UPDE dynamics = physical substrate for Active Inference",
        ),
        "source_mechanisms": (
            "phase locking drives sine phase-error terms toward zero",
            "prediction error minimisation is identified with free-energy minimisation",
            "active-inference substrate status remains a source-bounded theoretical claim",
        ),
        "variables": ("Delta_theta", "prediction_error", "Free_Energy", "Active_Inference"),
        "validation_targets": (
            "compute bounded phase-locking error",
            "confirm locked phases minimise the phase-locking error metric",
            "reject simulator-only results as empirical active-inference validation",
        ),
        "null_controls": (
            "non-finite-phase control must be rejected",
            "unsupported-active-inference-evidence control must be rejected",
            "missing-phase-locking-corollary control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class HPCUPDEDerivationValidationSpec:
    """Validation spec promoted from Paper 0 HPC-UPDE derivation records."""

    key: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_statement: str
    source_equation_ids: tuple[str, ...]
    source_formulae: tuple[str, ...]
    source_mechanisms: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    source_block_indices: tuple[int, ...]
    anchor_math_ids: tuple[str, ...]
    structural_source_ledger_ids: tuple[str, ...]
    variables: tuple[str, ...]
    validation_targets: tuple[str, ...]
    executable_validation_targets: tuple[str, ...]
    null_controls: tuple[str, ...]
    claim_boundary: str
    implementation_status: str
    domain_review_status: str
    hardware_status: str


@dataclass(frozen=True, slots=True)
class HPCUPDEDerivationValidationSpecBundle:
    """HPC-UPDE derivation validation specs plus coverage summary."""

    specs: tuple[HPCUPDEDerivationValidationSpec, ...]
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


def build_hpc_upde_derivation_specs(
    source_records: list[dict[str, Any]],
) -> HPCUPDEDerivationValidationSpecBundle:
    """Build source-covered specs for the HPC-UPDE derivation block."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[HPCUPDEDerivationValidationSpec] = []
    for key, content in SPEC_CONTENT.items():
        specs.append(
            HPCUPDEDerivationValidationSpec(
                key=key,
                validation_protocol=str(content["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(content["canonical_statement"]),
                source_equation_ids=tuple(content["source_equation_ids"]),
                source_formulae=tuple(content["source_formulae"]),
                source_mechanisms=tuple(content["source_mechanisms"]),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(record["source_record_id"]) for record in anchors),
                source_block_indices=tuple(
                    int(record["source_block_index"]) for record in anchors
                ),
                anchor_math_ids=tuple(
                    math_id for record in anchors for math_id in tuple(record.get("math_ids", ()))
                ),
                structural_source_ledger_ids=STRUCTURAL_SOURCE_LEDGER_IDS,
                variables=tuple(content["variables"]),
                validation_targets=tuple(content["validation_targets"]),
                executable_validation_targets=tuple(content["validation_targets"]),
                null_controls=tuple(content["null_controls"]),
                claim_boundary=CLAIM_BOUNDARY,
                implementation_status="implemented",
                domain_review_status="source_promoted_requires_empirical_review",
                hardware_status="simulator_only_no_provider_submission",
            )
        )

    summary: dict[str, Any] = {
        "title": "Paper 0 HPC-UPDE Mathematical Bridge Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(anchors),
        "coverage_match": len(anchors) == len(SOURCE_LEDGER_IDS),
        "structural_source_ledger_ids": list(STRUCTURAL_SOURCE_LEDGER_IDS),
        "equation_source_ledger_ids": list(EQUATION_SOURCE_LEDGER_IDS),
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "hardware_status": "simulator_only_no_provider_submission",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return HPCUPDEDerivationValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: HPCUPDEDerivationValidationSpecBundle) -> str:
    """Render a compact Markdown report for human audit."""
    lines = [
        "# Paper 0 HPC-UPDE Mathematical Bridge Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records consumed: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
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
                "",
                spec.canonical_statement,
                "",
                "Formulae:",
                *[f"- {formula}" for formula in spec.source_formulae],
                "",
                "Mechanisms:",
                *[f"- {mechanism}" for mechanism in spec.source_mechanisms],
                "",
                "Null controls:",
                *[f"- {control}" for control in spec.null_controls],
            ]
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    *,
    bundle: HPCUPDEDerivationValidationSpecBundle,
    output_path: Path,
    report_path: Path,
) -> None:
    """Write JSON and Markdown artefacts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "summary": bundle.summary,
                "specs": [asdict(spec) for spec in bundle.specs],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path.write_text(build_validation_report(bundle), encoding="utf-8")


def main() -> int:
    """Build the default HPC-UPDE derivation validation-spec artefacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXTRACTION_DIR
        / "paper0_hpc_upde_derivation_validation_specs_2026-05-13.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_EXTRACTION_DIR
        / "paper0_hpc_upde_derivation_validation_specs_report_2026-05-13.md",
    )
    args = parser.parse_args()

    bundle = build_hpc_upde_derivation_specs(load_jsonl(args.ledger))
    write_outputs(bundle=bundle, output_path=args.output, report_path=args.report)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
