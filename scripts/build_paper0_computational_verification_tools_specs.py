#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 computational verification tools spec builder
"""Promote Paper 0 computational verification-tool records into specs."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(7006, 7073))
CLAIM_BOUNDARY = "source-bounded computational protocol; not empirical execution evidence"
HARDWARE_STATUS = "computational_protocol_no_claimed_execution"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "computational_verification_tools.chapter_boundary": {
        "tool_id": "chapter",
        "validation_protocol": "paper0.computational_verification_tools.chapter_boundary",
        "canonical_statement": "The section defines computational verification tools, not completed executions.",
        "source_equation_ids": (
            "P0R07006:section_boundary",
            "P0R07007:lattice_hmc_tool",
            "P0R07035:class_patch_tool",
            "P0R07052:lambda_eff_tool",
        ),
        "source_formulae": ("three computational verification tools are enumerated",),
        "test_protocols": ("preserve tool catalogue and no-execution boundary",),
        "null_results": ("tool availability is not evidence for any physical claim",),
        "variables": ("tool_id", "source_span", "execution_boundary"),
        "validation_targets": (
            "preserve computational verification tools section boundary",
            "preserve three-tool catalogue",
            "reject empirical-execution claims without external run artefacts",
        ),
        "null_controls": (
            "missing-tool control must be rejected",
            "execution-overclaim control must be rejected",
            "missing-source-span control must be rejected",
        ),
    },
    "computational_verification_tools.lattice_hmc_flat_line": {
        "tool_id": "23.1",
        "validation_protocol": "paper0.computational_verification_tools.lattice_hmc_flat_line",
        "canonical_statement": "A lattice Higgs-Yukawa polar-form protocol is specified for a non-perturbative flat-line check.",
        "source_equation_ids": (
            "P0R07008:polar_higgs_yukawa_field",
            "P0R07009:lattice_action",
            "P0R07015:pcac_mass_relation",
            "P0R07017:mass_ratio_target",
            "P0R07028:quenched_boundary",
        ),
        "source_formulae": (
            "Sigma_x = (1/sqrt(2))(v + rho_x) exp(i Phi_x/v)",
            "S_lat = sum_x[radial gradient + Goldstone gradient + lambda4 potential + Yukawa term]",
            "m_Psi = y v / sqrt(2)",
            "M_rho / m_Psi -> sqrt(2) on the tuned line",
        ),
        "test_protocols": (
            "check y^2 = lambda4/2 tuning",
            "evaluate radial and Goldstone action terms on finite periodic lattices",
            "label the source skeleton as a quenched flat-line test when fermions are omitted",
            "separate warm-up L=12 from decorrelation target L>=16",
        ),
        "null_results": (
            "detuned starts drift off the diagonal beta-flow",
            "quenched skeleton cannot validate fermionic PCAC dynamics alone",
        ),
        "variables": ("rho", "Phi", "v", "lambda4", "y", "a", "L", "Ntraj"),
        "validation_targets": (
            "preserve polar Higgs-Yukawa field definition",
            "preserve source-stated lattice action terms",
            "preserve PCAC mass and mass-ratio targets",
            "preserve quenched-execution boundary",
        ),
        "null_controls": (
            "fermion-omission-overclaim control must be rejected",
            "detuned-coupling control must be rejected",
            "single-axis-gradient control must be rejected",
        ),
    },
    "computational_verification_tools.class_goldstone_eos": {
        "tool_id": "23.2",
        "validation_protocol": "paper0.computational_verification_tools.class_goldstone_eos",
        "canonical_statement": "A CLASS background patch is specified for Goldstone oscillatory early dark energy.",
        "source_equation_ids": (
            "P0R07037:oscillatory_equation_of_state",
            "P0R07040:rho_phi_patch",
            "P0R07041:pressure_patch",
            "P0R07043:class_parameters",
            "P0R07051:planck_washout_boundary",
        ),
        "source_formulae": (
            "w(a) = -1 + eps cos(omega_log ln(a) + phi)",
            "rho_phi = rho_phi0 a^-3 (1 + eps cos(omega_log ln(a) + phi0))",
            "p_phi = w(a) rho_phi",
        ),
        "test_protocols": (
            "preserve eps_phi, omega_log_phi, and phi0_phi parameter interface",
            "verify positive scale-factor domain",
            "preserve small-epsilon and high-frequency washout boundary",
        ),
        "null_results": (
            "small eps <= 1e-3 and omega_log >= 500 wash out consistently with Planck",
            "large non-washed oscillations raise chi-squared in cosmology fits",
        ),
        "variables": ("a", "eps_phi", "omega_log_phi", "phi0_phi", "rho_phi0"),
        "validation_targets": (
            "preserve oscillatory equation of state",
            "preserve CLASS background density and pressure mapping",
            "preserve Planck washout boundary",
        ),
        "null_controls": (
            "non-positive-scale-factor control must be rejected",
            "missing-CLASS-parameter control must be rejected",
            "washout-overclaim control must be rejected",
        ),
    },
    "computational_verification_tools.lambda_eff_utility": {
        "tool_id": "23.3",
        "validation_protocol": "paper0.computational_verification_tools.lambda_eff_utility",
        "canonical_statement": "A dark-energy-matched Lambda_eff utility is specified for CAMB/CLASS pipelines.",
        "source_equation_ids": (
            "P0R07056:lambda_psi_g",
            "P0R07058:natural_units",
            "P0R07059:canonical_lambda_0",
            "P0R07062:reduced_planck_mass",
            "P0R07067:lambda_psi_density",
            "P0R07070:lambda_eff",
        ),
        "source_formulae": (
            "LAMBDA_PSI_G = 1.068935e-122",
            "M_PL = 2.435e18 GeV",
            "LAMBDA_0 = 1.1056e-52 GeV^4",
            "rho_Lambda_psi(t) = LAMBDA_PSI_G psi(t)^2 M_PL^2",
            "Lambda_eff(t) = LAMBDA_0 + rho_Lambda_psi(t)",
        ),
        "test_protocols": (
            "preserve natural-unit convention",
            "preserve GeV^4 energy-density output",
            "reject parameter sets with inconsistent constants or units",
        ),
        "null_results": (
            "Lambda_eff utility output alone is not a cosmological fit",
            "unit mismatch invalidates CAMB/CLASS handoff",
        ),
        "variables": ("psi_t", "M_PL", "LAMBDA_0", "LAMBDA_PSI_G"),
        "validation_targets": (
            "preserve dark-energy-matched coupling value",
            "preserve reduced Planck mass value",
            "preserve Lambda_eff additive form",
        ),
        "null_controls": (
            "lambda-units-mismatch control must be rejected",
            "missing-bare-Lambda control must be rejected",
            "negative-density-overclaim control must be rejected",
        ),
    },
    "computational_verification_tools.execution_boundaries": {
        "tool_id": "execution",
        "validation_protocol": "paper0.computational_verification_tools.execution_boundaries",
        "canonical_statement": "Skeleton code and patches are promoted as executable boundary fixtures, not as completed HMC, CLASS, CAMB, or Planck runs.",
        "source_equation_ids": (
            "P0R07018:skeleton_code_boundary",
            "P0R07034:quick_2d_experiment_note",
            "P0R07038:class_background_patch_boundary",
            "P0R07057:camb_class_utility_boundary",
        ),
        "source_formulae": ("source code snippets require independent execution artefacts",),
        "test_protocols": (
            "require run artefacts before empirical status changes",
            "label fast 2D experiments separately from 4D decorrelation target",
            "label utility calculations separately from external solver runs",
        ),
        "null_results": (
            "without run artefacts the section remains protocol-only",
            "without external solver output no cosmological-data claim is supported",
        ),
        "variables": ("run_artifact", "solver_output", "claim_status"),
        "validation_targets": (
            "preserve no-execution boundary",
            "preserve distinction between protocol snippets and run evidence",
            "reject claiming validation from source snippets alone",
        ),
        "null_controls": (
            "source-snippet-as-result control must be rejected",
            "missing-run-artifact control must be rejected",
            "2d-warmup-as-4d-result control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class ComputationalVerificationToolsSpec:
    """Computational verification-tool spec promoted from Paper 0 records."""

    key: str
    tool_id: str
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
class ComputationalVerificationToolsSpecBundle:
    """Computational verification-tool specs plus coverage summary."""

    specs: tuple[ComputationalVerificationToolsSpec, ...]
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


def build_computational_verification_tools_specs(
    source_records: list[dict[str, Any]],
) -> ComputationalVerificationToolsSpecBundle:
    """Build source-covered computational verification-tool specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[ComputationalVerificationToolsSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ComputationalVerificationToolsSpec(
                key=key,
                tool_id=str(metadata["tool_id"]),
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
        "title": "Paper 0 Computational Verification Tools Specs",
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(SOURCE_LEDGER_IDS),
        "coverage_match": True,
        "unconsumed_source_ledger_ids": [],
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "spec_count": len(specs),
        "tool_count": 3,
        "spec_keys": [spec.key for spec in specs],
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "all_specs_have_null_controls": all(bool(spec.null_controls) for spec in specs),
        "all_specs_have_null_results": all(bool(spec.null_results) for spec in specs),
        "all_specs_are_source_anchored": all(
            spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in specs
        ),
    }
    return ComputationalVerificationToolsSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: ComputationalVerificationToolsSpecBundle) -> str:
    """Render a compact Markdown report for internal review."""
    lines = [
        "# Paper 0 Computational Verification Tools Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records consumed: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Tool count: {bundle.summary['tool_count']}",
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
                f"- Tool: {spec.tool_id}",
                f"- Protocol: {spec.validation_protocol}",
                f"- Statement: {spec.canonical_statement}",
                f"- Null results: {len(spec.null_results)}",
                f"- Null controls: {len(spec.null_controls)}",
            ]
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    bundle: ComputationalVerificationToolsSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the computational verification-tool specs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_computational_verification_tools_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_computational_verification_tools_validation_specs_report_{date_tag}.md"
    )
    payload = {"specs": [asdict(spec) for spec in bundle.specs], "summary": bundle.summary}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_validation_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> int:
    """Build computational verification-tool specs from the canonical review ledger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args()

    bundle = build_computational_verification_tools_specs(load_jsonl(args.ledger))
    write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
