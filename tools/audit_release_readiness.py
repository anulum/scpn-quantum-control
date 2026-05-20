# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — release readiness audit helper
"""Compose release-blocker gates into one deterministic tag-readiness audit."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audit_coverage_gaps import audit_coverage_gaps, load_justified_exclusions
from audit_test_behaviour import audit_test_tree, evaluate_quality_gate

VERSION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("pyproject.toml", re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)),
    (
        "src/scpn_quantum_control/__init__.py",
        re.compile(r'^__version__\s*=\s*"([^"]+)"', re.MULTILINE),
    ),
    ("CITATION.cff", re.compile(r'^version:\s*"([^"]+)"', re.MULTILINE)),
    (".zenodo.json", re.compile(r'"version":\s*"([^"]+)"')),
)

REQUIRED_RELEASE_ARTIFACTS: tuple[str, ...] = (
    "docs/paper0/paper0_validation_register.md",
    "docs/paper0/paper0_experimental_pathway.md",
    "docs/paper0/paper0_knm_measured_coupling_evidence_checklist.md",
    "data/knm_physical_validation/eeg_alpha_plv_knm_comparison.json",
    "data/knm_physical_validation/power_grid_ieee5bus_knm_comparison.json",
    "data/knm_physical_validation/measured_couplings_power_grid_ieee14bus.json",
    "data/knm_physical_validation/power_grid_ieee14bus_knm_comparison.json",
    "data/s2_advantage_scaling/s2_scaling_protocol_2026-05-06.json",
    "data/s2_advantage_scaling/s2_slice_progress_report_2026-05-07.json",
    "data/s2_advantage_scaling/s2_scaling_claim_boundary_2026-05-06.json",
    "data/hardware_result_packs/manifest.json",
    "docs/hardware_result_packs.md",
    "docs/hardware_result_pack_release_checklist.md",
    "data/synchronisation_benchmarks/synchronisation_benchmark_registry.json",
    "data/synchronisation_benchmarks/kuramoto_ring_n4_linear_omega_reference_rows.json",
    "data/synchronisation_benchmarks/kuramoto_chain_n8_decay_omega_reference_rows.json",
    "data/stable_core/backend_capability_matrix.json",
    "data/s7_logical_dla_parity/logical_dla_parity_roadmap_2026-05-20.json",
    "data/s8_adaptive_branching/adaptive_branching_readiness_2026-05-20.json",
    "data/s9_quantum_thermo/quantum_thermo_readiness_2026-05-20.json",
    "data/s10_analog_native/analog_native_readiness_2026-05-20.json",
    "data/s11_quantum_sensing/quantum_sensing_readiness_2026-05-20.json",
    "docs/synchronisation_benchmark_suite.md",
    "docs/synchronisation_benchmark_kuramoto_ring_n4.md",
    "docs/synchronisation_benchmark_kuramoto_chain_n8.md",
    "docs/stable_core_backend_capability_matrix.md",
    "docs/logical_dla_parity.md",
    "docs/adaptive_branching.md",
    "docs/quantum_thermo.md",
    "docs/analog_native_readiness.md",
    "docs/quantum_sensing.md",
)


@dataclass(frozen=True)
class ReleaseCheck:
    """One release-readiness check."""

    name: str
    valid: bool
    details: dict[str, Any]
    blockers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialise the check."""
        return {
            "name": self.name,
            "valid": self.valid,
            "details": self.details,
            "blockers": list(self.blockers),
        }


def _read_version(path: Path, pattern: re.Pattern[str]) -> tuple[str | None, str | None]:
    """Return a version string or blocker for one carrier file."""
    if not path.exists():
        return None, f"{path.as_posix()}: file not found"
    match = pattern.search(path.read_text(encoding="utf-8"))
    if match is None:
        return None, f"{path.as_posix()}: version pattern not found"
    return match.group(1), None


def check_versions(project_root: Path) -> ReleaseCheck:
    """Check that version carrier files agree."""
    versions: dict[str, str | None] = {}
    blockers: list[str] = []
    canonical: str | None = None
    for rel_path, pattern in VERSION_PATTERNS:
        version, blocker = _read_version(project_root / rel_path, pattern)
        versions[rel_path] = version
        if blocker is not None:
            blockers.append(blocker)
            continue
        if canonical is None:
            canonical = version
        elif version != canonical:
            blockers.append(f"{rel_path}: {version} does not match {canonical}")
    return ReleaseCheck(
        name="version_consistency",
        valid=not blockers,
        details={"canonical_version": canonical, "versions": versions},
        blockers=tuple(blockers),
    )


def check_required_artifacts(project_root: Path) -> ReleaseCheck:
    """Check that release-blocker evidence artefacts are present."""
    missing = tuple(
        rel_path
        for rel_path in REQUIRED_RELEASE_ARTIFACTS
        if not (project_root / rel_path).exists()
    )
    return ReleaseCheck(
        name="required_release_artifacts",
        valid=not missing,
        details={"required": list(REQUIRED_RELEASE_ARTIFACTS), "missing": list(missing)},
        blockers=tuple(f"missing required release artefact: {path}" for path in missing),
    )


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest for a file."""

    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_hardware_result_pack_evidence(
    project_root: Path, evidence_path: Path | None
) -> ReleaseCheck:
    """Check optional hardware result-pack evidence packet for release claims."""

    if evidence_path is None:
        return ReleaseCheck(
            name="hardware_result_pack_evidence",
            valid=True,
            details={
                "evidence_path": None,
                "hardware_evidence_cited": None,
                "requirement": "Provide --hardware-result-pack-evidence when release surfaces cite promoted IBM hardware evidence.",
            },
        )
    if not evidence_path.exists():
        return ReleaseCheck(
            name="hardware_result_pack_evidence",
            valid=False,
            details={"evidence_path": evidence_path.as_posix()},
            blockers=(f"hardware result-pack evidence packet missing: {evidence_path}",),
        )

    packet = json.loads(evidence_path.read_text(encoding="utf-8"))
    blockers: list[str] = []
    if packet.get("schema_version") != 1:
        blockers.append("hardware result-pack evidence schema_version must be 1")
    cited = packet.get("hardware_evidence_cited")
    if not isinstance(cited, bool):
        blockers.append("hardware_evidence_cited must be boolean")
    if cited is False:
        if not packet.get("reason"):
            blockers.append("non-citing hardware evidence packet must include reason")
        return ReleaseCheck(
            name="hardware_result_pack_evidence",
            valid=not blockers,
            details={
                "evidence_path": evidence_path.as_posix(),
                "hardware_evidence_cited": False,
                "reason": packet.get("reason"),
            },
            blockers=tuple(blockers),
        )

    verifier_rel = packet.get("verifier_summary_path")
    export_rel = packet.get("export_summary_path")
    reproduction_logs = packet.get("reproduction_logs")
    if not isinstance(verifier_rel, str):
        blockers.append("verifier_summary_path is required when hardware evidence is cited")
    if not isinstance(export_rel, str):
        blockers.append("export_summary_path is required when hardware evidence is cited")
    if not isinstance(reproduction_logs, list) or not reproduction_logs:
        blockers.append("reproduction_logs must contain at least one log entry")

    verifier_summary: dict[str, Any] = {}
    export_summary: dict[str, Any] = {}
    if isinstance(verifier_rel, str):
        verifier_path = project_root / verifier_rel
        if not verifier_path.exists():
            blockers.append(f"verifier summary missing: {verifier_rel}")
        else:
            verifier_summary = json.loads(verifier_path.read_text(encoding="utf-8"))
            if int(verifier_summary.get("pack_count", 0)) <= 0:
                blockers.append("verifier summary must report at least one pack")
    if isinstance(export_rel, str):
        export_path = project_root / export_rel
        if not export_path.exists():
            blockers.append(f"export summary missing: {export_rel}")
        else:
            export_summary = json.loads(export_path.read_text(encoding="utf-8"))
            if not export_summary.get("exports"):
                blockers.append("export summary must include exports")

    cited_pack_ids: set[str] = set()
    if isinstance(reproduction_logs, list):
        for index, item in enumerate(reproduction_logs):
            if not isinstance(item, dict):
                blockers.append(f"reproduction_logs[{index}] must be an object")
                continue
            pack_id = item.get("pack_id")
            command = item.get("command")
            log_rel = item.get("log_path")
            digest = item.get("sha256")
            if not isinstance(pack_id, str) or not pack_id:
                blockers.append(f"reproduction_logs[{index}].pack_id is required")
            else:
                cited_pack_ids.add(pack_id)
            if not isinstance(command, str) or not command:
                blockers.append(f"reproduction_logs[{index}].command is required")
            if not isinstance(log_rel, str):
                blockers.append(f"reproduction_logs[{index}].log_path is required")
                continue
            log_path = project_root / log_rel
            if not log_path.exists():
                blockers.append(f"reproduction log missing: {log_rel}")
                continue
            if not isinstance(digest, str) or _sha256(log_path) != digest:
                blockers.append(f"reproduction log digest mismatch: {log_rel}")

    export_pack_ids = {
        str(item.get("id")) for item in export_summary.get("exports", []) if isinstance(item, dict)
    }
    missing_exports = sorted(cited_pack_ids - export_pack_ids)
    if missing_exports:
        blockers.append(f"cited packs missing export digests: {missing_exports}")

    return ReleaseCheck(
        name="hardware_result_pack_evidence",
        valid=not blockers,
        details={
            "evidence_path": evidence_path.as_posix(),
            "hardware_evidence_cited": cited,
            "cited_pack_ids": sorted(cited_pack_ids),
            "verifier_pack_count": verifier_summary.get("pack_count"),
            "export_count": len(export_summary.get("exports", [])) if export_summary else 0,
            "reproduction_log_count": len(reproduction_logs)
            if isinstance(reproduction_logs, list)
            else 0,
        },
        blockers=tuple(blockers),
    )


def check_coverage_gate(
    *,
    project_root: Path,
    coverage_xml: Path,
    min_file_percent: float,
    min_aggregate_percent: float,
    justified_exclusions: Path | None,
    fail_on_file_gap: bool,
) -> ReleaseCheck:
    """Check release coverage gate from a coverage.py XML report."""
    if not coverage_xml.exists():
        return ReleaseCheck(
            name="coverage_gap_gate",
            valid=False,
            details={
                "coverage_xml": coverage_xml.as_posix(),
                "min_file_percent": min_file_percent,
                "min_aggregate_percent": min_aggregate_percent,
            },
            blockers=(f"coverage XML report missing: {coverage_xml.as_posix()}",),
        )
    audits = audit_coverage_gaps(
        project_root=project_root,
        source_root=project_root / "src" / "scpn_quantum_control",
        coverage_xml=coverage_xml,
        min_file_percent=min_file_percent,
        justified_exclusions=load_justified_exclusions(justified_exclusions),
    )
    gaps = tuple(item for item in audits if item.is_gap)
    missing = tuple(item for item in gaps if item.status == "missing_from_report")
    low = tuple(item for item in gaps if item.status == "below_threshold")
    measured = tuple(
        item
        for item in audits
        if item.valid_lines is not None and item.status != "justified_exclusion"
    )
    covered_lines = sum(item.covered_lines or 0 for item in measured)
    valid_lines = sum(item.valid_lines or 0 for item in measured)
    aggregate = 100.0 * covered_lines / valid_lines if valid_lines else None
    blockers: list[str] = []
    if aggregate is None:
        blockers.append("coverage XML does not contain measured package source lines")
    elif aggregate < min_aggregate_percent:
        blockers.append(
            f"aggregate coverage {aggregate:.2f} below minimum {min_aggregate_percent:.2f}"
        )
    blockers.extend(f"{item.path}: {item.status}" for item in missing)
    if fail_on_file_gap:
        blockers.extend(f"{item.path}: {item.status}" for item in low)
    return ReleaseCheck(
        name="coverage_gap_gate",
        valid=not blockers,
        details={
            "coverage_xml": coverage_xml.as_posix(),
            "source_files": len(audits),
            "gap_files": len(gaps),
            "below_threshold_files": len(low),
            "missing_from_report_files": len(missing),
            "aggregate_line_percent_in_report": aggregate,
            "min_aggregate_percent": min_aggregate_percent,
            "min_file_percent": min_file_percent,
            "fail_on_file_gap": fail_on_file_gap,
        },
        blockers=tuple(blockers[:50]),
    )


def check_behaviour_gate(
    *,
    tests_root: Path,
    min_assertion_density: float,
    min_raises_contract_density: float,
) -> ReleaseCheck:
    """Check behavioural-test density gate."""
    audits = audit_test_tree(tests_root)
    gate = evaluate_quality_gate(
        audits,
        min_assertion_density=min_assertion_density,
        min_raises_contract_density=min_raises_contract_density,
    )
    return ReleaseCheck(
        name="behavioural_quality_gate",
        valid=gate.valid,
        details=gate.to_dict(),
        blockers=gate.blockers,
    )


def audit_release_readiness(
    *,
    project_root: Path,
    coverage_xml: Path,
    min_file_percent: float,
    min_aggregate_percent: float,
    justified_exclusions: Path | None,
    fail_on_file_gap: bool,
    min_assertion_density: float,
    min_raises_contract_density: float,
    hardware_result_pack_evidence: Path | None = None,
) -> dict[str, Any]:
    """Run all release-readiness checks and return a deterministic payload."""
    project_root = project_root.resolve()
    if justified_exclusions is None:
        default_exclusions = (
            project_root
            / "docs"
            / "internal"
            / "audits"
            / "release_readiness"
            / "coverage_justified_exclusions_2026-05-18.json"
        )
        justified_exclusions = default_exclusions if default_exclusions.exists() else None
    checks = (
        check_versions(project_root),
        check_required_artifacts(project_root),
        check_coverage_gate(
            project_root=project_root,
            coverage_xml=coverage_xml,
            min_file_percent=min_file_percent,
            min_aggregate_percent=min_aggregate_percent,
            justified_exclusions=justified_exclusions,
            fail_on_file_gap=fail_on_file_gap,
        ),
        check_behaviour_gate(
            tests_root=project_root / "tests",
            min_assertion_density=min_assertion_density,
            min_raises_contract_density=min_raises_contract_density,
        ),
        check_hardware_result_pack_evidence(project_root, hardware_result_pack_evidence),
    )
    blockers = tuple(blocker for check in checks for blocker in check.blockers)
    return {
        "audit": "release_readiness",
        "ready_for_tag": not blockers,
        "blockers": list(blockers),
        "checks": [check.to_dict() for check in checks],
    }


def format_release_readiness(payload: dict[str, Any]) -> str:
    """Render a compact release-readiness summary."""
    lines = [
        "Release readiness audit summary:",
        f"- ready_for_tag: {payload['ready_for_tag']}",
        f"- blockers: {len(payload['blockers'])}",
    ]
    for check in payload["checks"]:
        lines.append(f"- {check['name']}: {'PASS' if check['valid'] else 'FAIL'}")
        for blocker in check["blockers"][:10]:
            lines.append(f"- blocker: {blocker}")
        if len(check["blockers"]) > 10:
            lines.append(f"- additional_{check['name']}_blockers: {len(check['blockers']) - 10}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--coverage-xml", type=Path, default=default_root / "coverage.xml")
    parser.add_argument("--min-file-percent", type=float, default=95.0)
    parser.add_argument("--min-aggregate-percent", type=float, default=95.0)
    parser.add_argument("--justified-exclusions", type=Path, default=None)
    parser.add_argument(
        "--fail-on-file-gap",
        action="store_true",
        help="Also block release on source files below the per-file threshold.",
    )
    parser.add_argument("--min-assertion-density", type=float, default=1.0)
    parser.add_argument("--min-raises-contract-density", type=float, default=0.05)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--fail-on-blocker",
        action="store_true",
        help="Return non-zero when release readiness is blocked.",
    )
    parser.add_argument(
        "--hardware-result-pack-evidence",
        type=Path,
        default=None,
        help="Optional hardware result-pack evidence packet required for promoted hardware claims.",
    )
    args = parser.parse_args(argv)
    payload = audit_release_readiness(
        project_root=args.project_root,
        coverage_xml=args.coverage_xml,
        min_file_percent=args.min_file_percent,
        min_aggregate_percent=args.min_aggregate_percent,
        justified_exclusions=args.justified_exclusions,
        fail_on_file_gap=args.fail_on_file_gap,
        min_assertion_density=args.min_assertion_density,
        min_raises_contract_density=args.min_raises_contract_density,
        hardware_result_pack_evidence=args.hardware_result_pack_evidence,
    )
    print(
        json.dumps(payload, indent=2, sort_keys=True)
        if args.json
        else format_release_readiness(payload)
    )
    return 1 if args.fail_on_blocker and not payload["ready_for_tag"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
