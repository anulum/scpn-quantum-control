# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- benchmark reproducibility command line interface
"""One-command benchmark artefact regeneration for the methods papers."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExecutionSurfacePolicy:
    """Execution policy for a fixed reproducibility harness."""

    classification: str
    network_allowed: bool
    credential_allowed: bool
    hardware_submission_allowed: bool
    allowed_write_roots: tuple[str, ...]
    subprocess_allowed: bool
    ci_blocking: bool


OFFLINE_HARNESS_POLICY = ExecutionSurfacePolicy(
    classification="trusted_offline_executable",
    network_allowed=False,
    credential_allowed=False,
    hardware_submission_allowed=False,
    allowed_write_roots=(
        "data/rust_vqe_methods",
        "data/scpn_fim_hamiltonian",
        "data/s1_feedback_loop",
        "data/s2_scaling",
        "data/s3_pulse_ansatz_design",
        "data/s4_multi_hardware_control",
        "data/s5_benchmark_harness",
        "data/s6_quantum_kuramoto_split",
        "data/stable_core",
        "data/synchronisation_benchmarks",
        "data/symmetry_sector_mitigation",
        "docs",
    ),
    subprocess_allowed=True,
    ci_blocking=True,
)


@dataclass(frozen=True)
class Harness:
    """A reproducibility harness script and its execution policy."""

    label: str
    script: str
    groups: frozenset[str]
    optional_flag: str | None = None
    policy: ExecutionSurfacePolicy = OFFLINE_HARNESS_POLICY


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

HARNESS_REGISTRY: tuple[Harness, ...] = (
    Harness("methods-rust-core", "scripts/benchmark_rust_core_methods.py", frozenset({"methods"})),
    Harness("methods-ansatz", "scripts/benchmark_ansatz_methods.py", frozenset({"methods"})),
    Harness("methods-vqe", "scripts/benchmark_vqe_methods.py", frozenset({"methods"})),
    Harness(
        "methods-multilang",
        "scripts/benchmark_multilang_knm_methods.py",
        frozenset({"methods"}),
    ),
    Harness(
        "methods-gpu",
        "scripts/benchmark_gpu_methods.py",
        frozenset({"methods"}),
        optional_flag="gpu",
    ),
    Harness(
        "methods-combined-summary",
        "scripts/summarise_rust_vqe_method_artifacts.py",
        frozenset({"methods"}),
    ),
    Harness(
        "methods-ansatz-scaling-tn",
        "scripts/benchmark_ansatz_scaling_tn.py",
        frozenset({"methods"}),
        optional_flag="scaling",
    ),
    Harness("fim-spectrum", "scripts/analyse_fim_spectrum.py", frozenset({"fim"})),
    Harness("fim-level-spacing", "scripts/analyse_fim_level_spacing.py", frozenset({"fim"})),
    Harness("fim-entanglement", "scripts/analyse_fim_entanglement.py", frozenset({"fim"})),
    Harness("fim-sector-survival", "scripts/analyse_fim_sector_survival.py", frozenset({"fim"})),
    Harness("fim-vqe", "scripts/benchmark_fim_vqe_ground_state.py", frozenset({"fim"})),
    Harness("fim-ibm-pilot-analysis", "scripts/analyse_fim_ibm_pilot.py", frozenset({"fim"})),
    Harness(
        "fim-ibm-repeated-analysis",
        "scripts/analyse_fim_ibm_repeated_followup.py",
        frozenset({"fim"}),
    ),
    Harness(
        "fim-readout-matrix-mitigation",
        "scripts/analyse_fim_readout_matrix_mitigation.py",
        frozenset({"fim"}),
        optional_flag="readout",
    ),
    Harness("s1-feedback-loop", "scripts/benchmark_s1_feedback_loop.py", frozenset({"s1"})),
    Harness(
        "s1-feedback-readiness",
        "scripts/reproduce_s1_feedback_readiness.py",
        frozenset({"s1-ready"}),
    ),
    Harness("s2-scaling-protocol", "scripts/export_s2_scaling_protocol.py", frozenset({"s2"})),
    Harness("s2-scaling-lite", "scripts/bench_s2_scaling_lite.py", frozenset({"s2"})),
    Harness(
        "s2-claim-boundary",
        "scripts/report_s2_scaling_claim_boundary.py",
        frozenset({"s2"}),
    ),
    Harness(
        "s3-design-readiness",
        "scripts/export_s3_design_readiness.py",
        frozenset({"s3"}),
    ),
    Harness(
        "s3-design-surrogate",
        "scripts/train_s3_design_surrogate.py",
        frozenset({"s3-surrogate"}),
    ),
    Harness(
        "s3-ansatz-observables",
        "scripts/validate_s3_ansatz_observables.py",
        frozenset({"s3-observables"}),
    ),
    Harness(
        "s3-pulse-feasibility",
        "scripts/probe_s3_pulse_feasibility.py",
        frozenset({"s3-pulse"}),
    ),
    Harness(
        "s3-hardware-dossiers",
        "scripts/export_s3_hardware_dossiers.py",
        frozenset({"s3-dossiers"}),
    ),
    Harness(
        "s4-multi-hardware-readiness",
        "scripts/export_s4_multi_hardware_readiness.py",
        frozenset({"s4"}),
    ),
    Harness(
        "s4-provider-preregistration",
        "scripts/export_s4_provider_preregistration.py",
        frozenset({"s4-provider"}),
    ),
    Harness(
        "s4-neutral-atom-preregistration",
        "scripts/export_s4_neutral_atom_preregistration.py",
        frozenset({"s4-neutral"}),
    ),
    Harness(
        "s5-benchmark-suite",
        "scripts/run_benchmark_suite.py",
        frozenset({"s5"}),
    ),
    Harness(
        "s5-benchmark-registry",
        "scripts/export_benchmark_registry.py",
        frozenset({"s5-registry"}),
    ),
    Harness(
        "s6-split-audit",
        "scripts/audit_quantum_kuramoto_split.py",
        frozenset({"s6"}),
    ),
    Harness(
        "s6-boundary-review",
        "scripts/export_quantum_kuramoto_boundary_review.py",
        frozenset({"s6-review"}),
    ),
    Harness(
        "s6-api-contract",
        "scripts/export_quantum_kuramoto_api_contract.py",
        frozenset({"s6-contract"}),
    ),
    Harness(
        "sync-benchmark-registry",
        "scripts/export_synchronisation_benchmark_registry.py",
        frozenset({"sync-registry"}),
    ),
    Harness(
        "sync-benchmark-run",
        "scripts/run_synchronisation_benchmark.py",
        frozenset({"sync-run"}),
    ),
    Harness(
        "sync-benchmark-compare",
        "scripts/compare_synchronisation_benchmark.py",
        frozenset({"sync-compare"}),
    ),
    Harness(
        "sync-benchmark-gate",
        "scripts/run_synchronisation_benchmark_gate.py",
        frozenset({"sync-gate"}),
    ),
    Harness(
        "symmetry-sector-mitigation-gate",
        "scripts/run_symmetry_sector_mitigation_gate.py",
        frozenset({"symmetry-sector-gate"}),
    ),
    Harness(
        "stable-core-capability-matrix",
        "scripts/export_stable_core_capability_matrix.py",
        frozenset({"stable-core"}),
    ),
    Harness(
        "stable-core-capability-gate",
        "scripts/run_stable_core_capability_gate.py",
        frozenset({"stable-core-gate"}),
    ),
    Harness(
        "stable-core-contract-gate",
        "scripts/run_stable_core_contract_gate.py",
        frozenset({"stable-core-contract-gate"}),
    ),
    Harness(
        "stable-core-preflight-gate",
        "scripts/run_stable_core_preflight_gate.py",
        frozenset({"stable-core-preflight-gate"}),
    ),
    Harness(
        "stable-core-release-gate",
        "scripts/run_stable_core_release_gate.py",
        frozenset({"stable-core-release-gate"}),
    ),
    Harness(
        "paper0-lane-registry-gate",
        "scripts/run_paper0_lane_registry_gate.py",
        frozenset({"paper0-lane-registry-gate"}),
    ),
)

ARTEFACT_PATHS = (
    "data/rust_vqe_methods",
    "data/scpn_fim_hamiltonian",
    "data/s1_feedback_loop",
    "data/s2_scaling",
    "data/s3_pulse_ansatz_design",
    "data/s4_multi_hardware_control",
    "data/s5_benchmark_harness",
    "data/s6_quantum_kuramoto_split",
    "data/stable_core",
    "data/synchronisation_benchmarks",
    "data/symmetry_sector_mitigation",
    "docs/stable_core_backend_capability_matrix.md",
)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scpn-bench",
        description=(
            "Regenerate committed benchmark artefacts and compare them with "
            "the repository state. Hardware submission scripts are deliberately "
            "excluded; IBM raw-count analyses are offline only."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    reproduce = subparsers.add_parser(
        "reproduce-methods",
        help="Regenerate Rust/VQE methods-paper artefacts.",
    )
    _add_run_options(reproduce, default_group="methods")

    fim = subparsers.add_parser(
        "fim-all",
        help="Regenerate SCPN/FIM paper artefacts from committed data.",
    )
    _add_run_options(fim, default_group="fim")

    all_parser = subparsers.add_parser(
        "all",
        help="Regenerate methods and FIM artefacts in one run.",
    )
    _add_run_options(all_parser, default_group="all")

    stable_core = subparsers.add_parser(
        "stable-core-capability-matrix",
        help="Regenerate the stable-core capability matrix artifacts.",
    )
    _add_run_options(stable_core, default_group="stable-core")

    stable_core_gate = subparsers.add_parser(
        "stable-core-capability-gate",
        help="Run and check stable-core capability gate fixtures.",
    )
    _add_run_options(stable_core_gate, default_group="stable-core-gate")

    stable_core_contract_gate = subparsers.add_parser(
        "stable-core-contract-gate",
        help="Run and check stable-core contract gate fixtures.",
    )
    _add_run_options(stable_core_contract_gate, default_group="stable-core-contract-gate")

    stable_core_preflight_gate = subparsers.add_parser(
        "stable-core-preflight-gate",
        help="Run and check stable-core preflight gate fixtures.",
    )
    _add_run_options(
        stable_core_preflight_gate,
        default_group="stable-core-preflight-gate",
    )

    stable_core_release_gate = subparsers.add_parser(
        "stable-core-release-gate",
        help="Run and check the stable-core release gate fixture set.",
    )
    _add_run_options(stable_core_release_gate, default_group="stable-core-release-gate")

    paper0_lane_registry_gate = subparsers.add_parser(
        "paper0-lane-registry-gate",
        help="Run and check the Paper 0 lane registry gate.",
    )
    _add_run_options(
        paper0_lane_registry_gate,
        default_group="paper0-lane-registry-gate",
    )

    s1 = subparsers.add_parser(
        "s1-feedback",
        help="Regenerate no-QPU S1 feedback-loop latency artefacts.",
    )
    _add_run_options(s1, default_group="s1")

    s1_ready = subparsers.add_parser(
        "s1-feedback-ready",
        help="Regenerate the complete no-QPU S1 readiness bundle.",
    )
    _add_run_options(s1_ready, default_group="s1-ready")

    s2 = subparsers.add_parser(
        "s2-scaling-lite",
        help="Regenerate no-QPU S2 scaling protocol and lite rows.",
    )
    _add_run_options(s2, default_group="s2")

    s3 = subparsers.add_parser(
        "s3-design-ready",
        help="Regenerate no-QPU S3 pulse/ansatz design-readiness artefacts.",
    )
    _add_run_options(s3, default_group="s3")

    s3_surrogate = subparsers.add_parser(
        "s3-design-surrogate",
        help="Regenerate no-QPU S3 design-surrogate rehearsal artefacts.",
    )
    _add_run_options(s3_surrogate, default_group="s3-surrogate")

    s3_observables = subparsers.add_parser(
        "s3-ansatz-observables",
        help="Regenerate no-QPU S3 ansatz observable-validation artefacts.",
    )
    _add_run_options(s3_observables, default_group="s3-observables")

    s3_pulse = subparsers.add_parser(
        "s3-pulse-feasibility",
        help="Regenerate no-submit S3 pulse feasibility artefacts.",
    )
    _add_run_options(s3_pulse, default_group="s3-pulse")

    s3_dossiers = subparsers.add_parser(
        "s3-hardware-dossiers",
        help="Regenerate no-submit S3 hardware-job dossier artefacts.",
    )
    _add_run_options(s3_dossiers, default_group="s3-dossiers")

    s4 = subparsers.add_parser(
        "s4-multi-hardware-ready",
        help="Regenerate no-submit S4 multi-hardware readiness artefacts.",
    )
    _add_run_options(s4, default_group="s4")

    s4_provider = subparsers.add_parser(
        "s4-provider-preregistration",
        help="Regenerate the no-submit S4 IBM pulse-level preregistration dossier.",
    )
    _add_run_options(s4_provider, default_group="s4-provider")

    s4_neutral = subparsers.add_parser(
        "s4-neutral-atom-preregistration",
        help="Regenerate the no-submit S4 neutral-atom preregistration dossier.",
    )
    _add_run_options(s4_neutral, default_group="s4-neutral")

    s5 = subparsers.add_parser(
        "s5-benchmark-suite",
        help="Regenerate the no-QPU S5 open-data benchmark harness artefacts.",
    )
    _add_run_options(s5, default_group="s5")

    s5_registry = subparsers.add_parser(
        "s5-benchmark-registry",
        help="Regenerate the S5 benchmark-harness registry artefacts.",
    )
    _add_run_options(s5_registry, default_group="s5-registry")

    s6 = subparsers.add_parser(
        "s6-split-audit",
        help="Regenerate the S6 quantum-kuramoto split audit artefacts.",
    )
    _add_run_options(s6, default_group="s6")

    s6_review = subparsers.add_parser(
        "s6-boundary-review",
        help="Regenerate the S6 quantum-kuramoto boundary-review artefacts.",
    )
    _add_run_options(s6_review, default_group="s6-review")

    s6_contract = subparsers.add_parser(
        "s6-api-contract",
        help="Regenerate the S6 quantum-kuramoto API-contract artefacts.",
    )
    _add_run_options(s6_contract, default_group="s6-contract")

    sync_registry = subparsers.add_parser(
        "sync-benchmark-registry",
        help="Regenerate the standardised synchronisation benchmark registry.",
    )
    _add_run_options(sync_registry, default_group="sync-registry")

    sync_run = subparsers.add_parser(
        "sync-benchmark-run",
        help="Regenerate no-QPU synchronisation benchmark reference rows.",
    )
    _add_run_options(sync_run, default_group="sync-run")

    sync_compare = subparsers.add_parser(
        "sync-benchmark-compare",
        help="Compare regenerated synchronisation benchmark rows against committed rows.",
    )
    _add_run_options(sync_compare, default_group="sync-compare")

    sync_gate = subparsers.add_parser(
        "sync-benchmark-gate",
        help="Regenerate all synchronisation benchmark artefacts and compare them.",
    )
    _add_run_options(sync_gate, default_group="sync-gate")

    symmetry_sector_gate = subparsers.add_parser(
        "symmetry-sector-mitigation-gate",
        help="Regenerate and compare symmetry-sector mitigation planner fixtures.",
    )
    _add_run_options(symmetry_sector_gate, default_group="symmetry-sector-gate")

    return parser.parse_args(argv)


def _add_run_options(parser: argparse.ArgumentParser, *, default_group: str) -> None:
    parser.set_defaults(group=default_group)
    parser.add_argument(
        "--include-gpu",
        action="store_true",
        help="Include GPU benchmark harnesses that may require CUDA/CuPy/PyTorch.",
    )
    parser.add_argument(
        "--include-scaling",
        action="store_true",
        help="Include heavier n=4--12 ansatz-scaling and tensor-network diagnostics.",
    )
    parser.add_argument(
        "--include-readout",
        action="store_true",
        help="Include full-basis offline readout-matrix mitigation cross-checks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected harnesses without executing them.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue running later harnesses after a failure.",
    )
    parser.add_argument(
        "--no-diff",
        action="store_true",
        help="Skip git diff summary after regeneration.",
    )


def _selected_harnesses(
    group: str,
    *,
    include_gpu: bool,
    include_scaling: bool = False,
    include_readout: bool = False,
) -> list[Harness]:
    wanted_groups = {"methods", "fim"} if group == "all" else {group}
    selected: list[Harness] = []
    for harness in HARNESS_REGISTRY:
        if not harness.groups.intersection(wanted_groups):
            continue
        if harness.optional_flag == "gpu" and not include_gpu:
            continue
        if harness.optional_flag == "scaling" and not include_scaling:
            continue
        if harness.optional_flag == "readout" and not include_readout:
            continue
        selected.append(harness)
    return selected


def _run_harness(harness: Harness) -> int:
    try:
        _validate_harness_policy(harness)
    except FileNotFoundError as exc:
        print(f"[scpn-bench] missing harness script: {exc}", file=sys.stderr)
        return 2
    script_path = REPO_ROOT / harness.script
    command = [PYTHON, str(script_path)]
    print(f"[scpn-bench] run {harness.label}: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return completed.returncode


def _validate_harness_policy(harness: Harness) -> None:
    """Fail closed before launching a fixed benchmark harness."""
    if harness.policy.classification != "trusted_offline_executable":
        raise PermissionError(f"harness {harness.label!r} is not executable by scpn-bench")
    if harness.policy.network_allowed:
        raise PermissionError(f"harness {harness.label!r} allows network access")
    if harness.policy.credential_allowed:
        raise PermissionError(f"harness {harness.label!r} allows credential access")
    if harness.policy.hardware_submission_allowed:
        raise PermissionError(f"harness {harness.label!r} allows hardware submission")
    if not harness.policy.subprocess_allowed:
        raise PermissionError(f"harness {harness.label!r} disallows subprocess execution")
    script_path = (REPO_ROOT / harness.script).resolve()
    if not script_path.is_relative_to(REPO_ROOT):
        raise ValueError(f"harness script must stay inside repository: {harness.script}")
    if not script_path.exists():
        raise FileNotFoundError(f"harness script does not exist: {harness.script}")
    for root in harness.policy.allowed_write_roots:
        root_path = (REPO_ROOT / root).resolve()
        if not root_path.is_relative_to(REPO_ROOT):
            raise ValueError(f"allowed write root must stay inside repository: {root}")


def _print_diff_summary() -> int:
    command = ["git", "diff", "--stat", "--", *ARTEFACT_PATHS]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False, text=True)
    if completed.returncode != 0:
        return completed.returncode
    name_only = subprocess.run(
        ["git", "diff", "--name-only", "--", *ARTEFACT_PATHS],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if name_only.returncode != 0:
        return name_only.returncode
    changed = [line for line in name_only.stdout.splitlines() if line.strip()]
    if changed:
        print("[scpn-bench] regenerated artefacts differ from committed files:")
        for path in changed:
            print(f"  {path}")
        return 2
    print("[scpn-bench] regenerated artefacts match committed files.")
    return 0


def run(argv: Sequence[str] | None = None) -> int:
    ns = _parse_args(sys.argv[1:] if argv is None else argv)
    harnesses = _selected_harnesses(
        ns.group,
        include_gpu=ns.include_gpu,
        include_scaling=ns.include_scaling,
        include_readout=ns.include_readout,
    )
    if not harnesses:
        print("[scpn-bench] no harnesses selected", file=sys.stderr)
        return 2

    print("[scpn-bench] selected harnesses:")
    for harness in harnesses:
        suffix = " (optional)" if harness.optional_flag is not None else ""
        print(f"  - {harness.label}: {harness.script}{suffix}")
    if ns.dry_run:
        return 0

    failures: list[tuple[Harness, int]] = []
    for harness in harnesses:
        returncode = _run_harness(harness)
        if returncode != 0:
            failures.append((harness, returncode))
            print(
                f"[scpn-bench] failed {harness.label} with exit code {returncode}",
                file=sys.stderr,
            )
            if not ns.keep_going:
                break

    diff_status = 0 if ns.no_diff else _print_diff_summary()
    if failures:
        print("[scpn-bench] failed harnesses:", file=sys.stderr)
        for harness, returncode in failures:
            print(f"  - {harness.label}: {returncode}", file=sys.stderr)
        return 1
    return diff_status


def main() -> int:
    """Console-script entry point."""

    return run()


if __name__ == "__main__":
    raise SystemExit(main())
