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
class Harness:
    """A reproducibility harness script and its execution policy."""

    label: str
    script: str
    groups: frozenset[str]
    optional_flag: str | None = None


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
)

ARTEFACT_PATHS = (
    "data/rust_vqe_methods",
    "data/scpn_fim_hamiltonian",
    "data/s1_feedback_loop",
    "data/s2_scaling",
    "data/s3_pulse_ansatz_design",
    "data/s4_multi_hardware_control",
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
    script_path = REPO_ROOT / harness.script
    command = [PYTHON, str(script_path)]
    print(f"[scpn-bench] run {harness.label}: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return completed.returncode


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
