# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase QNode Affinity Benchmark CLI
"""Write Phase-QNode affinity benchmark metadata as JSON."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast


class _AffinityBenchmarkResult(Protocol):
    """Result surface consumed by the lightweight benchmark CLI."""

    evidence_label: str
    isolation_failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-ready benchmark payload."""
        ...


class _AffinityBenchmarkModule(Protocol):
    """Typed entry point exposed by the dynamically loaded phase leaf."""

    def run_phase_qnode_affinity_benchmark(
        self,
        *,
        repetitions: int,
        warmups: int,
        reserved_cpus: tuple[int, ...],
        command: str,
    ) -> _AffinityBenchmarkResult:
        """Run the bounded Phase-QNode benchmark."""
        ...


class _LeanLoaderModule(Protocol):
    """Typed surface loaded from the sibling lean-loader tool."""

    def load_phase_module(self, submodule: str) -> ModuleType:
        """Load one phase leaf without executing package initializers."""
        ...


def _load_lean_loader() -> _LeanLoaderModule:
    """Load the sibling lean-loader tool without relying on ambient import paths."""
    module_path = Path(__file__).with_name("lean_phase_import.py")
    spec = importlib.util.spec_from_file_location("scpn_lean_phase_import", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load lean phase importer from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(_LeanLoaderModule, module)


def _canonical_command(
    *,
    repetitions: int,
    warmups: int,
    reserved_cpus: str,
    output: str,
    require_isolated: bool,
) -> str:
    """Return a shell-escaped command that reproduces the requested run."""
    command = [] if not reserved_cpus else ["taskset", "-c", reserved_cpus]
    command.extend(
        [
            "python",
            "tools/run_phase_qnode_affinity_benchmark.py",
            "--repetitions",
            str(repetitions),
            "--warmups",
            str(warmups),
            "--reserved-cpus",
            reserved_cpus,
            "--output",
            output,
        ]
    )
    if require_isolated:
        command.append("--require-isolated")
    return shlex.join(command)


def main() -> None:
    """Run the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--reserved-cpus", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--recorded-command",
        default="",
        help=(
            "Exact outer command to record when an orchestrator adds chrt or other "
            "admitted isolation controls. Defaults to a reproducible taskset command."
        ),
    )
    parser.add_argument(
        "--require-isolated",
        action="store_true",
        help="Exit non-zero unless the written evidence is classified as isolated_affinity.",
    )
    args = parser.parse_args()
    reserved = tuple(int(item.strip()) for item in args.reserved_cpus.split(",") if item.strip())
    module = cast(
        _AffinityBenchmarkModule,
        _load_lean_loader().load_phase_module("qnode_affinity_benchmark"),
    )
    command = args.recorded_command or _canonical_command(
        repetitions=args.repetitions,
        warmups=args.warmups,
        reserved_cpus=args.reserved_cpus,
        output=args.output,
        require_isolated=args.require_isolated,
    )
    result = module.run_phase_qnode_affinity_benchmark(
        repetitions=args.repetitions,
        warmups=args.warmups,
        reserved_cpus=reserved,
        command=command,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    if args.require_isolated and result.evidence_label != "isolated_affinity":
        raise SystemExit(
            "isolated_affinity evidence was required but benchmark classified as "
            f"{result.evidence_label}: {', '.join(result.isolation_failures)}"
        )


if __name__ == "__main__":
    main()
