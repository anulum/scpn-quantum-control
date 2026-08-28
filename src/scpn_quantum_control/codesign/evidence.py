# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — co-design functional evidence
"""Functional non-isolated evidence writer for the bounded co-design workflow."""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Final

import numpy as np

from ..control.closed_loop_analysis import ClosedLoopExecutionPolicy
from .components import (
    ExponentialOrderEstimator,
    GradientFeedbackController,
    PhaseObjectiveSimulator,
)
from .contracts import CODESIGN_CLAIM_BOUNDARY, CoDesignMode, LoopStepInput
from .loop import CoDesignLoop
from .policies import LatencyPolicy, SafetyEnvelope
from .replay import record_replay_trace, verify_replay_trace

EVIDENCE_SCHEMA: Final[str] = "quantum_classical_codesign.evidence.v1"
EVIDENCE_CLASSIFICATION: Final[str] = "functional_non_isolated"


@dataclass(frozen=True, slots=True)
class FunctionalEvidence:
    """Measured local workflow and deterministic replay evidence."""

    classification: str
    isolated: bool
    iterations: int
    steps_per_iteration: int
    elapsed_ms: tuple[float, ...]
    median_elapsed_ms: float
    throughput_steps_per_second: float
    replay_verified: bool
    trace_digest: str
    python_version: str
    platform: str
    command: tuple[str, ...]
    provider_execution: bool = False
    hardware_execution: bool = False
    schema: str = EVIDENCE_SCHEMA
    claim_boundary: str = CODESIGN_CLAIM_BOUNDARY

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready evidence mapping."""
        return {
            "classification": self.classification,
            "isolated": self.isolated,
            "iterations": self.iterations,
            "steps_per_iteration": self.steps_per_iteration,
            "elapsed_ms": list(self.elapsed_ms),
            "median_elapsed_ms": self.median_elapsed_ms,
            "throughput_steps_per_second": self.throughput_steps_per_second,
            "replay_verified": self.replay_verified,
            "trace_digest": self.trace_digest,
            "python_version": self.python_version,
            "platform": self.platform,
            "command": list(self.command),
            "provider_execution": self.provider_execution,
            "hardware_execution": self.hardware_execution,
            "schema": self.schema,
            "claim_boundary": self.claim_boundary,
        }


def build_demo_loop() -> CoDesignLoop:
    """Build a fresh deterministic, simulation-only co-design loop."""
    return CoDesignLoop(
        estimator=ExponentialOrderEstimator(alpha=0.5),
        evaluator=PhaseObjectiveSimulator(policy=ClosedLoopExecutionPolicy(round_budget=8)),
        controller=GradientFeedbackController(learning_rate=0.15, feedback_gain=0.5),
        latency_policy=LatencyPolicy(max_age_ms=5.0),
        safety_envelope=SafetyEnvelope(
            max_abs_parameter=float(np.pi),
            max_update_norm=0.25,
            max_gradient_norm=2.0,
        ),
    )


def demo_inputs() -> tuple[LoopStepInput, ...]:
    """Return a fixed two-step hybrid-replay workflow."""
    return (
        LoopStepInput(
            step=0,
            observed_at_ms=0.0,
            apply_at_ms=2.0,
            parameters=(0.0, 0.7, 1.4),
            measurement=0.72,
            target_order_parameter=0.9,
            mode=CoDesignMode.HYBRID_REPLAY,
        ),
        LoopStepInput(
            step=1,
            observed_at_ms=10.0,
            apply_at_ms=12.0,
            parameters=(-0.01, 0.7, 1.41),
            measurement=0.74,
            target_order_parameter=0.9,
            mode=CoDesignMode.HYBRID_REPLAY,
        ),
    )


def run_functional_evidence(*, iterations: int = 20) -> FunctionalEvidence:
    """Measure bounded local loop throughput and verify deterministic replay.

    The timings are functional workstation evidence only. The process is not
    CPU-isolated and the values must not be promoted to production latency.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    inputs = demo_inputs()
    elapsed_ms: list[float] = []
    last_digest = ""
    for _ in range(iterations):
        start = time.perf_counter_ns()
        trace, _outputs = record_replay_trace(build_demo_loop(), inputs)
        elapsed_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)
        last_digest = trace.digest
    trace, _outputs = record_replay_trace(build_demo_loop(), inputs)
    replayed = verify_replay_trace(build_demo_loop(), trace)
    total_seconds = sum(elapsed_ms) / 1000.0
    return FunctionalEvidence(
        classification=EVIDENCE_CLASSIFICATION,
        isolated=False,
        iterations=iterations,
        steps_per_iteration=len(inputs),
        elapsed_ms=tuple(elapsed_ms),
        median_elapsed_ms=float(median(elapsed_ms)),
        throughput_steps_per_second=(iterations * len(inputs)) / total_seconds,
        replay_verified=len(replayed) == len(inputs),
        trace_digest=last_digest,
        python_version=platform.python_version(),
        platform=platform.platform(),
        command=("python", "scripts/run_codesign_loop_evidence.py"),
    )


def write_functional_evidence(path: Path, *, iterations: int = 20) -> FunctionalEvidence:
    """Write one measured evidence payload to ``path``."""
    evidence = run_functional_evidence(iterations=iterations)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(evidence.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return evidence


def validate_functional_evidence(payload: object) -> tuple[str, ...]:
    """Return fail-closed findings for a co-design functional evidence payload."""
    if not isinstance(payload, dict):
        return ("payload must be a JSON object",)
    findings: list[str] = []
    expected = {
        "schema": EVIDENCE_SCHEMA,
        "classification": EVIDENCE_CLASSIFICATION,
        "isolated": False,
        "provider_execution": False,
        "hardware_execution": False,
        "replay_verified": True,
        "claim_boundary": CODESIGN_CLAIM_BOUNDARY,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            findings.append(f"{key} must equal {value!r}")
    numeric_positive = (
        "iterations",
        "steps_per_iteration",
        "median_elapsed_ms",
        "throughput_steps_per_second",
    )
    for key in numeric_positive:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            findings.append(f"{key} must be positive")
    digest = payload.get("trace_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        findings.append("trace_digest must be a SHA-256 hexadecimal value")
    timings = payload.get("elapsed_ms")
    if not isinstance(timings, list) or not timings:
        findings.append("elapsed_ms must be a non-empty array")
    elif not all(
        not isinstance(value, bool) and isinstance(value, (int, float)) and value > 0
        for value in timings
    ):
        findings.append("elapsed_ms values must be positive")
    return tuple(findings)


def main(argv: list[str] | None = None) -> int:
    """Run the evidence writer CLI without importing the standalone script."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/differentiable_phase_qnode/codesign_loop_evidence.json"),
    )
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args(argv)
    evidence = write_functional_evidence(args.output, iterations=args.iterations)
    print(args.output)
    print(f"classification={evidence.classification}")
    print("No provider or QPU execution")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "EVIDENCE_CLASSIFICATION",
    "EVIDENCE_SCHEMA",
    "FunctionalEvidence",
    "build_demo_loop",
    "demo_inputs",
    "main",
    "run_functional_evidence",
    "validate_functional_evidence",
    "write_functional_evidence",
]
