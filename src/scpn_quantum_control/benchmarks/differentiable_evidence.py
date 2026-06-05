# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable benchmark evidence metadata.
"""CI-only benchmark evidence metadata and artefact writers."""

from __future__ import annotations

import csv
import json
import os
import platform
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkIsolationMetadata:
    """Isolation metadata required before benchmark evidence can be promoted."""

    command: tuple[str, ...]
    runner_type: str
    runner_name: str
    runner_labels: tuple[str, ...]
    github_run_id: str | None
    commit_sha: str | None
    runner_os: str
    python_version: str
    platform: str
    cpu_count: int | None
    cpu_affinity: str | None
    isolation_method: str | None
    load_before: tuple[float, float, float] | None
    load_after: tuple[float, float, float] | None
    governor: str | None
    frequency_mhz: float | None
    heavy_jobs_running: bool
    classification: str
    failure_class: str | None
    production_eligible: bool
    gap_reason: str | None

    @classmethod
    def from_ci_environment(
        cls,
        env: Mapping[str, str],
        *,
        command: Sequence[str],
        cpu_affinity: str | None,
        isolation_method: str | None,
        load_before: tuple[float, float, float] | None,
        load_after: tuple[float, float, float] | None,
        governor: str | None,
        frequency_mhz: float | None,
        heavy_jobs_running: bool,
    ) -> BenchmarkIsolationMetadata:
        """Classify a CI benchmark run from runner metadata."""

        labels = tuple(
            label.strip() for label in env.get("RUNNER_LABELS", "").split(",") if label.strip()
        )
        runner_environment = env.get("RUNNER_ENVIRONMENT", "")
        runner_type = "self-hosted" if runner_environment == "self-hosted" else "github-hosted"
        has_isolated_label = "isolated-benchmark" in labels
        cpu_count = os.cpu_count()
        has_required_context = all(
            (
                cpu_affinity,
                isolation_method,
                load_before is not None,
                load_after is not None,
                governor,
                frequency_mhz is not None,
            )
        )
        production_eligible = (
            runner_type == "self-hosted"
            and has_isolated_label
            and has_required_context
            and not heavy_jobs_running
        )
        if production_eligible:
            classification = "isolated_affinity"
            failure_class = None
            gap_reason = None
        elif runner_type == "self-hosted" and has_isolated_label:
            classification = "hard_gap"
            failure_class = "insufficient_isolation_metadata"
            gap_reason = (
                "Self-hosted isolated benchmark runner did not provide complete CPU "
                "affinity, host load, governor/frequency, and heavy-job metadata."
            )
        else:
            classification = "functional_non_isolated"
            failure_class = "non_isolated_runner"
            gap_reason = (
                "Production benchmark promotion requires a self-hosted isolated benchmark "
                "runner, CPU affinity, isolation method, host load, governor/frequency "
                "context, and no concurrent heavy jobs."
            )
        return cls(
            command=tuple(command),
            runner_type=runner_type,
            runner_name=env.get("RUNNER_NAME", ""),
            runner_labels=labels,
            github_run_id=env.get("GITHUB_RUN_ID"),
            commit_sha=env.get("GITHUB_SHA"),
            runner_os=env.get("RUNNER_OS", platform.system()),
            python_version=platform.python_version(),
            platform=platform.platform(),
            cpu_count=cpu_count,
            cpu_affinity=cpu_affinity,
            isolation_method=isolation_method,
            load_before=load_before,
            load_after=load_after,
            governor=governor,
            frequency_mhz=frequency_mhz,
            heavy_jobs_running=heavy_jobs_running,
            classification=classification,
            failure_class=failure_class,
            production_eligible=production_eligible,
            gap_reason=gap_reason,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready metadata."""

        return {
            "command": list(self.command),
            "runner_type": self.runner_type,
            "runner_name": self.runner_name,
            "runner_labels": list(self.runner_labels),
            "github_run_id": self.github_run_id,
            "commit_sha": self.commit_sha,
            "runner_os": self.runner_os,
            "python_version": self.python_version,
            "platform": self.platform,
            "cpu_count": self.cpu_count,
            "cpu_affinity": self.cpu_affinity,
            "isolation_method": self.isolation_method,
            "load_before": list(self.load_before) if self.load_before is not None else None,
            "load_after": list(self.load_after) if self.load_after is not None else None,
            "governor": self.governor,
            "frequency_mhz": self.frequency_mhz,
            "heavy_jobs_running": self.heavy_jobs_running,
            "classification": self.classification,
            "failure_class": self.failure_class,
            "production_eligible": self.production_eligible,
            "gap_reason": self.gap_reason,
        }


@dataclass(frozen=True)
class DifferentiableBenchmarkEvidenceBundle:
    """Paths and metadata for one written differentiable benchmark bundle."""

    artifact_id: str
    classification: str
    raw_json_path: Path
    csv_path: Path
    markdown_path: Path
    generated_at_epoch: float


def write_differentiable_benchmark_evidence_bundle(
    output_dir: Path,
    *,
    metadata: BenchmarkIsolationMetadata,
    timing_rows: Sequence[Mapping[str, object]],
    artifact_id: str | None = None,
) -> DifferentiableBenchmarkEvidenceBundle:
    """Write raw JSON, CSV timing rows, and Markdown summary for benchmark evidence."""

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = time.time()
    resolved_artifact_id = artifact_id or (
        f"diff-qnode-{metadata.classification}-{metadata.github_run_id or 'local'}"
    )
    raw_path = output_dir / f"{resolved_artifact_id}.json"
    csv_path = output_dir / f"{resolved_artifact_id}.csv"
    markdown_path = output_dir / f"{resolved_artifact_id}.md"
    rows = [dict(row) for row in timing_rows]
    payload = {
        "schema": "scpn_qc_differentiable_benchmark_evidence_v1",
        "artifact_id": resolved_artifact_id,
        "generated_at_epoch": generated_at,
        "metadata": metadata.to_dict(),
        "timing_rows": rows,
        "evidence_artifact_ids": [
            resolved_artifact_id,
            "diff-qnode-external-comparison-schema-v1",
        ],
        "claim_boundary": (
            "CI benchmark evidence only. functional_non_isolated rows are parity/local "
            "regression artefacts and cannot support production performance claims."
        ),
    }
    raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(
        "\n".join(
            (
                "# Differentiable Phase-QNode Benchmark Evidence",
                "",
                f"- Artefact ID: `{resolved_artifact_id}`",
                f"- Classification: `{metadata.classification}`",
                f"- Production eligible: `{metadata.production_eligible}`",
                f"- Failure class: {metadata.failure_class or 'none'}",
                f"- Gap reason: {metadata.gap_reason or 'none'}",
                "",
                "No provider or QPU execution is performed by this evidence writer.",
                "",
            )
        ),
        encoding="utf-8",
    )
    return DifferentiableBenchmarkEvidenceBundle(
        artifact_id=resolved_artifact_id,
        classification=metadata.classification,
        raw_json_path=raw_path,
        csv_path=csv_path,
        markdown_path=markdown_path,
        generated_at_epoch=generated_at,
    )


def capture_host_load() -> tuple[float, float, float] | None:
    """Return host load averages when the platform exposes them."""

    try:
        one, five, fifteen = os.getloadavg()
        return (float(one), float(five), float(fifteen))
    except (AttributeError, OSError):
        return None


def read_cpu_governor(cpu_index: int = 0) -> str | None:
    """Read Linux CPU frequency governor metadata when available."""

    path = Path(f"/sys/devices/system/cpu/cpu{cpu_index}/cpufreq/scaling_governor")
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8", errors="replace").strip()
    return value or None


def read_cpu_frequency_mhz(cpu_index: int = 0) -> float | None:
    """Read Linux CPU frequency metadata in MHz when available."""

    sysfs_path = Path(f"/sys/devices/system/cpu/cpu{cpu_index}/cpufreq/scaling_cur_freq")
    if sysfs_path.exists():
        value = sysfs_path.read_text(encoding="utf-8", errors="replace").strip()
        if value:
            return float(value) / 1000.0
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("cpu mhz"):
                return float(line.split(":", maxsplit=1)[1].strip())
    return None


def infer_heavy_jobs_running(load: tuple[float, float, float] | None) -> bool:
    """Infer whether current host load is too high for production promotion."""

    cpu_count = os.cpu_count() or 1
    return bool(load and load[0] > max(1.0, cpu_count * 0.75))


__all__ = [
    "BenchmarkIsolationMetadata",
    "DifferentiableBenchmarkEvidenceBundle",
    "capture_host_load",
    "infer_heavy_jobs_running",
    "read_cpu_frequency_mhz",
    "read_cpu_governor",
    "write_differentiable_benchmark_evidence_bundle",
]
