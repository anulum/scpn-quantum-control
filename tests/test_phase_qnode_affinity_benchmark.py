# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase QNode Affinity Benchmark
"""Tests for phase/qnode_affinity_benchmark.py metadata and labels."""

from __future__ import annotations

from scpn_quantum_control.phase.qnode_affinity_benchmark import (
    classify_affinity_evidence,
    run_phase_qnode_affinity_benchmark,
)


def test_affinity_benchmark_downgrades_without_reserved_cpu_and_low_load() -> None:
    result = run_phase_qnode_affinity_benchmark(
        repetitions=3,
        warmups=1,
        reserved_cpus=(),
        host_load_before=(5.0, 5.0, 5.0),
        host_load_after=(5.0, 5.0, 5.0),
    )

    assert result.evidence_label == "functional_non_isolated"
    assert not result.production_benchmark
    assert result.metadata.command
    assert result.metadata.repetitions == 3
    assert result.metadata.warmups == 1
    assert result.metadata.cpu_model
    assert result.metadata.python_version
    assert result.raw_timing_rows
    assert "reserved CPU affinity" in result.isolation_failures
    assert result.to_dict()["evidence_label"] == "functional_non_isolated"


def test_affinity_label_requires_all_isolation_criteria() -> None:
    assert (
        classify_affinity_evidence(
            reserved_cpus=(2, 3),
            host_load_before=(0.1, 0.1, 0.1),
            host_load_after=(0.1, 0.1, 0.1),
            command="taskset -c 2,3 python -m bench",
            heavy_concurrent_jobs=False,
        )
        == "isolated_affinity"
    )
    assert (
        classify_affinity_evidence(
            reserved_cpus=(2, 3),
            host_load_before=(3.0, 3.0, 3.0),
            host_load_after=(0.1, 0.1, 0.1),
            command="taskset -c 2,3 python -m bench",
            heavy_concurrent_jobs=False,
        )
        == "functional_non_isolated"
    )
