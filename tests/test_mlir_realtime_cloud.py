# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- tests for MLIR, realtime, and cloud-native surfaces
"""Tests for production-grade compiler/runtime/deployment surfaces."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.compiler.mlir import (
    MLIRCompileConfig,
    compile_kuramoto_to_mlir,
)
from scpn_quantum_control.control.realtime_runtime import (
    RealtimeRuntimeConfig,
    VirtualRealtimeClock,
    run_realtime_control_loop,
)
from scpn_quantum_control.deployment.cloud_native import (
    CloudDeploymentSpec,
    ContainerResources,
    generate_cloud_manifests,
)
from scpn_quantum_control.kuramoto_core import build_kuramoto_problem


def _problem():
    return build_kuramoto_problem(
        np.array(
            [
                [0.0, 0.25, 0.0],
                [0.25, 0.0, -0.5],
                [0.0, -0.5, 0.0],
            ],
            dtype=np.float64,
        ),
        np.array([0.1, -0.2, 0.3], dtype=np.float64),
        metadata={"experiment": "compiler-smoke"},
    )


def test_kuramoto_mlir_emits_deterministic_text_digest_and_resources() -> None:
    """MLIR export should be deterministic and explicit about compiler resources."""

    module = compile_kuramoto_to_mlir(
        _problem(),
        MLIRCompileConfig(time=0.4, trotter_steps=2, trotter_order=2),
    )
    repeat = compile_kuramoto_to_mlir(
        _problem(),
        MLIRCompileConfig(time=0.4, trotter_steps=2, trotter_order=2),
    )

    assert module.text == repeat.text
    assert module.sha256 == repeat.sha256
    assert module.resource_counts["n_oscillators"] == 3
    assert module.resource_counts["coupling_terms"] == 2
    assert module.resource_counts["trotter_steps"] == 2
    assert 'scpn.module = "kuramoto_xy"' in module.text
    assert "scpn.omega" in module.text
    assert "scpn.coupling" in module.text
    assert "scpn.trotter_evolve" in module.text


def test_kuramoto_mlir_rejects_invalid_compile_config() -> None:
    """MLIR config should fail closed before emitting misleading IR."""

    with pytest.raises(ValueError, match="trotter_steps"):
        MLIRCompileConfig(time=0.1, trotter_steps=0)
    with pytest.raises(ValueError, match="time"):
        MLIRCompileConfig(time=float("nan"))


def test_realtime_control_loop_records_deadline_jitter_and_misses() -> None:
    """Realtime runtime should account for deterministic deadline misses."""

    clock = VirtualRealtimeClock()
    config = RealtimeRuntimeConfig(
        sample_period_s=0.01,
        deadline_s=0.005,
        jitter_budget_s=0.002,
        max_missed_deadlines=2,
    )

    durations = [0.001, 0.006, 0.004]

    def step(index: int):
        clock.advance(durations[index])
        return {"index": float(index)}

    result = run_realtime_control_loop(3, step, config=config, clock=clock)

    assert result.completed
    assert result.missed_deadlines == 1
    assert result.max_latency_s == pytest.approx(0.006)
    assert result.records[1].deadline_missed
    assert result.records[2].jitter_s == pytest.approx(0.0)


def test_realtime_control_loop_fails_when_deadline_budget_is_exceeded() -> None:
    """Realtime runtime should abort instead of hiding repeated deadline misses."""

    clock = VirtualRealtimeClock()
    config = RealtimeRuntimeConfig(
        sample_period_s=0.01,
        deadline_s=0.002,
        max_missed_deadlines=0,
    )

    def step(_index: int):
        clock.advance(0.003)
        return {}

    with pytest.raises(RuntimeError, match="deadline"):
        run_realtime_control_loop(1, step, config=config, clock=clock)


def test_cloud_native_manifests_are_deterministic_and_secret_free() -> None:
    """Cloud deployment export should generate usable Kubernetes and Compose specs."""

    spec = CloudDeploymentSpec(
        name="scpn-qc",
        image="registry.example/scpn-quantum-control:0.9.7",
        command=("scpn-bench", "stable-core-contract-gate"),
        replicas=2,
        resources=ContainerResources(cpu="1000m", memory="1Gi"),
        env={"SCPN_EXECUTION_MODE": "offline"},
    )

    bundle = generate_cloud_manifests(spec)

    assert bundle.sha256 == generate_cloud_manifests(spec).sha256
    assert "deployment.yaml" in bundle.files
    assert "service.yaml" in bundle.files
    assert "docker-compose.yaml" in bundle.files
    assert "replicas: 2" in bundle.files["deployment.yaml"]
    assert "readOnlyRootFilesystem: true" in bundle.files["deployment.yaml"]
    assert "stable-core-contract-gate" in bundle.files["docker-compose.yaml"]


def test_cloud_native_manifests_reject_secret_like_env_and_bad_resources() -> None:
    """Deployment specs must not turn credentials into public manifests."""

    with pytest.raises(ValueError, match="secret"):
        CloudDeploymentSpec(
            name="bad",
            image="repo/scpn:latest",
            env={"IBM_TOKEN": "leak"},
        )
    with pytest.raises(ValueError, match="memory"):
        ContainerResources(cpu="500m", memory="1TB")


def test_compiler_realtime_and_deployment_api_exported_from_package_root() -> None:
    """The new production surfaces should be stable package-root imports."""

    import scpn_quantum_control as scpn

    assert scpn.MLIRCompileConfig is MLIRCompileConfig
    assert scpn.compile_kuramoto_to_mlir is compile_kuramoto_to_mlir
    assert scpn.RealtimeRuntimeConfig is RealtimeRuntimeConfig
    assert scpn.run_realtime_control_loop is run_realtime_control_loop
    assert scpn.CloudDeploymentSpec is CloudDeploymentSpec
    assert scpn.generate_cloud_manifests is generate_cloud_manifests
