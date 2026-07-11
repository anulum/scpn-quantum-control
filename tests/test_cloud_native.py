# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Cloud-Native Deployment Tests
"""Manifest and package-export tests for cloud-native deployment."""

from __future__ import annotations

import pytest

import scpn_quantum_control as scpn
from scpn_quantum_control.deployment.cloud_native import (
    CloudDeploymentSpec,
    ContainerResources,
    generate_cloud_manifests,
)


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


def test_cloud_native_api_exported_from_package_root() -> None:
    """Cloud deployment surfaces should remain stable package-root imports."""

    assert scpn.CloudDeploymentSpec is CloudDeploymentSpec
    assert scpn.generate_cloud_manifests is generate_cloud_manifests
