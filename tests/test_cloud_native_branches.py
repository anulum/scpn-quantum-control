# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the cloud-native manifest generator
"""Guard and serialisation tests for the cloud-native deployment manifests.

Covers the container-resource and deployment-spec field guards, the manifest
bundle digest check, the empty YAML sequence/mapping branches, the compose CPU
formatting, the shell-quoting branch and the DNS-1123 name validator.
"""

from __future__ import annotations

from typing import Any

import pytest

from scpn_quantum_control.deployment.cloud_native import (
    CloudDeploymentSpec,
    CloudManifestBundle,
    ContainerResources,
    _compose_cpu,
    _shell_quote,
    _validate_name,
    _yaml_env,
    _yaml_mapping,
)


def _spec(**overrides: Any) -> CloudDeploymentSpec:
    kwargs: dict[str, Any] = {"name": "app", "image": "repo/img:1"}
    kwargs.update(overrides)
    return CloudDeploymentSpec(**kwargs)


def test_resources_reject_bad_cpu() -> None:
    """A malformed CPU request is rejected."""
    with pytest.raises(ValueError, match="cpu must be an integer core count"):
        ContainerResources(cpu="fast")


def test_spec_rejects_image_with_whitespace() -> None:
    """An image reference containing whitespace is rejected."""
    with pytest.raises(ValueError, match="image must be a non-empty image reference"):
        _spec(image="repo / img")


def test_spec_rejects_non_positive_replicas() -> None:
    """A non-positive replica count is rejected."""
    with pytest.raises(ValueError, match="replicas must be a positive integer"):
        _spec(replicas=0)


def test_spec_rejects_out_of_range_port() -> None:
    """A port outside the valid range is rejected."""
    with pytest.raises(ValueError, match=r"port must be in \[1, 65535\]"):
        _spec(port=70000)


def test_spec_rejects_empty_command_item() -> None:
    """A command with an empty entry is rejected."""
    with pytest.raises(ValueError, match="command must contain at least one non-empty item"):
        _spec(command=["run", ""])


def test_spec_rejects_bad_env_name() -> None:
    """An environment variable name with invalid characters is rejected."""
    with pytest.raises(ValueError, match="environment variable names must be"):
        _spec(env={"bad-name": "value"})


def test_spec_rejects_multiline_env_value() -> None:
    """A multi-line environment variable value is rejected."""
    with pytest.raises(ValueError, match="environment variable values must be single-line"):
        _spec(env={"GOOD": "line1\nline2"})


def test_manifest_bundle_rejects_digest_mismatch() -> None:
    """A manifest bundle whose digest does not match its files is rejected."""
    with pytest.raises(ValueError, match="sha256 must match manifest files"):
        CloudManifestBundle(files={"a.yaml": "x"}, sha256="0" * 64, claim_boundary="none")


def test_yaml_env_renders_empty_sequence() -> None:
    """An empty environment renders as an empty YAML sequence."""
    assert _yaml_env({}, indent=4) == "    []\n"


def test_yaml_mapping_renders_empty_mapping() -> None:
    """An empty mapping renders as an empty YAML mapping."""
    assert _yaml_mapping({}, indent=2) == "  {}\n"


def test_compose_cpu_passes_through_core_count() -> None:
    """An integer core count is passed through unchanged."""
    assert _compose_cpu("2") == "2"


def test_shell_quote_wraps_special_values() -> None:
    """A value with shell-special characters is quoted."""
    assert _shell_quote("a b") == '"a b"'


def test_validate_name_rejects_non_dns1123() -> None:
    """A non-DNS-1123 name is rejected."""
    with pytest.raises(ValueError, match="must be a DNS-1123 compatible name"):
        _validate_name("Bad_Name", "name")
