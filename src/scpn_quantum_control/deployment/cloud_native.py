# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- cloud-native deployment manifests
"""Deterministic cloud-native manifest generation.

The generator emits Kubernetes and Docker Compose manifests for offline SCPN
workloads. It rejects secret-like environment variables and does not read local
credentials, create clusters, or contact cloud APIs.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

_NAME_RE = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
_CPU_RE = re.compile(r"^([1-9][0-9]*m|[1-9][0-9]*)$")
_MEMORY_RE = re.compile(r"^[1-9][0-9]*(Mi|Gi)$")
_SECRET_ENV_RE = re.compile(r"(TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY|PRIVATE)", re.I)


@dataclass(frozen=True)
class ContainerResources:
    """CPU and memory requests/limits for one SCPN container."""

    cpu: str = "500m"
    memory: str = "512Mi"

    def __post_init__(self) -> None:
        if not _CPU_RE.match(self.cpu):
            raise ValueError("cpu must be an integer core count or Kubernetes millicore value")
        if not _MEMORY_RE.match(self.memory):
            raise ValueError("memory must use Mi or Gi units")


@dataclass(frozen=True)
class CloudDeploymentSpec:
    """Cloud-native deployment request for an offline SCPN workload."""

    name: str
    image: str
    command: Sequence[str] = ("scpn-bench", "stable-core-contract-gate")
    replicas: int = 1
    port: int = 8080
    resources: ContainerResources = field(default_factory=ContainerResources)
    env: Mapping[str, str] = field(default_factory=dict)
    service_account: str = "scpn-runtime"
    namespace: str = "default"

    def __post_init__(self) -> None:
        _validate_name(self.name, "name")
        _validate_name(self.service_account, "service_account")
        _validate_name(self.namespace, "namespace")
        if not self.image or any(char.isspace() for char in self.image):
            raise ValueError("image must be a non-empty image reference without whitespace")
        if not isinstance(self.replicas, int) or self.replicas < 1:
            raise ValueError("replicas must be a positive integer")
        if not isinstance(self.port, int) or not (1 <= self.port <= 65535):
            raise ValueError("port must be in [1, 65535]")
        if not self.command or any(not str(item) for item in self.command):
            raise ValueError("command must contain at least one non-empty item")
        env = dict(self.env)
        for key, value in env.items():
            if not key or not key.replace("_", "").isalnum():
                raise ValueError("environment variable names must be alphanumeric/underscore")
            if _SECRET_ENV_RE.search(key):
                raise ValueError("secret-like environment variables are not allowed in manifests")
            if "\n" in value:
                raise ValueError("environment variable values must be single-line strings")
        object.__setattr__(self, "command", tuple(str(item) for item in self.command))
        object.__setattr__(self, "env", MappingProxyType(env))


@dataclass(frozen=True)
class CloudManifestBundle:
    """Generated cloud-native manifest files plus provenance digest."""

    files: Mapping[str, str]
    sha256: str
    claim_boundary: str

    def __post_init__(self) -> None:
        files = dict(self.files)
        encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        expected = hashlib.sha256(encoded).hexdigest()
        if self.sha256 != expected:
            raise ValueError("sha256 must match manifest files")
        object.__setattr__(self, "files", MappingProxyType(files))


def generate_cloud_manifests(spec: CloudDeploymentSpec) -> CloudManifestBundle:
    """Generate Kubernetes and Docker Compose manifests for ``spec``."""
    files = {
        "deployment.yaml": _deployment_yaml(spec),
        "service.yaml": _service_yaml(spec),
        "docker-compose.yaml": _compose_yaml(spec),
    }
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return CloudManifestBundle(
        files=files,
        sha256=hashlib.sha256(encoded).hexdigest(),
        claim_boundary=(
            "deterministic manifest generation only; no cluster creation, "
            "credential loading, or hardware submission"
        ),
    )


def _deployment_yaml(spec: CloudDeploymentSpec) -> str:
    env_block = _yaml_env(spec.env, indent=10)
    command_block = _yaml_sequence(spec.command, indent=10)
    return (
        "apiVersion: apps/v1\n"
        "kind: Deployment\n"
        "metadata:\n"
        f"  name: {spec.name}\n"
        f"  namespace: {spec.namespace}\n"
        "spec:\n"
        f"  replicas: {spec.replicas}\n"
        "  selector:\n"
        "    matchLabels:\n"
        f"      app: {spec.name}\n"
        "  template:\n"
        "    metadata:\n"
        "      labels:\n"
        f"        app: {spec.name}\n"
        "    spec:\n"
        f"      serviceAccountName: {spec.service_account}\n"
        "      securityContext:\n"
        "        runAsNonRoot: true\n"
        "      containers:\n"
        f"        - name: {spec.name}\n"
        f"          image: {spec.image}\n"
        "          imagePullPolicy: IfNotPresent\n"
        "          command:\n"
        f"{command_block}"
        "          ports:\n"
        f"            - containerPort: {spec.port}\n"
        "          env:\n"
        f"{env_block}"
        "          resources:\n"
        "            requests:\n"
        f"              cpu: {spec.resources.cpu}\n"
        f"              memory: {spec.resources.memory}\n"
        "            limits:\n"
        f"              cpu: {spec.resources.cpu}\n"
        f"              memory: {spec.resources.memory}\n"
        "          securityContext:\n"
        "            allowPrivilegeEscalation: false\n"
        "            readOnlyRootFilesystem: true\n"
        "            capabilities:\n"
        "              drop:\n"
        "                - ALL\n"
    )


def _service_yaml(spec: CloudDeploymentSpec) -> str:
    return (
        "apiVersion: v1\n"
        "kind: Service\n"
        "metadata:\n"
        f"  name: {spec.name}\n"
        f"  namespace: {spec.namespace}\n"
        "spec:\n"
        "  type: ClusterIP\n"
        "  selector:\n"
        f"    app: {spec.name}\n"
        "  ports:\n"
        "    - name: http\n"
        f"      port: {spec.port}\n"
        f"      targetPort: {spec.port}\n"
    )


def _compose_yaml(spec: CloudDeploymentSpec) -> str:
    env_block = _yaml_mapping(spec.env, indent=6)
    command = " ".join(_shell_quote(item) for item in spec.command)
    return (
        "services:\n"
        f"  {spec.name}:\n"
        f"    image: {spec.image}\n"
        f"    command: {command}\n"
        "    read_only: true\n"
        "    security_opt:\n"
        "      - no-new-privileges:true\n"
        "    cap_drop:\n"
        "      - ALL\n"
        "    environment:\n"
        f"{env_block}"
        "    deploy:\n"
        "      resources:\n"
        "        limits:\n"
        f'          cpus: "{_compose_cpu(spec.resources.cpu)}"\n'
        f"          memory: {spec.resources.memory}\n"
    )


def _yaml_sequence(values: Sequence[str], *, indent: int) -> str:
    spaces = " " * indent
    return "".join(f'{spaces}- "{_escape_yaml(value)}"\n' for value in values)


def _yaml_env(env: Mapping[str, str], *, indent: int) -> str:
    if not env:
        return " " * indent + "[]\n"
    spaces = " " * indent
    return "".join(
        f'{spaces}- name: {key}\n{spaces}  value: "{_escape_yaml(env[key])}"\n'
        for key in sorted(env)
    )


def _yaml_mapping(env: Mapping[str, str], *, indent: int) -> str:
    if not env:
        return " " * indent + "{}\n"
    spaces = " " * indent
    return "".join(f'{spaces}{key}: "{_escape_yaml(env[key])}"\n' for key in sorted(env))


def _compose_cpu(cpu: str) -> str:
    if cpu.endswith("m"):
        return format(int(cpu[:-1]) / 1000.0, ".3f").rstrip("0").rstrip(".")
    return cpu


def _shell_quote(value: str) -> str:
    if re.match(r"^[A-Za-z0-9_./:=+-]+$", value):
        return value
    return '"' + _escape_yaml(value) + '"'


def _escape_yaml(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _validate_name(value: str, field_name: str) -> None:
    if not _NAME_RE.match(value):
        raise ValueError(f"{field_name} must be a DNS-1123 compatible name")


__all__ = [
    "CloudDeploymentSpec",
    "CloudManifestBundle",
    "ContainerResources",
    "generate_cloud_manifests",
]
