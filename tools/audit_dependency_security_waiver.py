# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — constrained dependency advisory waiver audit
"""Validate the single dependency advisory waiver used by CI.

The repository cannot raise ``setuptools`` above 81.0.0 while the Amazon
Braket simulator and schema distributions require that exact version. This
audit keeps the temporary exception narrow: the project must build with
Hatchling, all hashed CI locks must retain the source-verified pin, both
installed Braket distributions must declare it, and CI may ignore only
``PYSEC-2026-3447``.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import tomllib
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

ADVISORY_ID = "PYSEC-2026-3447"
SETUPTOOLS_VERSION = "81.0.0"
FIXED_SETUPTOOLS_VERSION = Version("83.0.0")
BUILD_BACKEND = "hatchling.build"
BRAKET_DISTRIBUTIONS = (
    "amazon-braket-default-simulator",
    "amazon-braket-schemas",
)
LOCK_PATHS = (
    "requirements-ci-py311-linux.txt",
    "requirements-ci-py312-linux.txt",
    "requirements-ci-py313-linux.txt",
)
SETUPTOOLS_HASHES = frozenset(
    {
        "487b53915f52501f0a79ccfd0c02c165ffe06631443a886740b91af4b7a5845a",
        "fdd925d5c5d9f62e4b74b30d6dd7828ce236fd6ed998a08d81de62ce5a6310d6",
    }
)
EXPECTED_PIP_AUDIT_COMMAND = (
    "pip-audit",
    "-r",
    "requirements-ci-py312-linux.txt",
    "--no-deps",
    "--desc",
    "on",
    "--ignore-vuln",
    ADVISORY_ID,
)
WAIVER_GATE_COMMAND = "python tools/audit_dependency_security_waiver.py"
WAIVER_DOC_HEADING = "### Braket-constrained setuptools advisory waiver"
DEPENDABOT_CONFIG_PATH = ".github/dependabot.yml"
DEPENDABOT_WAIVER_DEPENDENCY = "setuptools"
_RUN_KEY_RE = re.compile(
    r"^(?P<indent>\s*)(?P<sequence>-\s+)?(?P<quote>['\"]?)run(?P=quote):\s*"
    r"(?P<value>.*)$"
)
_JOBS_KEY_RE = re.compile(r"^jobs:\s*(?:#.*)?$")
_SECURITY_JOB_KEY_RE = re.compile(r"^  security:\s*(?:#.*)?$")
_SECURITY_STEPS_KEY_RE = re.compile(r"^    steps:\s*(?:#.*)?$")
_SECURITY_STEP_ITEM_RE = re.compile(r"^      -\s+")
_TOP_LEVEL_DEFAULTS_RE = re.compile(r"^(?P<quote>['\"]?)defaults(?P=quote)\s*:")
_SECURITY_EXECUTION_CONTROL_RE = re.compile(
    r"^\s*(?:-\s+)?(?P<quote>['\"]?)"
    r"(?P<key>if|continue-on-error|shell|defaults|<<)(?P=quote)\s*:"
)


class _YamlMappingKeyAuditModule(Protocol):
    """Typed dynamic import surface supporting script and package execution."""

    def has_escaped_double_quoted_mapping_key(self, source: str) -> bool:
        """Inspect mapping-key spelling in one YAML document."""


class _YamlComposer(Protocol):
    """Minimal typed PyYAML compose surface used for duplicate-safe validation."""

    def compose(self, stream: str) -> object | None:
        """Compose one YAML document without collapsing duplicate keys."""


def _load_yaml_mapping_key_audit() -> _YamlMappingKeyAuditModule:
    """Load the semantic helper in package and direct-script contexts."""
    try:
        module = importlib.import_module("tools.yaml_mapping_key_audit")
    except ModuleNotFoundError as exc:
        if exc.name not in {"tools", "tools.yaml_mapping_key_audit"}:
            raise
        module = importlib.import_module("yaml_mapping_key_audit")
    return cast(_YamlMappingKeyAuditModule, module)


_YAML_KEY_AUDIT = _load_yaml_mapping_key_audit()


def _yaml_attribute(node: object, name: str) -> object:
    """Return one dynamically provided PyYAML node attribute."""
    return cast(object, getattr(node, name, None))


def _yaml_kind(node: object) -> str:
    """Return the validated kind of one composed YAML node."""
    kind = _yaml_attribute(node, "id")
    if not isinstance(kind, str):
        raise ValueError("composed YAML node has no string id")
    return kind


def _yaml_scalar(node: object) -> str:
    """Return one scalar node value or fail closed on another node kind."""
    if _yaml_kind(node) != "scalar":
        raise ValueError("expected a scalar YAML node")
    value = _yaml_attribute(node, "value")
    if not isinstance(value, str):
        raise ValueError("composed YAML scalar has no string value")
    return value


def _yaml_sequence(node: object) -> list[object]:
    """Return duplicate-preserving sequence children."""
    if _yaml_kind(node) != "sequence":
        raise ValueError("expected a sequence YAML node")
    value = _yaml_attribute(node, "value")
    if not isinstance(value, list):
        raise ValueError("composed YAML sequence has invalid children")
    return cast(list[object], value)


def _yaml_mapping(node: object) -> list[tuple[object, object]]:
    """Return duplicate-preserving mapping entries."""
    if _yaml_kind(node) != "mapping":
        raise ValueError("expected a mapping YAML node")
    value = _yaml_attribute(node, "value")
    if not isinstance(value, list):
        raise ValueError("composed YAML mapping has invalid entries")
    entries: list[tuple[object, object]] = []
    for entry in cast(list[object], value):
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise ValueError("composed YAML mapping has an invalid entry")
        entries.append((entry[0], entry[1]))
    return entries


def _yaml_values(node: object, key: str) -> list[object]:
    """Return every value for a scalar mapping key, retaining duplicates."""
    return [value for candidate, value in _yaml_mapping(node) if _yaml_scalar(candidate) == key]


def audit_dependabot_config(config_text: str) -> tuple[str, ...]:
    """Require Dependabot to honor the active upstream-constrained waiver.

    Parameters
    ----------
    config_text:
        Complete ``.github/dependabot.yml`` source.

    Returns
    -------
    tuple[str, ...]
        Missing, duplicated, malformed, or weakened waiver conditions.

    """
    try:
        if _YAML_KEY_AUDIT.has_escaped_double_quoted_mapping_key(config_text):
            return ("Dependabot configuration contains an escaped mapping key",)
        composer = cast(_YamlComposer, importlib.import_module("yaml"))
        root = composer.compose(config_text)
        if root is None:
            raise ValueError("document is empty")
        update_nodes = _yaml_values(root, "updates")
        if len(update_nodes) != 1:
            raise ValueError("expected exactly one updates mapping")

        root_pip_updates: list[object] = []
        for update in _yaml_sequence(update_nodes[0]):
            ecosystems = _yaml_values(update, "package-ecosystem")
            directories = _yaml_values(update, "directory")
            if len(ecosystems) != 1 or len(directories) != 1:
                continue
            if _yaml_scalar(ecosystems[0]) == "pip" and _yaml_scalar(directories[0]) == "/":
                root_pip_updates.append(update)
        if len(root_pip_updates) != 1:
            raise ValueError("expected exactly one root pip update entry")

        ignore_nodes = _yaml_values(root_pip_updates[0], "ignore")
        if len(ignore_nodes) != 1:
            raise ValueError("root pip update must define exactly one ignore sequence")
        setuptools_rules: list[object] = []
        for rule in _yaml_sequence(ignore_nodes[0]):
            dependencies = _yaml_values(rule, "dependency-name")
            if len(dependencies) == 1 and _yaml_scalar(dependencies[0]) == (
                DEPENDABOT_WAIVER_DEPENDENCY
            ):
                setuptools_rules.append(rule)
        if len(setuptools_rules) != 1:
            raise ValueError("expected exactly one setuptools ignore rule")
        rule_keys = [_yaml_scalar(key) for key, _value in _yaml_mapping(setuptools_rules[0])]
        if rule_keys != ["dependency-name"]:
            raise ValueError("setuptools ignore must be unconditional and dependency-wide")
    except (ImportError, ValueError) as exc:
        return (f"Dependabot waiver configuration is invalid: {exc}",)
    return ()


@dataclass(frozen=True)
class DistributionRecord:
    """Installed distribution metadata relevant to the waiver.

    Parameters
    ----------
    name : str
        Canonical distribution name.
    version : str
        Installed PEP 440 version string.
    requirements : tuple[str, ...]
        Raw ``Requires-Dist`` values from package metadata.

    """

    name: str
    version: str
    requirements: tuple[str, ...]


@dataclass(frozen=True)
class LockPin:
    """One parsed ``setuptools`` lockfile stanza.

    Parameters
    ----------
    version : str
        Exact locked version.
    hashes : frozenset[str]
        SHA-256 distribution hashes accepted by pip.
    via : frozenset[str]
        Distributions recorded by pip-compile as pin owners.

    """

    version: str
    hashes: frozenset[str]
    via: frozenset[str]


@dataclass(frozen=True)
class WorkflowRun:
    """One parsed GitHub Actions ``run`` scalar.

    Parameters
    ----------
    command : str
        Unquoted inline value or normalised block-scalar body.
    starts_step : bool
        Whether the ``run`` key starts a sequence item.
    indent : int
        Number of leading spaces on the key line. Protected commands must be
        direct keys of a canonical security step, not nested lookalikes.

    """

    command: str
    starts_step: bool
    indent: int


@dataclass(frozen=True)
class WaiverAuditResult:
    """Outcome of validating the repository waiver boundary.

    Parameters
    ----------
    errors : tuple[str, ...]
        Deterministically ordered policy violations.

    """

    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether every waiver condition holds."""
        return not self.errors


def installed_distribution_record(name: str) -> DistributionRecord:
    """Read one installed distribution through standard metadata.

    Parameters
    ----------
    name:
        Distribution name accepted by ``importlib.metadata``.

    Returns
    -------
    DistributionRecord
        Version and dependency metadata used by the audit.

    Raises
    ------
    importlib.metadata.PackageNotFoundError
        If the CI lock did not install the requested distribution.

    """
    distribution = importlib.metadata.distribution(name)
    return DistributionRecord(
        name=distribution.metadata["Name"],
        version=distribution.version,
        requirements=tuple(distribution.requires or ()),
    )


def installed_distribution_records(
    names: Sequence[str] = BRAKET_DISTRIBUTIONS,
) -> tuple[tuple[DistributionRecord, ...], tuple[str, ...]]:
    """Read installed metadata without hiding missing distributions.

    Parameters
    ----------
    names:
        Distribution names expected in the active environment.

    Returns
    -------
    tuple[tuple[DistributionRecord, ...], tuple[str, ...]]
        Available metadata followed by explicit missing-distribution errors.

    """
    records: list[DistributionRecord] = []
    errors: list[str] = []
    for name in names:
        try:
            records.append(installed_distribution_record(name))
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"required installed distribution is missing: {name}")
    return tuple(records), tuple(errors)


def parse_setuptools_lock_pin(text: str) -> LockPin:
    """Parse the unique ``setuptools`` stanza from a pip-compile lock.

    Parameters
    ----------
    text:
        Complete UTF-8 lockfile content.

    Returns
    -------
    LockPin
        Exact version, hashes, and pip-compile owners.

    Raises
    ------
    ValueError
        If the lock omits or duplicates the pin, or its stanza is malformed.

    """
    lines = text.splitlines()
    pin_indices = [index for index, line in enumerate(lines) if line.startswith("setuptools==")]
    if len(pin_indices) != 1:
        raise ValueError(f"expected one setuptools pin, found {len(pin_indices)}")
    index = pin_indices[0]
    version_field = lines[index].removeprefix("setuptools==").split(maxsplit=1)
    if not version_field:
        raise ValueError("setuptools pin has no version")
    version = version_field[0]
    hashes: set[str] = set()
    cursor = index + 1
    while cursor < len(lines):
        stripped = lines[cursor].strip()
        prefix = "--hash=sha256:"
        if not stripped.startswith(prefix):
            break
        digest = stripped.removeprefix(prefix).removesuffix(" \\")
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"setuptools pin has invalid sha256: {digest}")
        hashes.add(digest)
        cursor += 1
    via: set[str] = set()
    while cursor < len(lines) and lines[cursor].lstrip().startswith("#"):
        owner = lines[cursor].lstrip().removeprefix("#").strip()
        if owner and owner != "via":
            via.add(canonicalize_name(owner))
        cursor += 1
    return LockPin(version=version, hashes=frozenset(hashes), via=frozenset(via))


def audit_build_system(pyproject_text: str) -> tuple[str, ...]:
    """Validate that this project neither builds with nor requires setuptools.

    Parameters
    ----------
    pyproject_text:
        Complete ``pyproject.toml`` content.

    Returns
    -------
    tuple[str, ...]
        Build-boundary violations.

    """
    payload = cast(dict[str, object], tomllib.loads(pyproject_text))
    build_system = payload.get("build-system")
    if not isinstance(build_system, dict):
        return ("pyproject.toml has no [build-system] table",)
    table = cast(dict[str, object], build_system)
    errors: list[str] = []
    if table.get("build-backend") != BUILD_BACKEND:
        errors.append(f"project build backend must remain {BUILD_BACKEND}")
    requirements = table.get("requires")
    if not isinstance(requirements, list) or not all(
        isinstance(item, str) for item in requirements
    ):
        errors.append("build-system.requires must be a string array")
        return tuple(errors)
    for raw in cast(list[str], requirements):
        try:
            requirement = Requirement(raw)
        except InvalidRequirement:
            errors.append(f"invalid build requirement: {raw}")
            continue
        if canonicalize_name(requirement.name) == "setuptools":
            errors.append("project build requirements must not include setuptools")
    return tuple(errors)


def audit_distribution_records(records: Sequence[DistributionRecord]) -> tuple[str, ...]:
    """Validate exact Braket ownership of the vulnerable transitive pin.

    Parameters
    ----------
    records:
        Installed metadata for the two Braket distributions.

    Returns
    -------
    tuple[str, ...]
        Metadata violations that invalidate the exception.

    """
    expected_names = frozenset(canonicalize_name(name) for name in BRAKET_DISTRIBUTIONS)
    actual_names = [canonicalize_name(record.name) for record in records]
    errors: list[str] = []
    if frozenset(actual_names) != expected_names or len(actual_names) != len(expected_names):
        errors.append("installed metadata must cover exactly both Braket pin owners")
    for record in records:
        canonical_name = canonicalize_name(record.name)
        try:
            Version(record.version)
        except InvalidVersion:
            errors.append(f"{canonical_name} has invalid installed version: {record.version}")
        setuptools_requirements: list[Requirement] = []
        for raw in record.requirements:
            try:
                requirement = Requirement(raw)
            except InvalidRequirement:
                errors.append(f"{canonical_name} has invalid Requires-Dist metadata: {raw}")
                continue
            if canonicalize_name(requirement.name) == "setuptools":
                setuptools_requirements.append(requirement)
        if len(setuptools_requirements) != 1:
            errors.append(
                f"{canonical_name} must declare exactly one setuptools requirement; "
                f"found {len(setuptools_requirements)}"
            )
            continue
        requirement = setuptools_requirements[0]
        if (
            str(requirement.specifier) != f"=={SETUPTOOLS_VERSION}"
            or requirement.marker is not None
        ):
            errors.append(f"{canonical_name} no longer hard-pins setuptools=={SETUPTOOLS_VERSION}")
    return tuple(errors)


def audit_lockfiles(lock_texts: dict[str, str]) -> tuple[str, ...]:
    """Validate the locked version, hashes, and transitive owners.

    Parameters
    ----------
    lock_texts:
        Mapping from canonical CI lock path to complete text.

    Returns
    -------
    tuple[str, ...]
        Missing or drifted lock conditions.

    """
    errors: list[str] = []
    expected_via = frozenset(canonicalize_name(name) for name in BRAKET_DISTRIBUTIONS)
    for path in LOCK_PATHS:
        text = lock_texts.get(path)
        if text is None:
            errors.append(f"missing CI lock: {path}")
            continue
        try:
            pin = parse_setuptools_lock_pin(text)
        except ValueError as exc:
            errors.append(f"{path}: {exc}")
            continue
        if pin.version != SETUPTOOLS_VERSION:
            errors.append(
                f"{path}: setuptools pin is {pin.version}, expected {SETUPTOOLS_VERSION}"
            )
        if pin.hashes != SETUPTOOLS_HASHES:
            errors.append(f"{path}: setuptools distribution hashes drifted")
        if pin.via != expected_via:
            errors.append(f"{path}: setuptools pin owners drifted: {sorted(pin.via)}")
    return tuple(errors)


def _workflow_runs(workflow_text: str) -> tuple[WorkflowRun, ...]:
    """Extract inline and block ``run`` scalars from a workflow."""
    lines = workflow_text.splitlines()
    runs: list[WorkflowRun] = []
    index = 0
    block_markers = frozenset({"|", "|-", "|+", ">", ">-", ">+"})
    while index < len(lines):
        match = _RUN_KEY_RE.match(lines[index])
        if match is None:
            index += 1
            continue
        value = match.group("value").strip()
        starts_step = match.group("sequence") is not None
        if value not in block_markers:
            runs.append(
                WorkflowRun(
                    command=value,
                    starts_step=starts_step,
                    indent=len(match.group("indent")),
                )
            )
            index += 1
            continue
        key_indent = len(match.group("indent"))
        index += 1
        block_lines: list[str] = []
        while index < len(lines):
            line = lines[index]
            if line.strip():
                indent = len(line) - len(line.lstrip())
                if indent <= key_indent:
                    break
            block_lines.append(line)
            index += 1
        non_empty_indents = [
            len(line) - len(line.lstrip()) for line in block_lines if line.strip()
        ]
        block_indent = min(non_empty_indents, default=key_indent + 2)
        runs.append(
            WorkflowRun(
                command="\n".join(
                    line[block_indent:] if line.strip() else "" for line in block_lines
                ),
                starts_step=starts_step,
                indent=key_indent,
            )
        )
    return tuple(runs)


def _workflow_run_values(workflow_text: str) -> tuple[str, ...]:
    """Return only command text from parsed workflow run scalars."""
    return tuple(run.command for run in _workflow_runs(workflow_text))


def _indented_yaml_block(
    lines: Sequence[str], header_index: int, header_indent: int
) -> tuple[str, ...]:
    """Return the lines nested below one canonical YAML mapping key."""
    block: list[str] = []
    for line in lines[header_index + 1 :]:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            indent = len(line) - len(line.lstrip())
            if indent <= header_indent:
                break
        block.append(line)
    return tuple(block)


def _security_job_text(workflow_text: str) -> str | None:
    """Return the canonical ``jobs.security`` body or ``None`` on ambiguity."""
    lines = workflow_text.splitlines()
    jobs_headers = [index for index, line in enumerate(lines) if _JOBS_KEY_RE.fullmatch(line)]
    if len(jobs_headers) != 1:
        return None
    jobs_block = _indented_yaml_block(lines, jobs_headers[0], 0)
    security_headers = [
        index for index, line in enumerate(jobs_block) if _SECURITY_JOB_KEY_RE.fullmatch(line)
    ]
    if len(security_headers) != 1:
        return None
    security_block = _indented_yaml_block(jobs_block, security_headers[0], 2)
    return "\n".join(security_block)


def _security_step_blocks(security_job_text: str) -> tuple[str, ...] | None:
    """Return canonical step bodies, preserving indentation and duplicate keys."""
    lines = security_job_text.splitlines()
    steps_headers = [
        index for index, line in enumerate(lines) if _SECURITY_STEPS_KEY_RE.fullmatch(line)
    ]
    if len(steps_headers) != 1:
        return None
    steps_block = _indented_yaml_block(lines, steps_headers[0], 4)
    step_starts = [
        index for index, line in enumerate(steps_block) if _SECURITY_STEP_ITEM_RE.match(line)
    ]
    if not step_starts:
        return ()
    return tuple(
        "\n".join(steps_block[start:end])
        for start, end in zip(step_starts, (*step_starts[1:], len(steps_block)), strict=True)
    )


def _protected_command_owns_step(
    step_blocks: Sequence[str], expected_tokens: tuple[str, ...]
) -> bool:
    """Return whether one canonical step contains only the protected run key."""
    matching_steps: list[tuple[WorkflowRun, ...]] = []
    for step_block in step_blocks:
        runs = _workflow_runs(step_block)
        if any(_shell_tokens(run.command) == expected_tokens for run in runs):
            matching_steps.append(runs)
    if len(matching_steps) != 1 or len(matching_steps[0]) != 1:
        return False
    run = matching_steps[0][0]
    direct_step_key = run.starts_step and run.indent == 6
    direct_named_step_key = not run.starts_step and run.indent == 8
    return direct_step_key or direct_named_step_key


def _shell_tokens(command: str) -> tuple[str, ...]:
    """Tokenise a workflow shell command while ignoring shell comments."""
    return tuple(shlex.split(command, comments=True, posix=True))


def audit_ci_workflow(workflow_text: str) -> tuple[str, ...]:
    """Validate exact, unconditional, blocking security-job commands.

    Parameters
    ----------
    workflow_text:
        Complete CI workflow YAML.

    Returns
    -------
    tuple[str, ...]
        Workflow wiring violations.

    """
    run_values = _workflow_run_values(workflow_text)
    errors: list[str] = []
    gate_tokens = _shell_tokens(WAIVER_GATE_COMMAND)
    tokenised_runs = tuple(_shell_tokens(value) for value in run_values)
    gate_runs = [
        tokens
        for tokens in tokenised_runs
        if any(
            tokens[index : index + len(gate_tokens)] == gate_tokens
            for index in range(len(tokens) - len(gate_tokens) + 1)
        )
    ]
    gate_command_valid = gate_runs == [gate_tokens]
    if not gate_command_valid:
        errors.append("CI must run the waiver audit exactly once")
    commands = [tokens for tokens in tokenised_runs if "pip-audit" in tokens]
    pip_audit_command_valid = commands == [EXPECTED_PIP_AUDIT_COMMAND]
    if not pip_audit_command_valid:
        errors.append(
            f"CI pip-audit command must scan the full lock and ignore only {ADVISORY_ID}"
        )
    try:
        escaped_mapping_key = _YAML_KEY_AUDIT.has_escaped_double_quoted_mapping_key(workflow_text)
    except ValueError:
        errors.append("CI workflow must be valid YAML for semantic mapping-key audit")
    else:
        if escaped_mapping_key:
            errors.append("CI must not encode mapping keys with YAML escapes")
    if any(_TOP_LEVEL_DEFAULTS_RE.match(line) for line in workflow_text.splitlines()):
        errors.append("CI must not override run defaults at workflow scope")

    security_job_text = _security_job_text(workflow_text)
    if security_job_text is None:
        errors.append("CI must define exactly one canonical jobs.security mapping")
        return tuple(errors)

    step_blocks = _security_step_blocks(security_job_text)
    if not step_blocks:
        errors.append("jobs.security must define one canonical non-empty steps sequence")
    else:
        if gate_command_valid and not _protected_command_owns_step(step_blocks, gate_tokens):
            errors.append("waiver audit must own a standalone jobs.security run step")
        if pip_audit_command_valid and not _protected_command_owns_step(
            step_blocks, EXPECTED_PIP_AUDIT_COMMAND
        ):
            errors.append("pip-audit must own a standalone jobs.security run step")

    execution_controls: list[str] = []
    for line in security_job_text.splitlines():
        match = _SECURITY_EXECUTION_CONTROL_RE.match(line)
        if match is not None and match.group("key") not in execution_controls:
            execution_controls.append(match.group("key"))
    for control in execution_controls:
        errors.append(f"jobs.security must not define execution control: {control}")
    return tuple(errors)


def _imports_setuptools(source: str) -> bool:
    """Return whether parsed Python imports setuptools or one of its modules."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import) and any(
            alias.name == "setuptools" or alias.name.startswith("setuptools.")
            for alias in node.names
        ):
            return True
        if (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and (node.module == "setuptools" or node.module.startswith("setuptools."))
        ):
            return True
    return False


def audit_setuptools_imports(repo_root: Path) -> tuple[str, ...]:
    """Reject project Python code that imports the waived build dependency.

    Parameters
    ----------
    repo_root:
        Repository root containing ``src``, ``scripts``, and ``tools``.

    Returns
    -------
    tuple[str, ...]
        Repository-relative Python files that import setuptools.

    """
    errors: list[str] = []
    for directory in ("src", "scripts", "tools"):
        root = repo_root / directory
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            relative_path = path.relative_to(repo_root).as_posix()
            try:
                imports_setuptools = _imports_setuptools(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError) as exc:
                errors.append(f"cannot audit Python imports in {relative_path}: {exc}")
                continue
            if imports_setuptools:
                errors.append(f"project code imports setuptools: {relative_path}")
    return tuple(errors)


def audit_operator_documentation(documentation_text: str) -> tuple[str, ...]:
    """Require the operator-facing threat boundary and removal condition.

    Parameters
    ----------
    documentation_text:
        Complete test-infrastructure guide.

    Returns
    -------
    tuple[str, ...]
        Missing stable documentation markers.

    """
    required_markers = (
        WAIVER_DOC_HEADING,
        ADVISORY_ID,
        f"setuptools=={SETUPTOOLS_VERSION}",
        BUILD_BACKEND,
        DEPENDABOT_CONFIG_PATH,
        "Remove the waiver",
    )
    missing = tuple(marker for marker in required_markers if marker not in documentation_text)
    return tuple(f"operator documentation is missing: {marker}" for marker in missing)


def audit_repository(
    repo_root: Path,
    records: Sequence[DistributionRecord] | None = None,
) -> WaiverAuditResult:
    """Audit the live repository and installed Braket dependency metadata.

    Parameters
    ----------
    repo_root:
        Repository root containing the CI locks and policy surfaces.
    records:
        Optional recorded distribution metadata. ``None`` reads the installed
        CI environment; explicit records support deterministic offline replay.

    Returns
    -------
    WaiverAuditResult
        All detected violations in stable check order.

    """
    errors: list[str] = []
    try:
        pyproject_text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
        errors.extend(audit_build_system(pyproject_text))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        errors.append(f"cannot read pyproject.toml: {exc}")

    lock_texts: dict[str, str] = {}
    for path in LOCK_PATHS:
        try:
            lock_texts[path] = (repo_root / path).read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"cannot read {path}: {exc}")
    errors.extend(audit_lockfiles(lock_texts))

    try:
        workflow_text = (repo_root / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        errors.extend(audit_ci_workflow(workflow_text))
    except OSError as exc:
        errors.append(f"cannot read CI workflow: {exc}")

    try:
        dependabot_text = (repo_root / DEPENDABOT_CONFIG_PATH).read_text(encoding="utf-8")
        errors.extend(audit_dependabot_config(dependabot_text))
    except OSError as exc:
        errors.append(f"cannot read Dependabot configuration: {exc}")

    try:
        documentation_text = (repo_root / "docs" / "test_infrastructure.md").read_text(
            encoding="utf-8"
        )
        errors.extend(audit_operator_documentation(documentation_text))
    except OSError as exc:
        errors.append(f"cannot read operator documentation: {exc}")

    errors.extend(audit_setuptools_imports(repo_root))
    if records is None:
        records, metadata_errors = installed_distribution_records()
        errors.extend(metadata_errors)
    errors.extend(audit_distribution_records(records))
    return WaiverAuditResult(errors=tuple(errors))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dependency advisory waiver audit.

    Parameters
    ----------
    argv:
        Optional command arguments. Defaults to ``sys.argv`` through argparse.

    Returns
    -------
    int
        Zero for a valid exception boundary, one for policy violations.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root to audit.",
    )
    args = parser.parse_args(argv)
    result = audit_repository(args.repo_root.resolve())
    if result.errors:
        print("dependency security waiver audit: FAIL")
        for error in result.errors:
            print(f"  - {error}")
        return 1
    print(
        "dependency security waiver audit: PASS "
        f"({ADVISORY_ID}; remove when Braket no longer requires "
        f"setuptools<{FIXED_SETUPTOOLS_VERSION})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
