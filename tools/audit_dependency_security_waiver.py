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
import importlib.metadata
import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

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
_RUN_KEY_RE = re.compile(r"^(?P<indent>\s*)(?:-\s+)?run:\s*(?P<value>.*)$")


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


def _workflow_run_values(workflow_text: str) -> tuple[str, ...]:
    """Extract inline and block ``run`` scalars from a workflow."""
    lines = workflow_text.splitlines()
    values: list[str] = []
    index = 0
    block_markers = frozenset({"|", "|-", "|+", ">", ">-", ">+"})
    while index < len(lines):
        match = _RUN_KEY_RE.match(lines[index])
        if match is None:
            index += 1
            continue
        value = match.group("value").strip()
        if value not in block_markers:
            values.append(value)
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
        values.append(
            "\n".join(line[block_indent:] if line.strip() else "" for line in block_lines)
        )
    return tuple(values)


def _shell_tokens(command: str) -> tuple[str, ...]:
    """Tokenise a workflow shell command while ignoring shell comments."""
    return tuple(shlex.split(command, comments=True, posix=True))


def audit_ci_workflow(workflow_text: str) -> tuple[str, ...]:
    """Validate the exact security-gate command and its policy precheck.

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
    if gate_runs != [gate_tokens]:
        errors.append("CI must run the waiver audit exactly once")
    commands = [tokens for tokens in tokenised_runs if "pip-audit" in tokens]
    if commands != [EXPECTED_PIP_AUDIT_COMMAND]:
        errors.append(
            f"CI pip-audit command must scan the full lock and ignore only {ADVISORY_ID}"
        )
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
