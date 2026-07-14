# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — CI build-environment closure audit
"""Validate the lock-faithful Hatchling wheel-build environment.

Release-wheel tests deliberately use ``python -m build --no-isolation`` so
they exercise the exact environment installed by CI and Docker. This audit
keeps the build frontend and backend in the authoritative development input,
all three hash-pinned Python locks, both Hatchling projects, and the active
CI/Docker consumers as one fail-closed contract.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import cast

import tomllib
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

BUILD_BACKEND = "hatchling.build"
DEVELOPMENT_INPUT = "requirements-dev.txt"
PROJECT_PATHS = ("pyproject.toml", "oscillatools/pyproject.toml")
LOCKS_BY_PYTHON = (
    ("3.11", "requirements-ci-py311-linux.txt"),
    ("3.12", "requirements-ci-py312-linux.txt"),
    ("3.13", "requirements-ci-py313-linux.txt"),
)
CI_WORKFLOW_PATH = ".github/workflows/ci.yml"
DOCKERFILE_PATH = "Dockerfile"
WHEEL_TEST_PATH = "tests/test_wheel_contents.py"
AUDIT_COMMAND = "python tools/audit_ci_build_environment.py"
LOCK_INSTALL_COMMAND = "python -m pip install --require-hashes -r ${{ matrix.requirements-file }}"
DOCKER_INSTALL_COMMAND = (
    "pip install --no-cache-dir --require-hashes -r requirements-ci-py312-linux.txt"
)
DOCKER_CONTEXT_COPIES = (
    "COPY Dockerfile Dockerfile",
    "COPY oscillatools/README.md oscillatools/README.md",
)
"""Build metadata that must survive into the reproduction image."""
SECURITY_JOB_ENVIRONMENT = (
    "  security:\n"
    "    needs: lint\n"
    "    runs-on: ubuntu-latest\n"
    "    env:\n"
    "      PYTHONPATH: ${{ github.workspace }}/src:"
    "${{ github.workspace }}/oscillatools/src"
)
_DIRECT_OWNER = frozenset({"-r requirements-dev.txt"})
_LOCK_PIN_RE = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[^\\\s]+)(?:\s+\\)?\s*$")
_ANY_LOCK_PIN_RE = re.compile(r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==[^\\\s]+(?:\s+\\)?\s*$")
_HASH_RE = re.compile(r"^\s+--hash=sha256:(?P<digest>[0-9a-f]{64})(?:\s+\\)?\s*$")
_VIA_RE = re.compile(r"^\s*#\s+via(?:\s+(?P<owner>.+))?\s*$")
_VIA_OWNER_RE = re.compile(r"^\s*#\s{3}(?P<owner>.+?)\s*$")


@dataclass(frozen=True)
class ExpectedBuildPin:
    """Expected direct build-tool pin and source-verified hashes.

    Parameters
    ----------
    name : str
        Canonical Python distribution name.
    version : str
        Exact version required by the development input and CI locks.
    hashes : frozenset[str]
        SHA-256 hashes published for that exact release on PyPI.

    """

    name: str
    version: str
    hashes: frozenset[str]


EXPECTED_BUILD_PINS = (
    ExpectedBuildPin(
        name="build",
        version="1.5.1",
        hashes=frozenset(
            {
                "94e17f1db803ab22f46049376c44c8437c52090f0dfdf1adc43df56542d644fb",
                "f1a58fe2e5af5b0238a07b9e70207492c79ddebbdb1ad954fc86d62a56be3e0d",
            }
        ),
    ),
    ExpectedBuildPin(
        name="hatchling",
        version="1.31.0",
        hashes=frozenset(
            {
                "6b48ad4068a482ed7239b3a8215bc55b47aad3345d58dfc94e553c5d2d46211b",
                "aac80bec8b6fe35e8480f1c335be8910fa210a0e6f735a139be205dadcacb544",
            }
        ),
    ),
)
HATCHLING_VERSION = next(pin.version for pin in EXPECTED_BUILD_PINS if pin.name == "hatchling")


@dataclass(frozen=True)
class LockPin:
    """Parsed build-tool stanza from one pip-compile lock.

    Parameters
    ----------
    version : str
        Exact locked distribution version.
    hashes : frozenset[str]
        SHA-256 hashes accepted by pip for the locked distribution.
    owners : frozenset[str]
        Pip-compile ``via`` owners recorded for the pin.

    """

    version: str
    hashes: frozenset[str]
    owners: frozenset[str]


@dataclass(frozen=True)
class BuildEnvironmentAuditResult:
    """Outcome of the CI build-environment closure audit.

    Parameters
    ----------
    errors : tuple[str, ...]
        Deterministically ordered contract violations.

    """

    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether every build-environment condition holds."""
        return not self.errors


def _mapping(value: object) -> dict[str, object] | None:
    """Return a string-keyed TOML mapping when ``value`` is one."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        return None
    return cast(dict[str, object], value)


def _string_array(value: object) -> tuple[str, ...] | None:
    """Return an immutable string array when every element is textual."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return tuple(cast(list[str], value))


def _audit_hatchling_requirement(
    requirements: tuple[str, ...],
    *,
    context: str,
) -> tuple[str, ...]:
    """Validate one unique Hatchling requirement admitting the locked version."""
    parsed: list[Requirement] = []
    errors: list[str] = []
    for text in requirements:
        try:
            parsed.append(Requirement(text))
        except InvalidRequirement:
            errors.append(f"{context} contains an invalid requirement: {text}")
    matches = [
        requirement for requirement in parsed if canonicalize_name(requirement.name) == "hatchling"
    ]
    if len(matches) != 1:
        errors.append(f"{context} must declare Hatchling exactly once; found {len(matches)}")
    elif not matches[0].specifier.contains(HATCHLING_VERSION, prereleases=True):
        errors.append(f"{context} does not admit the locked Hatchling {HATCHLING_VERSION}")
    return tuple(errors)


def audit_project_metadata(path: str, text: str) -> tuple[str, ...]:
    """Validate one Hatchling project's build metadata and root dev extra."""
    try:
        payload = cast(dict[str, object], tomllib.loads(text))
    except tomllib.TOMLDecodeError as exc:
        return (f"{path} is not valid TOML: {exc}",)

    errors: list[str] = []
    build_system = _mapping(payload.get("build-system"))
    if build_system is None:
        return (f"{path} has no valid [build-system] table",)
    if build_system.get("build-backend") != BUILD_BACKEND:
        errors.append(f"{path} build backend must be {BUILD_BACKEND}")
    build_requirements = _string_array(build_system.get("requires"))
    if build_requirements is None:
        errors.append(f"{path} build-system.requires must be a string array")
    else:
        errors.extend(
            _audit_hatchling_requirement(
                build_requirements,
                context=f"{path} build-system.requires",
            )
        )

    if path == PROJECT_PATHS[0]:
        project = _mapping(payload.get("project"))
        optional = _mapping(project.get("optional-dependencies")) if project else None
        dev = _string_array(optional.get("dev")) if optional else None
        if dev is None:
            errors.append(f"{path} project.optional-dependencies.dev must be a string array")
        else:
            errors.extend(
                _audit_hatchling_requirement(
                    dev,
                    context=f"{path} project.optional-dependencies.dev",
                )
            )
    return tuple(errors)


def parse_direct_requirements(text: str) -> tuple[Requirement, ...]:
    """Parse top-level requirements while ignoring includes and comments.

    Raises
    ------
    ValueError
        If a direct requirement is malformed.

    """
    parsed: list[Requirement] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "-r ")):
            continue
        try:
            parsed.append(Requirement(stripped))
        except InvalidRequirement as exc:
            raise ValueError(
                f"invalid direct requirement on line {line_number}: {stripped}"
            ) from exc
    return tuple(parsed)


def _exact_version(requirement: Requirement) -> str | None:
    """Return a single equality version, or ``None`` for a non-exact range."""
    specifiers = tuple(requirement.specifier)
    if len(specifiers) != 1 or specifiers[0].operator != "==":
        return None
    return specifiers[0].version


def audit_development_input(text: str) -> tuple[str, ...]:
    """Validate exact direct build-tool pins in ``requirements-dev.txt``."""
    try:
        requirements = parse_direct_requirements(text)
    except ValueError as exc:
        return (f"{DEVELOPMENT_INPUT}: {exc}",)

    errors: list[str] = []
    for expected in EXPECTED_BUILD_PINS:
        matches = [
            requirement
            for requirement in requirements
            if canonicalize_name(requirement.name) == expected.name
        ]
        if len(matches) != 1:
            errors.append(
                f"{DEVELOPMENT_INPUT} must pin {expected.name} exactly once; found {len(matches)}"
            )
            continue
        version = _exact_version(matches[0])
        if version != expected.version:
            errors.append(
                f"{DEVELOPMENT_INPUT} must pin {expected.name}=={expected.version}; "
                f"found {matches[0]}"
            )
    return tuple(errors)


def _lock_pin_matches(line: str, distribution: str) -> re.Match[str] | None:
    """Match a lock pin line for one canonical distribution name."""
    match = _LOCK_PIN_RE.fullmatch(line)
    if match is None or canonicalize_name(match.group("name")) != distribution:
        return None
    return match


def parse_lock_pin(text: str, distribution: str) -> LockPin:
    """Parse one unique hash-pinned distribution stanza from a CI lock.

    Raises
    ------
    ValueError
        If the pin is missing, duplicated, unhashed, or has malformed hashes.

    """
    lines = text.splitlines()
    matches = [
        (index, match)
        for index, line in enumerate(lines)
        if (match := _lock_pin_matches(line, distribution)) is not None
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {distribution} pin, found {len(matches)}")
    index, match = matches[0]
    end = len(lines)
    for candidate in range(index + 1, len(lines)):
        if _ANY_LOCK_PIN_RE.fullmatch(lines[candidate]):
            end = candidate
            break
    stanza = lines[index + 1 : end]
    hashes: set[str] = set()
    for line in stanza:
        if "--hash=" not in line:
            continue
        hash_match = _HASH_RE.fullmatch(line)
        if hash_match is None:
            raise ValueError(f"{distribution} pin contains an invalid sha256 hash")
        hashes.add(hash_match.group("digest"))
    if not hashes:
        raise ValueError(f"{distribution} pin has no sha256 hashes")

    owners: set[str] = set()
    collect_block_owners = False
    for line in stanza:
        via_match = _VIA_RE.fullmatch(line)
        if via_match is not None:
            collect_block_owners = via_match.group("owner") is None
            if via_match.group("owner") is not None:
                owners.add(via_match.group("owner"))
            continue
        owner_match = _VIA_OWNER_RE.fullmatch(line) if collect_block_owners else None
        if owner_match is not None:
            owners.add(owner_match.group("owner"))
        elif line.strip() and not line.lstrip().startswith("#"):
            collect_block_owners = False
    return LockPin(
        version=match.group("version"),
        hashes=frozenset(hashes),
        owners=frozenset(owners),
    )


def audit_lockfiles(lock_texts: Mapping[str, str]) -> tuple[str, ...]:
    """Validate interpreter provenance and build pins in every CI lock."""
    errors: list[str] = []
    for python_version, path in LOCKS_BY_PYTHON:
        text = lock_texts.get(path)
        if text is None:
            errors.append(f"missing CI lock: {path}")
            continue
        provenance = f"autogenerated by pip-compile with Python {python_version}"
        command = f"--output-file={path} requirements-dev.txt"
        if provenance not in text:
            errors.append(f"{path} does not record Python {python_version} provenance")
        if command not in text:
            errors.append(f"{path} does not record its canonical pip-compile command")
        for expected in EXPECTED_BUILD_PINS:
            try:
                pin = parse_lock_pin(text, expected.name)
            except ValueError as exc:
                errors.append(f"{path}: {exc}")
                continue
            if pin.version != expected.version:
                errors.append(
                    f"{path} must pin {expected.name}=={expected.version}; found {pin.version}"
                )
            if pin.hashes != expected.hashes:
                errors.append(f"{path} {expected.name} distribution hashes drifted")
            if pin.owners != _DIRECT_OWNER:
                errors.append(f"{path} {expected.name} pin owners drifted")
    return tuple(errors)


def audit_installed_backend(
    *,
    version_loader: Callable[[str], str] = importlib.metadata.version,
    module_loader: Callable[[str], ModuleType] = importlib.import_module,
) -> tuple[str, ...]:
    """Validate the active environment's Hatchling distribution and backend."""
    errors: list[str] = []
    try:
        installed_version = version_loader("hatchling")
    except importlib.metadata.PackageNotFoundError:
        return ("active environment does not install Hatchling",)
    if installed_version != HATCHLING_VERSION:
        errors.append(
            f"active environment must install Hatchling {HATCHLING_VERSION}; "
            f"found {installed_version}"
        )
    try:
        module_loader(BUILD_BACKEND)
    except ImportError:
        errors.append(f"active environment cannot import {BUILD_BACKEND}")
    return tuple(errors)


def audit_consumers(ci_text: str, dockerfile_text: str, wheel_test_text: str) -> tuple[str, ...]:
    """Validate CI, Docker, and wheel-test wiring to the locked environment."""
    errors: list[str] = []
    docker_lines = tuple(line.strip() for line in dockerfile_text.splitlines())
    for python_version, path in LOCKS_BY_PYTHON:
        mapping = f'python-version: "{python_version}"\n            requirements-file: {path}'
        if ci_text.count(mapping) != 1:
            errors.append(f"CI matrix must map Python {python_version} to {path} exactly once")
    if ci_text.count(LOCK_INSTALL_COMMAND) != 1:
        errors.append("CI test matrix must install its selected lock with --require-hashes")
    if ci_text.count(AUDIT_COMMAND) != 1:
        errors.append("CI lint job must execute the build-environment audit exactly once")
    if ci_text.count(SECURITY_JOB_ENVIRONMENT) != 1:
        errors.append("CI security job must expose both source trees through PYTHONPATH")
    if DOCKER_INSTALL_COMMAND not in dockerfile_text:
        errors.append("Docker must install the Python 3.12 CI lock with --require-hashes")
    for copy_command in DOCKER_CONTEXT_COPIES:
        if docker_lines.count(copy_command) != 1:
            copied_path = copy_command.removeprefix("COPY ").split(maxsplit=1)[0]
            errors.append(f"Docker reproduction context must copy {copied_path} exactly once")
    if wheel_test_text.count('"--no-isolation"') != 1:
        errors.append("real wheel tests must exercise the installed build environment")
    return tuple(errors)


def _read_required(root: Path, path: str, errors: list[str]) -> str | None:
    """Read one required UTF-8 repository file and record access failures."""
    try:
        return (root / path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        errors.append(f"cannot read {path}: {exc}")
        return None


def audit_repository(root: Path) -> BuildEnvironmentAuditResult:
    """Audit the complete repository and active CI build environment."""
    errors: list[str] = []
    for path in PROJECT_PATHS:
        text = _read_required(root, path, errors)
        if text is not None:
            errors.extend(audit_project_metadata(path, text))

    development_text = _read_required(root, DEVELOPMENT_INPUT, errors)
    if development_text is not None:
        errors.extend(audit_development_input(development_text))

    lock_texts: dict[str, str] = {}
    for _, path in LOCKS_BY_PYTHON:
        text = _read_required(root, path, errors)
        if text is not None:
            lock_texts[path] = text
    errors.extend(audit_lockfiles(lock_texts))

    ci_text = _read_required(root, CI_WORKFLOW_PATH, errors)
    dockerfile_text = _read_required(root, DOCKERFILE_PATH, errors)
    wheel_test_text = _read_required(root, WHEEL_TEST_PATH, errors)
    if ci_text is not None and dockerfile_text is not None and wheel_test_text is not None:
        errors.extend(audit_consumers(ci_text, dockerfile_text, wheel_test_text))
    errors.extend(audit_installed_backend())
    return BuildEnvironmentAuditResult(errors=tuple(errors))


def format_result(result: BuildEnvironmentAuditResult) -> str:
    """Render a deterministic operator-facing audit report."""
    if result.passed:
        return (
            "CI build-environment audit: PASS\n"
            f"Hatchling {HATCHLING_VERSION} is installed and locked for Python "
            "3.11, 3.12, and 3.13 wheel builds."
        )
    return "\n".join(("CI build-environment audit: FAIL", *(f"- {e}" for e in result.errors)))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the repository audit from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing CI build and lock inputs.",
    )
    args = parser.parse_args(argv)
    result = audit_repository(args.repo_root)
    print(format_result(result))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
