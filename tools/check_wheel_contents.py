# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Wheel Content Audit
"""Validate Python wheel identity, package contents, and RECORD integrity."""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import glob
import hashlib
import hmac
import io
import json
import re
import stat
import sys
import zipfile
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from email.message import Message
from email.parser import BytesParser
from email.policy import compat32
from pathlib import Path, PurePosixPath
from typing import Final, cast

import tomllib

_DISTRIBUTION_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_VERSION_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+!-]*$")
_MODULE_SEGMENT_PATTERN: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_COMPILED_SUFFIXES: Final = (".so", ".pyd", ".dylib")
_RECORD_SIGNATURE_SUFFIXES: Final = (".jws", ".p7s")


@dataclass(frozen=True)
class WheelExpectation:
    """Expected distribution identity and importable wheel surfaces.

    Parameters
    ----------
    distribution
        Project distribution name from ``pyproject.toml``.
    version
        Exact project version expected in the filename and METADATA.
    packages
        Dotted Python package roots that must contain a non-empty
        ``__init__.py``.
    extensions
        Dotted compiled-extension module paths that must resolve to one
        non-empty ``.so``, ``.pyd``, or ``.dylib`` member.

    """

    distribution: str
    version: str
    packages: tuple[str, ...]
    extensions: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject malformed or ambiguous expectation data.

        Raises
        ------
        ValueError
            If identity fields are invalid, a module path is malformed, a
            requirement is duplicated, or no importable surface is declared.

        """
        if not _DISTRIBUTION_PATTERN.fullmatch(self.distribution):
            raise ValueError(f"invalid distribution name: {self.distribution!r}")
        if not _VERSION_PATTERN.fullmatch(self.version):
            raise ValueError(f"invalid version: {self.version!r}")
        _validate_module_paths(self.packages, label="package")
        _validate_module_paths(self.extensions, label="extension")
        if not self.packages and not self.extensions:
            raise ValueError("at least one package or compiled extension is required")


@dataclass(frozen=True)
class WheelAudit:
    """Result of validating one wheel archive.

    Parameters
    ----------
    wheel
        Audited wheel path.
    distribution
        Expected distribution name.
    version
        Expected version.
    member_count
        Number of non-directory archive members.
    package_member_counts
        Required package roots paired with their file counts.
    extension_members
        Compiled members matched to the required extension modules.
    errors
        Fail-closed findings. An empty tuple means the wheel passed.

    """

    wheel: Path
    distribution: str
    version: str
    member_count: int
    package_member_counts: tuple[tuple[str, int], ...]
    extension_members: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether the wheel has no validation findings."""
        return not self.errors

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable audit payload.

        Returns
        -------
        dict[str, object]
            Stable result data for release evidence.

        """
        return {
            "wheel": self.wheel.as_posix(),
            "distribution": self.distribution,
            "version": self.version,
            "passed": self.passed,
            "member_count": self.member_count,
            "packages": dict(self.package_member_counts),
            "extensions": list(self.extension_members),
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class _WheelFilenameTags:
    """Expanded compatibility tags encoded by a wheel filename."""

    tags: frozenset[str]
    abi_tags: frozenset[str]
    platform_tags: frozenset[str]


def load_expectation(
    project_file: Path,
    *,
    packages: Sequence[str] = (),
    extensions: Sequence[str] = (),
) -> WheelExpectation:
    """Load wheel expectations from project metadata.

    Explicit package arguments take precedence over Hatch wheel-package
    inference. Compiled extensions are always explicit because their module
    path is a build-backend contract rather than PEP 621 project metadata.

    Parameters
    ----------
    project_file
        ``pyproject.toml`` carrying the distribution name and version.
    packages
        Optional dotted package roots. When empty, Hatch ``packages`` entries
        are inferred from the project file.
    extensions
        Optional dotted compiled-extension module paths.

    Returns
    -------
    WheelExpectation
        Validated release expectation.

    Raises
    ------
    OSError
        If the project file cannot be read.
    tomllib.TOMLDecodeError
        If the project file is not valid TOML.
    ValueError
        If required project metadata or package configuration is invalid.

    """
    payload: object = tomllib.loads(project_file.read_text(encoding="utf-8"))
    root = _require_mapping(payload, "pyproject root")
    project = _require_mapping(root.get("project"), "[project]")
    distribution = _require_string(project.get("name"), "[project].name")
    version = _require_string(project.get("version"), "[project].version")
    package_tuple = tuple(packages) or _hatch_packages(root)
    return WheelExpectation(
        distribution=distribution,
        version=version,
        packages=package_tuple,
        extensions=tuple(extensions),
    )


def audit_wheel(wheel: Path, expectation: WheelExpectation) -> WheelAudit:
    """Validate one wheel through its production archive surface.

    Parameters
    ----------
    wheel
        Wheel archive to inspect.
    expectation
        Distribution, package, and compiled-extension requirements.

    Returns
    -------
    WheelAudit
        Aggregate result. Archive and content errors are returned rather than
        raised so one command can report every release wheel.

    """
    errors = _wheel_path_errors(wheel)
    if errors:
        return _empty_audit(wheel, expectation, errors)

    try:
        with zipfile.ZipFile(wheel) as archive:
            corrupt_member = archive.testzip()
            if corrupt_member is not None:
                return _empty_audit(
                    wheel,
                    expectation,
                    [f"CRC validation failed for archive member: {corrupt_member}"],
                )
            return _audit_archive(wheel, archive, expectation)
    except (
        EOFError,
        OSError,
        RuntimeError,
        NotImplementedError,
        ValueError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        zlib.error,
    ) as exc:
        return _empty_audit(
            wheel,
            expectation,
            [f"wheel archive could not be read: {exc}"],
        )


def format_audit(audit: WheelAudit) -> str:
    """Render a human-readable wheel result.

    Parameters
    ----------
    audit
        Result to render.

    Returns
    -------
    str
        Multi-line release-gate summary.

    """
    status = "PASS" if audit.passed else "FAIL"
    lines = [
        f"{status}: {audit.wheel.as_posix()}",
        f"  distribution: {audit.distribution} {audit.version}",
        f"  archive members: {audit.member_count}",
    ]
    if audit.package_member_counts:
        packages = ", ".join(
            f"{package}={count}" for package, count in audit.package_member_counts
        )
        lines.append(f"  package members: {packages}")
    if audit.extension_members:
        lines.append(f"  compiled extensions: {', '.join(audit.extension_members)}")
    lines.extend(f"  error: {error}" for error in audit.errors)
    return "\n".join(lines)


def _validate_module_paths(paths: tuple[str, ...], *, label: str) -> None:
    if len(set(paths)) != len(paths):
        raise ValueError(f"duplicate {label} requirement")
    for path in paths:
        segments = path.split(".")
        if not segments or any(not _MODULE_SEGMENT_PATTERN.fullmatch(item) for item in segments):
            raise ValueError(f"invalid {label} module path: {path!r}")


def _require_mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{location} must be a TOML table")
    return cast(Mapping[str, object], value)


def _require_string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _hatch_packages(root: Mapping[str, object]) -> tuple[str, ...]:
    tool = _require_mapping(root.get("tool"), "[tool]")
    hatch = _require_mapping(tool.get("hatch"), "[tool.hatch]")
    build = _require_mapping(hatch.get("build"), "[tool.hatch.build]")
    targets = _require_mapping(build.get("targets"), "[tool.hatch.build.targets]")
    wheel = _require_mapping(targets.get("wheel"), "[tool.hatch.build.targets.wheel]")
    raw_packages = wheel.get("packages")
    if not isinstance(raw_packages, list) or not raw_packages:
        raise ValueError("[tool.hatch.build.targets.wheel].packages must be a non-empty list")

    inferred: list[str] = []
    for index, item in enumerate(raw_packages):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"[tool.hatch.build.targets.wheel].packages[{index}] must be a non-empty string"
            )
        normalised = item.replace("\\", "/").rstrip("/")
        leaf = PurePosixPath(normalised).name
        if not leaf:
            raise ValueError(f"wheel package path has no package name: {item!r}")
        inferred.append(leaf)
    return tuple(inferred)


def _wheel_path_errors(wheel: Path) -> list[str]:
    errors: list[str] = []
    if wheel.suffix.casefold() != ".whl":
        errors.append("artifact does not use the .whl suffix")
    if not wheel.exists():
        errors.append("wheel path does not exist")
    elif not wheel.is_file():
        errors.append("wheel path is not a regular file")
    return errors


def _empty_audit(
    wheel: Path,
    expectation: WheelExpectation,
    errors: Sequence[str],
) -> WheelAudit:
    return WheelAudit(
        wheel=wheel,
        distribution=expectation.distribution,
        version=expectation.version,
        member_count=0,
        package_member_counts=tuple((package, 0) for package in expectation.packages),
        extension_members=(),
        errors=tuple(errors),
    )


def _audit_archive(
    wheel: Path,
    archive: zipfile.ZipFile,
    expectation: WheelExpectation,
) -> WheelAudit:
    errors: list[str] = []
    infos = archive.infolist()
    names = [info.filename for info in infos]
    files = [info.filename for info in infos if not info.is_dir()]

    filename_tags = _validate_filename(wheel, expectation, errors)
    _validate_archive_members(infos, errors)
    dist_info = _validate_metadata(
        archive,
        files,
        expectation,
        filename_tags,
        errors,
    )
    package_counts = _validate_packages(archive, files, expectation.packages, errors)
    extension_members = _validate_extensions(
        archive,
        files,
        expectation.extensions,
        errors,
    )
    if dist_info is not None:
        _validate_record(archive, files, dist_info, errors)

    if len(set(names)) != len(names):
        duplicates = sorted(name for name in set(names) if names.count(name) > 1)
        errors.append(f"duplicate archive member names: {', '.join(duplicates)}")

    return WheelAudit(
        wheel=wheel,
        distribution=expectation.distribution,
        version=expectation.version,
        member_count=len(files),
        package_member_counts=package_counts,
        extension_members=extension_members,
        errors=tuple(errors),
    )


def _validate_filename(
    wheel: Path,
    expectation: WheelExpectation,
    errors: list[str],
) -> _WheelFilenameTags | None:
    parts = wheel.stem.split("-")
    if len(parts) not in {5, 6}:
        errors.append(
            "wheel filename must contain distribution, version, optional build, "
            "and three tag fields"
        )
        return None
    distribution_component, version_component = parts[:2]
    expected_distribution = _wheel_component(expectation.distribution)
    expected_version = _wheel_component(expectation.version)
    if distribution_component.casefold() != expected_distribution.casefold():
        errors.append(
            "wheel filename distribution mismatch: "
            f"expected {expected_distribution!r}, found {distribution_component!r}"
        )
    if version_component.casefold() != expected_version.casefold():
        errors.append(
            "wheel filename version mismatch: "
            f"expected {expected_version!r}, found {version_component!r}"
        )

    if len(parts) == 6 and (not parts[2] or not parts[2][0].isdigit()):
        errors.append(f"wheel filename build tag must start with a digit: {parts[2]!r}")

    python_component, abi_component, platform_component = parts[-3:]
    python_tags = _tag_component(python_component, "Python", errors)
    abi_tags = _tag_component(abi_component, "ABI", errors)
    platform_tags = _tag_component(platform_component, "platform", errors)
    if not python_tags or not abi_tags or not platform_tags:
        return None
    return _WheelFilenameTags(
        tags=frozenset(
            f"{python_tag}-{abi_tag}-{platform_tag}"
            for python_tag in python_tags
            for abi_tag in abi_tags
            for platform_tag in platform_tags
        ),
        abi_tags=frozenset(abi_tags),
        platform_tags=frozenset(platform_tags),
    )


def _tag_component(component: str, label: str, errors: list[str]) -> tuple[str, ...]:
    tags = tuple(component.split("."))
    if any(not tag or not re.fullmatch(r"[A-Za-z0-9_]+", tag) for tag in tags):
        errors.append(f"wheel filename {label} tag component is invalid: {component!r}")
        return ()
    return tags


def _wheel_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "_", value)


def _validate_archive_members(infos: Sequence[zipfile.ZipInfo], errors: list[str]) -> None:
    for info in infos:
        path_error = _unsafe_member_reason(info.filename)
        if path_error is not None:
            errors.append(f"unsafe archive member {info.filename!r}: {path_error}")
        if info.flag_bits & 0x1:
            errors.append(f"encrypted archive member is forbidden: {info.filename}")
        mode = (info.external_attr >> 16) & 0xFFFF
        if stat.S_ISLNK(mode):
            errors.append(f"symbolic-link archive member is forbidden: {info.filename}")


def _unsafe_member_reason(name: str) -> str | None:
    if not name:
        return "empty name"
    if "\x00" in name:
        return "NUL byte"
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        return "ASCII control character"
    if "\\" in name:
        return "backslash path separator"
    if name.startswith("/"):
        return "absolute path"
    stripped = name[:-1] if name.endswith("/") else name
    segments = stripped.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        return "empty, current-directory, or parent-directory segment"
    if re.match(r"^[A-Za-z]:", segments[0]):
        return "drive-qualified path"
    return None


def _validate_metadata(
    archive: zipfile.ZipFile,
    files: Sequence[str],
    expectation: WheelExpectation,
    filename_tags: _WheelFilenameTags | None,
    errors: list[str],
) -> str | None:
    candidates = sorted(
        {
            path.split("/", maxsplit=1)[0]
            for path in files
            if path.count("/") == 1
            and ".dist-info/" in path
            and path.rsplit("/", maxsplit=1)[1] in {"METADATA", "WHEEL", "RECORD"}
        }
    )
    if len(candidates) != 1:
        errors.append(
            f"expected one metadata-bearing .dist-info directory, found {len(candidates)}"
        )
        return None

    dist_info = candidates[0]
    expected_dist_info = (
        f"{_wheel_component(expectation.distribution)}-"
        f"{_wheel_component(expectation.version)}.dist-info"
    )
    if dist_info.casefold() != expected_dist_info.casefold():
        errors.append(
            f"dist-info directory mismatch: expected {expected_dist_info!r}, found {dist_info!r}"
        )

    required = {
        "METADATA": f"{dist_info}/METADATA",
        "WHEEL": f"{dist_info}/WHEEL",
        "RECORD": f"{dist_info}/RECORD",
    }
    for label, path in required.items():
        if path not in files:
            errors.append(f"missing required {label} member: {path}")

    metadata_path = required["METADATA"]
    if metadata_path in files:
        metadata = _message(archive.read(metadata_path), metadata_path, errors)
        if metadata is not None:
            actual_name = _header(metadata, "Name")
            actual_version = _header(metadata, "Version")
            if actual_name is None:
                errors.append("METADATA has no Name field")
            elif _normalise_distribution(actual_name) != _normalise_distribution(
                expectation.distribution
            ):
                errors.append(
                    f"METADATA Name mismatch: expected {expectation.distribution!r}, "
                    f"found {actual_name!r}"
                )
            if actual_version is None:
                errors.append("METADATA has no Version field")
            elif actual_version != expectation.version:
                errors.append(
                    f"METADATA Version mismatch: expected {expectation.version!r}, "
                    f"found {actual_version!r}"
                )

    wheel_path = required["WHEEL"]
    if wheel_path in files:
        wheel_headers = _message(archive.read(wheel_path), wheel_path, errors)
        if wheel_headers is not None:
            wheel_version = _header(wheel_headers, "Wheel-Version")
            purelib = _header(wheel_headers, "Root-Is-Purelib")
            generator = _header(wheel_headers, "Generator")
            tags = _headers(wheel_headers, "Tag")
            if wheel_version is None or wheel_version.split(".", maxsplit=1)[0] != "1":
                errors.append(f"unsupported or missing Wheel-Version: {wheel_version!r}")
            if purelib not in {"true", "false"}:
                errors.append(f"invalid or missing Root-Is-Purelib: {purelib!r}")
            if generator is None or not generator.strip():
                errors.append("WHEEL has no Generator field")
            if not tags or any(not tag.strip() for tag in tags):
                errors.append("WHEEL has no usable Tag field")
            elif len(set(tags)) != len(tags):
                errors.append("WHEEL contains duplicate Tag fields")
            elif filename_tags is not None:
                wheel_tags = frozenset(tag.strip() for tag in tags)
                missing_tags = sorted(filename_tags.tags - wheel_tags)
                extra_tags = sorted(wheel_tags - filename_tags.tags)
                if missing_tags:
                    errors.append(
                        f"WHEEL omits filename compatibility tags: {', '.join(missing_tags)}"
                    )
                if extra_tags:
                    errors.append(
                        f"WHEEL lists tags absent from the filename: {', '.join(extra_tags)}"
                    )
            if expectation.extensions and purelib != "false":
                errors.append("a wheel with required compiled extensions must not be purelib")
            if filename_tags is not None and expectation.extensions:
                if "none" in filename_tags.abi_tags:
                    errors.append("a wheel with required compiled extensions needs an ABI tag")
                if "any" in filename_tags.platform_tags:
                    errors.append("a wheel with required compiled extensions needs a platform tag")
    return dist_info


def _message(payload: bytes, path: str, errors: list[str]) -> Message | None:
    if not payload:
        errors.append(f"metadata member is empty: {path}")
        return None
    return BytesParser(policy=compat32).parsebytes(payload)


def _header(message: Message, name: str) -> str | None:
    return message.get(name)


def _headers(message: Message, name: str) -> list[str]:
    return message.get_all(name) or []


def _normalise_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _validate_packages(
    archive: zipfile.ZipFile,
    files: Sequence[str],
    packages: Sequence[str],
    errors: list[str],
) -> tuple[tuple[str, int], ...]:
    counts: list[tuple[str, int]] = []
    for package in packages:
        prefix = package.replace(".", "/") + "/"
        members = [path for path in files if path.startswith(prefix)]
        counts.append((package, len(members)))
        if not members:
            errors.append(f"required package has no wheel members: {package}")
            continue
        init_path = f"{prefix}__init__.py"
        if init_path not in members:
            errors.append(f"required package is missing __init__.py: {package}")
        elif not archive.read(init_path):
            errors.append(f"required package has an empty __init__.py: {package}")
        python_members = [path for path in members if path.endswith(".py")]
        if not python_members or not any(
            archive.getinfo(path).file_size > 0 for path in python_members
        ):
            errors.append(f"required package has no non-empty Python module: {package}")
    return tuple(counts)


def _validate_extensions(
    archive: zipfile.ZipFile,
    files: Sequence[str],
    extensions: Sequence[str],
    errors: list[str],
) -> tuple[str, ...]:
    matched: list[str] = []
    for extension in extensions:
        expected = extension.replace(".", "/")
        parent = PurePosixPath(expected).parent
        stem = PurePosixPath(expected).name
        candidates = [
            path
            for path in files
            if PurePosixPath(path).parent == parent
            and PurePosixPath(path).name.startswith(f"{stem}.")
            and path.casefold().endswith(_COMPILED_SUFFIXES)
        ]
        if len(candidates) != 1:
            errors.append(
                f"required compiled extension {extension!r} matched {len(candidates)} members"
            )
            continue
        path = candidates[0]
        if archive.getinfo(path).file_size == 0:
            errors.append(f"required compiled extension is empty: {path}")
            continue
        matched.append(path)
    return tuple(matched)


def _validate_record(
    archive: zipfile.ZipFile,
    files: Sequence[str],
    dist_info: str,
    errors: list[str],
) -> None:
    record_path = f"{dist_info}/RECORD"
    if record_path not in files:
        return
    try:
        record_text = archive.read(record_path).decode("utf-8")
    except UnicodeDecodeError as exc:
        errors.append(f"RECORD is not UTF-8: {exc}")
        return

    entries: dict[str, tuple[str, str]] = {}
    try:
        rows = csv.reader(io.StringIO(record_text, newline=""), strict=True)
        for line_number, row in enumerate(rows, start=1):
            if len(row) != 3:
                errors.append(f"RECORD row {line_number} must contain exactly three columns")
                continue
            path, digest, size = row
            if path in entries:
                errors.append(f"duplicate RECORD path: {path}")
                continue
            path_error = _unsafe_member_reason(path)
            if path_error is not None:
                errors.append(f"unsafe RECORD path {path!r}: {path_error}")
            entries[path] = (digest, size)
    except csv.Error as exc:
        errors.append(f"RECORD is not valid CSV: {exc}")

    archive_files = set(files)
    signatures = {
        path
        for path in archive_files
        if path.startswith(f"{record_path}.")
        and path.casefold().endswith(_RECORD_SIGNATURE_SUFFIXES)
    }
    expected_paths = archive_files - signatures
    listed_signatures = sorted(signatures & entries.keys())
    if listed_signatures:
        errors.append(
            f"RECORD must not list its detached signatures: {', '.join(listed_signatures)}"
        )
    missing = sorted(expected_paths - entries.keys())
    extra = sorted(entries.keys() - archive_files)
    if missing:
        errors.append(f"RECORD omits archive members: {', '.join(missing)}")
    if extra:
        errors.append(f"RECORD lists absent archive members: {', '.join(extra)}")

    for path in sorted(expected_paths & entries.keys()):
        digest, size = entries[path]
        if path == record_path:
            if digest or size:
                errors.append("RECORD must leave its own hash and size empty")
            continue
        payload = archive.read(path)
        _validate_record_size(path, size, len(payload), errors)
        _validate_record_digest(path, digest, payload, errors)


def _validate_record_size(path: str, encoded: str, actual: int, errors: list[str]) -> None:
    if not encoded:
        errors.append(f"RECORD size is missing for {path}")
        return
    try:
        expected = int(encoded)
    except ValueError:
        errors.append(f"RECORD size is not an integer for {path}: {encoded!r}")
        return
    if expected != actual:
        errors.append(f"RECORD size mismatch for {path}: expected {expected}, found {actual}")


def _validate_record_digest(
    path: str,
    encoded: str,
    payload: bytes,
    errors: list[str],
) -> None:
    if not encoded or "=" not in encoded:
        errors.append(f"RECORD digest is missing or malformed for {path}")
        return
    algorithm, value = encoded.split("=", maxsplit=1)
    if algorithm != "sha256":
        errors.append(f"RECORD digest algorithm must be sha256 for {path}: {algorithm!r}")
        return
    if not value or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        errors.append(f"RECORD digest is not valid URL-safe base64 for {path}")
        return
    try:
        expected = base64.b64decode(
            value + "=" * (-len(value) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, ValueError):
        errors.append(f"RECORD digest is not valid URL-safe base64 for {path}")
        return
    if len(expected) != hashlib.sha256().digest_size:
        errors.append(f"RECORD sha256 digest has the wrong length for {path}")
        return
    actual = hashlib.sha256(payload).digest()
    if not hmac.compare_digest(expected, actual):
        errors.append(f"RECORD digest mismatch for {path}")


def _expand_wheel_paths(patterns: Sequence[str]) -> tuple[Path, ...]:
    expanded: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise ValueError(f"wheel path or glob matched no files: {pattern}")
        expanded.extend(Path(match) for match in matches)
    unique = tuple(dict.fromkeys(expanded))
    if not unique:
        raise ValueError("no wheel paths were provided")
    return unique


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail closed when built wheel identity, package contents, compiled "
            "extensions, or RECORD integrity drift from project metadata."
        )
    )
    parser.add_argument("wheels", nargs="+", help="Wheel paths or glob patterns")
    parser.add_argument(
        "--project",
        type=Path,
        required=True,
        help="pyproject.toml supplying project name and version",
    )
    parser.add_argument(
        "--package",
        action="append",
        default=[],
        help="Required dotted Python package root; repeat as needed",
    )
    parser.add_argument(
        "--extension",
        action="append",
        default=[],
        help="Required dotted compiled-extension module; repeat as needed",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON release evidence")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the wheel-content release gate.

    Parameters
    ----------
    argv
        Optional command-line arguments excluding the executable name.

    Returns
    -------
    int
        ``0`` when every wheel passes, ``1`` for wheel findings, and ``2`` for
        invalid command or project configuration.

    """
    args = _build_parser().parse_args(argv)
    try:
        expectation = load_expectation(
            args.project,
            packages=cast(list[str], args.package),
            extensions=cast(list[str], args.extension),
        )
        wheels = _expand_wheel_paths(cast(list[str], args.wheels))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, ValueError) as exc:
        print(f"wheel-content audit configuration error: {exc}", file=sys.stderr)
        return 2

    audits = [audit_wheel(wheel, expectation) for wheel in wheels]
    if cast(bool, args.json):
        print(json.dumps([audit.as_dict() for audit in audits], indent=2, sort_keys=True))
    else:
        print("\n".join(format_audit(audit) for audit in audits))
    return 0 if all(audit.passed for audit in audits) else 1


if __name__ == "__main__":
    raise SystemExit(main())
