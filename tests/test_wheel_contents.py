# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Wheel Content Audit Tests
"""Test the fail-closed release-wheel content audit."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import shutil
import stat
import subprocess
import sys
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

import tools.check_wheel_contents as wheel_tool
from tools.check_wheel_contents import (
    WheelAudit,
    WheelExpectation,
    audit_wheel,
    format_audit,
    load_expectation,
    main,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MAIN_PROJECT = _REPO_ROOT / "pyproject.toml"
_OSCILLATOOLS_PROJECT = _REPO_ROOT / "oscillatools" / "pyproject.toml"
_Mutation = Callable[[dict[str, bytes]], None]


@dataclass(frozen=True)
class BuiltWheels:
    """Real wheels built from the two Hatch release projects."""

    main: Path
    oscillatools: Path


class _CorruptArchive:
    def __enter__(self) -> _CorruptArchive:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def testzip(self) -> str:
        return "corrupt.py"


@pytest.fixture(scope="module")
def built_wheels(tmp_path_factory: pytest.TempPathFactory) -> BuiltWheels:
    """Build production Hatch wheels without an isolated dependency download."""
    output_root = tmp_path_factory.mktemp("release-wheels")
    main_output = output_root / "main"
    oscillatools_output = output_root / "oscillatools"
    main_output.mkdir()
    oscillatools_output.mkdir()
    for project, output in (
        (_REPO_ROOT, main_output),
        (_REPO_ROOT / "oscillatools", oscillatools_output),
    ):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(output),
                str(project),
            ],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    return BuiltWheels(
        main=_only_wheel(main_output),
        oscillatools=_only_wheel(oscillatools_output),
    )


def _only_wheel(directory: Path) -> Path:
    wheels = tuple(directory.glob("*.whl"))
    if len(wheels) != 1:
        raise AssertionError(f"expected one wheel in {directory}, found {len(wheels)}")
    return wheels[0]


def _read_members(wheel: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(wheel) as archive:
        return {
            info.filename: archive.read(info) for info in archive.infolist() if not info.is_dir()
        }


def _record_path(members: dict[str, bytes]) -> str:
    matches = [path for path in members if path.endswith(".dist-info/RECORD")]
    if len(matches) != 1:
        raise AssertionError(f"expected one RECORD member, found {len(matches)}")
    return matches[0]


def _record_digest(payload: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
    return f"sha256={digest.decode('ascii')}"


def _rewrite_record(members: dict[str, bytes]) -> None:
    record_path = _record_path(members)
    signature_paths = {f"{record_path}.jws", f"{record_path}.p7s"}
    rows = [
        [path, _record_digest(payload), str(len(payload))]
        for path, payload in sorted(members.items())
        if path != record_path and path not in signature_paths
    ]
    rows.append([record_path, "", ""])
    members[record_path] = _encode_record(rows)


def _decode_record(payload: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(payload.decode("utf-8"), newline="")))


def _encode_record(rows: Sequence[Sequence[str]]) -> bytes:
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    return output.getvalue().encode("utf-8")


def _mutated_wheel(
    source: Path,
    directory: Path,
    mutation: _Mutation,
    *,
    repair_record: bool = True,
    filename: str | None = None,
) -> Path:
    members = _read_members(source)
    mutation(members)
    if repair_record:
        _rewrite_record(members)
    target = directory / (filename or source.name)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, payload in members.items():
            archive.writestr(path, payload)
    return target


def _identity_mutation(_members: dict[str, bytes]) -> None:
    return None


def _dist_info_member(members: dict[str, bytes], leaf: str) -> str:
    matches = [path for path in members if path.endswith(f".dist-info/{leaf}")]
    if len(matches) != 1:
        raise AssertionError(f"expected one {leaf} member, found {len(matches)}")
    return matches[0]


def _replace_header(payload: bytes, name: str, value: str | None) -> bytes:
    prefix = f"{name}:".casefold()
    lines = [
        line
        for line in payload.decode("utf-8").splitlines()
        if not line.casefold().startswith(prefix)
    ]
    if value is not None:
        lines.insert(0, f"{name}: {value}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _set_header(members: dict[str, bytes], leaf: str, name: str, value: str | None) -> None:
    path = _dist_info_member(members, leaf)
    members[path] = _replace_header(members[path], name, value)


def _rename_dist_info(members: dict[str, bytes], replacement: str) -> None:
    original = _dist_info_member(members, "RECORD").split("/", maxsplit=1)[0]
    renamed = {
        path.replace(f"{original}/", f"{replacement}/", 1): payload
        for path, payload in members.items()
    }
    members.clear()
    members.update(renamed)


def _main_expectation() -> WheelExpectation:
    return load_expectation(_MAIN_PROJECT)


def _assert_error(audit: WheelAudit, fragment: str) -> None:
    assert not audit.passed
    assert any(fragment in error for error in audit.errors), audit.errors


def test_real_hatch_wheels_pass_the_complete_release_audit(
    built_wheels: BuiltWheels,
) -> None:
    """The parent and oscillatools production wheels must pass unmodified."""
    main_expectation = _main_expectation()
    oscillatools_expectation = load_expectation(_OSCILLATOOLS_PROJECT)

    main_audit = audit_wheel(built_wheels.main, main_expectation)
    oscillatools_audit = audit_wheel(
        built_wheels.oscillatools,
        oscillatools_expectation,
    )

    assert main_expectation.packages == ("scpn_quantum_control", "scpn")
    assert oscillatools_expectation.packages == ("oscillatools",)
    assert main_audit.passed, main_audit.errors
    assert oscillatools_audit.passed, oscillatools_audit.errors
    assert main_audit.member_count > 500
    assert dict(main_audit.package_member_counts)["scpn_quantum_control"] > 500


def test_real_wheel_repack_accepts_a_platform_extension(
    built_wheels: BuiltWheels,
    tmp_path: Path,
) -> None:
    """A non-empty ABI-tagged extension passes in a non-pure platform wheel."""

    def add_extension(members: dict[str, bytes]) -> None:
        members["scpn_quantum_control/native.abi3.so"] = b"ELF-placeholder"
        _set_header(members, "WHEEL", "Root-Is-Purelib", "false")
        _set_header(
            members,
            "WHEEL",
            "Tag",
            "cp312-abi3-manylinux_2_17_x86_64",
        )

    wheel = _mutated_wheel(
        built_wheels.main,
        tmp_path,
        add_extension,
        filename=("scpn_quantum_control-1.0.0-cp312-abi3-manylinux_2_17_x86_64.whl"),
    )
    expectation = WheelExpectation(
        "scpn-quantum-control",
        "1.0.0",
        ("scpn_quantum_control", "scpn"),
        ("scpn_quantum_control.native",),
    )

    audit = audit_wheel(wheel, expectation)

    assert audit.passed, audit.errors
    assert audit.extension_members == ("scpn_quantum_control/native.abi3.so",)
    assert "compiled extensions" in format_audit(audit)


@pytest.mark.parametrize(
    ("distribution", "version", "packages", "extensions", "message"),
    [
        ("", "1", ("pkg",), (), "invalid distribution"),
        ("pkg", "", ("pkg",), (), "invalid version"),
        ("pkg", "1", ("bad-path",), (), "invalid package module path"),
        ("pkg", "1", (), ("bad-path",), "invalid extension module path"),
        ("pkg", "1", ("pkg", "pkg"), (), "duplicate package"),
        ("pkg", "1", (), ("pkg.native", "pkg.native"), "duplicate extension"),
        ("pkg", "1", (), (), "at least one package"),
    ],
)
def test_expectation_rejects_malformed_release_contracts(
    distribution: str,
    version: str,
    packages: tuple[str, ...],
    extensions: tuple[str, ...],
    message: str,
) -> None:
    """Expectation construction rejects invalid or ambiguous requirements."""
    with pytest.raises(ValueError, match=message):
        WheelExpectation(distribution, version, packages, extensions)


def test_project_loader_accepts_explicit_packages_without_hatch_config(
    tmp_path: Path,
) -> None:
    """An explicit package contract supports non-Hatch build backends."""
    project = tmp_path / "pyproject.toml"
    project.write_text('[project]\nname = "demo-dist"\nversion = "1.2.3"\n', encoding="utf-8")

    expectation = load_expectation(project, packages=("demo",), extensions=("demo.native",))

    assert expectation == WheelExpectation(
        "demo-dist",
        "1.2.3",
        ("demo",),
        ("demo.native",),
    )


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("name = 'x'\n", r"\[project\] must be a TOML table"),
        ("[project]\nversion = '1'\n", r"\[project\]\.name"),
        ("[project]\nname = 'x'\nversion = []\n", r"\[project\]\.version"),
        (
            "[project]\nname='x'\nversion='1'\n",
            r"\[tool\] must be a TOML table",
        ),
        (
            "[project]\nname='x'\nversion='1'\n[tool.hatch.build.targets.wheel]\npackages=[]\n",
            "packages must be a non-empty list",
        ),
        (
            "[project]\nname='x'\nversion='1'\n[tool.hatch.build.targets.wheel]\npackages=[1]\n",
            r"packages\[0\] must be a non-empty string",
        ),
        (
            "[project]\nname='x'\nversion='1'\n[tool.hatch.build.targets.wheel]\npackages=['/']\n",
            "has no package name",
        ),
    ],
)
def test_project_loader_rejects_incomplete_metadata(
    tmp_path: Path,
    body: str,
    message: str,
) -> None:
    """Invalid PEP 621 or Hatch metadata fails before any wheel is trusted."""
    project = tmp_path / "pyproject.toml"
    project.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_expectation(project)


def test_project_mapping_guard_rejects_non_string_keys() -> None:
    """The internal TOML-table guard rejects non-string mapping keys."""
    with pytest.raises(ValueError, match="must be a TOML table"):
        wheel_tool._require_mapping({1: "value"}, "mapping")


def test_path_and_archive_failures_return_audits(tmp_path: Path) -> None:
    """Missing, mis-suffixed, directory, and malformed artifacts fail cleanly."""
    expectation = WheelExpectation("demo", "1", ("demo",), ())
    missing = audit_wheel(tmp_path / "missing.whl", expectation)
    text_file = tmp_path / "demo.txt"
    text_file.write_text("not a wheel", encoding="utf-8")
    wrong_suffix = audit_wheel(text_file, expectation)
    directory = tmp_path / "folder.whl"
    directory.mkdir()
    not_file = audit_wheel(directory, expectation)
    malformed_path = tmp_path / "demo-1-py3-none-any.whl"
    malformed_path.write_bytes(b"not a zip")
    malformed = audit_wheel(malformed_path, expectation)

    _assert_error(missing, "does not exist")
    _assert_error(wrong_suffix, "does not use the .whl suffix")
    _assert_error(not_file, "not a regular file")
    _assert_error(malformed, "archive could not be read")
    assert missing.package_member_counts == (("demo", 0),)


def test_crc_failure_identifies_the_corrupt_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ZIP CRC failure identifies its member without entering content validation."""
    wheel = tmp_path / "demo-1-py3-none-any.whl"
    wheel.write_bytes(b"placeholder")
    monkeypatch.setattr(zipfile, "ZipFile", lambda _path: _CorruptArchive())

    audit = audit_wheel(wheel, WheelExpectation("demo", "1", ("demo",), ()))

    _assert_error(audit, "CRC validation failed for archive member: corrupt.py")


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        ("short.whl", "must contain distribution"),
        ("wrong-0.10.0-py3-none-any.whl", "distribution mismatch"),
        ("scpn_quantum_control-9-py3-none-any.whl", "version mismatch"),
        (
            "scpn_quantum_control-0.10.0-build-py3-none-any.whl",
            "build tag must start with a digit",
        ),
        (
            "scpn_quantum_control-0.10.0-py3..py2-none-any.whl",
            "Python tag component is invalid",
        ),
        (
            "scpn_quantum_control-0.10.0-py3-bad!-any.whl",
            "ABI tag component is invalid",
        ),
        (
            "scpn_quantum_control-0.10.0-py3-none-bad!.whl",
            "platform tag component is invalid",
        ),
    ],
)
def test_filename_identity_and_tag_failures_are_reported(
    built_wheels: BuiltWheels,
    tmp_path: Path,
    filename: str,
    message: str,
) -> None:
    """Wheel filenames must carry the expected identity and valid tag grammar."""
    target = tmp_path / filename
    shutil.copyfile(built_wheels.main, target)

    _assert_error(audit_wheel(target, _main_expectation()), message)


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("", "empty name"),
        ("bad\x00name", "NUL byte"),
        ("bad\nname", "control character"),
        (r"pkg\module.py", "backslash"),
        ("/pkg/module.py", "absolute"),
        ("pkg//module.py", "empty, current-directory"),
        ("pkg/./module.py", "empty, current-directory"),
        ("pkg/../module.py", "empty, current-directory"),
        ("C:pkg/module.py", "drive-qualified"),
    ],
)
def test_member_path_guard_rejects_unsafe_names(name: str, reason: str) -> None:
    """Archive and RECORD paths reject traversal and cross-platform ambiguity."""
    assert reason in str(wheel_tool._unsafe_member_reason(name))


def test_member_path_guard_accepts_a_normal_directory_member() -> None:
    """A normal POSIX directory member is safe."""
    assert wheel_tool._unsafe_member_reason("pkg/data/") is None


def test_archive_member_policy_flags_encryption_and_symbolic_links() -> None:
    """Encryption and symbolic links cannot hide release-wheel payloads."""
    encrypted = zipfile.ZipInfo("pkg/encrypted.py")
    encrypted.flag_bits = 0x1
    symbolic_link = zipfile.ZipInfo("pkg/link.py")
    symbolic_link.external_attr = stat.S_IFLNK << 16
    errors: list[str] = []

    wheel_tool._validate_archive_members((encrypted, symbolic_link), errors)

    assert any("encrypted archive member" in error for error in errors)
    assert any("symbolic-link archive member" in error for error in errors)


def test_actual_archive_rejects_path_traversal_and_duplicate_members(
    built_wheels: BuiltWheels,
    tmp_path: Path,
) -> None:
    """Unsafe and duplicate names are rejected through a real ZIP archive."""
    members = _read_members(built_wheels.main)
    members["../escape.py"] = b"escape"
    _rewrite_record(members)
    target = tmp_path / built_wheels.main.name
    duplicate = next(iter(members))
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, payload in members.items():
            archive.writestr(path, payload)
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr(duplicate, members[duplicate])

    audit = audit_wheel(target, _main_expectation())

    _assert_error(audit, "unsafe archive member")
    _assert_error(audit, "duplicate archive member names")


@pytest.mark.parametrize("leaf", ["METADATA", "WHEEL", "RECORD"])
def test_required_dist_info_members_cannot_be_omitted(
    built_wheels: BuiltWheels,
    tmp_path: Path,
    leaf: str,
) -> None:
    """Each required dist-info control member is release blocking."""

    def remove_member(members: dict[str, bytes]) -> None:
        del members[_dist_info_member(members, leaf)]

    wheel = _mutated_wheel(
        built_wheels.main,
        tmp_path,
        remove_member,
        repair_record=leaf != "RECORD",
    )

    _assert_error(audit_wheel(wheel, _main_expectation()), f"missing required {leaf}")


def test_metadata_directory_must_be_unique_and_match_identity(
    built_wheels: BuiltWheels,
    tmp_path: Path,
) -> None:
    """Multiple or wrongly named metadata directories fail closed."""

    def add_second_metadata(members: dict[str, bytes]) -> None:
        members["other-1.dist-info/METADATA"] = b"Name: other\nVersion: 1\n"

    multiple = _mutated_wheel(built_wheels.main, tmp_path, add_second_metadata)
    _assert_error(
        audit_wheel(multiple, _main_expectation()),
        "expected one metadata-bearing .dist-info directory, found 2",
    )

    wrong_dir = tmp_path / "wrong"
    wrong_dir.mkdir()
    renamed = _mutated_wheel(
        built_wheels.main,
        wrong_dir,
        lambda members: _rename_dist_info(members, "wrong-1.dist-info"),
    )
    _assert_error(audit_wheel(renamed, _main_expectation()), "dist-info directory mismatch")


@pytest.mark.parametrize(
    ("leaf", "header", "value", "message"),
    [
        ("METADATA", "Name", None, "METADATA has no Name"),
        ("METADATA", "Name", "other", "METADATA Name mismatch"),
        ("METADATA", "Version", None, "METADATA has no Version"),
        ("METADATA", "Version", "9", "METADATA Version mismatch"),
        ("WHEEL", "Wheel-Version", None, "unsupported or missing Wheel-Version"),
        ("WHEEL", "Wheel-Version", "2.0", "unsupported or missing Wheel-Version"),
        ("WHEEL", "Root-Is-Purelib", None, "invalid or missing Root-Is-Purelib"),
        ("WHEEL", "Root-Is-Purelib", "yes", "invalid or missing Root-Is-Purelib"),
        ("WHEEL", "Generator", None, "WHEEL has no Generator"),
        ("WHEEL", "Generator", "", "WHEEL has no Generator"),
        ("WHEEL", "Tag", None, "WHEEL has no usable Tag"),
        ("WHEEL", "Tag", "", "WHEEL has no usable Tag"),
        ("WHEEL", "Tag", "py2-none-any", "omits filename compatibility tags"),
        ("WHEEL", "Tag", "py3-none-other", "lists tags absent from the filename"),
    ],
)
def test_metadata_and_wheel_headers_are_fail_closed(
    built_wheels: BuiltWheels,
    tmp_path: Path,
    leaf: str,
    header: str,
    value: str | None,
    message: str,
) -> None:
    """Identity, generator, purity, and compatibility headers are mandatory."""

    def change_header(members: dict[str, bytes]) -> None:
        _set_header(members, leaf, header, value)

    wheel = _mutated_wheel(built_wheels.main, tmp_path, change_header)

    _assert_error(audit_wheel(wheel, _main_expectation()), message)


@pytest.mark.parametrize("leaf", ["METADATA", "WHEEL"])
def test_metadata_control_files_cannot_be_empty(
    built_wheels: BuiltWheels,
    tmp_path: Path,
    leaf: str,
) -> None:
    """Empty metadata control files are rejected before header parsing."""

    def empty_member(members: dict[str, bytes]) -> None:
        members[_dist_info_member(members, leaf)] = b""

    wheel = _mutated_wheel(built_wheels.main, tmp_path, empty_member)

    _assert_error(audit_wheel(wheel, _main_expectation()), "metadata member is empty")


def test_wheel_rejects_duplicate_compatibility_tags(
    built_wheels: BuiltWheels,
    tmp_path: Path,
) -> None:
    """Duplicate WHEEL Tag fields cannot obscure compatibility identity."""

    def duplicate_tag(members: dict[str, bytes]) -> None:
        wheel_path = _dist_info_member(members, "WHEEL")
        members[wheel_path] += b"Tag: py3-none-any\n"

    wheel = _mutated_wheel(built_wheels.main, tmp_path, duplicate_tag)

    _assert_error(audit_wheel(wheel, _main_expectation()), "duplicate Tag fields")


def test_compiled_extension_contract_rejects_missing_empty_and_ambiguous_members(
    built_wheels: BuiltWheels,
    tmp_path: Path,
) -> None:
    """A required extension resolves to exactly one non-empty native member."""
    expectation = WheelExpectation(
        "scpn-quantum-control",
        "0.10.0",
        ("scpn_quantum_control",),
        ("scpn_quantum_control.native",),
    )
    missing = audit_wheel(built_wheels.main, expectation)
    _assert_error(missing, "matched 0 members")
    _assert_error(missing, "must not be purelib")
    _assert_error(missing, "needs an ABI tag")
    _assert_error(missing, "needs a platform tag")

    platform_dir = tmp_path / "platform"
    platform_dir.mkdir()

    def configure_platform(members: dict[str, bytes]) -> None:
        _set_header(members, "WHEEL", "Root-Is-Purelib", "false")
        _set_header(members, "WHEEL", "Tag", "cp312-abi3-linux_x86_64")

    def add_empty(members: dict[str, bytes]) -> None:
        configure_platform(members)
        members["scpn_quantum_control/native.abi3.so"] = b""

    empty = _mutated_wheel(
        built_wheels.main,
        platform_dir,
        add_empty,
        filename="scpn_quantum_control-0.10.0-cp312-abi3-linux_x86_64.whl",
    )
    _assert_error(audit_wheel(empty, expectation), "compiled extension is empty")

    ambiguous_dir = tmp_path / "ambiguous"
    ambiguous_dir.mkdir()

    def add_two(members: dict[str, bytes]) -> None:
        configure_platform(members)
        members["scpn_quantum_control/native.abi3.so"] = b"one"
        members["scpn_quantum_control/native.cpython-312-x86_64-linux-gnu.so"] = b"two"

    ambiguous = _mutated_wheel(
        built_wheels.main,
        ambiguous_dir,
        add_two,
        filename="scpn_quantum_control-0.10.0-cp312-abi3-linux_x86_64.whl",
    )
    _assert_error(audit_wheel(ambiguous, expectation), "matched 2 members")


def test_package_contract_rejects_missing_init_and_empty_python_surfaces(
    built_wheels: BuiltWheels,
    tmp_path: Path,
) -> None:
    """Required packages must be importable and contain executable Python."""
    missing_expectation = WheelExpectation("scpn-quantum-control", "0.10.0", ("absent",), ())
    _assert_error(
        audit_wheel(built_wheels.main, missing_expectation),
        "required package has no wheel members",
    )

    def add_without_init(members: dict[str, bytes]) -> None:
        members["demo_pkg/module.py"] = b"VALUE = 1\n"

    missing_init_wheel = _mutated_wheel(built_wheels.main, tmp_path, add_without_init)
    missing_init = audit_wheel(
        missing_init_wheel,
        WheelExpectation("scpn-quantum-control", "0.10.0", ("demo_pkg",), ()),
    )
    _assert_error(missing_init, "missing __init__.py")

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    def add_empty_package(members: dict[str, bytes]) -> None:
        members["demo_pkg/__init__.py"] = b""
        members["demo_pkg/data.bin"] = b"data"

    empty_wheel = _mutated_wheel(built_wheels.main, empty_dir, add_empty_package)
    empty_package = audit_wheel(
        empty_wheel,
        WheelExpectation("scpn-quantum-control", "0.10.0", ("demo_pkg",), ()),
    )
    _assert_error(empty_package, "empty __init__.py")
    _assert_error(empty_package, "no non-empty Python module")


def _break_record(members: dict[str, bytes], case: str) -> None:
    record_path = _record_path(members)
    rows = _decode_record(members[record_path])
    payload_index = next(index for index, row in enumerate(rows) if row[0] != record_path)
    payload_path = rows[payload_index][0]
    if case == "columns":
        rows[payload_index] = [payload_path, "only-two"]
    elif case == "duplicate":
        rows.append(list(rows[payload_index]))
    elif case == "unsafe":
        rows[payload_index][0] = "../escape.py"
    elif case == "missing":
        del rows[payload_index]
    elif case == "extra":
        rows.append(["absent.py", _record_digest(b"absent"), "6"])
    elif case == "own-fields":
        own = next(row for row in rows if row[0] == record_path)
        own[1:] = [_record_digest(b"record"), "6"]
    elif case == "size-missing":
        rows[payload_index][2] = ""
    elif case == "size-text":
        rows[payload_index][2] = "large"
    elif case == "size-mismatch":
        rows[payload_index][2] = "999999"
    elif case == "digest-missing":
        rows[payload_index][1] = ""
    elif case == "digest-algorithm":
        rows[payload_index][1] = "md5=abc"
    elif case == "digest-base64":
        rows[payload_index][1] = "sha256=***"
    elif case == "digest-decode":
        rows[payload_index][1] = "sha256=A"
    elif case == "digest-length":
        rows[payload_index][1] = "sha256=YQ"
    elif case == "digest-mismatch":
        rows[payload_index][1] = _record_digest(b"different")
    else:
        raise AssertionError(f"unsupported RECORD corruption case: {case}")
    members[record_path] = _encode_record(rows)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("columns", "exactly three columns"),
        ("duplicate", "duplicate RECORD path"),
        ("unsafe", "unsafe RECORD path"),
        ("missing", "RECORD omits archive members"),
        ("extra", "RECORD lists absent archive members"),
        ("own-fields", "leave its own hash and size empty"),
        ("size-missing", "RECORD size is missing"),
        ("size-text", "RECORD size is not an integer"),
        ("size-mismatch", "RECORD size mismatch"),
        ("digest-missing", "digest is missing or malformed"),
        ("digest-algorithm", "digest algorithm must be sha256"),
        ("digest-base64", "digest is not valid URL-safe base64"),
        ("digest-decode", "digest is not valid URL-safe base64"),
        ("digest-length", "sha256 digest has the wrong length"),
        ("digest-mismatch", "RECORD digest mismatch"),
    ],
)
def test_record_integrity_corruptions_are_release_blocking(
    built_wheels: BuiltWheels,
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    """Every RECORD membership, size, and sha256 failure is reported."""
    wheel = _mutated_wheel(
        built_wheels.main,
        tmp_path,
        lambda members: _break_record(members, case),
        repair_record=False,
    )

    _assert_error(audit_wheel(wheel, _main_expectation()), message)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"\xff", "RECORD is not UTF-8"),
        (b'"unterminated', "RECORD is not valid CSV"),
    ],
)
def test_record_text_must_be_utf8_and_valid_csv(
    built_wheels: BuiltWheels,
    tmp_path: Path,
    payload: bytes,
    message: str,
) -> None:
    """RECORD decoding and CSV syntax fail without a traceback."""

    def replace_record(members: dict[str, bytes]) -> None:
        members[_record_path(members)] = payload

    wheel = _mutated_wheel(
        built_wheels.main,
        tmp_path,
        replace_record,
        repair_record=False,
    )

    _assert_error(audit_wheel(wheel, _main_expectation()), message)


def test_detached_record_signatures_are_omitted_from_record(
    built_wheels: BuiltWheels,
    tmp_path: Path,
) -> None:
    """Detached RECORD signatures may exist but must not list themselves."""

    def add_signature(members: dict[str, bytes]) -> None:
        members[f"{_record_path(members)}.jws"] = b"signature"

    valid = _mutated_wheel(built_wheels.main, tmp_path, add_signature)
    valid_audit = audit_wheel(valid, _main_expectation())
    assert valid_audit.passed, valid_audit.errors

    listed_dir = tmp_path / "listed"
    listed_dir.mkdir()

    def list_signature(members: dict[str, bytes]) -> None:
        add_signature(members)
        _rewrite_record(members)
        record_path = _record_path(members)
        rows = _decode_record(members[record_path])
        rows.insert(0, [f"{record_path}.jws", _record_digest(b"signature"), "9"])
        members[record_path] = _encode_record(rows)

    listed = _mutated_wheel(
        built_wheels.main,
        listed_dir,
        list_signature,
        repair_record=False,
    )
    _assert_error(
        audit_wheel(listed, _main_expectation()), "must not list its detached signatures"
    )


def test_cli_emits_human_and_json_evidence_and_deduplicates_globs(
    built_wheels: BuiltWheels,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI handles direct paths, globs, deduplication, and both formats."""
    assert main(["--project", str(_MAIN_PROJECT), str(built_wheels.main)]) == 0
    human = capsys.readouterr().out
    assert "PASS:" in human
    assert "package members:" in human

    assert (
        main(
            [
                "--project",
                str(_MAIN_PROJECT),
                "--json",
                str(built_wheels.main),
                str(built_wheels.main.parent / "*.whl"),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert len(payload) == 1
    assert payload[0]["passed"] is True


def test_cli_returns_one_for_wheel_findings(
    built_wheels: BuiltWheels,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A content finding produces a human-readable failure and exit one."""

    def remove_metadata(members: dict[str, bytes]) -> None:
        del members[_dist_info_member(members, "METADATA")]

    failing = _mutated_wheel(
        built_wheels.main,
        tmp_path,
        remove_metadata,
    )

    assert main(["--project", str(_MAIN_PROJECT), str(failing)]) == 1
    assert "FAIL:" in capsys.readouterr().out


def test_cli_returns_two_for_project_and_path_configuration_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unreadable metadata and unmatched wheel patterns return exit two."""
    missing_project = tmp_path / "missing.toml"
    assert main(["--project", str(missing_project), "missing.whl"]) == 2
    assert "configuration error" in capsys.readouterr().err

    invalid_utf8 = tmp_path / "pyproject.toml"
    invalid_utf8.write_bytes(b"\xff")
    assert main(["--project", str(invalid_utf8), "missing.whl"]) == 2
    assert "configuration error" in capsys.readouterr().err

    invalid_toml = tmp_path / "invalid.toml"
    invalid_toml.write_text("[project", encoding="utf-8")
    assert main(["--project", str(invalid_toml), "missing.whl"]) == 2
    assert "configuration error" in capsys.readouterr().err

    assert main(["--project", str(_MAIN_PROJECT), str(tmp_path / "*.whl")]) == 2
    assert "matched no files" in capsys.readouterr().err


def test_parser_and_empty_expansion_reject_missing_wheels() -> None:
    """Both argparse and the expansion helper reject an empty wheel set."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--project", str(_MAIN_PROJECT)])
    assert exc_info.value.code == 2
    with pytest.raises(ValueError, match="no wheel paths"):
        wheel_tool._expand_wheel_paths(())


def test_format_and_json_payload_include_failure_evidence(tmp_path: Path) -> None:
    """Machine and human evidence expose stable failure fields."""
    audit = WheelAudit(
        wheel=tmp_path / "demo.whl",
        distribution="demo",
        version="1",
        member_count=2,
        package_member_counts=(),
        extension_members=(),
        errors=("broken",),
    )

    assert audit.passed is False
    assert audit.as_dict()["errors"] == ["broken"]
    assert "FAIL:" in format_audit(audit)
    assert "error: broken" in format_audit(audit)


def _assert_ordered(text: str, *needles: str) -> None:
    offsets = [text.index(needle) for needle in needles]
    assert offsets == sorted(offsets)


def test_all_publish_workflows_block_on_the_wheel_audit() -> None:
    """Every Python and Rust publication path runs the gate before upload."""
    publish = (_REPO_ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")
    oscillatools = (_REPO_ROOT / ".github/workflows/oscillatools-publish.yml").read_text(
        encoding="utf-8"
    )
    rust = (_REPO_ROOT / ".github/workflows/rust-wheels.yml").read_text(encoding="utf-8")
    publish_compact = " ".join(publish.split())
    oscillatools_compact = " ".join(oscillatools.split())
    rust_compact = " ".join(rust.split())
    _assert_ordered(
        publish_compact,
        "python -m build",
        "python tools/check_wheel_contents.py --project pyproject.toml 'dist/*.whl'",
        "pypa/gh-action-pypi-publish@",
    )
    _assert_ordered(
        oscillatools_compact,
        "python -m build oscillatools/",
        "--project oscillatools/pyproject.toml 'oscillatools/dist/*.whl'",
        "pypa/gh-action-pypi-publish@",
    )
    rust_command = (
        "--project scpn_quantum_engine/pyproject.toml --package scpn_quantum_engine "
        "--extension scpn_quantum_engine.scpn_quantum_engine 'dist/*.whl'"
    )
    assert rust_compact.count(rust_command) == 2
    _assert_ordered(rust_compact, "name: Build wheels", rust_command, "name: Upload wheels")
    rust_publish = rust_compact[rust_compact.index("publish: needs:") :]
    _assert_ordered(
        rust_publish,
        "name: Download all artifacts",
        rust_command,
        "name: Publish to PyPI",
    )
    assert 'tags: ["v*"]' in publish
    assert 'tags: ["oscillatools-v*"]' in oscillatools
    assert 'tags: ["engine-v*"]' in rust


def test_ci_and_strict_typing_policy_own_the_wheel_gate() -> None:
    """CI permanently enforces strict typing, NumPy docstrings, and test typing."""
    ci = (_REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    policy = json.loads((_REPO_ROOT / "tools/test_typing_policy.json").read_text(encoding="utf-8"))
    repository_policy = next(
        cohort for cohort in policy["cohorts"] if cohort["id"] == "repository_policy"
    )
    assert "mypy --strict --explicit-package-bases" in ci
    assert "tools/check_wheel_contents.py" in ci
    assert "tests/test_wheel_contents.py" in ci
    assert "ruff check --isolated --select D,D413" in ci
    assert "tests/test_wheel_contents.py" in repository_policy["files"]


def test_release_documentation_exposes_each_distribution_command() -> None:
    """Release operators receive exact parent, oscillatools, and Rust commands."""
    readiness = (_REPO_ROOT / "docs/release_readiness.md").read_text(encoding="utf-8")
    for fragment in (
        "tools/check_wheel_contents.py --project pyproject.toml",
        "--project oscillatools/pyproject.toml",
        "--project scpn_quantum_engine/pyproject.toml",
        "Wheel content",
    ):
        assert fragment in readiness
