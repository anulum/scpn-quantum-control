# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Studio browser environment tests
"""Exercise public profile generation and source custody on the real filesystem."""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
from pathlib import Path

import pytest

from tools import studio_browser_environment as environment


@pytest.fixture
def workbench_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """Write source, preview and valid minimal WASM inputs for byte custody.

    Parameters
    ----------
    tmp_path : Path
        Isolated filesystem location.

    Returns
    -------
    tuple[Path, Path]
        Repository and temporary build roots.

    """
    original = tmp_path / "original"
    temporary = tmp_path / "temporary"
    for name in ("docs", "data", "tests", "studio-web/node_modules"):
        (original / name).mkdir(parents=True)
    for name in ("src/owners", "public", "browser-tests", "test-support", "dist/wasm"):
        (original / "studio-web" / name).mkdir(parents=True)
    (original / "studio-web/public/favicon.svg").write_bytes(
        (Path(__file__).resolve().parents[2] / "studio-web/public/favicon.svg").read_bytes()
    )
    for name, body in {
        "src/owners/source.ts": "export const source = 7;\n",
        "src/owners/metadata.json": '{"schema_version":1}\n',
        "browser-tests/workspace.html": "<main>Workspace</main>\n",
        "test-support/source.ts": "export const original = true;\n",
        "dist/index.html": "<main>Preview</main>\n",
        "package.json": '{"private":true}\n',
        "vite.config.ts": "export default {};\n",
        "index.html": "<main>Original</main>\n",
        "README.md": "Original source host\n",
    }.items():
        (original / "studio-web" / name).write_text(body)
    # Minimal version-one WASM modules are copied and hashed, never executed here.
    for name in (
        "scpn_quantum_studio_wasm_kernel.wasm",
        "scpn_quantum_studio_program_ad_wasm.wasm",
    ):
        (original / "studio-web/dist/wasm" / name).write_bytes(b"\x00asm\x01\x00\x00\x00")
    (temporary / "studio-federation-host").mkdir(parents=True)
    (temporary / "studio-federation-host/index.html").write_text("<main>Federation</main>\n")
    return original, temporary


def test_profile_binds_real_executable_and_appends_environment(tmp_path: Path) -> None:
    """Bind the actual Python executable path without changing host policy.

    Parameters
    ----------
    tmp_path : Path
        Existing output directory.

    """
    github_env = tmp_path / "github-env"
    github_env.write_text("EXISTING=preserved\n")
    selected = Path(sys.executable).resolve()
    profile = environment.configure_chromium_sandbox(selected, tmp_path, github_env)
    assert profile.read_text() == (
        "abi <abi/4.0>,\ninclude <tunables/global>\n"
        f'profile scpn-studio-chromium "{selected}" flags=(unconfined) {{\n'
        "  userns,\n}\n"
    )
    assert github_env.read_text() == (
        f"EXISTING=preserved\nSTUDIO_NATIVE_BROWSER_EXECUTABLE={selected}\n"
    )


@pytest.mark.parametrize("character", ['"', "\\", "\r", "\n", "{", "}", "[", "]", "*", "?"])
def test_profile_refuses_syntax_in_existing_executable_path(
    tmp_path: Path, character: str
) -> None:
    """Refuse each profile metacharacter before creating output or environment.

    Parameters
    ----------
    tmp_path : Path
        Real output directory.
    character : str
        Unsafe character in a real executable filename.

    """
    executable = tmp_path / f"browser{character}path"
    executable.touch(mode=0o700)
    with pytest.raises(ValueError, match="metacharacters"):
        environment.configure_chromium_sandbox(executable, tmp_path, tmp_path / "github-env")
    assert not (tmp_path / "studio-chromium.apparmor").exists()
    assert not (tmp_path / "github-env").exists()


@pytest.mark.parametrize("kind", ["directory", "nonexecutable", "missing"])
def test_profile_refuses_invalid_executable(tmp_path: Path, kind: str) -> None:
    """Refuse actual missing, directory and non-executable inputs.

    Parameters
    ----------
    tmp_path : Path
        Real output directory.
    kind : str
        Invalid input type.

    """
    executable = tmp_path / "browser"
    if kind == "directory":
        executable.mkdir()
    elif kind == "nonexecutable":
        executable.touch(mode=0o600)
    with pytest.raises((ValueError, FileNotFoundError)):
        environment.configure_chromium_sandbox(executable, tmp_path, tmp_path / "github-env")
    assert not (tmp_path / "studio-chromium.apparmor").exists()


def test_workbench_preserves_original_sources_and_hashes(
    workbench_inputs: tuple[Path, Path],
) -> None:
    """Verify real source/preview copies, original links and both kernel digests.

    Parameters
    ----------
    workbench_inputs : tuple[Path, Path]
        Repository and build roots containing actual files.

    """
    original, temporary = workbench_inputs
    result = environment.prepare_source_workbench(original, temporary)
    isolated = temporary / "studio-workbench-source"
    assert "public/favicon.svg" in result["original_source_sha256"]
    for name in ("docs", "data", "tests", "studio-web/node_modules"):
        assert (isolated / name).resolve() == (original / name).resolve()
    for relative, digest in result["original_source_sha256"].items():
        body = (original / "studio-web" / relative).read_bytes()
        assert (isolated / "studio-web" / relative).read_bytes() == body
        assert digest == hashlib.sha256(body).hexdigest()
    for name, digest in result["kernel_sha256"].items():
        body = (original / "studio-web/dist/wasm" / name).read_bytes()
        assert (isolated / "studio-web/src/wasm" / name).read_bytes() == body
        assert digest == hashlib.sha256(body).hexdigest()
    for name in ("package.json", "vite.config.ts", "index.html"):
        assert (isolated / "studio-web" / name).read_bytes() == (
            original / "studio-web" / name
        ).read_bytes()
    assert not (isolated / "studio-web/README.md").exists()
    assert (temporary / "studio-browser-preview/acceptance-host/index.html").read_text() == (
        temporary / "studio-federation-host/index.html"
    ).read_text()
    assert (
        json.loads((temporary / "studio-workbench-source-qualification.json").read_text())
        == result
    )
    with pytest.raises(FileExistsError):
        environment.main(["workbench", "--root", str(original), "--temporary", str(temporary)])


@pytest.mark.parametrize("kind", ["missing", "unexpected"])
def test_workbench_refuses_incomplete_original_kernel_cohort(
    workbench_inputs: tuple[Path, Path], kind: str
) -> None:
    """Refuse missing or extra kernels before publishing qualification.

    Parameters
    ----------
    workbench_inputs : tuple[Path, Path]
        Real original and temporary roots.
    kind : str
        Kernel cohort defect.

    """
    original, temporary = workbench_inputs
    kernels = original / "studio-web/dist/wasm"
    if kind == "missing":
        (kernels / "scpn_quantum_studio_program_ad_wasm.wasm").unlink()
    else:
        (kernels / "unexpected.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00")
    with pytest.raises(AssertionError):
        environment.prepare_source_workbench(original, temporary)
    assert not (temporary / "studio-workbench-source-qualification.json").exists()


def test_workbench_cli_publishes_original_byte_custody(
    workbench_inputs: tuple[Path, Path],
) -> None:
    """Complete the real workbench CLI and read its original qualification file.

    Parameters
    ----------
    workbench_inputs : tuple[Path, Path]
        Real repository and output directories.

    """
    original, temporary = workbench_inputs
    assert (
        environment.main(["workbench", "--root", str(original), "--temporary", str(temporary)])
        == 0
    )
    qualification = json.loads(
        (temporary / "studio-workbench-source-qualification.json").read_text()
    )
    assert (
        qualification["original_source_sha256"]["src/owners/source.ts"]
        == hashlib.sha256((original / "studio-web/src/owners/source.ts").read_bytes()).hexdigest()
    )
    assert set(qualification["kernel_sha256"]) == {
        "scpn_quantum_studio_wasm_kernel.wasm",
        "scpn_quantum_studio_program_ad_wasm.wasm",
    }


def test_public_module_entry_writes_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the original module entry with explicit public CLI arguments.

    Parameters
    ----------
    tmp_path : Path
        Actual output directory.
    monkeypatch : pytest.MonkeyPatch
        Supplies process arguments without replacing filesystem operations.

    """
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "studio_browser_environment",
            "sandbox",
            "--executable",
            sys.executable,
            "--temporary",
            str(tmp_path),
            "--github-env",
            str(tmp_path / "github-env"),
        ],
    )
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(Path(environment.__file__)), run_name="__main__")
    assert result.value.code == 0
    assert (tmp_path / "studio-chromium.apparmor").exists()
