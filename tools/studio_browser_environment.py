# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Studio browser environment preparation
"""Prepare an executable-specific sandbox profile and qualified browser sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path


def configure_chromium_sandbox(executable: Path, temporary: Path, github_env: Path) -> Path:
    """Write the Ubuntu namespace profile for one existing browser executable.

    Parameters
    ----------
    executable : Path
        Installed Chromium selected by the caller from the locked browser package.
    temporary : Path
        Existing task-owned directory for the profile.
    github_env : Path
        GitHub environment file receiving the same explicit executable path.

    Returns
    -------
    Path
        Written profile, ready for the runner's explicit AppArmor loading step.

    Raises
    ------
    OSError
        If the executable or output paths cannot be read or written.
    ValueError
        If the target is not executable or contains profile syntax metacharacters.

    """
    resolved = executable.resolve(strict=True)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError("Chromium must be an executable file")
    selected = str(resolved)
    if any(character in selected for character in '\\"\r\n{}[]*?'):
        raise ValueError("Chromium executable cannot contain AppArmor metacharacters")
    profile = temporary / "studio-chromium.apparmor"
    profile.write_text(
        "abi <abi/4.0>,\ninclude <tunables/global>\n"
        f'profile scpn-studio-chromium "{selected}" flags=(unconfined) {{\n'
        "  userns,\n}\n"
    )
    with github_env.open("a") as environment:
        environment.write(f"STUDIO_NATIVE_BROWSER_EXECUTABLE={selected}\n")
    return profile


def prepare_source_workbench(original: Path, temporary: Path) -> dict[str, dict[str, str]]:
    """Copy the original preview, source owners and both built kernels with hashes.

    Parameters
    ----------
    original : Path
        Original repository with its completed Studio and federation builds.
    temporary : Path
        Existing task-owned directory receiving separate preview and source hosts.

    Returns
    -------
    dict[str, dict[str, str]]
        Original source and kernel SHA-256 maps also written to qualification JSON.

    Raises
    ------
    OSError
        If an input is missing or an output cannot be created.
    AssertionError
        If copied bytes differ or the original two-kernel cohort is incomplete.

    """
    original = original.resolve()
    preview = temporary / "studio-browser-preview"
    shutil.copytree(original / "studio-web/dist", preview)
    shutil.copytree(temporary / "studio-federation-host", preview / "acceptance-host")
    isolated = temporary / "studio-workbench-source"
    isolated.mkdir()
    for name in ("docs", "data", "tests"):
        (isolated / name).symlink_to(original / name, target_is_directory=True)
    web = isolated / "studio-web"
    web.mkdir()
    for name in ("src", "browser-tests", "test-support"):
        shutil.copytree(original / "studio-web" / name, web / name)
    for source in (original / "studio-web").iterdir():
        if source.is_file() and source.suffix in {".ts", ".json", ".html"}:
            shutil.copy2(source, web / source.name)
    (web / "node_modules").symlink_to(
        original / "studio-web/node_modules", target_is_directory=True
    )
    shutil.copytree(original / "studio-web/dist/wasm", web / "src/wasm")
    source_hashes = {}
    for source in (original / "studio-web/src").rglob("*"):
        if source.is_file():
            relative = source.relative_to(original / "studio-web")
            assert source.read_bytes() == (web / relative).read_bytes()
            source_hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    kernels = {}
    for source in (original / "studio-web/dist/wasm").glob("*.wasm"):
        assert source.read_bytes() == (web / "src/wasm" / source.name).read_bytes()
        kernels[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    assert set(kernels) == {
        "scpn_quantum_studio_wasm_kernel.wasm",
        "scpn_quantum_studio_program_ad_wasm.wasm",
    }
    qualification = {"original_source_sha256": source_hashes, "kernel_sha256": kernels}
    (temporary / "studio-workbench-source-qualification.json").write_text(
        json.dumps(qualification, indent=2) + "\n"
    )
    return qualification


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare one browser environment through its explicit public CLI.

    Parameters
    ----------
    argv : Sequence[str] or None
        Supplied arguments, or the process arguments when omitted.

    Returns
    -------
    int
        Zero after successful preparation; invalid inputs raise before success.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    sandbox = commands.add_parser("sandbox")
    sandbox.add_argument("--executable", type=Path, required=True)
    sandbox.add_argument("--temporary", type=Path, required=True)
    sandbox.add_argument("--github-env", type=Path, required=True)
    workbench = commands.add_parser("workbench")
    workbench.add_argument("--root", type=Path, required=True)
    workbench.add_argument("--temporary", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.operation == "sandbox":
        configure_chromium_sandbox(arguments.executable, arguments.temporary, arguments.github_env)
    else:
        prepare_source_workbench(arguments.root, arguments.temporary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
