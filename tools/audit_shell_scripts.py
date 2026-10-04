# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — shell script lint gate
"""Fail when a tracked shell script has a ShellCheck finding or is not formatted.

The repository's lint gates read Python, Rust, Julia and TypeScript. Shell
scripts provision runners, drive hardware campaigns and guard pushes, and no
gate read them. This gate lists every tracked shell script, runs ShellCheck
over all of them and fails on any finding, down to style level, and fails on
any script that the ``shfmt`` formatter would change.

A tracked file is a shell script when its suffix is ``.sh`` or ``.bash``, or
when it has no suffix and its first line names ``sh``, ``bash``, ``dash`` or
``ksh`` as the interpreter. The second rule finds hook scripts such as
``pre-push`` that a suffix search misses.

Findings depend on the ShellCheck release. The release is pinned in
``requirements-ci-shell-lint.txt`` and the gate refuses to run with another
one instead of reporting a result that the pinned release might not give.
The gate uses the ShellCheck installed next to the running interpreter, which
is where that requirement file puts it, and falls back to the one on ``PATH``.

The formatter is pinned in the same file as the ``shfmt-py`` distribution, whose
version differs from the version of the binary it carries. The gate therefore
checks that the distribution installed beside the running interpreter is the
pinned one and uses that binary. The layout follows the repository's editor
configuration, which the formatter reads by itself.

The gate reports what the two tools report. It does not rewrite scripts and
it does not run them.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

PIN_FILE = Path("requirements-ci-shell-lint.txt")
DISTRIBUTION = "shellcheck-py"
FORMATTER_DISTRIBUTION = "shfmt-py"
SHELL_SUFFIXES = frozenset({".sh", ".bash"})
_VERSION = re.compile(r"^version: (\d+\.\d+\.\d+)$", re.MULTILINE)
_INTERPRETER = re.compile(rb"^#![^\n]*?(?:/|\s)(?:ba|da|k)?sh(?:\s|$)")


def _run(repo: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run a fixed command in ``repo`` and return the finished process.

    Raises
    ------
    ValueError
        If the command cannot be started.

    """
    try:
        return subprocess.run(  # noqa: S603 - fixed argument vectors, no shell
            list(arguments), capture_output=True, text=True, cwd=repo, check=False
        )
    except OSError as error:
        raise ValueError(f"cannot run {arguments[0]}: {error}") from error


def environment_executable(interpreter: Path) -> str:
    """Return the ShellCheck that belongs to an interpreter's environment.

    Parameters
    ----------
    interpreter
        Path of a Python executable.

    Returns
    -------
    str
        The ``shellcheck`` file beside ``interpreter`` when there is one,
        otherwise the bare name ``"shellcheck"`` to be resolved on ``PATH``.

    """
    candidate = interpreter.parent / "shellcheck"
    return str(candidate) if candidate.is_file() else "shellcheck"


def pinned_version(repo: Path, distribution: str) -> str:
    """Return the version a distribution is pinned to in the linter lock.

    Parameters
    ----------
    repo
        Repository root.
    distribution
        Distribution name as written in the lock.

    Returns
    -------
    str
        The complete pinned version.

    Raises
    ------
    ValueError
        If the pin file is unreadable or has no pin for ``distribution``.

    """
    path = repo / PIN_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read the linter pin: {error}") from error
    match = re.search(rf"^{re.escape(distribution)}==([0-9][0-9A-Za-z.]*)", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"{PIN_FILE} does not pin {distribution}")
    return match.group(1)


def pinned_release(repo: Path) -> str:
    """Return the ShellCheck release the repository pins.

    Returns
    -------
    str
        The first three components of the pinned ``shellcheck-py`` version,
        which are the ShellCheck release the wheel carries.

    Raises
    ------
    ValueError
        If the pin file is unreadable or has no such pin.

    """
    return ".".join(pinned_version(repo, DISTRIBUTION).split(".")[:3])


def pinned_formatter(
    repo: Path, interpreter: Path, distribution: str = FORMATTER_DISTRIBUTION
) -> str:
    """Return the pinned formatter that belongs to an interpreter's environment.

    Parameters
    ----------
    repo
        Repository root.
    interpreter
        Path of the Python executable whose environment is inspected. The
        installed version is read from the environment of the running
        interpreter, so this is the running interpreter in normal use.
    distribution
        Distribution that carries the formatter.

    Returns
    -------
    str
        Path of the ``shfmt`` file beside ``interpreter``.

    Raises
    ------
    ValueError
        If the distribution is not installed, is installed in another version
        than the pinned one, or left no ``shfmt`` beside the interpreter.

    """
    pinned = pinned_version(repo, distribution)
    try:
        installed = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError(f"{distribution} is not installed; install {PIN_FILE}") from error
    if installed != pinned:
        raise ValueError(
            f"{distribution} {installed} is not the pinned {pinned}; install {PIN_FILE}"
        )
    candidate = interpreter.parent / "shfmt"
    if not candidate.is_file():
        raise ValueError(f"{distribution} left no shfmt beside {interpreter}")
    return str(candidate)


def installed_release(repo: Path, executable: str) -> str:
    """Return the release of the ShellCheck executable.

    Raises
    ------
    ValueError
        If the executable cannot run or prints no version.

    """
    completed = _run(repo, [executable, "--version"])
    match = _VERSION.search(completed.stdout)
    if completed.returncode != 0 or match is None:
        raise ValueError(f"{executable} --version reported no release")
    return match.group(1)


def is_shell_script(repo: Path, name: str) -> bool:
    """Tell whether a tracked path is a shell script.

    Parameters
    ----------
    repo
        Repository root.
    name
        Path relative to the repository root, with forward slashes.

    Returns
    -------
    bool
        True for a ``.sh`` or ``.bash`` file, and for a regular file without a
        suffix whose first line names a supported shell as the interpreter.

    """
    suffix = PurePosixPath(name).suffix.lower()
    if suffix in SHELL_SUFFIXES:
        return True
    path = repo / name
    if suffix or not path.is_file():
        return False
    with path.open("rb") as handle:
        return _INTERPRETER.match(handle.readline(256)) is not None


def shell_scripts(repo: Path) -> list[str]:
    """Return the tracked shell scripts of ``repo`` in sorted order.

    Raises
    ------
    ValueError
        If Git cannot list the tracked files.

    """
    completed = _run(repo, ["git", "ls-files", "-z"])
    if completed.returncode != 0:
        raise ValueError(f"git ls-files failed: {completed.stderr.strip()}")
    names = [name for name in completed.stdout.split("\0") if name]
    return sorted(name for name in names if is_shell_script(repo, name))


def lint(repo: Path, scripts: Sequence[str], executable: str) -> list[str]:
    """Run ShellCheck over ``scripts`` and return one message per finding.

    Parameters
    ----------
    repo
        Repository root; the paths in ``scripts`` are relative to it.
    scripts
        Shell scripts to read.
    executable
        ShellCheck executable.

    Returns
    -------
    list[str]
        Messages of the form ``path:line:column: level: SCnnnn: text``; empty
        when there is no script or no finding.

    Raises
    ------
    ValueError
        If ShellCheck could not read a script or failed for another reason.

    """
    if not scripts:
        return []
    completed = _run(repo, [executable, "--format=json1", "--severity=style", "--", *scripts])
    if completed.returncode not in (0, 1):
        raise ValueError(f"{executable} failed: {' '.join(completed.stderr.split())}")
    comments = json.loads(completed.stdout)["comments"]
    return [
        f"{comment['file']}:{comment['line']}:{comment['column']}: {comment['level']}: "
        f"SC{comment['code']}: {comment['message']}"
        for comment in comments
    ]


def unformatted(repo: Path, scripts: Sequence[str], executable: str) -> list[str]:
    """Return one message per script that the formatter would change.

    Parameters
    ----------
    repo
        Repository root; the paths in ``scripts`` are relative to it.
    scripts
        Shell scripts to read.
    executable
        ``shfmt`` executable.

    Returns
    -------
    list[str]
        Messages of the form ``path: not formatted; run shfmt -w path``; empty
        when there is no script or every script is formatted.

    Raises
    ------
    ValueError
        If the formatter cannot parse or read a script. The formatter's exit
        status is not used: it is non-zero both for an error and for a
        script that merely needs formatting; only an error writes to the
        error stream.

    """
    if not scripts:
        return []
    completed = _run(repo, [executable, "-l", "--", *scripts])
    if completed.stderr.strip():
        raise ValueError(f"{executable} failed: {' '.join(completed.stderr.split())}")
    return [
        f"{name}: not formatted; run shfmt -w {name}" for name in completed.stdout.splitlines()
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Lint and format-check every tracked shell script with the pinned tools.

    Parameters
    ----------
    argv
        Command-line arguments; ``None`` reads the process arguments.

    Returns
    -------
    int
        Zero when no script has a finding and none needs formatting; one on
        a finding, on a tool other than the pinned one, or when a tool or
        Git cannot run.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--shellcheck",
        default=environment_executable(Path(sys.executable)),
        help="ShellCheck executable (default: the one beside this interpreter, else on PATH)",
    )
    args = parser.parse_args(argv)
    repo: Path = args.repo
    try:
        pinned = pinned_release(repo)
        installed = installed_release(repo, args.shellcheck)
        if installed != pinned:
            raise ValueError(
                f"ShellCheck {installed} is not the pinned release {pinned}; install {PIN_FILE}"
            )
        formatter = pinned_formatter(repo, Path(sys.executable))
        scripts = shell_scripts(repo)
        findings = lint(repo, scripts, args.shellcheck)
        findings += unformatted(repo, scripts, formatter)
    except ValueError as error:
        print(f"shell script lint failed: {error}", file=sys.stderr)
        return 1
    for message in findings:
        print(message, file=sys.stderr)
    print(
        f"Shell script lint: {len(scripts)} scripts; ShellCheck {installed}; "
        f"{FORMATTER_DISTRIBUTION} {pinned_version(repo, FORMATTER_DISTRIBUTION)}; "
        f"{len(findings)} findings"
    )
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
