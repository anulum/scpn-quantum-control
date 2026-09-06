# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Reproducible documentation-debt baseline
"""Measure the full-scope documentation-rule baseline with its provenance.

The documentation lane's recorded figure could not be reproduced from what was
kept beside it. The 2026-09-04 review recorded 8,114 findings across 699 files
and listed ``D413`` among the rules; the same scope measured with Ruff 0.16.4
reports ``D421`` and no ``D413`` at all, because the preview rule set changed
between the two runs. A count without its tool version is not a baseline, it is
an anecdote.

This module emits the count together with everything needed to obtain it again:
the Ruff and Python versions, the exact argument vector, the source commit and
the scopes. It deliberately shells out to Ruff rather than reimplementing the
rules, so the number is Ruff's and not a second opinion.

It does not compete with ``tools/audit_documentation_surface.py``. That auditor
walks the AST over the published surface and excludes tests and oscillatools by
design, which is why it can correctly report zero while this scan reports
thousands. They measure different things and both are wanted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

DOCUMENTATION_DEBT_SCOPES: Final = (
    "src",
    "tests",
    "oscillatools",
    "scripts",
    "tools",
    "examples",
    "data",
    "figures",
    "paper",
)
"""Directories the full-scope scan covers, in the order the review named them."""

PYDOCSTYLE_CONVENTION: Final = 'lint.pydocstyle.convention = "numpy"'
"""Inline configuration applied instead of the repository's own settings.

The scan runs ``--isolated`` so the repository's per-file ignores do not hide
debt in tests and oscillatools, which is the whole point of a full-scope
baseline. A repository-configured run answers a different question and gives a
different number; both are legitimate and they must not be quoted as if they
were one measurement.
"""


@dataclass(frozen=True)
class DocumentationDebtBaseline:
    """One measurement of the documentation-rule debt, with its provenance."""

    ruff_version: str
    python_version: str
    source_sha: str
    scopes: tuple[str, ...]
    command: tuple[str, ...]
    total_findings: int
    total_files: int
    by_scope: dict[str, int]
    by_rule: dict[str, int]

    def to_json(self) -> str:
        """Return the baseline as stable, sorted JSON.

        Returns
        -------
        str
            Two-space indented JSON ending in a newline.

        """
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def baseline_command(python: str | None = None) -> tuple[str, ...]:
    """Return the exact argument vector the baseline is measured with.

    Parameters
    ----------
    python
        Interpreter to run Ruff through; defaults to the current one, so the
        recorded command names the interpreter that produced the figure.

    Returns
    -------
    tuple
        The argument vector, suitable for :func:`subprocess.run`.

    """
    return (
        python or sys.executable,
        "-m",
        "ruff",
        "check",
        "--isolated",
        "--preview",
        "--select",
        "D",
        "--config",
        PYDOCSTYLE_CONVENTION,
        "--output-format",
        "json",
        *DOCUMENTATION_DEBT_SCOPES,
    )


def _tool_version(python: str, module: str) -> str:
    """Return a tool's reported version string.

    Parameters
    ----------
    python
        Interpreter to ask.
    module
        Module exposing ``--version``.

    Returns
    -------
    str
        The version line, stripped.

    """
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [python, "-m", module, "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _source_sha(root: Path) -> str:
    """Return the commit the measurement describes.

    Parameters
    ----------
    root
        Repository root.

    Returns
    -------
    str
        The full commit SHA, or ``"unknown"`` when Git cannot answer.

    """
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _scope_of(filename: str, root: Path) -> str:
    """Return the top-level scope a finding belongs to.

    Parameters
    ----------
    filename
        Absolute path reported by Ruff.
    root
        Repository root, stripped from the front.

    Returns
    -------
    str
        The first path segment, or ``"<outside>"`` for anything unexpected.

    """
    try:
        relative = Path(filename).resolve().relative_to(root.resolve())
    except ValueError:
        return "<outside>"
    return relative.parts[0] if relative.parts else "<outside>"


def measure_documentation_debt(
    root: Path | None = None, python: str | None = None
) -> DocumentationDebtBaseline:
    """Run the scan and return the baseline with its provenance.

    Parameters
    ----------
    root
        Repository root; defaults to the current working directory.
    python
        Interpreter to run Ruff through; defaults to the current one.

    Returns
    -------
    DocumentationDebtBaseline
        The measurement, including the command that produced it.

    Raises
    ------
    RuntimeError
        If Ruff fails in a way that is not "findings were reported". Exit code
        1 means findings exist, which is the expected outcome here; anything
        else is a broken invocation and must not be recorded as a count.

    """
    repository_root = Path.cwd() if root is None else root
    interpreter = python or sys.executable
    command = baseline_command(interpreter)
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        command,
        capture_output=True,
        text=True,
        cwd=repository_root,
        check=False,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError(
            f"ruff exited {completed.returncode}, which is not a findings result: "
            f"{completed.stderr.strip()[:200]}"
        )
    findings: list[dict[str, Any]] = json.loads(completed.stdout or "[]")
    return DocumentationDebtBaseline(
        ruff_version=_tool_version(interpreter, "ruff"),
        python_version=f"Python {sys.version.split()[0]}",
        source_sha=_source_sha(repository_root),
        scopes=DOCUMENTATION_DEBT_SCOPES,
        command=command[1:],
        total_findings=len(findings),
        total_files=len({finding["filename"] for finding in findings}),
        by_scope=dict(
            sorted(
                Counter(
                    _scope_of(finding["filename"], repository_root) for finding in findings
                ).items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        by_rule=dict(
            sorted(
                Counter(finding["code"] for finding in findings).items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """Print the baseline as JSON.

    Parameters
    ----------
    argv
        Command-line arguments; defaults to :data:`sys.argv`.

    Returns
    -------
    int
        Process exit status, zero on success.

    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the baseline here instead of standard output",
    )
    arguments = parser.parse_args(argv)
    baseline = measure_documentation_debt()
    if arguments.output is None:
        sys.stdout.write(baseline.to_json())
    else:
        arguments.output.write_text(baseline.to_json(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
