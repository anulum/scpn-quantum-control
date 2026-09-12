# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — documentation scope gate
"""Keep the documentation scopes that reached zero at zero.

Closing a scope is not the same as holding it closed. Every non-test Python
scope in this repository now reports zero findings under the isolated preview
documentation scan, and until this gate existed nothing stopped the next file
added to one of them from putting the count back to one, silently: the repeated
scan is a lane measurement that someone runs, not a check that fails a build.

The other languages already had this. Rust denies ``missing_docs`` per crate,
typedoc fails the docs build on an undocumented reflection, and the Julia tier
has a gate of its own. Python's closed scopes were the exception, which is the
gap this closes.

The scan is deliberately the isolated preview one rather than the repository's
configured Ruff run. The configured run passes today while the isolated scan
finds thousands, because per-file ignores and non-preview rules hide them; a
gate measuring the lenient view would report a scope clean while the lane it is
meant to protect still grows.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

#: Scopes required to hold at zero. Test trees are absent by intent: they carry
#: a measured, openly recorded debt and closing them is its own lane.
ENFORCED_SCOPES: Final[tuple[str, ...]] = (
    "src",
    "tools",
    "scripts",
    "oscillatools/src",
    "figures",
    "paper",
    "examples",
    "notebooks",
    "data",
)

#: Files exempt from the rule, each with the reason it cannot comply.
#:
#: These are frozen benchmark runners. Their SHA-256 is recorded in the README
#: beside them as evidence of what was executed, so adding a docstring would
#: change the digest and falsify that record. They are recorded evidence rather
#: than maintained source, and the scan simply reaches a directory holding the
#: former.
EXEMPT: Final[dict[str, str]] = {
    "data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/enzyme_jax_runner.py": "frozen benchmark runner; its SHA-256 is recorded beside it",
    "data/differentiable_phase_qnode/ml350_full_framework_catalyst_baseline_20260705/runners/catalyst_qjit_runner.py": "frozen benchmark runner; its SHA-256 is recorded beside it",
}

#: Explicit NumPy profile shared with the stricter owned documentation cohorts.
#: Ruff validates structure, not semantic completeness of contract descriptions.
SCAN: Final[tuple[str, ...]] = (
    "-m",
    "ruff",
    "check",
    "--isolated",
    "--preview",
    "--select",
    "D,D413,D417,D420",
    "--config",
    "lint.explicit-preview-rules = true",
    "--config",
    'lint.pydocstyle.convention = "numpy"',
    "--output-format",
    "json",
)


def scan(repo: Path, scopes: Sequence[str]) -> list[dict[str, object]]:
    """Return the documentation findings across ``scopes``.

    Parameters
    ----------
    repo
        Repository root.
    scopes
        Directories to scan, relative to ``repo``.

    Returns
    -------
    list[dict[str, object]]
        One entry per finding, as Ruff reports it.

    Raises
    ------
    RuntimeError
        When Ruff fails, emits an invalid finding list, or reports an exit
        status inconsistent with that list.
    ValueError
        When the repository is absent, no scopes are selected, a selected
        scope is not a directory, or the entire inventory has no Python source.
        An existing non-Python scope is valid; a missing scope is not.

    """
    if not repo.is_dir():
        raise ValueError(f"repository is not a directory: {repo}")
    if not scopes:
        raise ValueError("select at least one documentation scope")
    for scope in scopes:
        directory = repo / scope
        if not directory.is_dir():
            raise ValueError(f"required documentation scope is not a directory: {scope}")
    if not any(path.is_file() for scope in scopes for path in (repo / scope).rglob("*.py")):
        raise ValueError("documentation inventory contains no Python files")
    completed = subprocess.run(  # noqa: S603
        [sys.executable, *SCAN, *scopes],
        capture_output=True,
        text=True,
        cwd=repo,
        check=False,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(f"ruff failed: {completed.stderr.strip()}")
    try:
        decoded: object = json.loads(completed.stdout)
    except ValueError as error:
        raise RuntimeError(f"ruff emitted invalid JSON: {completed.stderr.strip()}") from error
    if not isinstance(decoded, list) or any(not isinstance(item, dict) for item in decoded):
        raise RuntimeError("ruff did not emit a list of finding objects")
    findings: list[dict[str, object]] = decoded
    if bool(findings) != (completed.returncode == 1):
        raise RuntimeError("ruff exit status disagrees with its finding list")
    return findings


def unexempt(findings: Sequence[dict[str, object]], repo: Path | None = None) -> list[str]:
    """Return one line per finding that no exemption covers.

    ``repo`` anchors the reported paths. Splitting the absolute path on a
    directory name would silently stop matching the exemption keys the moment
    the checkout is renamed or the gate is exercised from a temporary tree.
    """
    root = (repo or Path(__file__).resolve().parent.parent).resolve()
    out: list[str] = []
    for finding in findings:
        filename = str(finding.get("filename", ""))
        try:
            relative = str(Path(filename).resolve().relative_to(root))
        except ValueError:
            relative = filename
        if relative in EXEMPT:
            continue
        location = finding.get("location") or {}
        row = location.get("row", "?") if isinstance(location, dict) else "?"
        code = finding.get("code") or "D"
        out.append(f"{relative}:{row}: {code} {finding.get('message', '')}")
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the gate and return a process exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    arguments = parser.parse_args(argv)

    try:
        findings = scan(arguments.repo, ENFORCED_SCOPES)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"documentation scan could not complete: {error}", file=sys.stderr)
        return 2
    offenders = unexempt(findings, arguments.repo)
    if offenders:
        print(f"{len(offenders)} documentation finding(s) in a scope required to hold at zero:")
        for line in offenders:
            print(f"    {line}")
        print("\nEvery non-test scope stands at zero. Document the new code rather than")
        print("re-opening the scope; an exemption needs a reason recorded in EXEMPT.")
        return 1
    print(
        f"documentation scopes clean: {len(ENFORCED_SCOPES)} scopes inspected, "
        f"{len(EXEMPT)} recorded exemption(s)"
    )
    for scope in ENFORCED_SCOPES:
        count = sum(path.is_file() for path in (arguments.repo / scope).rglob("*.py"))
        print(f"    {scope}: {count} Python file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
