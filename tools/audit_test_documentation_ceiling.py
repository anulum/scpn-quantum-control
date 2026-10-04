# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — test documentation ceiling gate
"""Hold the documentation debt of the test suites under a ceiling that only falls.

The default Ruff profile exempts ``tests/**`` from every documentation rule, and
the documentation scope gate deliberately leaves the test suites out, so nothing
rejected an undocumented test. This gate measures the two test scopes with the
same isolated profile as the scope gate and compares every file with a recorded
per-file ceiling.

Three things fail it. A test file that is absent from the ceiling and has
findings: new tests arrive documented. A file whose finding count rose above its
ceiling: existing debt does not grow. A ceiling that is higher than the
measurement, or names a file that is now clean or gone: the recorded figure must
be lowered so it stays a measurement rather than an allowance.

With ``--changed-against`` every test file that differs from the named revision
must have no finding at all, whatever its ceiling: a file that is touched is
brought to the full standard in the same change.

The counts depend on the Ruff release, because the preview rule set changes
between releases. The ceiling records the release it was measured with and the
gate refuses to compare across releases; ``--rebaseline`` records a new
measurement only in that situation and keeps the previous totals as history.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.audit_documentation_scopes import scan

TEST_SCOPES: Final[tuple[str, ...]] = ("tests", "oscillatools/tests")
DEFAULT_CEILING: Final[Path] = Path("tools/test_documentation_ceiling.json")
SCHEMA: Final[str] = "test_documentation_ceiling_v1"


@dataclass(frozen=True)
class Ceiling:
    """Recorded per-file documentation finding counts and their provenance."""

    ruff_version: str
    measured_commit: str
    files: dict[str, int]
    history: tuple[dict[str, object], ...]

    @property
    def total(self) -> int:
        """Sum of the recorded per-file ceilings."""
        return sum(self.files.values())


def _in_scope(path: str) -> bool:
    """Return whether ``path`` is a Python file inside one of the test scopes."""
    return path.endswith(".py") and any(path.startswith(scope + "/") for scope in TEST_SCOPES)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while refusing a repeated key."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key in ceiling: {key}")
        result[key] = value
    return result


def load_ceiling(path: Path) -> Ceiling:
    """Read and validate a recorded ceiling.

    Parameters
    ----------
    path
        JSON file with schema ``test_documentation_ceiling_v1``.

    Returns
    -------
    Ceiling
        The recorded counts, Ruff release, source commit and history.

    Raises
    ------
    ValueError
        If the schema, the provenance fields, a path or a count is invalid, or
        a key is repeated.

    """
    raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise ValueError("unsupported test documentation ceiling schema")
    ruff_version = raw.get("ruff_version")
    measured_commit = raw.get("measured_commit")
    if not isinstance(ruff_version, str) or not ruff_version.strip():
        raise ValueError("ceiling must record the Ruff release it was measured with")
    if not isinstance(measured_commit, str) or not measured_commit.strip():
        raise ValueError("ceiling must record the commit it was measured at")
    if raw.get("scopes") != list(TEST_SCOPES):
        raise ValueError("ceiling scopes differ from the gate's test scopes")
    files = raw.get("files")
    if not isinstance(files, dict):
        raise ValueError("ceiling files must be an object")
    counts: dict[str, int] = {}
    for name, count in files.items():
        if not _in_scope(name) or ".." in Path(name).parts:
            raise ValueError(f"ceiling path is outside the test scopes: {name}")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(f"ceiling count must be a positive integer: {name}")
        counts[name] = count
    history = raw.get("history", [])
    if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
        raise ValueError("ceiling history must be a list of objects")
    return Ceiling(
        ruff_version, measured_commit, counts, tuple(cast(list[dict[str, object]], history))
    )


def write_ceiling(path: Path, ceiling: Ceiling) -> None:
    """Write ``ceiling`` as sorted, indented JSON ending in a newline.

    Parameters
    ----------
    path
        Destination file.
    ceiling
        Counts and provenance to record.

    """
    payload = {
        "schema": SCHEMA,
        "ruff_version": ceiling.ruff_version,
        "measured_commit": ceiling.measured_commit,
        "scopes": list(TEST_SCOPES),
        "total_findings": ceiling.total,
        "total_files": len(ceiling.files),
        "files": dict(sorted(ceiling.files.items())),
        "history": list(ceiling.history),
    }
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def _output(repo: Path, arguments: Sequence[str]) -> str:
    """Run a fixed command in ``repo`` and return its standard output.

    Raises
    ------
    ValueError
        If the command cannot start or exits with a failure status.

    """
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argument vectors, no shell
            list(arguments), capture_output=True, text=True, cwd=repo, check=False
        )
    except OSError as error:
        raise ValueError(f"cannot run {arguments[0]}: {error}") from error
    if completed.returncode != 0:
        raise ValueError(f"{' '.join(arguments)} failed: {completed.stderr.strip()}")
    return completed.stdout


def ruff_version(repo: Path) -> str:
    """Return the version line of the Ruff release the gate measures with."""
    return _output(repo, [sys.executable, "-m", "ruff", "--version"]).strip()


def measure(repo: Path) -> dict[str, int]:
    """Count documentation findings per test file in the working tree.

    Parameters
    ----------
    repo
        Repository root.

    Returns
    -------
    dict[str, int]
        Finding count for every test file that has at least one finding, keyed
        by its path relative to ``repo``.

    """
    root = repo.resolve()
    counts: Counter[str] = Counter()
    for finding in scan(repo, TEST_SCOPES):
        counts[Path(str(finding["filename"])).resolve().relative_to(root).as_posix()] += 1
    return dict(counts)


def changed_test_files(repo: Path, revision: str) -> list[str]:
    """List the test files that differ between ``revision`` and the working tree.

    Parameters
    ----------
    repo
        Repository root.
    revision
        Commit to compare with; it must resolve in the repository.

    Returns
    -------
    list[str]
        Added, modified or renamed Python test files, sorted.

    Raises
    ------
    ValueError
        If ``revision`` does not resolve or Git cannot produce the comparison.

    """
    names = _output(
        repo,
        ["git", "diff", "--name-only", "--diff-filter=AMR", revision, "--", *TEST_SCOPES],
    )
    return sorted(name for name in names.splitlines() if _in_scope(name))


def compare(measured: dict[str, int], ceiling: Ceiling, current_ruff: str) -> list[str]:
    """Return every way the measurement departs from the recorded ceiling.

    Parameters
    ----------
    measured
        Finding counts of the working tree.
    ceiling
        Recorded ceiling.
    current_ruff
        Version line of the running Ruff release.

    Returns
    -------
    list[str]
        One message per problem; empty when the measurement equals the ceiling.

    """
    if current_ruff != ceiling.ruff_version:
        return [
            f"ceiling was measured with {ceiling.ruff_version}, this is {current_ruff}; "
            "counts are not comparable across releases, record a new measurement with --rebaseline"
        ]
    errors: list[str] = []
    for name, count in sorted(measured.items()):
        recorded = ceiling.files.get(name)
        if recorded is None:
            errors.append(f"undocumented test file outside the ceiling: {name} ({count} findings)")
        elif count > recorded:
            errors.append(f"documentation findings grew: {name}: {recorded} -> {count}")
        elif count < recorded:
            errors.append(
                f"ceiling is above the measurement: {name}: {recorded} -> {count}; lower it with --lower"
            )
    for name in sorted(ceiling.files.keys() - measured.keys()):
        errors.append(f"ceiling row has no findings left: {name}; remove it with --lower")
    return errors


def check_changed(measured: dict[str, int], changed: Sequence[str]) -> list[str]:
    """Require every changed test file to be free of documentation findings.

    Parameters
    ----------
    measured
        Finding counts of the working tree.
    changed
        Test files that differ from the comparison revision.

    Returns
    -------
    list[str]
        One message per changed file that still has findings.

    """
    return [
        f"changed test file must be fully documented: {name} ({measured[name]} findings)"
        for name in changed
        if name in measured
    ]


def lower(measured: dict[str, int], ceiling: Ceiling, commit: str) -> Ceiling:
    """Return the ceiling reduced to the measurement.

    Parameters
    ----------
    measured
        Finding counts of the working tree.
    ceiling
        Recorded ceiling, measured with the running Ruff release.
    commit
        Commit the new figures describe.

    Returns
    -------
    Ceiling
        The measurement as the new ceiling, with unchanged history.

    Raises
    ------
    ValueError
        If any file is new or above its ceiling: lowering never admits debt.

    """
    raised = sorted(name for name, count in measured.items() if count > ceiling.files.get(name, 0))
    if raised:
        raise ValueError(f"cannot lower the ceiling while findings grew: {', '.join(raised)}")
    return Ceiling(ceiling.ruff_version, commit, dict(measured), ceiling.history)


def rebaseline(
    measured: dict[str, int], ceiling: Ceiling, current_ruff: str, commit: str
) -> Ceiling:
    """Record a new measurement after the Ruff release changed.

    Parameters
    ----------
    measured
        Finding counts of the working tree under the running release.
    ceiling
        Ceiling recorded with the previous release.
    current_ruff
        Version line of the running Ruff release.
    commit
        Commit the new figures describe.

    Returns
    -------
    Ceiling
        The new measurement, with the previous totals appended to its history.

    Raises
    ------
    ValueError
        If the release did not change: under one release the ceiling only falls.

    """
    if current_ruff == ceiling.ruff_version:
        raise ValueError("the Ruff release did not change; the ceiling can only be lowered")
    previous: dict[str, object] = {
        "ruff_version": ceiling.ruff_version,
        "measured_commit": ceiling.measured_commit,
        "total_findings": ceiling.total,
        "total_files": len(ceiling.files),
    }
    return Ceiling(current_ruff, commit, dict(measured), (*ceiling.history, previous))


def main(argv: Sequence[str] | None = None) -> int:
    """Compare the test suites with their ceiling, or update the ceiling.

    Parameters
    ----------
    argv
        Command-line arguments; ``None`` reads the process arguments.

    Returns
    -------
    int
        Zero when the measurement equals the ceiling (or the requested update
        was written), one otherwise.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ceiling", type=Path, default=DEFAULT_CEILING)
    parser.add_argument(
        "--changed-against",
        metavar="REVISION",
        help="also require every test file that differs from REVISION to have no finding",
    )
    update = parser.add_mutually_exclusive_group()
    update.add_argument(
        "--lower", action="store_true", help="reduce the ceiling to the measurement"
    )
    update.add_argument(
        "--rebaseline",
        action="store_true",
        help="record a new measurement after the Ruff release changed",
    )
    args = parser.parse_args(argv)
    repo: Path = args.repo
    path = repo / args.ceiling
    try:
        ceiling = load_ceiling(path)
        current_ruff = ruff_version(repo)
        measured = measure(repo)
        if args.lower or args.rebaseline:
            commit = _output(repo, ["git", "rev-parse", "HEAD"]).strip()
            if args.rebaseline:
                updated = rebaseline(measured, ceiling, current_ruff, commit)
            elif current_ruff != ceiling.ruff_version:
                raise ValueError("the Ruff release changed; use --rebaseline, not --lower")
            else:
                updated = lower(measured, ceiling, commit)
            write_ceiling(path, updated)
            print(
                f"Test documentation ceiling written: {updated.total} findings in {len(updated.files)} files"
            )
            return 0
        errors = compare(measured, ceiling, current_ruff)
        if args.changed_against is not None:
            errors.extend(check_changed(measured, changed_test_files(repo, args.changed_against)))
    except (OSError, ValueError, RuntimeError) as error:
        print(f"test documentation ceiling failed: {error}", file=sys.stderr)
        return 1
    for message in errors:
        print(message, file=sys.stderr)
    print(
        f"Test documentation ceiling: {sum(measured.values())} findings in {len(measured)} files; "
        f"ceiling {ceiling.total} in {len(ceiling.files)}; {len(errors)} problems"
    )
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
