# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — web source lint and format ceiling gate
"""Hold the lint and format debt of the web workspace under a ceiling that only falls.

The web workspace is type-checked, tested and built, but no gate linted its
TypeScript, stylesheets and pages or checked their layout, and nothing read
the script and stylesheet of the documentation site. The sources were never
formatted by a tool, so a plain "everything must pass" gate would demand one
mass rewrite. This gate measures every tracked web source with the pinned
Biome release and compares each file with a recorded ceiling instead.

Three things fail it. A source that is absent from the ceiling and has a lint
finding or is not formatted: new files arrive clean. A file whose finding
count rose above its ceiling, or a formatted file that lost its layout:
existing debt does not grow. A ceiling that is higher than the measurement, or
names a file that is now clean, formatted or gone: the recorded figure must be
lowered so it stays a measurement and not an allowance.

With ``--changed-against`` every web source that differs from the named
revision must have no finding and be formatted, whatever its ceiling: a file
that is touched is brought to the full standard in the same change.

Findings depend on the Biome release. The release is pinned as an exact
development dependency of the web workspace; the gate uses the executable that
installation puts into the workspace, refuses any other release, and refuses
to compare with a ceiling measured by another release. ``--rebaseline`` records
a new measurement only in that situation and keeps the previous totals as
history. A diagnostic that is neither a lint finding nor a format difference,
such as a parse error, is a failure of the gate, never a counted finding.

Pages are linted only: the pinned release does not format HTML unless an
experimental formatter is switched on, and the gate does not switch it on, so
the layout of a page is not checked.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, cast

WEB_ROOT: Final[str] = "studio-web"
SOURCE_ROOTS: Final[tuple[str, ...]] = (WEB_ROOT, "docs/css", "docs/js")
SOURCE_SUFFIXES: Final[frozenset[str]] = frozenset({".ts", ".tsx", ".css", ".js", ".html"})
DISTRIBUTION: Final[str] = "@biomejs/biome"
DEFAULT_CEILING: Final[Path] = Path("tools/web_source_ceiling.json")
SCHEMA: Final[str] = "web_source_ceiling_v1"
_EXACT_RELEASE: Final[re.Pattern[str]] = re.compile(r"\d+\.\d+\.\d+")
_VERSION_LINE: Final[re.Pattern[str]] = re.compile(r"^Version: (\d+\.\d+\.\d+)$", re.MULTILINE)


@dataclass(frozen=True)
class Measurement:
    """Lint finding counts and unformatted files of the working tree."""

    lint: dict[str, int]
    unformatted: tuple[str, ...]


@dataclass(frozen=True)
class Ceiling:
    """Recorded per-file lint counts, unformatted files and their provenance."""

    release: str
    measured_commit: str
    lint: dict[str, int]
    unformatted: tuple[str, ...]
    history: tuple[dict[str, object], ...]

    @property
    def total(self) -> int:
        """Sum of the recorded per-file lint ceilings."""
        return sum(self.lint.values())


def _in_scope(path: str) -> bool:
    """Return whether ``path`` is a web source the gate measures."""
    pure = PurePosixPath(path)
    return (
        ".." not in pure.parts
        and pure.suffix in SOURCE_SUFFIXES
        and any(pure.is_relative_to(root) for root in SOURCE_ROOTS)
    )


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
        JSON file with schema ``web_source_ceiling_v1``.

    Returns
    -------
    Ceiling
        The recorded counts, unformatted files, Biome release, commit and history.

    Raises
    ------
    ValueError
        If the schema, a provenance field, a path, a count or the order of the
        unformatted files is invalid, or a key is repeated.

    """
    raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise ValueError("unsupported web source ceiling schema")
    release = raw.get("release")
    measured_commit = raw.get("measured_commit")
    if not isinstance(release, str) or _EXACT_RELEASE.fullmatch(release) is None:
        raise ValueError("ceiling must record the Biome release it was measured with")
    if not isinstance(measured_commit, str) or not measured_commit.strip():
        raise ValueError("ceiling must record the commit it was measured at")
    lint = raw.get("lint")
    if not isinstance(lint, dict):
        raise ValueError("ceiling lint counts must be an object")
    counts: dict[str, int] = {}
    for name, count in lint.items():
        if not _in_scope(name):
            raise ValueError(f"ceiling path is not a web source: {name}")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(f"ceiling count must be a positive integer: {name}")
        counts[name] = count
    unformatted = raw.get("unformatted")
    if not isinstance(unformatted, list) or any(not isinstance(name, str) for name in unformatted):
        raise ValueError("ceiling unformatted files must be a list of paths")
    names = cast(list[str], unformatted)
    for name in names:
        if not _in_scope(name):
            raise ValueError(f"ceiling path is not a web source: {name}")
    if names != sorted(set(names)):
        raise ValueError("ceiling unformatted files must be sorted and unique")
    history = raw.get("history", [])
    if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
        raise ValueError("ceiling history must be a list of objects")
    return Ceiling(
        release,
        measured_commit,
        counts,
        tuple(names),
        tuple(cast(list[dict[str, object]], history)),
    )


def write_ceiling(path: Path, ceiling: Ceiling) -> None:
    """Write ``ceiling`` as sorted, indented JSON ending in a newline.

    Parameters
    ----------
    path
        Destination file.
    ceiling
        Counts, unformatted files and provenance to record.

    """
    payload = {
        "schema": SCHEMA,
        "release": ceiling.release,
        "measured_commit": ceiling.measured_commit,
        "total_lint_findings": ceiling.total,
        "total_lint_files": len(ceiling.lint),
        "total_unformatted_files": len(ceiling.unformatted),
        "lint": dict(sorted(ceiling.lint.items())),
        "unformatted": sorted(ceiling.unformatted),
        "history": list(ceiling.history),
    }
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def _run(directory: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run a fixed command in ``directory`` and return the finished process.

    Raises
    ------
    ValueError
        If the command cannot be started.

    """
    try:
        return subprocess.run(  # noqa: S603 - fixed argument vectors, no shell
            list(arguments), capture_output=True, text=True, cwd=directory, check=False
        )
    except OSError as error:
        raise ValueError(f"cannot run {arguments[0]}: {error}") from error


def _git(repo: Path, arguments: Sequence[str]) -> str:
    """Return the standard output of a Git command that must succeed.

    Raises
    ------
    ValueError
        If Git cannot start or exits with a failure status.

    """
    completed = _run(repo, ["git", *arguments])
    if completed.returncode != 0:
        raise ValueError(f"git {' '.join(arguments)} failed: {completed.stderr.strip()}")
    return completed.stdout


def pinned_release(repo: Path) -> str:
    """Return the Biome release the web workspace pins.

    Raises
    ------
    ValueError
        If the workspace manifest is unreadable or does not pin one exact release.

    """
    manifest = repo / WEB_ROOT / "package.json"
    try:
        package = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read the web workspace manifest: {error}") from error
    pinned = package.get("devDependencies", {}).get(DISTRIBUTION)
    if not isinstance(pinned, str) or _EXACT_RELEASE.fullmatch(pinned) is None:
        raise ValueError(f"{WEB_ROOT}/package.json does not pin one exact {DISTRIBUTION} release")
    return pinned


def workspace_executable(repo: Path) -> Path:
    """Return the Biome executable installed in the web workspace.

    Raises
    ------
    ValueError
        If the workspace is not installed.

    """
    executable = repo / WEB_ROOT / "node_modules" / ".bin" / "biome"
    if not executable.is_file():
        raise ValueError(
            f"{DISTRIBUTION} is not installed in {WEB_ROOT}; run pnpm install --frozen-lockfile there"
        )
    return executable


def installed_release(repo: Path, executable: Path) -> str:
    """Return the release the installed Biome executable reports.

    Raises
    ------
    ValueError
        If the executable cannot run or prints no release.

    """
    completed = _run(repo / WEB_ROOT, [str(executable), "--version"])
    match = _VERSION_LINE.search(completed.stdout)
    if completed.returncode != 0 or match is None:
        raise ValueError(f"{executable.name} --version reported no release")
    return match.group(1)


def tracked_sources(repo: Path) -> list[str]:
    """Return the tracked web sources in sorted order, relative to ``repo``.

    Raises
    ------
    ValueError
        If Git cannot list the tracked files.

    """
    names = _git(repo, ["ls-files", "-z", "--", *SOURCE_ROOTS]).split("\0")
    return sorted(name for name in names if name and _in_scope(name))


def _diagnostics(
    repo: Path, executable: Path, command: str, sources: Sequence[str]
) -> list[tuple[str, str]]:
    """Run one Biome command over ``sources`` and return ``(category, path)`` pairs.

    Biome runs in the web workspace, where its configuration lives. It reports
    a file inside the workspace by its path relative to the workspace and a file
    outside it, such as the documentation site's script, by its absolute path.

    Raises
    ------
    ValueError
        If the report is not the expected JSON document, omits diagnostics, or
        names a file outside ``sources``.

    """
    root = repo.resolve()
    relative = [os.path.relpath(root / name, root / WEB_ROOT) for name in sources]
    completed = _run(
        repo / WEB_ROOT,
        [str(executable), command, "--reporter=json", "--max-diagnostics=none", "--", *relative],
    )
    try:
        report = json.loads(completed.stdout)
        summary = report["summary"]
        rows = report["diagnostics"]
        if summary["diagnosticsNotPrinted"] != 0:
            raise ValueError("the report omits diagnostics")
        pairs = [(str(row["category"]), str(row["location"]["path"])) for row in rows]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        detail = " ".join(completed.stderr.split())[:300]
        raise ValueError(f"biome {command} gave no usable report: {error} {detail}") from error
    known = set(sources)
    located: list[tuple[str, str]] = []
    for category, path in pairs:
        name = PurePosixPath(os.path.relpath(root / WEB_ROOT / path, root)).as_posix()
        if name not in known:
            raise ValueError(f"biome {command} reported {category} outside the sources: {path}")
        located.append((category, name))
    return located


def measure(repo: Path, executable: Path) -> Measurement:
    """Count lint findings per web source and list the unformatted sources.

    Parameters
    ----------
    repo
        Repository root.
    executable
        Biome executable of the pinned release.

    Returns
    -------
    Measurement
        Lint finding count for every source with at least one finding, and the
        sorted sources whose layout the formatter would change.

    Raises
    ------
    ValueError
        If a report is unusable, or a diagnostic is neither a lint finding nor
        a format difference (a parse error, for instance).

    """
    sources = tracked_sources(repo)
    if not sources:
        return Measurement({}, ())
    counts: Counter[str] = Counter()
    for category, path in _diagnostics(repo, executable, "lint", sources):
        if not category.startswith("lint/"):
            raise ValueError(f"biome lint reported {category} in {path}; fix it before the gate")
        counts[path] += 1
    unformatted: set[str] = set()
    for category, path in _diagnostics(repo, executable, "format", sources):
        if category != "format":
            raise ValueError(f"biome format reported {category} in {path}; fix it before the gate")
        unformatted.add(path)
    return Measurement(dict(counts), tuple(sorted(unformatted)))


def changed_sources(repo: Path, revision: str) -> list[str]:
    """List the web sources that differ between ``revision`` and the working tree.

    Parameters
    ----------
    repo
        Repository root.
    revision
        Commit to compare with; it must resolve in the repository.

    Returns
    -------
    list[str]
        Added, modified or renamed web sources, sorted.

    Raises
    ------
    ValueError
        If ``revision`` does not resolve or Git cannot produce the comparison.

    """
    names = _git(repo, ["diff", "--name-only", "--diff-filter=AMR", revision, "--", *SOURCE_ROOTS])
    return sorted(name for name in names.splitlines() if _in_scope(name))


def compare(measured: Measurement, ceiling: Ceiling, release: str) -> list[str]:
    """Return every way the measurement departs from the recorded ceiling.

    Parameters
    ----------
    measured
        Lint counts and unformatted sources of the working tree.
    ceiling
        Recorded ceiling.
    release
        Release of the running Biome executable.

    Returns
    -------
    list[str]
        One message per problem; empty when the measurement equals the ceiling.

    """
    if release != ceiling.release:
        return [
            f"ceiling was measured with {DISTRIBUTION} {ceiling.release}, this is {release}; "
            "findings are not comparable across releases, record a new measurement with --rebaseline"
        ]
    errors: list[str] = []
    for name, count in sorted(measured.lint.items()):
        recorded = ceiling.lint.get(name)
        if recorded is None:
            errors.append(f"web source outside the ceiling has lint findings: {name} ({count})")
        elif count > recorded:
            errors.append(f"lint findings grew: {name}: {recorded} -> {count}")
        elif count < recorded:
            errors.append(
                f"ceiling is above the measurement: {name}: {recorded} -> {count}; lower it with --lower"
            )
    for name in sorted(ceiling.lint.keys() - measured.lint.keys()):
        errors.append(f"ceiling row has no lint findings left: {name}; remove it with --lower")
    recorded_unformatted = set(ceiling.unformatted)
    for name in measured.unformatted:
        if name not in recorded_unformatted:
            errors.append(f"not formatted: {name}; run biome format --write on it")
    for name in sorted(recorded_unformatted - set(measured.unformatted)):
        errors.append(f"ceiling lists a formatted or removed file: {name}; remove it with --lower")
    return errors


def check_changed(measured: Measurement, changed: Sequence[str]) -> list[str]:
    """Require every changed web source to be free of findings and formatted.

    Parameters
    ----------
    measured
        Lint counts and unformatted sources of the working tree.
    changed
        Web sources that differ from the comparison revision.

    Returns
    -------
    list[str]
        One message per changed source and remaining defect.

    """
    unformatted = set(measured.unformatted)
    errors: list[str] = []
    for name in changed:
        if name in measured.lint:
            errors.append(
                f"changed web source must have no lint finding: {name} ({measured.lint[name]})"
            )
        if name in unformatted:
            errors.append(f"changed web source must be formatted: {name}")
    return errors


def lower(measured: Measurement, ceiling: Ceiling, commit: str) -> Ceiling:
    """Return the ceiling reduced to the measurement.

    Parameters
    ----------
    measured
        Lint counts and unformatted sources of the working tree.
    ceiling
        Recorded ceiling, measured with the running Biome release.
    commit
        Commit the new figures describe.

    Returns
    -------
    Ceiling
        The measurement as the new ceiling, with unchanged history.

    Raises
    ------
    ValueError
        If any file is new, above its ceiling or newly unformatted: lowering
        never admits debt.

    """
    raised = sorted(
        {name for name, count in measured.lint.items() if count > ceiling.lint.get(name, 0)}
        | (set(measured.unformatted) - set(ceiling.unformatted))
    )
    if raised:
        raise ValueError(f"cannot lower the ceiling while debt grew: {', '.join(raised)}")
    return Ceiling(
        ceiling.release, commit, dict(measured.lint), measured.unformatted, ceiling.history
    )


def rebaseline(measured: Measurement, ceiling: Ceiling, release: str, commit: str) -> Ceiling:
    """Record a new measurement after the Biome release changed.

    Parameters
    ----------
    measured
        Lint counts and unformatted sources under the running release.
    ceiling
        Ceiling recorded with the previous release.
    release
        Release of the running Biome executable.
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
    if release == ceiling.release:
        raise ValueError("the Biome release did not change; the ceiling can only be lowered")
    previous: dict[str, object] = {
        "release": ceiling.release,
        "measured_commit": ceiling.measured_commit,
        "total_lint_findings": ceiling.total,
        "total_lint_files": len(ceiling.lint),
        "total_unformatted_files": len(ceiling.unformatted),
    }
    return Ceiling(
        release, commit, dict(measured.lint), measured.unformatted, (*ceiling.history, previous)
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Compare the web sources with their ceiling, or update the ceiling.

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
        help="also require every web source that differs from REVISION to be clean and formatted",
    )
    update = parser.add_mutually_exclusive_group()
    update.add_argument(
        "--lower", action="store_true", help="reduce the ceiling to the measurement"
    )
    update.add_argument(
        "--rebaseline",
        action="store_true",
        help="record a new measurement after the Biome release changed",
    )
    args = parser.parse_args(argv)
    repo: Path = args.repo
    path = repo / args.ceiling
    try:
        ceiling = load_ceiling(path)
        executable = workspace_executable(repo)
        release = installed_release(repo, executable)
        pinned = pinned_release(repo)
        if release != pinned:
            raise ValueError(
                f"{DISTRIBUTION} {release} is not the pinned release {pinned}; reinstall {WEB_ROOT}"
            )
        measured = measure(repo, executable)
        if args.lower or args.rebaseline:
            commit = _git(repo, ["rev-parse", "HEAD"]).strip()
            if args.rebaseline:
                updated = rebaseline(measured, ceiling, release, commit)
            elif release != ceiling.release:
                raise ValueError("the Biome release changed; use --rebaseline, not --lower")
            else:
                updated = lower(measured, ceiling, commit)
            write_ceiling(path, updated)
            print(
                f"Web source ceiling written: {updated.total} lint findings in "
                f"{len(updated.lint)} files; {len(updated.unformatted)} unformatted files"
            )
            return 0
        errors = compare(measured, ceiling, release)
        if args.changed_against is not None:
            errors.extend(check_changed(measured, changed_sources(repo, args.changed_against)))
    except (OSError, ValueError) as error:
        print(f"web source ceiling failed: {error}", file=sys.stderr)
        return 1
    for message in errors:
        print(message, file=sys.stderr)
    print(
        f"Web source ceiling: {sum(measured.lint.values())} lint findings in "
        f"{len(measured.lint)} files, {len(measured.unformatted)} unformatted files; ceiling "
        f"{ceiling.total} in {len(ceiling.lint)}, {len(ceiling.unformatted)} unformatted; "
        f"{len(errors)} problems"
    )
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
