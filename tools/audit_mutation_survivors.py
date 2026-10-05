# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — mutation survivor ceiling gate
"""Run the mutation targets and hold their survivors under a ceiling that only falls.

Mutation testing changes one operator or constant of a module at a time and
runs the module's tests; a mutant that the tests do not notice survives. The
weekly job ran the tool with its failure discarded, and the runner script was
not executable, so the job could report success without testing a mutant.

This gate runs each recorded target with the pinned mutmut release and
compares the outcome with ``tools/mutation_survivor_ceiling.json``. It fails
when the tool cannot run, when any mutant is left untested or skipped, when
the number of generated mutants differs from the recorded one, when the
survivor or timeout count rose above its ceiling, and when a count fell below
it without the ceiling being lowered. A run that tests nothing therefore
fails instead of looking clean. A mutant that the tests noticed slowly is
reported as suspicious and counts as noticed: that label depends on how busy
the machine was, so it is not held to a ceiling. A timeout depends on the
machine as well: the tool's limit is ten times the duration of the unmutated
tests at the start of the run. A timed-out mutant is therefore applied again
and its tests run once under a fixed limit; it is then counted as survived,
noticed, or, if it exceeds that limit too, timed out.

mutmut rewrites the target module on disk while it works. The gate never does
that in the working tree: it exports the committed tree of ``HEAD`` into a
directory of its own and runs there, so a local run tests the committed
sources and cannot leave a mutated file behind.

The mutants depend on the target's source and on the mutmut release. The
ceiling records the digest of the source and the release; when either
changes the counts are not comparable and the gate asks for ``--rebaseline``,
which records a new measurement and keeps the previous one as history. The
gate counts survivors; it does not classify them. A recorded survivor may be
an equivalent mutant or a real gap in the tests.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, cast

DEFAULT_POLICY: Final[Path] = Path("tools/mutation_survivor_ceiling.json")
PIN_FILE: Final[Path] = Path("requirements-ci-mutation.txt")
DISTRIBUTION: Final[str] = "mutmut"
SCHEMA: Final[str] = "mutation_survivor_ceiling_v1"
CEILINGS: Final[tuple[str, ...]] = ("survived", "timeout")
STATUSES: Final[tuple[str, ...]] = ("killed", *CEILINGS, "suspicious", "skipped", "untested")
RETEST_LIMIT_SECONDS: Final[float] = 300.0
CACHE_FILE: Final[str] = ".mutmut-cache"
REPORT_SCHEMA: Final[str] = "mutation_survivor_report_v1"
_PIN: Final[re.Pattern[str]] = re.compile(r"^mutmut==(\d+\.\d+\.\d+)", re.MULTILINE)
_DIGEST: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class Target:
    """One mutation target with the recorded outcome of its last measurement."""

    name: str
    module: str
    runner: str
    source_sha256: str
    mutants: int
    survived: int
    timeout: int


@dataclass(frozen=True)
class Ceiling:
    """Recorded targets, the release they were measured with and earlier measurements."""

    release: str
    targets: tuple[Target, ...]
    history: tuple[dict[str, object], ...]


def _relative(row: dict[str, object], field: str) -> str:
    """Return a required repository-relative path field of a target row.

    Raises
    ------
    ValueError
        If the field is not a non-empty relative path inside the repository.

    """
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"mutation target needs a non-empty {field}: {row}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"mutation target {field} must stay inside the repository: {value}")
    return value


def _count(row: dict[str, object], field: str) -> int:
    """Return a required non-negative integer field of a target row.

    Raises
    ------
    ValueError
        If the field is absent, not an integer or negative.

    """
    value = row.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"mutation target {field} must be a non-negative integer: {row}")
    return value


def load_ceiling(path: Path) -> Ceiling:
    """Read and validate the recorded mutation ceiling.

    Parameters
    ----------
    path
        JSON file with schema ``mutation_survivor_ceiling_v1``.

    Returns
    -------
    Ceiling
        The recorded release, targets and history.

    Raises
    ------
    ValueError
        If the schema, the release, a target field or the history is invalid,
        a name is repeated, or the counts of a target exceed its mutants.

    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise ValueError("unsupported mutation survivor ceiling schema")
    release = raw.get("release")
    if not isinstance(release, str) or re.fullmatch(r"\d+\.\d+\.\d+", release) is None:
        raise ValueError("ceiling must record the mutmut release it was measured with")
    rows = raw.get("targets")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("ceiling targets must be a list of objects")
    targets: list[Target] = []
    for row in cast(list[dict[str, object]], rows):
        name = row.get("name")
        digest = row.get("source_sha256")
        if not isinstance(name, str) or re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) is None:
            raise ValueError(f"mutation target needs a lower-case name: {row}")
        if any(target.name == name for target in targets):
            raise ValueError(f"mutation target is recorded twice: {name}")
        if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
            raise ValueError(f"mutation target needs the digest of its source: {name}")
        target = Target(
            name,
            _relative(row, "module"),
            _relative(row, "runner"),
            digest,
            _count(row, "mutants"),
            _count(row, "survived"),
            _count(row, "timeout"),
        )
        if target.survived + target.timeout > target.mutants:
            raise ValueError(f"mutation target counts exceed its mutants: {name}")
        targets.append(target)
    history = raw.get("history", [])
    if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
        raise ValueError("ceiling history must be a list of objects")
    return Ceiling(release, tuple(targets), tuple(cast(list[dict[str, object]], history)))


def write_ceiling(path: Path, ceiling: Ceiling) -> None:
    """Write ``ceiling`` as indented JSON ending in a newline.

    Parameters
    ----------
    path
        Destination file.
    ceiling
        Release, targets and history to record.

    """
    payload = {
        "schema": SCHEMA,
        "release": ceiling.release,
        "targets": [
            {
                "name": target.name,
                "module": target.module,
                "runner": target.runner,
                "source_sha256": target.source_sha256,
                "mutants": target.mutants,
                "survived": target.survived,
                "timeout": target.timeout,
            }
            for target in ceiling.targets
        ],
        "history": list(ceiling.history),
    }
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def pinned_release(repo: Path) -> str:
    """Return the mutmut release the repository pins.

    Raises
    ------
    ValueError
        If the pin file is unreadable or has no pin for mutmut.

    """
    try:
        text = (repo / PIN_FILE).read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read the mutation tool pin: {error}") from error
    match = _PIN.search(text)
    if match is None:
        raise ValueError(f"{PIN_FILE} does not pin {DISTRIBUTION}")
    return match.group(1)


def installed_release(distribution: str = DISTRIBUTION) -> str:
    """Return the release of the mutation tool installed beside this interpreter.

    Raises
    ------
    ValueError
        If the distribution is not installed.

    """
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError(f"{distribution} is not installed; install {PIN_FILE}") from error


def _run(
    directory: Path, arguments: Sequence[str], environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess[bytes]:
    """Run a fixed command in ``directory`` and return the finished process.

    Raises
    ------
    ValueError
        If the command cannot be started.

    """
    try:
        return subprocess.run(  # noqa: S603 - fixed argument vectors, no shell
            list(arguments), capture_output=True, cwd=directory, env=environment, check=False
        )
    except OSError as error:
        raise ValueError(f"cannot run {arguments[0]}: {error}") from error


def _git(repo: Path, arguments: Sequence[str]) -> bytes:
    """Return the standard output of a Git command that must succeed.

    Raises
    ------
    ValueError
        If Git cannot start or exits with a failure status.

    """
    completed = _run(repo, ["git", *arguments])
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise ValueError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout


def export_head(repo: Path, destination: Path) -> None:
    """Write the committed tree of ``HEAD`` into the empty directory ``destination``.

    Raises
    ------
    ValueError
        If ``destination`` is not an empty directory or Git cannot produce the archive.

    """
    if not destination.is_dir() or any(destination.iterdir()):
        raise ValueError(f"export directory must exist and be empty: {destination}")
    archive = _git(repo, ["archive", "--format=tar", "HEAD"])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(destination, filter="data")


def measure_statuses(
    tree: Path, target: Target, interpreter: Path, retest_limit: float = RETEST_LIMIT_SECONDS
) -> dict[str, list[str]]:
    """Run one mutation target in an exported tree and list its mutants by status.

    Parameters
    ----------
    tree
        Exported repository tree; the target module in it is rewritten during
        the run and restored afterwards by the tool.
    target
        Module and runner to use.
    interpreter
        Python executable that has the pinned mutmut and the test dependencies.
    retest_limit
        Seconds the tests of a timed-out mutant may take when it is tested again.

    Returns
    -------
    dict[str, list[str]]
        The tool's mutant identifiers per status, for every status the tool
        knows, after the timed-out mutants were tested again. The identifiers
        are the tool's own numbering for this module and release.

    Raises
    ------
    ValueError
        If the module or the runner is missing, the runner is not executable,
        the tool reports a fatal error, or a timed-out mutant cannot be applied.

    """
    if not (tree / target.module).is_file():
        raise ValueError(f"mutation target module is missing: {target.module}")
    runner = tree / target.runner
    if not runner.is_file() or not os.access(runner, os.X_OK):
        raise ValueError(f"mutation runner is missing or not executable: {target.runner}")
    environment = dict(os.environ)
    environment.update(
        {
            "VENV_PY": str(interpreter),
            "PYTHONPATH": os.pathsep.join(
                str(tree / part) for part in ("src", "oscillatools/src", ".")
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    # mutmut keeps one result cache per directory and lists every mutant in it,
    # so the cache of an earlier target would be counted with this one.
    (tree / CACHE_FILE).unlink(missing_ok=True)
    tool = [str(interpreter), "-m", DISTRIBUTION]
    command = f"./{target.runner}"
    completed = _run(
        tree,
        [
            *tool,
            "run",
            "--paths-to-mutate",
            target.module,
            "--tests-dir",
            "tests/",
            "--runner",
            command,
            "--no-progress",
            "--CI",
        ],
        environment,
    )
    if completed.returncode != 0:
        output = (completed.stdout + completed.stderr).decode("utf-8", "replace")
        detail = " ".join(output.split())[-400:]
        raise ValueError(f"mutmut could not run {target.name}: {detail}")
    listed: dict[str, list[str]] = {}
    for status in STATUSES:
        listing = _run(tree, [*tool, "result-ids", status], environment)
        if listing.returncode != 0:
            raise ValueError(f"mutmut could not list the {status} mutants of {target.name}")
        listed[status] = listing.stdout.decode("ascii", "replace").split()
    module = tree / target.module
    original = module.read_bytes()
    for mutant in tuple(listed["timeout"]):
        # The tool calls a mutant timed out when its tests ran ten times longer
        # than the unmutated tests did at the start of the run, so a busy
        # machine turns noticed and surviving mutants into timeouts. Each one
        # is applied again and its tests run once under a fixed limit.
        applied = _run(tree, [*tool, "apply", mutant], environment)
        try:
            if applied.returncode != 0 or module.read_bytes() == original:
                raise ValueError(f"mutmut could not apply mutant {mutant} of {target.name}")
            finished = subprocess.run(  # noqa: S603 - the target's recorded runner, no shell
                [command],
                capture_output=True,
                cwd=tree,
                env=environment,
                timeout=retest_limit,
                check=False,
            )
        except subprocess.TimeoutExpired:
            continue
        finally:
            module.write_bytes(original)
        listed["timeout"].remove(mutant)
        listed["survived" if finished.returncode == 0 else "killed"].append(mutant)
    return listed


def compare(target: Target, counts: dict[str, int], digest: str) -> list[str]:
    """Return every way a measured target departs from its recorded ceiling.

    Parameters
    ----------
    target
        Recorded target.
    counts
        Measured number of mutants per status.
    digest
        SHA-256 of the target module that was mutated.

    Returns
    -------
    list[str]
        One message per problem; empty when the measurement equals the record.

    """
    if digest != target.source_sha256:
        return [
            f"{target.name}: the source of {target.module} changed; "
            "record a new measurement with --rebaseline"
        ]
    errors: list[str] = []
    untested = counts["untested"] + counts["skipped"]
    if untested:
        errors.append(f"{target.name}: {untested} mutants were not tested")
    total = sum(counts.values())
    if total != target.mutants:
        errors.append(
            f"{target.name}: {total} mutants were generated, {target.mutants} are recorded"
        )
    for status in CEILINGS:
        recorded = cast(int, getattr(target, status))
        if counts[status] > recorded:
            errors.append(f"{target.name}: {status} mutants grew: {recorded} -> {counts[status]}")
        elif counts[status] < recorded:
            errors.append(
                f"{target.name}: ceiling is above the measurement: {status} "
                f"{recorded} -> {counts[status]}; lower it with --lower"
            )
    return errors


def lowered(target: Target, counts: dict[str, int], digest: str) -> Target:
    """Return ``target`` with its ceilings reduced to a complete measurement.

    Raises
    ------
    ValueError
        If the source changed, the run is incomplete, the mutant count differs
        or any count rose: lowering never admits debt.

    """
    blocked = [
        message for message in compare(target, counts, digest) if "lower it with" not in message
    ]
    if blocked:
        raise ValueError(f"cannot lower the ceiling: {'; '.join(blocked)}")
    return replace(target, survived=counts["survived"], timeout=counts["timeout"])


def rebaselined(target: Target, counts: dict[str, int], digest: str) -> Target:
    """Return ``target`` recorded anew from a complete measurement.

    Raises
    ------
    ValueError
        If any mutant was left untested or skipped.

    """
    untested = counts["untested"] + counts["skipped"]
    if untested:
        raise ValueError(f"cannot record {target.name}: {untested} mutants were not tested")
    return replace(
        target,
        source_sha256=digest,
        mutants=sum(counts.values()),
        survived=counts["survived"],
        timeout=counts["timeout"],
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the mutation targets and compare them with the ceiling, or update it.

    Parameters
    ----------
    argv
        Command-line arguments; ``None`` reads the process arguments.

    Returns
    -------
    int
        Zero when every selected target equals its record (or the requested
        update was written), one otherwise.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--target", action="append", help="target to run; repeat it, or omit it for all targets"
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help="directory for the exported tree (default: the temporary directory)",
    )
    parser.add_argument(
        "--retest-limit",
        type=float,
        default=RETEST_LIMIT_SECONDS,
        metavar="SECONDS",
        help="time the tests of a timed-out mutant may take when it is tested again",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="write the mutant identifiers per target and status to this JSON file",
    )
    update = parser.add_mutually_exclusive_group()
    update.add_argument(
        "--lower", action="store_true", help="reduce the ceilings to the measurement"
    )
    update.add_argument(
        "--rebaseline",
        action="store_true",
        help="record a new measurement after the source or the mutmut release changed",
    )
    args = parser.parse_args(argv)
    repo: Path = args.repo
    path = repo / args.policy
    errors: list[str] = []
    try:
        ceiling = load_ceiling(path)
        release = installed_release()
        if release != pinned_release(repo):
            raise ValueError(
                f"{DISTRIBUTION} {release} is not the pinned release {pinned_release(repo)}; "
                f"install {PIN_FILE}"
            )
        if release != ceiling.release and not args.rebaseline:
            raise ValueError(
                f"ceiling was measured with {DISTRIBUTION} {ceiling.release}, this is {release}; "
                "record a new measurement with --rebaseline"
            )
        names = [target.name for target in ceiling.targets]
        selected = list(dict.fromkeys(args.target or names))
        unknown = sorted(set(selected) - set(names))
        if unknown:
            raise ValueError(f"unknown mutation target: {', '.join(unknown)}")
        updated: list[Target] = []
        report: dict[str, dict[str, list[str]]] = {}
        with tempfile.TemporaryDirectory(dir=args.workspace, prefix="mutation-tree-") as scratch:
            tree = Path(scratch)
            export_head(repo, tree)
            for target in ceiling.targets:
                if target.name not in selected:
                    updated.append(target)
                    continue
                digest = hashlib.sha256((tree / target.module).read_bytes()).hexdigest()
                statuses = measure_statuses(tree, target, Path(sys.executable), args.retest_limit)
                report[target.name] = statuses
                counts = {status: len(identifiers) for status, identifiers in statuses.items()}
                summary = ", ".join(f"{counts[status]} {status}" for status in STATUSES)
                print(f"{target.name}: {sum(counts.values())} mutants: {summary}")
                if args.rebaseline:
                    updated.append(rebaselined(target, counts, digest))
                elif args.lower:
                    updated.append(lowered(target, counts, digest))
                else:
                    errors.extend(compare(target, counts, digest))
        if args.report is not None:
            document = {"schema": REPORT_SCHEMA, "release": release, "targets": report}
            args.report.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
        if args.lower or args.rebaseline:
            history = ceiling.history
            if args.rebaseline:
                changed = [
                    {
                        "release": ceiling.release,
                        "name": old.name,
                        "source_sha256": old.source_sha256,
                        "mutants": old.mutants,
                        "survived": old.survived,
                        "timeout": old.timeout,
                    }
                    for old, new in zip(ceiling.targets, updated, strict=True)
                    if old != new or release != ceiling.release
                ]
                history = (*history, *changed)
            write_ceiling(path, Ceiling(release, tuple(updated), history))
            print(f"Mutation survivor ceiling written for {len(selected)} targets")
            return 0
    except (OSError, ValueError) as error:
        print(f"mutation survivor gate failed: {error}", file=sys.stderr)
        return 1
    for message in errors:
        print(message, file=sys.stderr)
    print(f"Mutation survivors: {len(selected)} targets; {len(errors)} problems")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
