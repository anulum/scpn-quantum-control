# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tracked file size ceiling gate
"""Hold every large tracked file under a recorded ceiling.

Nothing rejected a large file: a dataset, a checkpoint or a serialised circuit
set could be committed and would then stay in the history for good. This gate
sizes every file in the Git index and compares the large ones with a recorded
list.

A file is large when it reaches the recorded threshold. Every large file must
have a row that states its ceiling in whole mebibytes, and the row must be the
smallest ceiling that holds the file. Four things fail the gate. A large file
without a row: large data belongs on the owner's storage, with a manifest,
digest and provenance in the repository, unless it is recorded here on purpose.
A file above its ceiling: recorded files do not grow past a whole mebibyte
unnoticed. A ceiling above what the file needs: the record stays a measurement
rather than an allowance. A row whose file is gone or fell under the threshold.

Sizes are those of the blobs in the index, so the result does not depend on the
checkout's line endings or on uncommitted edits. Submodule links have no blob
and are not sized.

``--lower`` rewrites the rows that are too high or no longer needed. It never
adds a row and never raises one: admitting a large file is an edit a reviewer
sees in the ceiling file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

DEFAULT_CEILING: Final[Path] = Path("tools/tracked_file_size_ceiling.json")
SCHEMA: Final[str] = "tracked_file_size_ceiling_v1"
MEBIBYTE: Final[int] = 1024 * 1024
SUBMODULE_MODE: Final[str] = "160000"


@dataclass(frozen=True)
class Ceiling:
    """Recorded threshold and the ceiling of every large tracked file."""

    threshold_bytes: int
    files: dict[str, int]


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
        JSON file with schema ``tracked_file_size_ceiling_v1``.

    Returns
    -------
    Ceiling
        The threshold in bytes and the ceiling of every recorded file in
        mebibytes.

    Raises
    ------
    ValueError
        If the schema, the threshold, a path or a ceiling is invalid, or a key
        is repeated.

    """
    raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise ValueError("unsupported tracked file size ceiling schema")
    threshold = raw.get("threshold_bytes")
    if not isinstance(threshold, int) or isinstance(threshold, bool) or threshold <= 0:
        raise ValueError("ceiling threshold must be a positive integer number of bytes")
    files = raw.get("files")
    if not isinstance(files, dict):
        raise ValueError("ceiling files must be an object")
    limits: dict[str, int] = {}
    for name, limit in files.items():
        parts = Path(name).parts
        if not parts or Path(name).is_absolute() or ".." in parts:
            raise ValueError(f"ceiling path is not a path inside the repository: {name}")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError(f"ceiling must be a positive integer number of mebibytes: {name}")
        limits[name] = limit
    return Ceiling(threshold, limits)


def write_ceiling(path: Path, ceiling: Ceiling) -> None:
    """Write ``ceiling`` as sorted, indented JSON ending in a newline.

    Parameters
    ----------
    path
        Destination file.
    ceiling
        Threshold and per-file ceilings to record.

    """
    payload = {
        "schema": SCHEMA,
        "threshold_bytes": ceiling.threshold_bytes,
        "total_files": len(ceiling.files),
        "total_mebibytes": sum(ceiling.files.values()),
        "files": dict(sorted(ceiling.files.items())),
    }
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def _output(repo: Path, arguments: Sequence[str], stdin: str = "") -> str:
    """Run a fixed command in ``repo`` and return its standard output.

    Raises
    ------
    ValueError
        If the command cannot start or exits with a failure status.

    """
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argument vectors, no shell
            list(arguments),
            capture_output=True,
            input=stdin,
            encoding="utf-8",
            errors="backslashreplace",
            cwd=repo,
            check=False,
        )
    except OSError as error:
        raise ValueError(f"cannot run {arguments[0]}: {error}") from error
    if completed.returncode != 0:
        raise ValueError(f"{' '.join(arguments)} failed: {completed.stderr.strip()}")
    return completed.stdout


def tracked_sizes(repo: Path) -> dict[str, int]:
    """Return the blob size of every file in the index of ``repo``.

    Parameters
    ----------
    repo
        Repository root.

    Returns
    -------
    dict[str, int]
        Size in bytes keyed by the path relative to ``repo``. A path that is
        present in several stages of an unfinished merge carries its largest
        stage. Submodule links are left out.

    Raises
    ------
    ValueError
        If Git cannot list the index or cannot size one of its blobs.

    """
    entries: list[tuple[str, str]] = []
    for record in _output(repo, ["git", "ls-files", "--stage", "-z"]).split("\0"):
        if not record:
            continue
        meta, name = record.split("\t", 1)
        mode, blob, _stage = meta.split(" ")
        if mode != SUBMODULE_MODE:
            entries.append((name, blob))
    answers = _output(
        repo,
        ["git", "cat-file", "--batch-check=%(objectsize)"],
        "".join(f"{blob}\n" for _name, blob in entries),
    ).splitlines()
    sizes: dict[str, int] = {}
    for (name, _blob), answer in zip(entries, answers, strict=True):
        if not answer.isdigit():
            raise ValueError(f"git cannot size the tracked file {name}: {answer}")
        sizes[name] = max(sizes.get(name, 0), int(answer))
    return sizes


def needed_mebibytes(size: int) -> int:
    """Return the smallest whole number of mebibytes that holds ``size`` bytes."""
    return -(-size // MEBIBYTE)


def compare(sizes: dict[str, int], ceiling: Ceiling) -> list[str]:
    """Compare the sizes of the tracked files with the recorded ceiling.

    Parameters
    ----------
    sizes
        Blob size of every tracked file.
    ceiling
        Recorded threshold and per-file ceilings.

    Returns
    -------
    list[str]
        One message per large file without a row, per file above its ceiling,
        per ceiling above the file and per row that is no longer needed.

    """
    errors: list[str] = []
    for name, size in sorted(sizes.items()):
        if size >= ceiling.threshold_bytes and name not in ceiling.files:
            errors.append(
                f"large file outside the ceiling: {name} ({size} bytes); keep large data on the "
                "owner's storage with a manifest in the repository, or record the file on purpose"
            )
    for name, limit in sorted(ceiling.files.items()):
        size = sizes.get(name, 0)
        needed = needed_mebibytes(size)
        if name not in sizes:
            errors.append(f"ceiling row names a file that is not tracked: {name}; remove it")
        elif size < ceiling.threshold_bytes:
            errors.append(
                f"ceiling row is no longer needed: {name} is {size} bytes, under the threshold; "
                "remove it with --lower"
            )
        elif needed > limit:
            errors.append(
                f"tracked file grew above its ceiling: {name}: {limit} MiB -> {size} bytes"
            )
        elif needed < limit:
            errors.append(
                f"ceiling is above the file: {name}: {limit} MiB -> {needed} MiB; "
                "lower it with --lower"
            )
    return errors


def lower(sizes: dict[str, int], ceiling: Ceiling) -> Ceiling:
    """Return the ceiling with every row reduced to what its file needs.

    Parameters
    ----------
    sizes
        Blob size of every tracked file.
    ceiling
        Recorded threshold and per-file ceilings.

    Returns
    -------
    Ceiling
        The same threshold; rows of files that are gone or under the threshold
        are dropped, the others carry the smaller of the recorded and the needed
        ceiling. No row is added and none is raised.

    """
    return Ceiling(
        ceiling.threshold_bytes,
        {
            name: min(limit, needed_mebibytes(sizes[name]))
            for name, limit in ceiling.files.items()
            if sizes.get(name, 0) >= ceiling.threshold_bytes
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Compare the tracked files with their ceiling, or lower the ceiling.

    Parameters
    ----------
    argv
        Command-line arguments; ``None`` reads the process arguments.

    Returns
    -------
    int
        Zero when every large file is held by a tight row, one otherwise.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ceiling", type=Path, default=DEFAULT_CEILING)
    parser.add_argument(
        "--lower",
        action="store_true",
        help="reduce rows that are too high and drop rows that are no longer needed",
    )
    args = parser.parse_args(argv)
    repo: Path = args.repo
    path = repo / args.ceiling
    try:
        ceiling = load_ceiling(path)
        sizes = tracked_sizes(repo)
        if args.lower:
            ceiling = lower(sizes, ceiling)
            write_ceiling(path, ceiling)
        errors = compare(sizes, ceiling)
    except (OSError, ValueError) as error:
        print(f"tracked file size ceiling failed: {error}", file=sys.stderr)
        return 1
    for message in errors:
        print(message, file=sys.stderr)
    large = [size for size in sizes.values() if size >= ceiling.threshold_bytes]
    print(
        f"Tracked file size ceiling: {len(sizes)} files, {len(large)} of at least "
        f"{ceiling.threshold_bytes} bytes ({sum(large)} bytes); "
        f"{len(ceiling.files)} rows; {len(errors)} problems"
    )
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
