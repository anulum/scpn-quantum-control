# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Complete CI test shard admission
"""Schedule collected tests and admit complete, source-bound shard evidence."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import sqlite3
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import coverage
from coverage.exceptions import CoverageException


@dataclass(frozen=True)
class RunIdentity:
    """Bind a suite's workers to one interpreter, source and workflow attempt.

    Parameters
    ----------
    source_sha
        Checked-out Git commit expected by the aggregate.
    run_id
        Actual workflow run or explicitly local integration invocation.
    attempt
        Workflow attempt that produced the indexed artifacts.
    python
        Interpreter major and minor version.
    suite
        Independent admitted test suite, such as Python or Studio browser tests.

    """

    source_sha: str
    run_id: str
    attempt: str
    python: str
    suite: str = "python"

    def directory(self, index: int) -> str:
        """Return the indexed artifact name for this source and attempt.

        Parameters
        ----------
        index
            One-based worker index.

        Returns
        -------
        str
            Same-run artifact directory name.

        """
        return f"test-shard-{self.suite}-{self.python}-{index}-{self.source_sha}-{self.attempt}"


def write_json(path: Path, payload: object) -> None:
    """Atomically write an evidence or timing document.

    Parameters
    ----------
    path
        Destination on the worker's own filesystem.
    payload
        JSON-serialisable document.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _durations(payload: object) -> dict[str, float]:
    """Refuse malformed timing weights before they influence scheduling."""
    if not isinstance(payload, dict):
        raise ValueError("timing seconds must be an object")
    result: dict[str, float] = {}
    for node, value in payload.items():
        if not isinstance(node, str) or not node:
            raise ValueError("timing identity must be a nonempty string")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("timing duration must be numeric")
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError("timing duration must be positive and finite")
        result[node] = number
    return result


def load_timings(path: Path) -> dict[str, float]:
    """Read optional scheduling hints without making them a test inventory.

    Parameters
    ----------
    path
        Shared immutable snapshot, or a missing cache entry for a cold run.

    Returns
    -------
    dict[str, float]
        Positive finite observed durations keyed by actual pytest node ID.

    Raises
    ------
    ValueError
        If an existing snapshot has an unsupported or malformed schema.

    """
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
    ):
        raise ValueError("unsupported timing snapshot schema")
    return _durations(payload.get("seconds"))


def _inventory(value: object) -> list[str]:
    """Read a unique sorted list of actual collected test identities."""
    if not isinstance(value, list) or not value:
        raise ValueError("test inventory must be a nonempty list")
    if any(not isinstance(node, str) or not node for node in value):
        raise ValueError("test identities must be nonempty strings")
    nodes = cast(list[str], value)
    if len(nodes) != len(set(nodes)):
        raise ValueError("duplicate test identities")
    return sorted(nodes)


def partition_tests(nodes: list[str], count: int, timings: dict[str, float]) -> list[list[str]]:
    """Partition the complete live inventory using deterministic duration hints.

    Parameters
    ----------
    nodes
        Actual admitted node IDs after normal pytest marker selection.
    count
        Positive number of workers, no greater than the admitted item count.
    timings
        Optional observed durations; unknown live tests receive the median.

    Returns
    -------
    list[list[str]]
        Sorted worker inventories with every live item present exactly once.

    Raises
    ------
    ValueError
        If identities, weights or the worker count are invalid.

    """
    live = _inventory(nodes)
    hints = _durations(timings)
    if type(count) is not int or not 1 <= count <= len(live):
        raise ValueError("worker count must fit the admitted test inventory")
    groups: list[list[str]] = [[] for _ in range(count)]
    measured = {node: hints[node] for node in live if node in hints}
    if not measured:
        return [live[index::count] for index in range(count)]
    estimate = statistics.median(measured.values())
    weights = {node: measured.get(node, estimate) for node in live}
    workers = [(0.0, index) for index in range(count)]
    heapq.heapify(workers)
    for node in sorted(live, key=lambda item: (-weights[item], item)):
        load, index = heapq.heappop(workers)
        groups[index].append(node)
        heapq.heappush(workers, (load + weights[node], index))
    return [sorted(group) for group in groups]


def _read_result(directory: Path) -> dict[str, Any]:
    """Decode one worker's production evidence document."""
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if (
        not isinstance(result, dict)
        or type(result.get("schema_version")) is not int
        or result.get("schema_version") != 1
    ):
        raise ValueError("unsupported shard evidence schema")
    return cast(dict[str, Any], result)


def _successful_selection(result: dict[str, Any]) -> list[str]:
    """Reject unsuccessful, collection-only or incomplete executions."""
    if (
        type(result.get("exitstatus")) is not int
        or result.get("exitstatus") != 0
        or result.get("collect_only") is not False
    ):
        raise ValueError("shard did not complete a successful test execution")
    selected = _inventory(result.get("selected"))
    if _inventory(result.get("completed")) != selected:
        raise ValueError("shard did not execute every selected test")
    return selected


def _coverage_digest(path: Path) -> str:
    """Require actual nonempty branch data before binding its digest."""
    data = coverage.CoverageData(basename=str(path))
    data.read()
    if not data.has_arcs() or not data.measured_files():
        raise ValueError("shard coverage must contain measured branch data")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def seal_shard(directory: Path, *, with_coverage: bool) -> None:
    """Seal a completed worker only after its mandatory checks have passed.

    Parameters
    ----------
    directory
        Worker-local result directory emitted by the actual pytest plugin.
    with_coverage
        Require and bind the indexed branch-data file before publication.

    Raises
    ------
    ValueError
        If the execution or branch data is incomplete.

    """
    result = _read_result(directory)
    _successful_selection(result)
    if with_coverage:
        path = directory / f".coverage.shard-{result['shard']}"
        result["coverage_sha256"] = _coverage_digest(path)
    result["sealed"] = True
    write_json(directory / "result.json", result)


def verify_shards(
    root: Path,
    identity: RunIdentity,
    count: int,
    *,
    coverage_output: Path | None = None,
    timings_output: Path | None = None,
) -> dict[str, object]:
    """Admit the exact indexed union before publishing aggregate evidence.

    Parameters
    ----------
    root
        Directory containing only the expected same-run worker artifacts.
    identity
        Expected source, attempt, interpreter and suite.
    count
        Exact mandatory worker count.
    coverage_output
        Optional destination for the complete branch-data union.
    timings_output
        Optional destination for newly observed scheduling hints.

    Returns
    -------
    dict[str, object]
        Verified execution counts and inventory digest.

    Raises
    ------
    ValueError
        If any worker, identity, execution, inventory or coverage input differs.

    """
    if count < 1:
        raise ValueError("worker count must be positive")
    expected = {identity.directory(index) for index in range(1, count + 1)}
    if {path.name for path in root.iterdir()} != expected:
        raise ValueError("missing or unexpected indexed shard artifact")
    inventory: list[str] | None = None
    snapshot: str | None = None
    selected: list[str] = []
    measured: dict[str, float] = {}
    data_files: list[str] = []
    sizes: list[int] = []
    for index in range(1, count + 1):
        directory = root / identity.directory(index)
        result = _read_result(directory)
        if (
            result.get("identity") != asdict(identity)
            or type(result.get("shard")) is not int
            or type(result.get("shards")) is not int
            or result.get("shard") != index
            or result.get("shards") != count
            or result.get("sealed") is not True
        ):
            raise ValueError("wrong or unsealed shard identity")
        live = _inventory(result.get("inventory"))
        timing_digest = result.get("timing_snapshot_sha256")
        if not isinstance(timing_digest, str) or len(timing_digest) != 64:
            raise ValueError("missing immutable timing snapshot digest")
        if inventory is None:
            inventory, snapshot = live, timing_digest
        if live != inventory or timing_digest != snapshot:
            raise ValueError("workers used different discovery or timing snapshots")
        owned = _successful_selection(result)
        if not set(owned) <= set(live):
            raise ValueError("selected tests are outside the actual inventory")
        durations = _durations(result.get("seconds"))
        if not durations.keys() <= set(owned):
            raise ValueError("timings contain unexecuted test identities")
        measured.update(durations)
        selected.extend(owned)
        sizes.append(len(owned))
        if coverage_output is not None:
            path = directory / f".coverage.shard-{index}"
            if _coverage_digest(path) != result.get("coverage_sha256"):
                raise ValueError("shard branch-data digest mismatch")
            data_files.append(str(path))
    if sorted(selected) != inventory:
        raise ValueError("test union contains duplicate or omitted identities")
    if coverage_output is not None:
        if coverage_output.exists():
            raise ValueError("aggregate coverage destination already exists")
        combined = coverage.Coverage(data_file=str(coverage_output))
        combined.combine(data_paths=data_files, strict=True, keep=True)
        combined.save()
        _coverage_digest(coverage_output)
    if timings_output is not None:
        write_json(timings_output, {"schema_version": 1, "seconds": measured})
    digest = hashlib.sha256(json.dumps(inventory, separators=(",", ":")).encode()).hexdigest()
    return {
        "identity": asdict(identity),
        "workers": count,
        "tests": len(selected),
        "per_worker": sizes,
        "inventory_sha256": digest,
    }


def main() -> int:
    """Run timing-snapshot, worker-sealing or aggregate-admission commands.

    Returns
    -------
    int
        Zero for admitted evidence, one for invalid or incomplete input.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--source", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    seal = commands.add_parser("seal")
    seal.add_argument("directory", type=Path)
    seal.add_argument("--coverage", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("root", type=Path)
    for option in ("source-sha", "run-id", "attempt", "python"):
        verify.add_argument("--" + option, required=True)
    verify.add_argument("--suite", default="python")
    verify.add_argument("--shards", type=int, required=True)
    verify.add_argument("--coverage-output", type=Path)
    verify.add_argument("--timings-output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "snapshot":
            write_json(args.output, {"schema_version": 1, "seconds": load_timings(args.source)})
        elif args.command == "seal":
            seal_shard(args.directory, with_coverage=args.coverage)
        else:
            identity = RunIdentity(
                args.source_sha, args.run_id, args.attempt, args.python, args.suite
            )
            result = verify_shards(
                args.root,
                identity,
                args.shards,
                coverage_output=args.coverage_output,
                timings_output=args.timings_output,
            )
            print(json.dumps(result, sort_keys=True))
    except (OSError, ValueError, KeyError, sqlite3.DatabaseError, CoverageException) as error:
        print(f"CI shard admission failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
