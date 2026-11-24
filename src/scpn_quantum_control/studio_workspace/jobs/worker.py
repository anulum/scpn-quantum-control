# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — worker process of one local job attempt
"""Run one admitted attempt in its own process and seal its record.

The service starts this module as ``python -m
scpn_quantum_control.studio_workspace.jobs.worker <attempt-directory>`` with its
memory, thread, CPU-time and file-size limits, which the worker applies first. The
directory is created and owned by the service and holds ``request.json``. The
worker runs the request through the executive spine with the local-only
registry and writes ``record.json`` atomically: a reader sees either no record
or the complete one. A record of a failed action is still a record; the exit
status is non-zero only when no record could be written.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
from collections.abc import Sequence
from pathlib import Path
from typing import Final

REQUEST_FILE = "request.json"
"""Request written by the service before the worker starts."""
RECORD_FILE = "record.json"
"""Sealed executive record written by the worker."""
THREAD_VARIABLES: Final[tuple[str, ...]] = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "RAYON_NUM_THREADS",
)
"""Thread counts read by the libraries a worker loads: OpenBLAS (NumPy and SciPy),
OpenMP, MKL and the Rayon pool of the Rust engine."""


def run_attempt(directory: Path) -> Path:
    """Run the attempt stored in ``directory`` and write its sealed record.

    Parameters
    ----------
    directory
        Service-owned attempt directory holding ``request.json``.

    Returns
    -------
    Path
        The written record.

    """
    # Imported here, so the resource limits apply before the numerical stack loads.
    from ...studio.executive import ExecutiveRequest, run_action
    from .registry import build_local_registry

    request = json.loads((directory / REQUEST_FILE).read_text(encoding="utf-8"))
    record = run_action(
        ExecutiveRequest(
            verb=request["verb"],
            action_id=request["action_id"],
            parameters=request["parameters"],
            backend=request["backend"],
        ),
        registry=build_local_registry(),
    )
    target = directory / RECORD_FILE
    partial = directory / (RECORD_FILE + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(record.to_dict(), handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(target)
    return target


def limit_resources(*, memory_bytes: int, threads: int, cpu_seconds: int, file_bytes: int) -> None:
    """Apply the service's limits to this process and every child it starts.

    Parameters
    ----------
    memory_bytes
        Address-space limit.
    threads
        Threads each numerical library may start. It replaces any inherited
        value, because each thread reserves address space and the libraries
        otherwise start one per core.
    cpu_seconds
        CPU-time limit; the kernel stops the process beyond it.
    file_bytes
        Largest file the process may write.

    Raises
    ------
    ValueError
        ``threads`` is below one, which the libraries would read as "one per core".

    """
    if threads < 1:
        raise ValueError("threads must be at least 1")
    # Set before the numerical stack loads: each library reads its count once, at start.
    for name in THREAD_VARIABLES:
        os.environ[name] = str(threads)
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def main(argv: Sequence[str] | None = None) -> int:
    """Apply the limits, then run the attempt named on the command line.

    Parameters
    ----------
    argv
        Argument vector: the attempt directory and the four limits.

    Returns
    -------
    int
        ``0`` once the record is written.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--memory-bytes", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--cpu-seconds", type=int, required=True)
    parser.add_argument("--file-bytes", type=int, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    limit_resources(
        memory_bytes=args.memory_bytes,
        threads=args.threads,
        cpu_seconds=args.cpu_seconds,
        file_bytes=args.file_bytes,
    )
    run_attempt(args.directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
