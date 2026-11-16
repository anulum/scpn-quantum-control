# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — private home for quality-gate coverage databases
"""Give the isolated coverage database of every quality gate a private per-process home.

The gate builders used to name their databases directly under ``/tmp``. Two local
runs of one gate then wrote the same file, and a predictable name in a shared
directory can be created first by someone else. Each process now makes one
private directory on first use, with an unpredictable name and owner-only
permissions, places every database inside it and removes it when the process
exits. A gate's coverage run and its threshold report are built in the same
process, so they still name the same file.

The hosted workflows keep their literal ``/tmp`` paths, because each job runs on
a fresh single-tenant runner. The workflow contract tests compare database names
rather than full paths.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from functools import cache
from pathlib import Path

QUALITY_COVERAGE_PREFIX = "scpn-qc-quality-coverage-"
"""Prefix of the private directory name; the rest of the name is random."""


@cache
def quality_coverage_dir() -> Path:
    """Return this process's private directory for gate coverage databases.

    Returns
    -------
    pathlib.Path
        A directory created with :func:`tempfile.mkdtemp` (mode ``0o700``) on the
        first call and removed when the interpreter exits. Later calls return the
        same directory.

    """
    directory = Path(tempfile.mkdtemp(prefix=QUALITY_COVERAGE_PREFIX))
    atexit.register(shutil.rmtree, directory, ignore_errors=True)
    return directory


def quality_coverage_file(name: str) -> str:
    """Return the path of one named coverage database in the private directory.

    Parameters
    ----------
    name
        Bare file name of the database, such as
        ``"scpn-qc-application-honesty-quality.coverage"``.

    Returns
    -------
    str
        The database path inside :func:`quality_coverage_dir`, ready for
        ``--data-file=``.

    Raises
    ------
    ValueError
        If ``name`` is empty, names a directory entry such as ``..``, or contains
        a path separator.

    """
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError(f"coverage database name must be a bare file name: {name!r}")
    return str(quality_coverage_dir() / name)


__all__ = ["QUALITY_COVERAGE_PREFIX", "quality_coverage_dir", "quality_coverage_file"]
