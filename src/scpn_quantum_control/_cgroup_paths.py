# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Process memory-controller path resolution
"""Resolve visible cgroup memory ancestors from process membership and mounts.

Paths are namespace-local: ancestors above a mounted subtree cannot be observed.
This module reads proc metadata only; it never moves processes or writes limits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class MemoryCgroup:
    """A visible controller directory and its relation to process membership.

    Attributes
    ----------
    directory
        Absolute filesystem path from the process mount namespace.
    version
        Memory controller version, one or two.
    leaf
        Whether this directory holds the process rather than an ancestor.
    """

    directory: Path
    version: int
    leaf: bool


def _path(value: str, *, escaped: bool = False) -> PurePosixPath:
    """Decode mountinfo escapes and reject paths that cannot be mapped safely."""
    if escaped:
        value = re.sub(r"\\(040|011|012|134)", lambda m: chr(int(m[1], 8)), value)
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or "\x00" in value:
        raise ValueError("invalid cgroup namespace path")
    return path


def memory_cgroup_paths(proc_root: Path) -> tuple[MemoryCgroup, ...] | None:
    """Resolve memory-controller membership and every visible ancestor.

    Parameters
    ----------
    proc_root
        Process proc directory, normally ``/proc/self``; injectable for tests.

    Returns
    -------
    tuple[MemoryCgroup, ...] | None
        De-duplicated paths, leaf first per mount, or ``None`` when proc files
        cannot be read. An empty tuple means no memory hierarchy was recorded.

    Raises
    ------
    ValueError
        Metadata is malformed or a recorded hierarchy has no matching visible
        mount. Callers must not interpret this as an unrestricted process.

    Notes
    -----
    Mountinfo fields four and five map hierarchy root to namespace mount point.
    See Linux ``proc_pid_mountinfo(5)`` and cgroup v2 membership documentation.
    No filesystem traversal above a reported mount point is performed.
    """
    try:
        membership = (proc_root / "cgroup").read_text(encoding="utf-8")
        mountinfo = (proc_root / "mountinfo").read_text(encoding="utf-8")
    except OSError:
        return None
    except UnicodeError as exc:
        raise ValueError("invalid proc metadata encoding") from exc
    members: dict[int, PurePosixPath] = {}
    for line in membership.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            raise ValueError("malformed cgroup membership")
        hierarchy, controllers, member = fields
        if (
            not hierarchy.isascii()
            or not hierarchy.isdecimal()
            or (not controllers and hierarchy != "0")
        ):
            raise ValueError("invalid cgroup hierarchy identifier")
        version = 2 if hierarchy == "0" and not controllers else 1
        if version == 1 and "memory" not in controllers.split(","):
            continue
        path = _path(member)
        if version in members and members[version] != path:
            raise ValueError("conflicting memory controller memberships")
        members[version] = path

    paths: list[MemoryCgroup] = []
    mapped: set[int] = set()
    for line in mountinfo.splitlines():
        before, separator, after = line.partition(" - ")
        fields, filesystem = before.split(), after.split()
        if not separator or len(fields) < 6 or len(filesystem) < 3:
            continue
        if filesystem[0] == "cgroup2":
            version = 2
        elif filesystem[0] == "cgroup" and "memory" in filesystem[2].split(","):
            version = 1
        else:
            continue
        if version not in members:
            continue
        root = _path(fields[3], escaped=True)
        mount = _path(fields[4], escaped=True)
        try:
            relative = members[version].relative_to(root)
        except ValueError:
            continue
        directory = Path(mount / relative)
        boundary = Path(mount)
        leaf = True
        while True:
            entry = MemoryCgroup(directory, version, leaf)
            if entry not in paths:
                paths.append(entry)
            if directory == boundary:
                break
            directory = directory.parent
            leaf = False
        mapped.add(version)
    if members.keys() - mapped:
        raise ValueError("memory membership is outside visible controller mounts")
    return tuple(paths)
