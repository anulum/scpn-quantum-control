# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Git location isolation for the test suite
"""Keep inherited Git repository-location variables out of the test session.

Git exports variables such as ``GIT_DIR`` and ``GIT_INDEX_FILE`` to its hooks,
and a caller may export them to address a repository from outside its work
tree. A test that builds a temporary repository would then run ``git init``
and ``git commit`` against the inherited repository instead of its own:
``git init`` rewrites ``core.worktree`` there and ``git commit`` writes to its
index and references. The shared test configuration therefore drops those
variables once, before any test module is collected.

A test that needs such a variable sets it itself, for its own child process.
"""

from __future__ import annotations

import os
import subprocess


def repository_local_variables() -> tuple[str, ...]:
    """Return the environment variables Git treats as local to one repository.

    The names come from ``git rev-parse --local-env-vars``, so the list follows
    the installed Git release instead of a copy kept here. The query needs no
    repository and answers even when the inherited location is unusable.

    Returns
    -------
    tuple[str, ...]
        Variable names in the order Git reports them. Empty when no ``git``
        executable can be started, in which case no test can reach a
        repository either.

    Raises
    ------
    subprocess.CalledProcessError
        If Git starts but cannot report the list. The session then stops
        instead of running without isolation.

    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--local-env-vars"],
            check=True,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ()
    return tuple(completed.stdout.split())


def drop_inherited_git_location() -> dict[str, str]:
    """Remove every repository-local Git variable from the process environment.

    Returns
    -------
    dict[str, str]
        The removed variables with the values they had, in Git's order.

    """
    return {
        name: os.environ.pop(name) for name in repository_local_variables() if name in os.environ
    }
