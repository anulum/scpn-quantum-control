# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Live container memory admission check
"""Check default dense admission inside an externally memory-limited container.

Run after installing the real package/dependencies. The caller supplies the
Docker memory limit in bytes; this command never creates a container, changes
its limits or allocates the estimated buffers. Exit zero prints JSON evidence;
an invalid limit, override or failed admission invariant exits nonzero.
Injected filesystem tests are not a substitute for the hosted container run.
"""

from __future__ import annotations

import argparse
import json
import os

from scpn_quantum_control.dense_budget import (
    DEFAULT_DENSE_BUDGET_ENV,
    DEFAULT_DENSE_RAM_FRACTION,
    DenseAllocationError,
    cgroup_headroom_bytes,
    dense_budget_bytes,
    require_dense_allocation,
)


def main(argv: list[str] | None = None) -> int:
    """Verify live memory admission and print byte-denominated evidence.

    Parameters
    ----------
    argv
        CLI arguments, or ``None`` to read the process arguments. Requires
        ``--limit-bytes`` matching the container's externally configured limit.

    Returns
    -------
    int
        Zero after all checks pass. Argument errors exit with status two.

    Raises
    ------
    RuntimeError
        An override masks the default, no positive finite cgroup allowance is
        visible, the budget exceeds the declared bound, or admission is wrong.

    Notes
    -----
    Readings are separate snapshots, not reservations. This checks admission,
    not total RSS, hidden namespace ancestors or immunity from concurrent OOM.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-bytes", type=int, required=True)
    args = parser.parse_args(argv)
    limit: int = args.limit_bytes
    if not 1024**2 <= limit <= 8 * 1024**3:
        parser.error("limit must be between 1 MiB and 8 GiB")
    if DEFAULT_DENSE_BUDGET_ENV in os.environ:
        raise RuntimeError("an environment override masks the default budget")
    headroom = cgroup_headroom_bytes()
    if headroom is None or not 0 < headroom <= limit:
        raise RuntimeError("no positive cgroup headroom within the declared limit")
    budget = dense_budget_bytes()
    if not 0 < budget <= int(limit * DEFAULT_DENSE_RAM_FRACTION):
        raise RuntimeError("default budget is outside the container admission bound")
    small = require_dense_allocation(1, rank=1, dtype="uint8")
    qubits = limit.bit_length()
    try:
        require_dense_allocation(qubits, rank=1, dtype="uint8")
    except DenseAllocationError:
        pass
    else:
        raise RuntimeError("above-limit dense allocation was admitted")
    print(
        json.dumps(
            {
                "limit_bytes": limit,
                "cgroup_headroom_bytes": headroom,
                "default_budget_bytes": budget,
                "small_admitted_bytes": small.bytes_required,
                "refused_bytes": 1 << qubits,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
