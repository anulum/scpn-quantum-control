# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — shared contract ownership and conformance
"""Run every registered real shared-contract consumer without optional fallbacks."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tools.ci_workflow_inventory import load_contract_cohorts, validate_contract_workflow

Gate = tuple[str, Path, list[str]]


def build_contract_conformance_gates(python: str, *, repo_root: Path | None = None) -> list[Gate]:
    """Bind each mandatory consumer to its existing executable test owner.

    Parameters
    ----------
    python
        Locked interpreter path, substituted only for the literal Python token.
    repo_root
        Selected local checkout, defaulting to this tool's repository.

    Returns
    -------
    list[Gate]
        Ordered labels, actual working directories and argument vectors.

    Raises
    ------
    ValueError
        If a cohort, owner or required CI aggregation is absent or bypassed.
    OSError
        If a source-bound registry or workflow cannot be read.

    """
    root = (Path(__file__).resolve().parents[1] if repo_root is None else repo_root).resolve()
    cohorts = load_contract_cohorts(repo_root=root)
    validate_contract_workflow(repo_root=root)
    return [
        (
            row["id"] + "/" + consumer["id"],
            root / consumer["cwd"],
            [python if part == "{python}" else part for part in consumer["command"]],
        )
        for row in cohorts
        for consumer in row["consumers"]
    ]


def main(argv: list[str] | None = None) -> int:
    """Check ownership, optionally running all required consumers in order.

    Parameters
    ----------
    argv
        Explicit CLI arguments for embedding, or the actual process arguments.
        `--run` executes the corpus; the default validates ownership only.

    Returns
    -------
    int
        Zero on complete success; first consumer's failure, 1 for invalid
        ownership, 124 for timeout or 127 for an unavailable runtime.

    Notes
    -----
    Commands use argument vectors without a shell, at most 120 seconds per
    consumer, with no retries or optional fallback. Consumer stdout and stderr
    remain observable. This command performs no provider submission.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Execute every actual corpus consumer")
    options = parser.parse_args(argv)
    try:
        gates = build_contract_conformance_gates(sys.executable)
    except (ValueError, KeyError, OSError) as error:
        print(f"Shared contract ownership refused: {error}", file=sys.stderr)
        return 1
    if options.run:
        for name, cwd, command in gates:
            print(f"Required shared contract: {name}", flush=True)
            try:
                result = subprocess.run(command, cwd=cwd, check=False, timeout=120)
            except subprocess.TimeoutExpired:
                print(f"Shared contract timed out: {name}", file=sys.stderr)
                return 124
            except OSError as error:
                print(f"Shared contract runtime unavailable: {name}: {error}", file=sys.stderr)
                return 127
            if result.returncode:
                return result.returncode if result.returncode > 0 else 1
    print(f"Shared contract ownership passed: {len(gates)} required real consumers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_contract_conformance_gates", "main"]
