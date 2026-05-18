#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- stable core capability matrix gate
"""Regenerate stable core capability matrix artefacts and compare against commit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARATOR_SCRIPT = Path("scripts") / "compare_stable_core_capability_matrix.py"
DEFAULT_EXPECTED_JSON = REPO_ROOT / "data" / "stable_core" / "backend_capability_matrix.json"
DEFAULT_EXPECTED_MARKDOWN = REPO_ROOT / "docs" / "stable_core_backend_capability_matrix.md"


def build_stable_core_capability_gate_commands(
    *,
    comparator_script: Path = COMPARATOR_SCRIPT,
    expected_json: Path = DEFAULT_EXPECTED_JSON,
    expected_markdown: Path = DEFAULT_EXPECTED_MARKDOWN,
) -> tuple[tuple[str, ...], ...]:
    """Return the deterministic command sequence for the stable-core gate."""

    return (
        (
            sys.executable,
            str(comparator_script),
            "--expected-json",
            str(expected_json),
            "--expected-markdown",
            str(expected_markdown),
        ),
    )


def run_command(command: tuple[str, ...]) -> None:
    """Run one gate command and fail closed on non-zero return code."""

    print(f"[stable-core-capability-gate] {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    """Run stable-core capability matrix gate."""

    for command in build_stable_core_capability_gate_commands():
        run_command(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
