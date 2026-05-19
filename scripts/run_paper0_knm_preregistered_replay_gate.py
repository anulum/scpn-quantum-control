#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 K_nm preregistered replay gate
"""Regenerate the Paper 0 K_nm replay artefacts and compare against commit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARATOR_SCRIPT = Path("scripts") / "compare_paper0_knm_preregistered_replay.py"
MEASURED_CANDIDATE_GATE_SCRIPT = Path("scripts") / "run_knm_measured_candidate_gate.py"
DEFAULT_EXPECTED_JSON = REPO_ROOT / "data" / "paper0_knm_preregistered_replay.json"
DEFAULT_EXPECTED_MARKDOWN = REPO_ROOT / "docs" / "paper0_knm_preregistered_replay.md"


def build_paper0_knm_preregistered_replay_gate_commands(
    *,
    comparator_script: Path = COMPARATOR_SCRIPT,
    expected_json: Path = DEFAULT_EXPECTED_JSON,
    expected_markdown: Path = DEFAULT_EXPECTED_MARKDOWN,
    measured_candidate_gate_script: Path = MEASURED_CANDIDATE_GATE_SCRIPT,
) -> tuple[tuple[str, ...], ...]:
    """Return deterministic Paper 0 K_nm replay gate commands."""

    return (
        (
            sys.executable,
            str(comparator_script),
            "--expected-json",
            str(expected_json),
            "--expected-markdown",
            str(expected_markdown),
        ),
        (sys.executable, str(measured_candidate_gate_script)),
    )


def run_command(command: tuple[str, ...]) -> None:
    """Run one Paper 0 K_nm replay gate command and fail closed."""

    print(f"[paper0-knm-preregistered-replay-gate] {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    """Run the Paper 0 K_nm replay gate."""

    for command in build_paper0_knm_preregistered_replay_gate_commands():
        run_command(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
