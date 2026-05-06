#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S1 readiness bundle
"""Regenerate all no-QPU S1 feedback-readiness artefacts."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
Entrypoint = Callable[..., int]


def _commands() -> tuple[tuple[str, Entrypoint, tuple[str, ...]], ...]:
    sys.path.insert(0, str(REPO_ROOT))
    from scripts import (
        analyse_s1_feedback_hardware,
        benchmark_s1_feedback_loop,
        export_s1_feedback_preregistration,
    )

    return (
        ("latency", benchmark_s1_feedback_loop.main, ()),
        ("preregistration", export_s1_feedback_preregistration.main, ()),
        (
            "synthetic-analysis",
            analyse_s1_feedback_hardware.main,
            ("data/s1_feedback_loop/s1_feedback_synthetic_raw_counts_2026-05-06.json",),
        ),
    )


def main() -> int:
    failures: list[tuple[str, int]] = []
    for label, entrypoint, args in _commands():
        command = f"{entrypoint.__module__}.{entrypoint.__name__}"
        print(f"[s1-readiness] run {label}: {sys.executable} -m {command}", flush=True)
        returncode = entrypoint(args) if args else entrypoint()
        if returncode != 0:
            failures.append((label, returncode))
            break
    if failures:
        for label, returncode in failures:
            print(f"[s1-readiness] failed {label}: {returncode}", file=sys.stderr)
        return 1
    print("[s1-readiness] regenerated all no-QPU S1 readiness artefacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
