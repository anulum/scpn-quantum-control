#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Rust replay/native-JIT decision-evidence writer
"""Write the frozen Rust replay/native-JIT decision-evidence artifact."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

REPO_ROOT = Path(__file__).resolve().parents[1]


class _DecisionEvidenceModule(Protocol):
    """Typed surface loaded from the repository-local evidence tool."""

    def capture_decision_evidence(
        self,
        *,
        stamp: str,
        rounds: int,
        repetitions: int,
        warmups: int,
        isolated: bool,
    ) -> dict[str, object]:
        """Capture one bounded comparison payload."""

    def write_decision_evidence(
        self,
        payload: Mapping[str, object],
        path: Path,
    ) -> Path:
        """Write one validated comparison payload."""


def _load_decision_evidence_module() -> _DecisionEvidenceModule:
    """Load the repository-local evidence tool without import-order suppressions."""
    repository = str(REPO_ROOT)
    if repository not in sys.path:
        sys.path.insert(0, repository)
    return cast(
        _DecisionEvidenceModule,
        import_module("tools.rust_llvm_jit_decision_evidence"),
    )


def main() -> int:
    """Capture the bounded comparison and write its canonical JSON record."""
    decision_evidence = _load_decision_evidence_module()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", default="20260726")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/differentiable_phase_qnode/rust_llvm_jit_decision_evidence_20260726.json"
        ),
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--warmups", type=int, default=10)
    args = parser.parse_args()
    payload = decision_evidence.capture_decision_evidence(
        stamp=args.stamp,
        rounds=args.rounds,
        repetitions=args.repetitions,
        warmups=args.warmups,
        isolated=False,
    )
    path = decision_evidence.write_decision_evidence(payload, args.output)
    print(f"wrote {path}")
    print(f"decision={payload['decision']} sha256={payload['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
