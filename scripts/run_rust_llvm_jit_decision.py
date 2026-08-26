#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — BL-38 decision evidence writer
"""Write the frozen Rust-JIT decision Rust LLVM/JIT decision evidence artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.rust_llvm_jit_decision import (  # noqa: E402
    capture_decision_evidence,
    write_decision_evidence,
)


def main() -> int:
    """Capture the bounded comparison and write its canonical JSON record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", default="20260726")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/differentiable_phase_qnode/rust_llvm_jit_decision_20260726.json"),
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--warmups", type=int, default=10)
    args = parser.parse_args()
    payload = capture_decision_evidence(
        stamp=args.stamp,
        rounds=args.rounds,
        repetitions=args.repetitions,
        warmups=args.warmups,
        isolated=False,
    )
    path = write_decision_evidence(payload, args.output)
    print(f"wrote {path}")
    print(f"decision={payload['decision']} sha256={payload['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
