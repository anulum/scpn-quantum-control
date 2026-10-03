# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — offline declared backend profile export
"""Export a dated declared catalogue without contacting any provider."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from scpn_quantum_control.hardware.provider_capability_discovery import build_backend_profiles


def main(argv: Sequence[str] | None = None) -> int:
    """Write source-owned offline metadata or verify an existing exact export.

    Parameters
    ----------
    argv
        Command arguments, or the process arguments when omitted.

    Returns
    -------
    int
        Zero on an exact export/check; one when committed metadata is stale.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = (
        json.dumps(
            build_backend_profiles(observed_at=args.observed_at), ensure_ascii=False, indent=2
        )
        + "\n"
    )
    if args.check:
        return int(not args.output.is_file() or args.output.read_text(encoding="utf-8") != text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
