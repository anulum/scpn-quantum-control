# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Complete retained-population comparison CLI
"""Export a complete native method-comparison population for read-only Studio import."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from scpn_quantum_control.studio.method_comparison_artifact import (
    MAX_COMPARISON_BYTES,
    export_method_comparison,
    read_method_comparison_population,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Compute the native all-run projection and write an exclusively new document.

    Parameters
    ----------
    argv : sequence of str or None
        Process arguments when absent. Required --input names an existing
        complete population JSON; --output names a new detached Studio JSON.

    Returns
    -------
    int
        Zero on export, one on a refused read/admission/write. Malformed input
        never creates an output, and existing input/output files are preserved.
        No solver, provider, external URL or execution service is invoked.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        with args.input.open("r", encoding="utf-8") as source:
            original = source.read(MAX_COMPARISON_BYTES + 1)
        runs, baseline, policy, selected = read_method_comparison_population(original)
        text = export_method_comparison(
            runs, baseline_method=baseline, missing_data_policy=policy, selected_run_id=selected
        )
        with args.output.open("x", encoding="utf-8") as destination:
            destination.write(text)
    except (OSError, ValueError) as error:
        print(f"Method comparison export refused: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
