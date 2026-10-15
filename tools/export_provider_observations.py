# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — existing provider observation CLI
"""Export one existing journal attempt for Studio; never submit or contact a provider."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path

from scpn_quantum_control.hardware.provider_job_journal import ProviderJobJournal
from scpn_quantum_control.studio.provider_observations import export_provider_observation


def main(argv: Sequence[str] | None = None) -> int:
    """Write a new detached snapshot of an existing original attempt.

    Parameters
    ----------
    argv
        Process arguments when omitted. Supply an existing ``--journal`` root,
        ``--attempt-id`` and a new ``--output`` JSON file. Refresh by exporting
        to a new file after explicit observation through the native runner.

    Returns
    -------
    int
        Zero on export, one on a refused read/write. Existing output files and
        original event history are preserved. No provider session is opened.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if not (args.journal / "provider_jobs.sqlite3").is_file():
            raise FileNotFoundError("an existing provider journal is required")
        text = export_provider_observation(ProviderJobJournal(args.journal), args.attempt_id)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    except (OSError, ValueError, KeyError, sqlite3.Error) as error:
        print(f"Provider observation export refused: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
