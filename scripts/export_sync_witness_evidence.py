# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Synchronisation Witness Evidence Export
"""Export BL-18 synchronisation-witness evidence rows."""

from __future__ import annotations

from pathlib import Path

from scpn_quantum_control.benchmarks.sync_witness_evidence import (
    write_sync_witness_evidence_artifact,
)

DEFAULT_OUTPUT = Path("data/differentiable_phase_qnode/sync_witness_evidence_20260709.json")


def main() -> int:
    """Write the committed BL-18 synchronisation-witness artifacts."""
    artifact = write_sync_witness_evidence_artifact(DEFAULT_OUTPUT)
    print(f"wrote {artifact.json_path}")
    print(f"wrote {artifact.markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
