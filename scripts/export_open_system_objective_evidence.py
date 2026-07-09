# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Open-System Objective Evidence Export
"""Export BL-16 open-system objective evidence rows."""

from __future__ import annotations

from pathlib import Path

from scpn_quantum_control.benchmarks.open_system_objective_evidence import (
    write_open_system_objective_evidence_artifact,
)

DEFAULT_OUTPUT = Path(
    "data/differentiable_phase_qnode/open_system_objective_evidence_20260709.json"
)


def main() -> int:
    """Write the committed BL-16 open-system objective artifacts."""
    artifact = write_open_system_objective_evidence_artifact(DEFAULT_OUTPUT)
    print(f"wrote {artifact.json_path}")
    print(f"wrote {artifact.markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
