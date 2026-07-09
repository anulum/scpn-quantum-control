# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Ground-State Optimizer Convergence Export
"""Export BL-15 ground-state optimizer convergence benchmark rows."""

from __future__ import annotations

from pathlib import Path

from scpn_quantum_control.benchmarks.differentiable_optimizer_convergence import (
    write_ground_state_optimizer_convergence_artifact,
)

DEFAULT_OUTPUT = Path(
    "data/differentiable_phase_qnode/ground_state_optimizer_convergence_20260709.json"
)


def main() -> int:
    """Write the committed BL-15 optimizer convergence artifacts."""
    artifact = write_ground_state_optimizer_convergence_artifact(DEFAULT_OUTPUT)
    print(f"wrote {artifact.json_path}")
    print(f"wrote {artifact.markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
