# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""First-path differentiable namespace example."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_quantum_control import diff


def phase_cost(params: NDArray[np.float64]) -> float:
    """Return a scalar local phase-control objective."""
    return float(np.sin(params[0]) + params[1] ** 2)


def main() -> None:
    """Run the canonical no-credential differentiable first path."""
    circuit = diff.differentiable_circuit(
        phase_cost,
        name="phase_cost_first_path",
        parameter_names=("theta", "bias"),
    )
    params = np.array([0.3, 0.5], dtype=np.float64)
    gradient = circuit.grad(params, method="finite_difference")
    jit_status = diff.jit_or_explain(circuit)

    print("canonical diff namespace")
    print(f"  value: {circuit(params):.8f}")
    print(f"  gradient: {gradient.tolist()}")
    print(f"  supported: {circuit.diagnostics.supported}")
    print(f"  jit fail_closed: {jit_status.fail_closed}")
    print(f"  claim boundary: {circuit.claim_boundary}")


if __name__ == "__main__":
    main()
