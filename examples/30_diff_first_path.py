# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — diff first path example
"""First-path differentiable namespace example."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from scpn_quantum_control import TraceADArray, diff, whole_program_value_and_grad
from scpn_quantum_control.differentiable import program_adjoint_replay_gradient


def phase_cost(params: NDArray[np.float64]) -> float:
    """Return a scalar local phase-control objective."""
    return float(np.sin(params[0]) + params[1] ** 2)


def effectful_program_demo() -> None:
    """Demonstrate repeated-index mutation and captured-coefficient replay."""
    coefficient = np.array([1.0], dtype=np.float64)

    def objective(values: TraceADArray) -> object:
        working = values.copy()
        np.add.at(working, [0, 0, 2], values)
        return cast(TraceADArray, working**2).sum() * coefficient[0]

    inputs = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    result = whole_program_value_and_grad(objective, inputs, trace=False)
    # working=(2*x+y,y,2*z), so its squared norm has gradient (16,12,24).
    np.testing.assert_allclose(result.gradient, [16.0, 12.0, 24.0], rtol=0.0, atol=1.0e-12)
    np.testing.assert_array_equal(inputs, [1.0, 2.0, 3.0])
    print("\nwhole-program effects")
    print(f"  value: {result.value:.8f}")
    print(f"  replay gradient: {program_adjoint_replay_gradient(result).tolist()}")

    coefficient[0] = 2.0
    try:
        program_adjoint_replay_gradient(result)
    except ValueError:
        print("  changed coefficient: replay refused")
    else:
        raise RuntimeError("replay accepted a changed captured coefficient")
    finally:
        coefficient[0] = 1.0
    np.testing.assert_allclose(
        program_adjoint_replay_gradient(result), [16.0, 12.0, 24.0], rtol=0.0, atol=1.0e-12
    )
    print("  restored coefficient: replay available")


def main() -> None:
    """Run the canonical no-credential differentiable first path."""
    circuit = diff.differentiable_circuit(
        phase_cost,
        name="phase_cost_first_path",
        parameter_names=("theta", "bias"),
        gradient_method="finite_difference",
    )
    params = np.array([0.3, 0.5], dtype=np.float64)
    gradient = circuit.grad(params)
    jit_status = diff.jit_or_explain(circuit)
    contract = diff.run_differentiable_circuit_contract_audit()

    print("canonical diff namespace")
    print(f"  value: {circuit(params):.8f}")
    print(f"  gradient: {gradient.tolist()}")
    print(f"  supported: {circuit.diagnostics.supported}")
    print(f"  jit fail_closed: {jit_status.fail_closed}")
    print(f"  contract audit passed: {contract.passed}")
    print(f"  claim boundary: {circuit.claim_boundary}")
    effectful_program_demo()


if __name__ == "__main__":
    main()
