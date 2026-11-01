# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — captured polynomial and linear-solve replay example
"""Capture one complete user program for native and browser WASM replay.

Run with ``PYTHONPATH=src:oscillatools/src python examples/whole_program_replay.py``.
The JSON carries actual effect-IR, input bindings, the portable replay record
and Python values/gradients. It contains no measured performance, physical
validation or promise of cross-platform bit-exact linear algebra.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from scpn_quantum_control.differentiable import whole_program_value_and_grad
from scpn_quantum_control.studio.program_ad_replay_artifact import encode_replay_input
from scpn_quantum_control.whole_program_ad_api import (
    capture_whole_program_replay,
)

_A = np.array([[3.0, 1.0], [1.0, 2.0]])
_C = np.array([0.5, -1.0])


def polynomial_solve_objective(p: Any) -> Any:
    """Compose a polynomial, a well-conditioned solve and a scalar objective.

    Parameters
    ----------
    p:
        Two dimensionless float64 parameters or the public AD trace array.

    Returns
    -------
    Any
        Scalar ``sum(x*x) + 3*sum(x)`` for ``A*x = p*p + 2*p + C``.

    """
    b = p * p + 2.0 * p + _C
    x = np.linalg.solve(_A, b)
    return np.sum(x * x) + 3.0 * np.sum(x)


def main() -> None:
    """Print actual captures for three inputs without files or provider calls."""
    cases: list[dict[str, object]] = []
    for point in ([0.7, -0.4], [-1.3, 0.25], [2.0, 1.5]):
        result = whole_program_value_and_grad(polynomial_solve_objective, np.array(point))
        record = capture_whole_program_replay(result)
        assert result.program_ir is not None
        ir = result.program_ir.serialization
        packed = encode_replay_input(ir, point)
        cases.append(
            {
                "inputs": point,
                "effect_ir": ir,
                "effect_ir_sha256": record.program_ir_sha256,
                "input_hex": packed.hex(),
                "input_sha256": hashlib.sha256(packed).hexdigest(),
                "value": float(result.value),
                "gradient": result.gradient.tolist(),
                "record": json.loads(record.to_json()),
            }
        )
    print(
        json.dumps({"schema": "whole_program_replay_example.v1", "cases": cases}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
