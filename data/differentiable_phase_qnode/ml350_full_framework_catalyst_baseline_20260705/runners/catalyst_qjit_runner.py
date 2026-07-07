#!/usr/bin/env python3
"""ML350 Catalyst qjit runner for SCPN bounded external comparison evidence."""

from __future__ import annotations

import json
import sys

import catalyst
import jax.numpy as jnp
from catalyst import grad, qjit


@qjit
def objective(x):
    return jnp.cos(x[0]) + 0.25 * jnp.sin(x[1])


@qjit
def objective_grad(x):
    return grad(objective)(x)


def main() -> int:
    try:
        request = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        print(f"invalid JSON request: {exc}", file=sys.stderr)
        return 2
    if request.get("schema") != "scpn_qc_catalyst_runner_request_v1":
        print("unsupported request schema", file=sys.stderr)
        return 2
    if request.get("case_id") != "bounded_phase_objective":
        print("unsupported Catalyst benchmark case", file=sys.stderr)
        return 2
    values = request.get("values")
    if (
        not isinstance(values, list)
        or len(values) != 2
        or not all(isinstance(value, (int, float)) for value in values)
    ):
        print("values must be two real numeric entries", file=sys.stderr)
        return 2
    x = jnp.asarray(values, dtype=jnp.float64)
    value = objective(x)
    gradient = objective_grad(x)
    print(
        json.dumps(
            {
                "value": float(value),
                "gradient": [float(item) for item in gradient],
                "toolchain": {
                    "catalyst": f"pennylane-catalyst=={getattr(catalyst, '__version__', 'unknown')}",
                    "mlir": "Catalyst bundled CLI, mlir-opt-18 installed on host",
                    "workflow": "qjit + catalyst.grad CPU float64",
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
