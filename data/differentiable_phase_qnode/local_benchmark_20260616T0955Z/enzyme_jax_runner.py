#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — local Enzyme-JAX benchmark runner

"""Run the bounded external-comparison objective through Enzyme-JAX.

This runner is intentionally narrow: it accepts only the benchmark request
schema emitted by ``run_differentiable_benchmark_evidence.py`` and exits
non-zero if Enzyme-JAX cannot execute the value-and-gradient path. That keeps
the committed artifact honest: a dependency installation that still fails at
runtime remains a hard gap, not a fabricated success row.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

ENZYME_PYTHON = Path("/home/anulum/.cache/scpn-qc-enzyme-py39/bin/python")


def main() -> int:
    try:
        request = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        print(f"invalid JSON request: {exc}", file=sys.stderr)
        return 2
    if request.get("schema") != "scpn_qc_enzyme_runner_request_v1":
        print("unsupported request schema", file=sys.stderr)
        return 2
    if request.get("case_id") != "bounded_phase_objective":
        print("unsupported Enzyme benchmark case", file=sys.stderr)
        return 2
    values = request.get("values")
    if (
        not isinstance(values, list)
        or len(values) != 2
        or not all(isinstance(value, (int, float)) for value in values)
    ):
        print("values must be two real numeric entries", file=sys.stderr)
        return 2
    if not ENZYME_PYTHON.exists():
        print(f"Enzyme Python runtime is missing: {ENZYME_PYTHON}", file=sys.stderr)
        return 2

    program = dedent(
        """
        import json
        import sys

        import jax
        import jax.numpy as jnp
        from enzyme_ad.jax import enzyme_jax_ir

        values = json.loads(sys.argv[1])

        @enzyme_jax_ir()
        def objective(x):
            return jnp.cos(x[0]) + 0.25 * jnp.sin(x[1])

        x = jnp.asarray(values, dtype=jnp.float64)
        value = objective(x)
        gradient = jax.grad(objective)(x)
        print(json.dumps({
            "value": float(value),
            "gradient": [float(item) for item in gradient],
            "toolchain": {
                "enzyme": "enzyme-ad==0.0.6 via Enzyme-JAX",
                "llvm": "Enzyme-JAX bundled native extension",
                "jax": jax.__version__,
            },
        }, sort_keys=True))
        """
    )
    completed = subprocess.run(
        [str(ENZYME_PYTHON), "-c", program, json.dumps(values)],
        text=True,
        capture_output=True,
        check=False,
        timeout=20.0,
        env={
            "JAX_ENABLE_X64": "1",
            "JAX_PLATFORMS": "cpu",
            "PYTHONNOUSERSITE": "1",
        },
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "no stderr"
        print(f"Enzyme-JAX execution failed: {stderr}", file=sys.stderr)
        return completed.returncode
    sys.stdout.write(completed.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
