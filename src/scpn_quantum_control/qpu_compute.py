# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — QPU compute module
# SCPN Quantum Control - QPU compute-unit facade
"""Public QPU compute façade and command-line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from scpn_quantum_control.qpu_compute_runtime import (
    execute_simulator_request,
    make_compute_request,
    read_compute_request,
    read_compute_result,
    read_fusion_result,
    read_node_descriptor,
    read_stream_delta,
    run_simulator_from_artifact,
    write_compute_request,
    write_compute_result,
    write_fusion_result,
    write_node_descriptor,
    write_stream_delta,
)
from scpn_quantum_control.qpu_compute_types import (
    FUSION_SCHEMA_VERSION,
    NODE_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    STREAM_DELTA_SCHEMA_VERSION,
    SUPPORTED_BACKEND_POLICIES,
    SUPPORTED_KERNELS,
    QPUComputeRequest,
    QPUComputeResult,
    QPUFusionResult,
    QPUNodeDescriptor,
    QPUStreamDelta,
    fuse_compute_results,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SCPN QPU compute-unit runner")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-simulator", help="Run a QPU artifact on the local simulator")
    run.add_argument("--artifact", required=True, help="Path to QPU data artifact JSON")
    run.add_argument("--request-out", required=True, help="Path for compute request JSON")
    run.add_argument("--result-out", required=True, help="Path for compute result JSON")
    run.add_argument("--kernel", default="sync_dla", choices=sorted(SUPPORTED_KERNELS))
    run.add_argument("--shots", type=int, default=1024)
    run.add_argument("--trotter-depth", type=int, default=2)
    run.add_argument("--time-step", type=float, default=0.1)
    run.add_argument("--lambda-fim", type=float, default=0.0)
    run.add_argument("--coupling-scale", type=float, default=1.0)
    run.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow synthetic/simulation/fixture artifacts for smoke runs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for QPU compute-unit commands."""
    args = _build_parser().parse_args(argv)
    if args.command == "run-simulator":
        result = run_simulator_from_artifact(
            args.artifact,
            request_out=args.request_out,
            result_out=args.result_out,
            kernel=args.kernel,
            shots=args.shots,
            trotter_depth=args.trotter_depth,
            time_step=args.time_step,
            lambda_fim=args.lambda_fim,
            coupling_scale=args.coupling_scale,
            require_publication_safe=not args.allow_synthetic,
        )
        print(result.to_json())
        return 0
    raise ValueError(f"unsupported command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FUSION_SCHEMA_VERSION",
    "NODE_SCHEMA_VERSION",
    "QPUComputeRequest",
    "QPUComputeResult",
    "QPUFusionResult",
    "QPUNodeDescriptor",
    "QPUStreamDelta",
    "REQUEST_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "STREAM_DELTA_SCHEMA_VERSION",
    "SUPPORTED_BACKEND_POLICIES",
    "SUPPORTED_KERNELS",
    "execute_simulator_request",
    "fuse_compute_results",
    "make_compute_request",
    "read_compute_request",
    "read_compute_result",
    "read_fusion_result",
    "read_node_descriptor",
    "read_stream_delta",
    "run_simulator_from_artifact",
    "write_compute_request",
    "write_compute_result",
    "write_fusion_result",
    "write_node_descriptor",
    "write_stream_delta",
]
