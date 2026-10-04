# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — offline operator policy conformance exports
"""Export dated synthetic policy decisions without contacting a provider."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from scpn_quantum_control.canonical_encoding import canonical_digest
from scpn_quantum_control.hardware.hal import (
    BackendCapabilities,
    BackendProfile,
    HardwareAbstractionLayer,
    QuantumWorkload,
)
from scpn_quantum_control.hardware.operator_policy_contracts import OperatorPolicy, PricingEstimate
from scpn_quantum_control.hardware.provider_semantics import WorkloadSemantics
from scpn_quantum_control.studio.workspace import (
    ResolvedSettings,
    assess_workspace_operator_policy,
    operator_request_from_settings,
    write_json,
)

EXAMPLE_CASES = (
    "unknown_price",
    "at_ceiling",
    "over_shots",
    "over_cost",
    "wrong_region",
    "expired_policy",
    "over_concurrency",
    "over_time",
)
"""Independent synthetic scenarios; none are live prices, availability or account evidence."""


def build_operator_policy_example(case: str = "unknown_price") -> dict[str, object]:
    """Produce an exact dated core verdict through the public workspace boundary.

    Parameters
    ----------
    case
        One named synthetic acceptance case; default deliberately exposes unknown pricing.

    Returns
    -------
    dict[str, object]
        Read-only v1 snapshot of a synthetic candidate and actual native policy verdict.
        No adapter is registered, SDK loaded or provider called.

    Raises
    ------
    ValueError
        If the synthetic case name is unsupported.

    """
    if case not in EXAMPLE_CASES:
        raise ValueError("unknown operator policy example")
    profile = BackendProfile(
        "synthetic-policy",
        "synthetic",
        "direct",
        "gate_model",
        "none",
        ("openqasm3",),
        BackendCapabilities(True, True, False, False, False, False),
        is_cloud=True,
        submit_requires_approval=True,
        region="eu-north1",
    )
    policy = OperatorPolicy(
        "synthetic-conformance-only",
        profile.backend_id,
        ("synthetic-device",),
        ("eu-north1",),
        1024,
        2,
        60000,
        "12.50",
        "USD",
        "2026-10-03T00:00:00Z",
        "2026-10-05T00:00:00Z",
    )
    if case == "expired_policy":
        policy = replace(policy, expires_at="2026-10-04T00:00:00Z")
    source = "OPENQASM 3.0; qubit q; bit c; c = measure q;"
    semantics = WorkloadSemantics(
        hashlib.sha256(source.encode()).hexdigest(),
        1,
        1,
        ((0, 0),),
        (("c", (0,)),),
        requested_target="synthetic-device",
    )
    shots = 1025 if case == "over_shots" else 1024
    work = QuantumWorkload(
        "synthetic-operator-plan", "openqasm3", source, 1, shots, semantics=semantics
    )
    values: dict[str, object] = {
        "backend": profile.backend_id,
        "device": "synthetic-device",
        "shots": shots,
        "region": "us-east1" if case == "wrong_region" else "eu-north1",
        "concurrency": 3 if case == "over_concurrency" else 2,
        "time_limit_ms": 60001 if case == "over_time" else 60000,
        "unattended": True,
        "seed": 9007199254740993,
        "theme": "dark",
    }
    # This is a candidate awaiting core admission, not a claim that F08 admitted
    # an over-ceiling layer. The original structural contract preserves its values.
    settings = ResolvedSettings(
        {
            "requested": values,
            "effective": values,
            "origins": {key: "run" for key in values},
            "policy_ref": {
                "schema": "operator_policy.v1",
                "sha256": canonical_digest("operator_policy.v1", policy.to_dict()),
                "media_type": "application/json",
            },
            "environment_ref": {
                "schema": "synthetic_environment.v1",
                "sha256": "0" * 64,
                "media_type": "application/json",
            },
            "rejected_fields": [],
        }
    )
    request = operator_request_from_settings(work, settings)
    estimate = PricingEstimate(
        request.sha256,
        None if case == "unknown_price" else "12.500000001" if case == "over_cost" else "12.50",
        "USD",
        "synthetic-conformance-only",
        "2026-10-03T00:00:00Z",
        "2026-10-05T00:00:00Z",
    )
    return assess_workspace_operator_policy(
        HardwareAbstractionLayer([profile], operator_policy=policy),
        work,
        settings,
        estimate=estimate,
        now="2026-10-04T00:00:00Z",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Write a new exact synthetic snapshot or check existing committed bytes.

    Parameters
    ----------
    argv
        Explicit command arguments, or the normal process command line.

    Returns
    -------
    int
        Zero on exact export/check, one on missing or stale committed bytes.

    Raises
    ------
    FileExistsError
        If export would overwrite existing evidence; check mode is read-only.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=EXAMPLE_CASES, default="unknown_price")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = write_json(build_operator_policy_example(args.case)) + "\n"
    if args.check:
        return int(not args.output.is_file() or args.output.read_text(encoding="utf-8") != text)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
