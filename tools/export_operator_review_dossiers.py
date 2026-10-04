# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — offline native operator review exports
"""Export synthetic native review evidence without a provider or price service."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import cast

from scpn_quantum_control.canonical_encoding import canonical_digest
from scpn_quantum_control.hardware.hal import (
    BackendCapabilities,
    BackendProfile,
    HardwareAbstractionLayer,
    QuantumWorkload,
)
from scpn_quantum_control.hardware.operator_policy_contracts import (
    OperatorPolicy,
    PricingEstimate,
    utc_second,
)
from scpn_quantum_control.hardware.provider_semantics import WorkloadSemantics
from scpn_quantum_control.studio.executive import ExecutiveRequest
from scpn_quantum_control.studio.executive_execute import ExecuteActionHandler
from scpn_quantum_control.studio.workspace import (
    ResolvedSettings,
    operator_request_from_settings,
    write_json,
)
from tools.export_operator_policy_decisions import build_operator_policy_example

EXAMPLE_CASES = (
    "pending",
    "unknown_price",
    "expired",
    "expires_soon",
    "theme_light",
    "changed_payload",
    "changed_target",
    "changed_shots",
    "changed_calibration",
    "changed_price",
    "changed_expiry",
    "unknown_calibration",
    "declared_shot_capacity",
)
"""Finite synthetic conformance inputs, never prices, calibration or account observations."""


def build_operator_review_example(
    case: str = "pending", *, as_of: str = "2026-10-04T00:00:00Z"
) -> dict[str, object]:
    """Produce native source, policy and verifier through the original public handler.

    Parameters
    ----------
    case
        Named synthetic acceptance input; all provider and calibration labels are synthetic.
    as_of
        Explicit replay UTC seconds; default freezes the committed example identity.
        Supplying a current clock produces fresh synthetic test evidence, not live pricing.

    Returns
    -------
    dict[str, object]
        Sealed export with original dossier text and its complete native no-submit verifier.

    Raises
    ------
    ValueError
        If the case or exact UTC spelling is unsupported. Nothing is persisted.

    """
    if case not in EXAMPLE_CASES:
        raise ValueError("unknown operator review example")
    instant = utc_second(as_of)
    before = (instant - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    after = (instant + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    source = build_operator_policy_example("at_ceiling")
    original = cast(Mapping[str, object], source["body"])
    raw_settings = cast(Mapping[str, object], original["settings"])
    settings_body = ResolvedSettings(cast(Mapping[str, object], raw_settings["body"])).to_dict()[
        "body"
    ]
    assert isinstance(settings_body, dict)
    verdict = cast(Mapping[str, object], original["decision"])
    policy = replace(
        OperatorPolicy.from_dict(cast(Mapping[str, object], verdict["policy"])),
        targets=("synthetic-device", "synthetic-device-b"),
        valid_from=before,
        expires_at=after,
    )
    target = "synthetic-device-b" if case == "changed_target" else "synthetic-device"
    shots = 512 if case == "changed_shots" else 1024
    for field in ("requested", "effective"):
        values = settings_body[field]
        assert isinstance(values, dict)
        values.update(
            device=target, shots=shots, theme="light" if case == "theme_light" else "dark"
        )
    settings_body["policy_ref"] = {
        "schema": "operator_policy.v1",
        "sha256": canonical_digest("operator_policy.v1", policy.to_dict()),
        "media_type": "application/json",
    }
    settings = ResolvedSettings(settings_body)
    profile = BackendProfile(
        policy.backend_id,
        "synthetic",
        "direct",
        "gate_model",
        "none",
        ("openqasm3",),
        BackendCapabilities(
            True,
            True,
            False,
            False,
            False,
            False,
            max_shots=2048 if case == "declared_shot_capacity" else None,
        ),
        is_cloud=True,
        submit_requires_approval=True,
        region="eu-north1",
    )
    program = "OPENQASM 3.0; qubit q; bit c; c = measure q;"
    work = QuantumWorkload(
        "synthetic-review-plan",
        "openqasm3",
        program,
        1,
        shots,
        semantics=WorkloadSemantics(
            hashlib.sha256(program.encode()).hexdigest(),
            1,
            1,
            ((0, 0),),
            (("c", (0,)),),
            requested_target=target,
        ),
    )
    operational = operator_request_from_settings(work, settings)
    estimate = PricingEstimate(
        operational.sha256,
        None if case == "unknown_price" else "12.40" if case == "changed_price" else "12.50",
        "USD",
        "synthetic-conformance-only",
        before,
        after,
    )
    decision = HardwareAbstractionLayer([profile], operator_policy=policy).assess_operator_policy(
        profile.backend_id,
        work,
        operational,
        estimate=estimate,
        now=as_of,
    )
    payload = (
        "compiled synthetic payload: "
        + target
        + (" changed" if case == "changed_payload" else "")
        + "\n"
    ).encode()
    parameters: dict[str, object] = {
        "provider": profile.provider,
        "endpoint": target,
        "shots": shots,
        "circuit_digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "circuit_ref": "synthetic/compiled-payload",
    }
    calibration: Mapping[str, object] | None = None
    if case != "unknown_calibration":
        parameters["calibration_ref"] = "synthetic/calibration"
        calibration = {
            "reference": "synthetic/calibration",
            "sha256": ("d" if case == "changed_calibration" else "c") * 64,
            "target": target,
            "observed_at": before,
            "expires_at": after,
        }
    expiry = (
        (instant + timedelta(seconds=1 if case == "expired" else 10)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        if case in ("expired", "expires_soon")
        else (instant + timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ")
        if case == "changed_expiry"
        else after
    )
    dossier = ExecuteActionHandler().prepare_review(
        ExecutiveRequest(
            "execute", "synthetic-operator-review", parameters, backend="provider-hal"
        ),
        profile=profile,
        workload=work,
        compiled_payload=payload,
        settings=settings,
        policy_decision=decision,
        calibration=calibration,
        created_at=as_of,
        expires_at=expiry,
    )
    return dossier.export_bundle()


def main(argv: Sequence[str] | None = None) -> int:
    """Write a new native export or check committed bytes without overwriting evidence.

    Parameters
    ----------
    argv
        Explicit arguments or the process command line.

    Returns
    -------
    int
        Zero on exact export/check, one on missing or stale checked source.

    Raises
    ------
    FileExistsError
        If writing would overwrite existing evidence; check mode is read-only.
    ValueError
        If synthetic source dates or fields are invalid.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=EXAMPLE_CASES, default="pending")
    parser.add_argument("--as-of", default="2026-10-04T00:00:00Z")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = write_json(build_operator_review_example(args.case, as_of=args.as_of)) + "\n"
    if args.check:
        return int(not args.output.is_file() or args.output.read_text(encoding="utf-8") != text)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
