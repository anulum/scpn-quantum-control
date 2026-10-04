# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — operator policy submission contracts
"""Exercise operator admission through the original public HAL boundary."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import cast

import pytest

from scpn_quantum_control.hardware.hal import (
    BackendCapabilities,
    BackendProfile,
    HardwareAbstractionLayer,
    LocalDeterministicSimulator,
    QuantumJobRef,
    QuantumJobResult,
    QuantumWorkload,
)
from scpn_quantum_control.hardware.operator_policy import workload_fingerprint
from scpn_quantum_control.hardware.operator_policy_contracts import (
    OperatorPolicy,
    OperatorPolicyRefused,
    OperatorRequest,
    PricingEstimate,
)
from scpn_quantum_control.hardware.provider_semantics import WorkloadSemantics

NOW = "2026-10-04T00:00:00Z"


def policy(**changes: object) -> OperatorPolicy:
    """Create a bounded test policy with independent literal ceilings."""
    values: dict[str, object] = dict(
        reference="test-policy",
        backend_id="policy-cloud",
        targets=("pinned-device",),
        regions=("eu-north1",),
        max_shots=1024,
        max_concurrency=2,
        max_time_limit_ms=60000,
        max_cost="12.50",
        currency="USD",
        valid_from="2020-01-01T00:00:00Z",
        expires_at="2099-01-01T00:00:00Z",
    )
    values.update(changes)
    return OperatorPolicy.from_dict(values)


def workload(shots: int = 1024) -> QuantumWorkload:
    """Pin actual source and target without loading a provider SDK."""
    source = "OPENQASM 3.0; qubit q; bit c; c = measure q;"
    semantics = WorkloadSemantics(
        hashlib.sha256(source.encode()).hexdigest(),
        1,
        1,
        ((0, 0),),
        (("c", (0,)),),
        requested_target="pinned-device",
    )
    return QuantumWorkload("policy-workload", "openqasm3", source, 1, shots, semantics=semantics)


def request(work: QuantumWorkload, **changes: object) -> OperatorRequest:
    """Declare the exact workload and operational plan without inferred values."""
    values: dict[str, object] = dict(
        workload_sha256=workload_fingerprint(work),
        backend_id="policy-cloud",
        target="pinned-device",
        region="eu-north1",
        shots=work.shots,
        concurrency=2,
        time_limit_ms=60000,
        unattended=True,
    )
    values.update(changes)
    return OperatorRequest.from_dict(values)


def price(plan: OperatorRequest, **changes: object) -> PricingEstimate:
    """Supply dated synthetic prices, never a live provider observation."""
    values: dict[str, object] = dict(
        request_sha256=plan.sha256,
        amount="12.50",
        currency="USD",
        source_ref="synthetic-only",
        observed_at="2020-01-01T00:00:00Z",
        expires_at="2099-01-01T00:00:00Z",
    )
    values.update(changes)
    return PricingEstimate.from_dict(values)


class RecordedAdapter:
    """Observe the actual HAL adapter boundary without a network or provider SDK."""

    backend_id = "policy-cloud"
    supports_provider_semantics = True

    def __init__(self) -> None:
        """Preserve every actually dispatched workload."""
        self.calls: list[QuantumWorkload] = []

    def submit(self, work: QuantumWorkload, *, approval_id: str | None = None) -> QuantumJobRef:
        """Record the literal request without returning fabricated device counts."""
        self.calls.append(work)
        return QuantumJobRef("recorded-job", self.backend_id, work.workload_id, "queued")

    def status(self, job: QuantumJobRef) -> str:
        """Return the original handle's observed lifecycle annotation."""
        return job.status

    def result(self, job: QuantumJobRef) -> QuantumJobResult:
        """Retain a pending handle with no observed sample total."""
        return QuantumJobResult(job, "queued", {}, 0)

    def cancel(self, job: QuantumJobRef) -> QuantumJobRef:
        """Retain identity without inventing confirmed cancellation."""
        return job


def route(
    restrictions: OperatorPolicy | None = None,
) -> tuple[HardwareAbstractionLayer, RecordedAdapter]:
    """Register one explicit cloud declaration and an observed adapter boundary."""
    profile = BackendProfile(
        backend_id="policy-cloud",
        provider="synthetic",
        broker="direct",
        modality="gate_model",
        sdk_package="none",
        ir_formats=("openqasm3",),
        capabilities=BackendCapabilities(True, True, False, False, False, False),
        is_cloud=True,
        region="eu-north1",
        submit_requires_approval=True,
    )
    hal = HardwareAbstractionLayer([profile], operator_policy=restrictions or policy())
    backend = RecordedAdapter()
    hal.register_backend(backend)
    return hal, backend


@pytest.mark.parametrize(
    "shots,amount,allowed",
    [(1024, "12.50", True), (1025, "12.50", False), (1024, "12.500000001", False)],
)
def test_operator_policy_decisions_01(shots: int, amount: str, allowed: bool) -> None:
    """Equality preserves shots and price; either ceiling excess stops transport."""
    hal, adapter = route()
    work = workload(shots)
    plan = request(work)
    estimate = price(plan, amount=amount)
    decision = hal.assess_operator_policy(
        adapter.backend_id, work, plan, estimate=estimate, now=NOW
    )
    assert decision.allowed is allowed
    assert decision.request.shots == work.shots == shots
    assert decision.estimate is not None and decision.estimate.amount == amount
    if allowed:
        hal.submit(
            adapter.backend_id,
            work,
            approval_id="explicit",
            operator_request=plan,
            pricing_estimate=estimate,
        )
        assert adapter.calls == [work]
    else:
        with pytest.raises(OperatorPolicyRefused) as error:
            hal.submit(
                adapter.backend_id,
                work,
                approval_id="explicit",
                operator_request=plan,
                pricing_estimate=estimate,
            )
        assert error.value.decision.request == plan
        assert adapter.calls == []


def test_operator_policy_decisions_02() -> None:
    """Region incompatibility stops before the actual adapter entry point."""
    hal, adapter = route()
    work = workload()
    plan = request(work, region="us-east1")
    with pytest.raises(OperatorPolicyRefused) as error:
        hal.submit(
            adapter.backend_id,
            work,
            approval_id="explicit",
            operator_request=plan,
            pricing_estimate=price(plan),
        )
    assert "region_forbidden" in error.value.decision.reasons
    assert "profile_region_mismatch" in error.value.decision.rejected_substitutions
    assert adapter.calls == []


def test_operator_policy_decisions_03() -> None:
    """An old passing verdict cannot authorise fresh submit after policy expiry."""
    hal, adapter = route(
        policy(valid_from="2020-01-01T00:00:00Z", expires_at="2021-01-01T00:00:00Z")
    )
    work = workload()
    plan = request(work)
    estimate = price(plan, observed_at="2020-01-01T00:00:00Z", expires_at="2022-01-01T00:00:00Z")
    historical = hal.assess_operator_policy(
        adapter.backend_id, work, plan, estimate=estimate, now="2020-06-01T00:00:00Z"
    )
    assert historical.allowed
    with pytest.raises(OperatorPolicyRefused) as error:
        hal.submit(
            adapter.backend_id,
            work,
            approval_id="explicit",
            operator_request=historical.request,
            pricing_estimate=historical.estimate,
        )
    assert "policy_expired" in error.value.decision.reasons
    assert adapter.calls == []


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"concurrency": 3}, "concurrency_ceiling"),
        ({"time_limit_ms": 60001}, "time_limit_ms_ceiling"),
        ({"backend_id": "substitute"}, "backend_forbidden"),
        ({"target": "other"}, "target_forbidden"),
        ({"shots": 2}, "workload_shots_mismatch"),
        ({"workload_sha256": "0" * 64}, "workload_source_mismatch"),
        ({"target": None}, "target_unbound"),
        ({"region": None}, "region_unknown"),
    ],
)
def test_operational_plan_refusal(changes: dict[str, object], reason: str) -> None:
    """Altered workload, route or operational limits never reach transport."""
    hal, adapter = route()
    work = workload()
    plan = request(work, **changes)
    with pytest.raises(OperatorPolicyRefused) as error:
        hal.submit(
            adapter.backend_id,
            work,
            approval_id="explicit",
            operator_request=plan,
            pricing_estimate=price(plan),
        )
    assert reason in error.value.decision.reasons
    assert adapter.calls == []


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"amount": None}, "price_unknown"),
        ({"currency": "EUR"}, "price_currency_mismatch"),
        ({"request_sha256": "0" * 64}, "price_request_mismatch"),
        ({"expires_at": "2026-10-03T12:00:00Z"}, "price_expired"),
        ({"observed_at": "2098-01-01T00:00:00Z"}, "price_future"),
    ],
)
def test_dated_estimate_refusal(changes: dict[str, object], reason: str) -> None:
    """Unknown, stale or mismatched estimates cannot authorise external execution."""
    hal, adapter = route()
    work = workload()
    plan = request(work)
    with pytest.raises(OperatorPolicyRefused) as error:
        hal.submit(
            adapter.backend_id,
            work,
            approval_id="explicit",
            operator_request=plan,
            pricing_estimate=price(plan, **changes),
        )
    assert reason in error.value.decision.reasons
    assert adapter.calls == []


def test_configured_policy_cannot_be_omitted() -> None:
    """Omitting a request or known price cannot bypass configured admission."""
    hal, adapter = route()
    work = workload()
    with pytest.raises(PermissionError, match="operator policy"):
        hal.submit(adapter.backend_id, work, approval_id="explicit")
    plan = request(work)
    with pytest.raises(OperatorPolicyRefused, match="price_unknown"):
        hal.submit(adapter.backend_id, work, approval_id="explicit", operator_request=plan)
    assert adapter.calls == []


def test_legacy_local_simulation_and_opt_in_refusal() -> None:
    """Original local simulation remains real, with no implicit policy authority."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = LocalDeterministicSimulator(hal.profile("local_statevector"))
    hal.register_backend(adapter)
    work = QuantumWorkload("legacy", "mlir", "module {}", 1, 17)
    job = hal.submit(adapter.backend_id, work)
    assert hal.result(job).shots == 17
    with pytest.raises(PermissionError, match="operator policy"):
        hal.submit(adapter.backend_id, work, operator_request=request(work))


def test_native_input_types_and_future_policy_refuse() -> None:
    """Unvalidated objects and a future policy cannot masquerade as admission."""
    hal, adapter = route()
    work = workload()
    plan = request(work)
    with pytest.raises(TypeError, match="immutable native"):
        hal.assess_operator_policy(adapter.backend_id, work, cast(OperatorRequest, {}), now=NOW)
    with pytest.raises(TypeError, match="PricingEstimate"):
        hal.assess_operator_policy(
            adapter.backend_id, work, plan, estimate=cast(PricingEstimate, {}), now=NOW
        )
    future, _ = route(policy(valid_from="2098-01-01T00:00:00Z"))
    decision = future.assess_operator_policy(
        adapter.backend_id, work, plan, estimate=price(plan), now=NOW
    )
    assert "policy_future" in decision.reasons
    with pytest.raises(TypeError, match="immutable OperatorPolicy"):
        HardwareAbstractionLayer([], operator_policy=cast(OperatorPolicy, {}))


def test_actual_local_policy_with_explicit_non_geographic_region() -> None:
    """Literal local plans admit without inventing provider geography or device pins."""
    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("local_statevector")
    restrictions = policy(backend_id=profile.backend_id, targets=(profile.backend_id,))
    hal = HardwareAbstractionLayer([profile], operator_policy=restrictions)
    adapter = LocalDeterministicSimulator(profile)
    hal.register_backend(adapter)
    work = QuantumWorkload("local-policy", "mlir", "module {}", 1, 17)
    plan = request(
        work,
        backend_id=profile.backend_id,
        target=profile.backend_id,
        region=None,
        shots=17,
        unattended=False,
    )
    estimate = price(plan, amount="0")
    decision = hal.assess_operator_policy(
        profile.backend_id, work, plan, estimate=estimate, now=NOW
    )
    assert decision.allowed and decision.request.region is None
    result = hal.result(
        hal.submit(profile.backend_id, work, operator_request=plan, pricing_estimate=estimate)
    )
    assert result.shots == 17
    substituted = replace(plan, target="different", region="different")
    refused = hal.assess_operator_policy(
        profile.backend_id, work, substituted, estimate=price(substituted), now=NOW
    )
    assert "native_target_mismatch" in refused.rejected_substitutions
    assert "profile_region_mismatch" in refused.rejected_substitutions


def test_cloud_native_target_and_region_unknown_remain_refused() -> None:
    """Unknown original transport bindings cannot be supplied only by an operator label."""
    hal, adapter = route()
    work = replace(workload(), semantics=None)
    plan = request(work)
    decision = hal.assess_operator_policy(
        adapter.backend_id, work, plan, estimate=price(plan), now=NOW
    )
    assert "target_unbound" in decision.reasons
    with pytest.raises(PermissionError, match="operator policy"):
        HardwareAbstractionLayer(hal.list_profiles()).assess_operator_policy(
            adapter.backend_id, work, plan, now=NOW
        )


def test_native_profile_capacity_precedes_operator_policy_and_transport() -> None:
    """A passing operator plan cannot override the original HAL's native shot ceiling."""
    original, adapter = route()
    profile = original.profile(adapter.backend_id)
    limited = replace(profile, capabilities=replace(profile.capabilities, max_shots=1023))
    hal = HardwareAbstractionLayer([limited], operator_policy=policy())
    hal.register_backend(adapter)
    work = workload()
    plan = request(work)
    with pytest.raises(ValueError, match="workload shots exceed"):
        hal.submit(
            adapter.backend_id,
            work,
            approval_id="explicit",
            operator_request=plan,
            pricing_estimate=price(plan),
        )
    assert adapter.calls == [] and work.shots == 1024
