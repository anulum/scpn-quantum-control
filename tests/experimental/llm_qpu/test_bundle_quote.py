# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental offline bundle quote
"""Offline quotes carry exact totals and jobs; every provider-only figure stays an explicit blocker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _llm_qpu_dev_chain import _REVISIONS, _planned, _target, _worker

from scpn_quantum_control.experimental.llm_qpu.contracts import (
    CompiledBundle,
    TargetSnapshot,
    validate_cost_quote,
)
from scpn_quantum_control.experimental.llm_qpu.contracts.target import ReportedValue
from scpn_quantum_control.experimental.llm_qpu.data.bundle_quote import quote_bundle
from scpn_quantum_control.experimental.llm_qpu.data.compile_records import _bundle
from scpn_quantum_control.experimental.llm_qpu.data.target_compiler import compile_experiment


def _compiled(**target_changes: Any) -> tuple[CompiledBundle, TargetSnapshot]:
    """Return the dev experiment compiled on the synthetic target, and that target."""
    planned, target = _planned(), _target(**target_changes)
    compiled = compile_experiment(
        planned.experiment, planned.plan_set, target, seed=7, revisions=_REVISIONS
    )
    return compiled.bundle, target


def test_an_offline_quote_states_exact_totals_jobs_and_its_blockers(tmp_path: Path) -> None:
    """Totals and jobs are exact; resource units and price are unknown with a reason, never zero."""
    bundle, target = _compiled(max_circuits_per_job=ReportedValue("known", 5, None))
    quote = quote_bundle(
        bundle, target, resource_unit="qpu_second", currency="USD", revisions=_REVISIONS
    )
    validate_cost_quote(quote, bundle, target)
    assert (quote.circuit_count, quote.total_shots) == (
        16,
        sum(cell.shots for cell in bundle.cells),
    )
    assert quote.job_count.value == 4
    assert (quote.overhead_executions.state, quote.overhead_executions.value) == ("known", 0)
    assert quote.quoted_options == ("optimization_level", "seed_transpiler")
    assert quote.priced_at_utc is None
    assert (quote.resource_derivation, quote.pricing_source) == (
        "provider_estimate",
        "not_read_offline",
    )
    assert (quote.resource_unit, quote.currency) == ("qpu_second", "USD")
    assert [name for name, _ in quote.blockers()] == ["resource_units", "price_minor_units"]
    request: dict[str, object] = {
        "op": "roundtrip_cost_quote",
        "target": target.to_wire(),
        "bundle": bundle.to_wire(),
        "quote": quote.to_wire(),
    }
    response = _worker(request, tmp_path)
    assert response.returncode == 0, response.stdout.decode()
    assert json.loads(response.stdout)["status"] == "validated_roundtrip_no_compute"


def test_unreported_limits_and_execution_options_become_blockers() -> None:
    """An unknown batch limit leaves the job count unknown; an execution option leaves overhead unknown."""
    limit = ReportedValue("unknown", None, "the SDK does not report a circuit batch limit")
    bundle, target = _compiled(max_circuits_per_job=limit)
    quote = quote_bundle(
        bundle, target, resource_unit="credit", currency="EUR", revisions=_REVISIONS
    )
    assert quote.job_count.state == "unknown"
    assert "the SDK does not report a circuit batch limit" in str(quote.job_count.blocker)
    names = (
        "target_snapshot_digest",
        "provider",
        "device_name",
        "sdk_lock_digest",
        "cells",
        "layout",
        "logical_gates",
        "compiled_gate_counts",
        "substitutions",
        "compiled_depth",
        "compile_log_digest",
    )
    fields = {name: getattr(bundle, name) for name in names}
    options = (("dynamical_decoupling", True), *bundle.provider_options)
    probed = _bundle({**fields, "provider_options": options}, target, _REVISIONS)
    overhead = quote_bundle(
        probed, target, resource_unit="credit", currency="EUR", revisions=_REVISIONS
    )
    assert overhead.overhead_executions.state == "unknown"
    assert (
        overhead.overhead_executions.blocker
        == "execution options may add runs: dynamical_decoupling"
    )
    assert overhead.quoted_options == (
        "dynamical_decoupling",
        "optimization_level",
        "seed_transpiler",
    )


def test_an_offline_quote_refuses_shots_as_credits_and_a_malformed_currency() -> None:
    """The contract's refusals hold for an offline quote too."""
    bundle, target = _compiled()
    for unit, currency, reason in (
        ("shot", "USD", "^provider credits are not shots$"),
        ("qpu_second", "usd", "^currency must be an ISO 4217 code$"),
    ):
        with pytest.raises(ValueError, match=reason):
            quote_bundle(
                bundle, target, resource_unit=unit, currency=currency, revisions=_REVISIONS
            )
