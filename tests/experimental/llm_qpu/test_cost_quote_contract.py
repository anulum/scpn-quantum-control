# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental cost quote contract
"""Exercise the provider-free cost quote wire against its bundle and target."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control.experimental.llm_qpu.contracts import (
    COMPILED_BUNDLE_SCHEMA,
    COST_QUOTE_SCHEMA,
    TARGET_SNAPSHOT_SCHEMA,
    ArtifactHeader,
    CompiledBundle,
    CompiledCell,
    CostQuote,
    QuotedCount,
    ReportedValue,
    TargetSnapshot,
    canonical_bytes,
    decode_contract,
    decode_cost_quote,
    validate_cost_quote,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKER = _REPO_ROOT / "experimental_workers/llm_qpu/protocol/worker.py"
_BASE = "de259e4837a92ecb09b63c8f4332dbcf3d21021c"
_REVISION = "0883d1e5204ffe4594bb2f5b6e7b6a5d0a915209"
_ORIGIN = {
    "synthetic_fixture": "synthetic_classical",
    "provider_target_query": "provider_metadata",
}
_COUNTS = ("job_count", "overhead_executions", "resource_units", "price_minor_units")


def _header(
    kind: str, content: dict[str, object], parents: tuple[str, ...], origin: str
) -> ArtifactHeader:
    """Return a design-only header that hashes exactly the given content."""
    return ArtifactHeader(
        object_kind=kind,
        content_digest=hashlib.sha256(canonical_bytes(content)).hexdigest(),
        parents=tuple(sorted(parents)),
        base_repo_commit=_BASE,
        implementation_revision=_REVISION,
        execution_origin="offline_design",
        data_origin=origin,
        claim_scope="design_only",
    )


def _target(
    provenance: str = "synthetic_fixture", *, batch_limit: ReportedValue | None = None
) -> TargetSnapshot:
    """Build a three-qubit line target, optionally with another circuit batch limit."""
    device = "garnet"
    known = {
        "calibration_id": ReportedValue("known", "cal-2026-11-15-0800", None),
        "calibration_time_utc": ReportedValue("known", "2026-11-15T08:00:00Z", None),
        "max_shots": ReportedValue("known", 100_000, None),
        "max_circuits_per_job": batch_limit or ReportedValue("known", 1, None),
    }
    plain: dict[str, Any] = {
        "provider": "iqm",
        "account_route": "iqm-resonance/owner-project",
        "device_name": device,
        "sdk_name": "iqm-client",
        "sdk_version": "35.0.3",
        "sdk_target_digest": "a" * 64,
        "capability_provenance": provenance,
        "num_qubits": 3,
    }
    content: dict[str, object] = {
        "schema": TARGET_SNAPSHOT_SCHEMA,
        "object_kind": "target_snapshot",
        **plain,
        "basis_gates": ["cz", "measure", "prx"],
        "coupling_edges": [[0, 1], [1, 2]],
        **{name: value.to_wire() for name, value in known.items()},
    }
    header = _header("target_snapshot", content, ("a" * 64,), _ORIGIN[provenance])
    return TargetSnapshot(
        **plain,
        basis_gates=("cz", "measure", "prx"),
        coupling_edges=((0, 1), (1, 2)),
        calibration_id=known["calibration_id"],
        calibration_time_utc=known["calibration_time_utc"],
        max_shots=known["max_shots"],
        max_circuits_per_job=known["max_circuits_per_job"],
        header=header,
    )


def _payload(fields: dict[str, Any]) -> dict[str, object]:
    """Return the approvable payload as an independent oracle would serialise it."""
    return {
        "cells": [
            cell.to_wire() if type(cell) is CompiledCell else cell for cell in fields["cells"]
        ],
        "layout": list(fields["layout"]),
        "logical_gates": list(fields["logical_gates"]),
        "compiled_gate_counts": [list(pair) for pair in fields["compiled_gate_counts"]],
        "substitutions": [[source, list(targets)] for source, targets in fields["substitutions"]],
        "compiled_depth": fields["compiled_depth"],
        "provider_options": [list(pair) for pair in fields["provider_options"]],
    }


def _bundle(
    target: TargetSnapshot | None = None, *, device: str | None = None, **changes: Any
) -> CompiledBundle:
    """Build a bundle that compiles cx and rz into the target's cz and prx basis."""
    target = target or _target()
    fields: dict[str, Any] = {
        "cells": (
            CompiledCell("1" * 64, "2" * 64, "3" * 64, "4" * 64, 1000),
            CompiledCell("5" * 64, "6" * 64, "7" * 64, "8" * 64, 2000),
        ),
        "layout": (2, 0, 1),
        "logical_gates": ("cx", "measure", "rz"),
        "compiled_gate_counts": (("cz", 4), ("measure", 3), ("prx", 10)),
        "substitutions": (("cx", ("cz", "prx")), ("rz", ("prx",))),
        "compiled_depth": 12,
        "provider_options": (
            ("layout_method", "sabre"),
            ("optimization_level", 1),
            ("readout_mitigation", False),
        ),
        **changes,
    }
    payload_digest = hashlib.sha256(canonical_bytes(_payload(fields))).hexdigest()
    identity: dict[str, Any] = {
        "target_snapshot_digest": hashlib.sha256(canonical_bytes(target.to_wire())).hexdigest(),
        "provider": target.provider,
        "device_name": device or target.device_name,
        "sdk_lock_digest": "b" * 64,
        "compile_log_digest": "c" * 64,
        "payload_digest": payload_digest,
    }
    content = {
        "schema": COMPILED_BUNDLE_SCHEMA,
        "object_kind": "compiled_bundle",
        **identity,
        **_payload(fields),
    }
    parents = (identity["target_snapshot_digest"], "b" * 64, "c" * 64)
    header = _header("compiled_bundle", content, parents, target.header.data_origin)
    return CompiledBundle(**identity, **fields, header=header)


def _worker(request: dict[str, object], tmp_path: Path) -> subprocess.CompletedProcess[bytes]:
    """Run the standalone protocol worker on one request with poisoned provider tokens."""
    return subprocess.run(
        [sys.executable, "-S", str(_WORKER)],
        input=json.dumps(request).encode(),
        cwd=tmp_path,
        env={"PYTHONPATH": str(tmp_path), "IQM_TOKEN": "poison", "IBM_QUANTUM_TOKEN": "poison"},
        capture_output=True,
        timeout=15,
        check=False,
    )


def _known(value: int) -> QuotedCount:
    """Return a known count."""
    return QuotedCount("known", value, None)


def _quote(bundle: CompiledBundle, **changes: Any) -> CostQuote:
    """Build a quote for the bundle: one cell per job, so two jobs, priced in EUR cents."""
    fields: dict[str, Any] = {
        "compiled_bundle_digest": hashlib.sha256(canonical_bytes(bundle.to_wire())).hexdigest(),
        "bundle_payload_digest": bundle.payload_digest,
        "circuit_count": len(bundle.cells),
        "total_shots": sum(cell.shots for cell in bundle.cells),
        "quoted_options": tuple(name for name, _ in bundle.provider_options),
        "job_count": _known(2),
        "overhead_executions": _known(0),
        "resource_unit": "iqm_credit",
        "resource_derivation": "published_rate_card",
        "resource_units": _known(45),
        "price_minor_units": _known(4500),
        "currency": "EUR",
        "pricing_source": "IQM Resonance price list, read 2026-11-15",
        "priced_at_utc": "2026-11-15T09:00:00Z",
        **changes,
    }
    content = {
        "schema": COST_QUOTE_SCHEMA,
        "object_kind": "cost_quote",
        **{
            name: value.to_wire()
            if type(value) is QuotedCount
            else list(value)
            if name == "quoted_options" and type(value) is tuple
            else value
            for name, value in fields.items()
        },
    }
    parents = (fields["compiled_bundle_digest"],)
    header = _header("cost_quote", content, parents, bundle.header.data_origin)
    return CostQuote(**fields, header=header)


@pytest.mark.parametrize("provenance", ["synthetic_fixture", "provider_target_query"])
def test_cost_quote_roundtrips_and_validates_against_its_bundle(
    provenance: str, tmp_path: Path
) -> None:
    """A quote with exact totals round-trips through the worker and validates."""
    target = _target(provenance)
    bundle = _bundle(target)
    quote = _quote(bundle)
    request: dict[str, object] = {
        "op": "roundtrip_cost_quote",
        "target": target.to_wire(),
        "bundle": bundle.to_wire(),
        "quote": quote.to_wire(),
    }

    response = _worker(request, tmp_path)

    assert response.returncode == 0, response.stdout.decode()
    payload = json.loads(response.stdout)
    assert payload["status"] == "validated_roundtrip_no_compute"
    assert payload["hardware_submission_enabled"] is False
    assert payload["quote_sha256"] == hashlib.sha256(canonical_bytes(quote.to_wire())).hexdigest()
    assert decode_cost_quote(canonical_bytes(payload["quote"])) == quote
    validate_cost_quote(quote, bundle, target)
    assert quote.blockers() == ()


def test_cost_quote_rounds_partial_batches_up_to_whole_jobs(tmp_path: Path) -> None:
    """Two cells under a batch limit of 200 are one job, never zero."""
    target = _target(batch_limit=ReportedValue("known", 200, None))
    bundle = _bundle(target)
    quote = _quote(bundle, job_count=_known(1))

    validate_cost_quote(quote, bundle, target)
    request: dict[str, object] = {
        "op": "roundtrip_cost_quote",
        "target": target.to_wire(),
        "bundle": bundle.to_wire(),
        "quote": quote.to_wire(),
    }
    response = _worker(request, tmp_path)
    assert response.returncode == 0, response.stdout.decode()


def test_cost_quote_keeps_unknown_totals_and_prices_unknown() -> None:
    """Without a reported batch limit the job count stays unknown, as does an unquoted price."""
    target = _target(batch_limit=ReportedValue("unknown", None, "the SDK reports no batch limit"))
    bundle = _bundle(target)
    unknown_jobs = QuotedCount("unknown", None, "the batch limit is not reported")
    unknown_price = QuotedCount("unknown", None, "no price list covers this device")
    quote = _quote(
        bundle, job_count=unknown_jobs, price_minor_units=unknown_price, priced_at_utc=None
    )

    validate_cost_quote(quote, bundle, target)
    assert quote.blockers() == (
        ("job_count", "the batch limit is not reported"),
        ("price_minor_units", "no price list covers this device"),
    )
    with pytest.raises(
        ValueError, match="^a job count cannot be known while the batch limit is unknown$"
    ):
        validate_cost_quote(_quote(bundle), bundle, target)
    with pytest.raises(
        ValueError, match="^an unknown count cannot carry a value; zero is not unknown$"
    ):
        QuotedCount("unknown", 0, "not quoted")


def test_cost_quote_refuses_shots_as_credits_untimed_prices_and_bad_counts() -> None:
    """Credits are never shots, a known price is timed and sourced, counts are typed."""
    bundle = _bundle()
    for changes, reason in (
        ({"resource_unit": "Shots"}, "^provider credits are not shots$"),
        ({"resource_derivation": "shots_as_credits"}, "^provider credits are not shots$"),
        ({"resource_derivation": "guess"}, "^unknown resource derivation$"),
        ({"currency": "eur"}, "^currency must be an ISO 4217 code$"),
        ({"priced_at_utc": None}, "^a known price needs its ISO-8601 UTC quote time$"),
        (
            {"priced_at_utc": "2026-11-15 09:00"},
            "^a known price needs its ISO-8601 UTC quote time$",
        ),
        ({"job_count": _known(0)}, "^a known job count must be positive$"),
        (
            {"quoted_options": ("optimization_level", "layout_method")},
            "^quoted options must be sorted and unique$",
        ),
        ({"resource_units": 45}, "^resource_units must be a QuotedCount$"),
        ({"total_shots": 0}, "quoted total shots"),
    ):
        with pytest.raises(ValueError, match=reason):
            _quote(bundle, **changes)
    unknown_price = QuotedCount("unknown", None, "no price list covers this device")
    with pytest.raises(ValueError, match="^an unknown price cannot carry a quote time$"):
        _quote(bundle, price_minor_units=unknown_price)
    valid = _quote(bundle)
    option_list: Any = ["layout_method"]
    with pytest.raises(ValueError, match="pricing source"):
        replace(valid, pricing_source="")
    with pytest.raises(ValueError, match="^quoted options must be a tuple$"):
        replace(valid, quoted_options=option_list)
    for state, value, blocker, reason in (
        ("known", -1, None, "^a known count must be a bounded non-negative integer$"),
        ("known", True, None, "^a known count must be a bounded non-negative integer$"),
        ("known", 3, "stale", "^a known count cannot carry a blocker$"),
        ("maybe", None, None, "^count state must be known or unknown$"),
    ):
        with pytest.raises(ValueError, match=reason):
            QuotedCount(state, value, blocker)
    quote = _quote(bundle)
    with pytest.raises(ValueError, match="^CostQuote requires its exact artifact header$"):
        replace(quote, header=replace(quote.header, object_kind="compiled_bundle"))
    with pytest.raises(
        ValueError, match="^a cost quote derives from provider metadata or a fixture$"
    ):
        replace(quote, header=replace(quote.header, data_origin="owner_dataset"))
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(quote, resource_units=_known(46))


def test_cost_quote_refuses_another_bundle_wrong_totals_and_hidden_options() -> None:
    """A quote must price this bundle exactly, name every option and chunk by the target."""
    target = _target()
    bundle = _bundle(target)
    other = _bundle(target, compiled_depth=13)
    for quote, reason in (
        (_quote(other), "^quote prices a different compiled bundle$"),
        (
            _quote(bundle, bundle_payload_digest="e" * 64),
            "^quote payload digest differs from the bundle$",
        ),
        (_quote(bundle, circuit_count=3), "^quoted circuit count differs from the bundle$"),
        (_quote(bundle, total_shots=2999), "^quoted total shots differ from the bundle$"),
        (
            _quote(bundle, quoted_options=("layout_method", "optimization_level")),
            "^quote does not name every provider option the bundle runs with$",
        ),
        (
            _quote(bundle, job_count=_known(1)),
            "^quoted job count differs from the target's circuit batch limit$",
        ),
    ):
        with pytest.raises(ValueError, match=reason):
            validate_cost_quote(quote, bundle, target)
    mixed = _quote(bundle)
    with pytest.raises(ValueError, match="^quote and bundle data origins differ$"):
        validate_cost_quote(
            replace(mixed, header=replace(mixed.header, data_origin="provider_metadata")),
            bundle,
            target,
        )
    not_a_quote: Any = "quote"
    with pytest.raises(ValueError, match="^quote validation needs a CostQuote$"):
        validate_cost_quote(not_a_quote, bundle, target)


def test_cost_quote_wire_is_exact_here_and_in_the_worker(tmp_path: Path) -> None:
    """Edited wires are refused by the decoder and the worker; decode_contract stays closed."""
    target = _target()
    bundle = _bundle(target)
    quote = _quote(bundle)
    with pytest.raises(ValueError, match="^unsupported contract schema$"):
        decode_contract(canonical_bytes(quote.to_wire()))
    wire_cases: tuple[tuple[str, object, str], ...] = (
        ("unexpected", True, "^CostQuote fields mismatch$"),
        ("schema", "scpn.experimental.llm_qpu.cost_quote.v2", "^unknown CostQuote schema$"),
        ("quoted_options", "layout_method", "^quoted options must be a list on wire$"),
        ("job_count", {"state": "known", "value": 2}, "^QuotedCount fields mismatch$"),
    )
    for name, value, reason in wire_cases:
        wire: dict[str, Any] = quote.to_wire()
        wire[name] = value
        with pytest.raises(ValueError, match=reason):
            decode_cost_quote(canonical_bytes(wire))
    worker_cases: tuple[tuple[tuple[str, ...], object, str], ...] = (
        (
            ("price_minor_units",),
            {"state": "unknown", "value": 0, "blocker": "x"},
            "zero is not unknown",
        ),
        (("resource_unit",), "shot", "credits are not shots"),
        (("priced_at_utc",), None, "needs its ISO-8601 UTC quote time"),
        (("header", "content_digest"), "f" * 64, "content digest mismatch"),
    )
    for path, value, reason in worker_cases:
        forged: dict[str, Any] = quote.to_wire()
        holder = forged
        for key in path[:-1]:
            holder = holder[key]
        holder[path[-1]] = value
        request: dict[str, object] = {
            "op": "roundtrip_cost_quote",
            "target": target.to_wire(),
            "bundle": bundle.to_wire(),
            "quote": forged,
        }
        response = _worker(request, tmp_path)
        assert response.returncode == 2, path
        assert reason in json.loads(response.stdout)["reason"], path
    for consistent, reason in (
        (
            _quote(bundle, quoted_options=("layout_method", "optimization_level")),
            "does not name every provider option",
        ),
        (_quote(bundle, job_count=_known(1)), "circuit batch limit"),
        (_quote(bundle, total_shots=2999), "total shots differ"),
    ):
        request = {
            "op": "roundtrip_cost_quote",
            "target": target.to_wire(),
            "bundle": bundle.to_wire(),
            "quote": consistent.to_wire(),
        }
        response = _worker(request, tmp_path)
        assert response.returncode == 2, reason
        assert reason in json.loads(response.stdout)["reason"], reason
