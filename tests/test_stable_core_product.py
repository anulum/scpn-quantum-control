# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for stable_core product surface
"""Real-surface tests for ``scpn_quantum_control.stable_core_product``."""

from __future__ import annotations

from typing import Any, cast

import pytest

import scpn_quantum_control.stable_core_product as stable_core_product
from scpn_quantum_control.stable_core import (
    build_backend,
    build_problem,
    build_result,
    classical_reference_backend,
)
from scpn_quantum_control.stable_core_product import (
    STABLE_CORE_MODEL_SCHEMA_VERSION,
    STABLE_CORE_PRODUCT_CLAIM_BOUNDARY,
    STABLE_CORE_PRODUCT_SCHEMA,
    StableCoreContractRow,
    StableCoreRoundTripResult,
    assert_stable_core_product_integrity,
    backend_from_dict,
    build_demo_experiment,
    build_stable_core_product_registry,
    canonical_json_bytes,
    deserialise_backend,
    deserialise_experiment,
    deserialise_problem,
    deserialise_result,
    digest_stable_core_payload,
    experiment_from_dict,
    get_stable_core_contract,
    iter_stable_core_contracts,
    list_stable_core_contract_ids,
    map_stable_core_public_surfaces,
    problem_from_dict,
    result_from_dict,
    round_trip_experiment,
    round_trip_problem,
    schema_version_policy,
    serialise_backend,
    serialise_experiment,
    serialise_problem,
    serialise_result,
    unwrap_model_envelope,
    validate_model_schema_version,
    wrap_model_envelope,
)


def test_list_contracts_and_filters() -> None:
    """List contracts deterministically and filter them by contract kind."""
    ids = list_stable_core_contract_ids()
    assert "schema_policy" in ids
    assert "experiment_contract" in ids
    assert "problem_contract" in ids
    assert ids == list_stable_core_contract_ids()
    problems = iter_stable_core_contracts(kind="problem")
    assert len(problems) == 1
    assert problems[0].kind == "problem"


def test_get_known_and_unknown_fail_closed() -> None:
    """Resolve known contracts and reject blank or unknown identifiers."""
    row = get_stable_core_contract("experiment_contract")
    assert row.api_stability_class == "stable_core"
    assert row.claim_boundary == STABLE_CORE_PRODUCT_CLAIM_BOUNDARY
    assert row.reproduction_kit_pointer
    assert row.scorecard_pointer
    with pytest.raises(ValueError, match="non-empty"):
        get_stable_core_contract("  ")
    with pytest.raises(ValueError, match="unknown contract_id"):
        get_stable_core_contract("not_a_contract")


def test_schema_version_policy() -> None:
    """Expose the supported schema and refuse silent field drops."""
    policy = schema_version_policy()
    assert STABLE_CORE_MODEL_SCHEMA_VERSION == "stable_core.experiment_model.v2"
    assert STABLE_CORE_PRODUCT_SCHEMA == "stable_core_product.v2"
    assert policy["model_schema_version"] == STABLE_CORE_MODEL_SCHEMA_VERSION
    assert policy["silent_field_drop_allowed"] is False
    assert validate_model_schema_version(STABLE_CORE_MODEL_SCHEMA_VERSION)
    with pytest.raises(ValueError, match="non-empty"):
        validate_model_schema_version("")
    with pytest.raises(ValueError, match="unknown model schema_version"):
        validate_model_schema_version("stable_core.experiment_model.v1")


def test_problem_round_trip_and_digest() -> None:
    """Round-trip a problem with stable canonical payload and digest output."""
    problem = build_problem(
        problem_id="p1",
        coupling_matrix=((0.0, 0.2), (0.2, 0.0)),
        omega=(0.0, 0.1),
        initial_state="01",
    )
    result = round_trip_problem(problem)
    assert result.matched is True
    assert result.kind == "problem"
    assert len(result.digest_sha256) == 64
    assert result.digest_sha256 == digest_stable_core_payload(cast(dict[str, Any], result.payload))
    rebuilt = deserialise_problem(cast(dict[str, Any], result.payload))
    assert rebuilt.problem_id == "p1"
    assert rebuilt.n_qubits == 2


def test_experiment_round_trip_demo() -> None:
    """Round-trip the deterministic no-hardware demonstration experiment."""
    experiment = build_demo_experiment()
    assert experiment.backend.kind == "classical_reference"
    assert experiment.backend.hardware_submission_allowed is False
    result = round_trip_experiment(experiment)
    assert result.matched is True
    assert len(result.digest_sha256) == 64
    rebuilt = deserialise_experiment(cast(dict[str, Any], result.payload))
    assert rebuilt.experiment_id == experiment.experiment_id
    assert rebuilt.problem.problem_id == experiment.problem.problem_id
    assert rebuilt.objective == "order_parameter"


def test_backend_and_result_serialise() -> None:
    """Serialise and rebuild backend and result envelopes without field loss."""
    backend = classical_reference_backend()
    env = serialise_backend(backend)
    assert env["kind"] == "backend"
    assert deserialise_backend(env).backend_id == backend.backend_id

    res = build_result(
        experiment_id="e1",
        backend_id=backend.backend_id,
        status="succeeded",
        observables={"order_parameter": 0.5},
    )
    renv = serialise_result(res)
    assert deserialise_result(renv).observables["order_parameter"] == 0.5

    blocked = build_result(
        experiment_id="e2",
        backend_id=backend.backend_id,
        status="blocked",
        observables={},
        blockers=("missing dependency",),
    )
    assert deserialise_result(serialise_result(blocked)).status == "blocked"


def test_from_dict_fail_closed() -> None:
    """Reject malformed mappings across all stable-core contract builders."""
    with pytest.raises(ValueError, match="mapping"):
        problem_from_dict(cast(Any, "nope"))
    with pytest.raises(ValueError, match="problem_id"):
        problem_from_dict({"problem_id": "", "coupling_matrix": [[0.0]], "omega": [0.0]})
    with pytest.raises(ValueError, match="coupling_matrix"):
        problem_from_dict({"problem_id": "p", "coupling_matrix": [], "omega": [0.0]})
    with pytest.raises(ValueError, match="backend_id"):
        backend_from_dict({"backend_id": "", "kind": "classical_reference", "capabilities": ["x"]})
    with pytest.raises(ValueError, match="capabilities"):
        backend_from_dict(
            {
                "backend_id": "b",
                "kind": "classical_reference",
                "capabilities": [],
            }
        )
    with pytest.raises(ValueError, match="experiment_id"):
        experiment_from_dict(
            {
                "experiment_id": "",
                "problem": build_problem(
                    problem_id="p",
                    coupling_matrix=((0.0,),),
                    omega=(0.0,),
                ).to_dict(),
                "backend": classical_reference_backend().to_dict(),
                "objective": "order_parameter",
                "seed": 0,
            }
        )
    with pytest.raises(ValueError, match="seed must be an int"):
        experiment_from_dict(
            {
                "experiment_id": "e",
                "problem": build_problem(
                    problem_id="p",
                    coupling_matrix=((0.0,),),
                    omega=(0.0,),
                ).to_dict(),
                "backend": classical_reference_backend().to_dict(),
                "objective": "order_parameter",
                "seed": "7",
            }
        )
    with pytest.raises(ValueError, match="observables must be a mapping"):
        result_from_dict(
            {
                "experiment_id": "e",
                "backend_id": "b",
                "status": "succeeded",
                "observables": "nope",
            }
        )


def test_envelope_wrap_unwrap_fail_closed() -> None:
    """Validate envelope round-trips and reject invalid schema or body data."""
    body = build_problem(
        problem_id="p",
        coupling_matrix=((0.0,),),
        omega=(0.0,),
    ).to_dict()
    env = wrap_model_envelope("problem", body)
    assert set(env) == {"schema_version", "kind", "body", "claim_boundary"}
    assert env["claim_boundary"] == STABLE_CORE_PRODUCT_CLAIM_BOUNDARY
    version, kind, unwrapped = unwrap_model_envelope(env)
    assert version == STABLE_CORE_MODEL_SCHEMA_VERSION
    assert kind == "problem"
    assert unwrapped["problem_id"] == "p"
    with pytest.raises(ValueError, match="envelope kind"):
        wrap_model_envelope(cast(Any, "schema_policy"), body)
    with pytest.raises(ValueError, match="non-empty mapping"):
        wrap_model_envelope("problem", {})
    with pytest.raises(ValueError, match="unknown model schema"):
        unwrap_model_envelope(
            {
                "schema_version": "nope.v0",
                "kind": "problem",
                "body": body,
                "claim_boundary": STABLE_CORE_PRODUCT_CLAIM_BOUNDARY,
            }
        )
    with pytest.raises(ValueError, match="envelope key drift"):
        unwrap_model_envelope(
            {
                "schema_version": STABLE_CORE_MODEL_SCHEMA_VERSION,
                "kind": "problem",
                "body": body,
            }
        )
    with pytest.raises(ValueError, match="envelope key drift"):
        unwrap_model_envelope({**env, "unexpected": True})
    with pytest.raises(ValueError, match="claim_boundary drift"):
        unwrap_model_envelope({**env, "claim_boundary": "drifted"})
    with pytest.raises(ValueError, match="unknown envelope kind"):
        unwrap_model_envelope(
            {
                "schema_version": STABLE_CORE_MODEL_SCHEMA_VERSION,
                "kind": "ghost",
                "body": body,
                "claim_boundary": STABLE_CORE_PRODUCT_CLAIM_BOUNDARY,
            }
        )


def test_kind_mismatch_on_deserialise() -> None:
    """Reject envelopes whose declared kind differs from the target model."""
    problem = build_problem(
        problem_id="p",
        coupling_matrix=((0.0,),),
        omega=(0.0,),
    )
    env = serialise_problem(problem)
    with pytest.raises(ValueError, match="expected backend"):
        deserialise_backend(env)
    with pytest.raises(ValueError, match="expected experiment"):
        deserialise_experiment(env)
    with pytest.raises(ValueError, match="expected result"):
        deserialise_result(env)
    with pytest.raises(ValueError, match="expected problem"):
        deserialise_problem(serialise_backend(classical_reference_backend()))


def test_public_surfaces_and_registry() -> None:
    """Publish a complete deterministic surface map and product registry."""
    surfaces = map_stable_core_public_surfaces()
    assert surfaces
    symbols = {row["symbol_name"] for row in surfaces}
    assert "Problem" in symbols
    assert "Experiment" in symbols
    for row in surfaces:
        assert row["api_stability_class"] == "stable_core"
        assert row["role"] == "stable_core_product_surface"

    registry = build_stable_core_product_registry()
    assert registry["schema"] == STABLE_CORE_PRODUCT_SCHEMA
    assert registry["blank_entry_count"] == 0
    assert registry["default_contract_id"] == "experiment_contract"
    validated = assert_stable_core_product_integrity(registry)
    assert validated["contract_count"] == len(list_stable_core_contract_ids())
    assert assert_stable_core_product_integrity()["blank_entry_count"] == 0


def test_integrity_rejects_drift() -> None:
    """Reject missing, duplicate, or schema-policy registry drift."""
    registry = build_stable_core_product_registry()
    contracts = list(cast(list[dict[str, object]], registry["contracts"]))
    broken = dict(registry)
    broken["contracts"] = contracts + [
        {
            "contract_id": "ghost",
            "kind": "problem",
            "title": "t",
            "summary": "s",
            "module_path": "m",
            "symbol_name": "X",
            "api_stability_class": "stable_core",
            "reproduction_kit_pointer": "a",
            "scorecard_pointer": "b",
            "as_of": "2026-07-24",
            "claim_boundary": STABLE_CORE_PRODUCT_CLAIM_BOUNDARY,
        }
    ]
    broken["contract_count"] = len(cast(list[object], broken["contracts"]))
    with pytest.raises(ValueError, match="drift"):
        assert_stable_core_product_integrity(broken)

    empty: dict[str, object] = {"contracts": [], "blank_entry_count": 0, "contract_count": 0}
    with pytest.raises(ValueError, match="non-empty contracts"):
        assert_stable_core_product_integrity(empty)

    no_policy = dict(registry)
    no_policy["schema_policy"] = {"silent_field_drop_allowed": True}
    with pytest.raises(ValueError, match="silent field drops"):
        assert_stable_core_product_integrity(no_policy)

    extra_key = dict(registry)
    extra_key["unexpected"] = True
    with pytest.raises(ValueError, match="registry key drift"):
        assert_stable_core_product_integrity(extra_key)

    old_schema = dict(registry)
    old_schema["schema"] = "stable_core_product.v1"
    with pytest.raises(ValueError, match="registry schema drift"):
        assert_stable_core_product_integrity(old_schema)

    claim_drift = dict(registry)
    claim_drift["claim_boundary"] = "drifted"
    with pytest.raises(ValueError, match="claim_boundary drift"):
        assert_stable_core_product_integrity(claim_drift)

    default_drift = dict(registry)
    default_drift["default_contract_id"] = "problem_contract"
    with pytest.raises(ValueError, match="default_contract_id drift"):
        assert_stable_core_product_integrity(default_drift)

    policy_drift = dict(registry)
    drifted_policy = dict(cast(dict[str, object], registry["schema_policy"]))
    drifted_policy["refuse_unknown_schema"] = False
    policy_drift["schema_policy"] = drifted_policy
    with pytest.raises(ValueError, match="schema_policy drift"):
        assert_stable_core_product_integrity(policy_drift)

    surface_drift = dict(registry)
    drifted_surfaces = [
        dict(row) for row in cast(list[dict[str, object]], registry["public_surfaces"])
    ]
    drifted_surfaces[0]["role"] = "drifted"
    surface_drift["public_surfaces"] = drifted_surfaces
    with pytest.raises(ValueError, match="public_surfaces drift"):
        assert_stable_core_product_integrity(surface_drift)

    row_drift = dict(registry)
    drifted_rows = [dict(row) for row in contracts]
    drifted_rows[0]["title"] = "Drifted"
    row_drift["contracts"] = drifted_rows
    with pytest.raises(ValueError, match="canonical contract rows drift"):
        assert_stable_core_product_integrity(row_drift)

    note_drift = dict(registry)
    note_drift["policy_note"] = "drifted"
    with pytest.raises(ValueError, match="policy_note drift"):
        assert_stable_core_product_integrity(note_drift)


def test_integrity_rejects_blank_invalid_and_metadata() -> None:
    """Reject blank, invalid, and inconsistent registry metadata."""
    registry = build_stable_core_product_registry()
    contracts = list(cast(list[dict[str, object]], registry["contracts"]))

    non_map = dict(registry)
    non_map["contracts"] = [cast(Any, "not-a-mapping")]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_stable_core_product_integrity(non_map)

    blank_id = dict(registry)
    blank_rows = [dict(row) for row in contracts]
    blank_rows[0]["contract_id"] = "  "
    blank_id["contracts"] = blank_rows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_stable_core_product_integrity(blank_id)

    bad_kind = dict(registry)
    kind_rows = [dict(row) for row in contracts]
    kind_rows[1]["kind"] = "nope"
    bad_kind["contracts"] = kind_rows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_stable_core_product_integrity(bad_kind)

    no_symbol = dict(registry)
    sym_rows = [dict(row) for row in contracts]
    sym_rows[0]["symbol_name"] = ""
    no_symbol["contracts"] = sym_rows
    with pytest.raises(ValueError, match="symbol_name"):
        assert_stable_core_product_integrity(no_symbol)

    no_default = dict(registry)
    renamed = [dict(row) for row in contracts]
    for row in renamed:
        if row.get("contract_id") == "experiment_contract":
            row["contract_id"] = "renamed"
    no_default["contracts"] = renamed
    with pytest.raises(ValueError, match="missing experiment_contract|drift"):
        assert_stable_core_product_integrity(no_default)

    dup = dict(registry)
    dup_rows = [dict(row) for row in contracts]
    dup_rows.append(dict(dup_rows[0]))
    dup["contracts"] = dup_rows
    dup["contract_count"] = len(dup_rows)
    with pytest.raises(ValueError, match="duplicate contract_id"):
        assert_stable_core_product_integrity(dup)

    blank_count = dict(registry)
    blank_count["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_stable_core_product_integrity(blank_count)

    count_mismatch = dict(registry)
    count_mismatch["contract_count"] = 0
    with pytest.raises(ValueError, match="contract_count"):
        assert_stable_core_product_integrity(count_mismatch)

    no_policy = dict(registry)
    no_policy["schema_policy"] = "nope"
    with pytest.raises(ValueError, match="schema_policy must be a mapping"):
        assert_stable_core_product_integrity(no_policy)


def test_module_exports() -> None:
    """Keep the documented stable-core product symbols publicly exported."""
    assert "round_trip_experiment" in stable_core_product.__all__
    assert "build_demo_experiment" in stable_core_product.__all__
    assert "schema_version_policy" in stable_core_product.__all__


def test_contract_row_validation() -> None:
    """Validate every required contract-row invariant independently."""
    base: dict[str, Any] = {
        "contract_id": "x",
        "kind": "problem",
        "title": "t",
        "summary": "s",
        "module_path": "m",
        "symbol_name": "Problem",
    }
    assert StableCoreContractRow(**base).contract_id == "x"
    with pytest.raises(ValueError, match="contract_id"):
        StableCoreContractRow(**{**base, "contract_id": ""})
    with pytest.raises(ValueError, match="kind"):
        StableCoreContractRow(**{**base, "kind": cast(Any, "nope")})
    with pytest.raises(ValueError, match="title"):
        StableCoreContractRow(**{**base, "title": ""})
    with pytest.raises(ValueError, match="summary"):
        StableCoreContractRow(**{**base, "summary": ""})
    with pytest.raises(ValueError, match="module_path"):
        StableCoreContractRow(**{**base, "module_path": ""})
    with pytest.raises(ValueError, match="symbol_name"):
        StableCoreContractRow(**{**base, "symbol_name": ""})
    with pytest.raises(ValueError, match="api_stability_class"):
        StableCoreContractRow(**{**base, "api_stability_class": ""})
    with pytest.raises(ValueError, match="as_of"):
        StableCoreContractRow(**{**base, "as_of": ""})
    with pytest.raises(ValueError, match="claim_boundary"):
        StableCoreContractRow(**{**base, "claim_boundary": "drifted"})


def test_round_trip_result_validation() -> None:
    """Validate round-trip result kind, schema, digest, and payload fields."""
    with pytest.raises(ValueError, match="kind"):
        StableCoreRoundTripResult(
            kind=cast(Any, "nope"),
            schema_version=STABLE_CORE_MODEL_SCHEMA_VERSION,
            digest_sha256="a" * 64,
            payload={"k": 1},
            matched=True,
        )
    with pytest.raises(ValueError, match="schema_version"):
        StableCoreRoundTripResult(
            kind="problem",
            schema_version="stable_core.experiment_model.v1",
            digest_sha256="a" * 64,
            payload={"k": 1},
            matched=True,
        )
    with pytest.raises(ValueError, match="digest_sha256"):
        StableCoreRoundTripResult(
            kind="problem",
            schema_version=STABLE_CORE_MODEL_SCHEMA_VERSION,
            digest_sha256="short",
            payload={"k": 1},
            matched=True,
        )
    with pytest.raises(ValueError, match="digest_sha256"):
        StableCoreRoundTripResult(
            kind="problem",
            schema_version=STABLE_CORE_MODEL_SCHEMA_VERSION,
            digest_sha256="A" * 64,
            payload={"k": 1},
            matched=True,
        )
    with pytest.raises(ValueError, match="payload"):
        StableCoreRoundTripResult(
            kind="problem",
            schema_version=STABLE_CORE_MODEL_SCHEMA_VERSION,
            digest_sha256="a" * 64,
            payload={},
            matched=True,
        )
    with pytest.raises(ValueError, match="claim_boundary"):
        StableCoreRoundTripResult(
            kind="problem",
            schema_version=STABLE_CORE_MODEL_SCHEMA_VERSION,
            digest_sha256="a" * 64,
            payload={"k": 1},
            matched=True,
            claim_boundary="drifted",
        )
    ok = StableCoreRoundTripResult(
        kind="problem",
        schema_version=STABLE_CORE_MODEL_SCHEMA_VERSION,
        digest_sha256="a" * 64,
        payload={"k": 1},
        matched=True,
    )
    assert ok.to_dict()["matched"] is True


def test_serialise_type_guards() -> None:
    """Reject values that do not match the requested serialisation model."""
    with pytest.raises(ValueError, match="Problem"):
        serialise_problem(cast(Any, "nope"))
    with pytest.raises(ValueError, match="Backend"):
        serialise_backend(cast(Any, "nope"))
    with pytest.raises(ValueError, match="Experiment"):
        serialise_experiment(cast(Any, "nope"))
    with pytest.raises(ValueError, match="Result"):
        serialise_result(cast(Any, "nope"))
    with pytest.raises(ValueError, match="mapping"):
        canonical_json_bytes(cast(Any, [1, 2]))


def test_catalogue_map_runtime_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject empty, blank, and duplicate internal catalogue definitions."""
    from scpn_quantum_control import stable_core_product as mod

    good = get_stable_core_contract("problem_contract")
    blank = StableCoreContractRow(
        contract_id="tmp",
        kind="problem",
        title="t",
        summary="s",
        module_path="m",
        symbol_name="X",
    )
    object.__setattr__(blank, "contract_id", "  ")
    monkeypatch.setattr(mod, "_CANONICAL_CONTRACTS", (blank,))
    with pytest.raises(RuntimeError, match="blank contract_id"):
        mod._catalogue_map()

    a = get_stable_core_contract("problem_contract")
    monkeypatch.setattr(mod, "_CANONICAL_CONTRACTS", (a, a))
    with pytest.raises(RuntimeError, match="duplicate contract_id"):
        mod._catalogue_map()

    monkeypatch.setattr(mod, "_CANONICAL_CONTRACTS", ())
    with pytest.raises(RuntimeError, match="non-empty"):
        mod._catalogue_map()

    monkeypatch.setattr(mod, "_CANONICAL_CONTRACTS", (good,))
    assert mod._catalogue_map()[good.contract_id].contract_id == good.contract_id


def test_to_dict_on_contract_row() -> None:
    """Render every contract-row field into a JSON-ready mapping."""
    row = get_stable_core_contract("schema_policy")
    payload = row.to_dict()
    assert payload["kind"] == "schema_policy"
    assert payload["contract_id"] == "schema_policy"


def test_backend_from_dict_hw_flag() -> None:
    """Preserve a valid hardware flag and reject non-boolean flag values."""
    backend = build_backend(
        backend_id="qiskit-runtime",
        kind="qiskit",
        capabilities=("order_parameter", "parity"),
        hardware_submission_allowed=False,
    )
    rebuilt = backend_from_dict(backend.to_dict())
    assert rebuilt.hardware_submission_allowed is False
    with pytest.raises(ValueError, match="hardware_submission_allowed must be a bool"):
        backend_from_dict(
            {
                "backend_id": "b",
                "kind": "classical_reference",
                "capabilities": ["order_parameter"],
                "hardware_submission_allowed": "yes",
            }
        )


def test_from_dict_extra_type_edges() -> None:
    """Reject additional sequence, mapping, scalar, and null type violations."""
    with pytest.raises(ValueError, match="unsupported problem kind"):
        problem_from_dict(
            {
                "problem_id": "p",
                "kind": "not_kuramoto",
                "coupling_matrix": [[0.0]],
                "omega": [0.0],
            }
        )
    with pytest.raises(ValueError, match="initial_state must be a string"):
        problem_from_dict(
            {
                "problem_id": "p",
                "coupling_matrix": [[0.0]],
                "omega": [0.0],
                "initial_state": 123,
            }
        )
    with pytest.raises(ValueError, match="metadata must be a mapping"):
        problem_from_dict(
            {
                "problem_id": "p",
                "coupling_matrix": [[0.0]],
                "omega": [0.0],
                "metadata": "nope",
            }
        )
    with pytest.raises(ValueError, match="kind must be a non-empty string"):
        backend_from_dict(
            {
                "backend_id": "b",
                "kind": "",
                "capabilities": ["order_parameter"],
            }
        )
    with pytest.raises(ValueError, match="metadata must be a mapping"):
        backend_from_dict(
            {
                "backend_id": "b",
                "kind": "classical_reference",
                "capabilities": ["order_parameter"],
                "metadata": ["x"],
            }
        )
    good_problem = build_problem(
        problem_id="p",
        coupling_matrix=((0.0,),),
        omega=(0.0,),
    ).to_dict()
    good_backend = classical_reference_backend().to_dict()
    with pytest.raises(ValueError, match="problem must be a mapping"):
        experiment_from_dict(
            {
                "experiment_id": "e",
                "problem": "nope",
                "backend": good_backend,
                "objective": "order_parameter",
                "seed": 0,
            }
        )
    with pytest.raises(ValueError, match="backend must be a mapping"):
        experiment_from_dict(
            {
                "experiment_id": "e",
                "problem": good_problem,
                "backend": "nope",
                "objective": "order_parameter",
                "seed": 0,
            }
        )
    with pytest.raises(ValueError, match="objective must be a non-empty string"):
        experiment_from_dict(
            {
                "experiment_id": "e",
                "problem": good_problem,
                "backend": good_backend,
                "objective": "  ",
                "seed": 0,
            }
        )
    with pytest.raises(ValueError, match="shots must be an int"):
        experiment_from_dict(
            {
                "experiment_id": "e",
                "problem": good_problem,
                "backend": good_backend,
                "objective": "order_parameter",
                "seed": 0,
                "shots": "100",
            }
        )
    with pytest.raises(ValueError, match="metadata must be a mapping"):
        experiment_from_dict(
            {
                "experiment_id": "e",
                "problem": good_problem,
                "backend": good_backend,
                "objective": "order_parameter",
                "seed": 0,
                "metadata": 1,
            }
        )
    with pytest.raises(ValueError, match="status must be a non-empty string"):
        result_from_dict(
            {
                "experiment_id": "e",
                "backend_id": "b",
                "status": "",
                "observables": {},
                "blockers": ("x",),
            }
        )
    with pytest.raises(ValueError, match="artifacts must be a sequence"):
        result_from_dict(
            {
                "experiment_id": "e",
                "backend_id": "b",
                "status": "succeeded",
                "observables": {"o": 1.0},
                "artifacts": "nope",
            }
        )
    with pytest.raises(ValueError, match="blockers must be a sequence"):
        result_from_dict(
            {
                "experiment_id": "e",
                "backend_id": "b",
                "status": "blocked",
                "observables": {},
                "blockers": "nope",
            }
        )
    with pytest.raises(ValueError, match="metadata must be a mapping"):
        result_from_dict(
            {
                "experiment_id": "e",
                "backend_id": "b",
                "status": "succeeded",
                "observables": {"o": 1.0},
                "metadata": "nope",
            }
        )


def test_envelope_kind_and_body_edges() -> None:
    """Reject unsupported kinds and empty or non-mapping envelope bodies."""
    with pytest.raises(ValueError, match="kind must be a non-empty string"):
        unwrap_model_envelope(
            {
                "schema_version": STABLE_CORE_MODEL_SCHEMA_VERSION,
                "kind": "  ",
                "body": {"problem_id": "p"},
                "claim_boundary": STABLE_CORE_PRODUCT_CLAIM_BOUNDARY,
            }
        )
    with pytest.raises(ValueError, match="body must be a non-empty mapping"):
        unwrap_model_envelope(
            {
                "schema_version": STABLE_CORE_MODEL_SCHEMA_VERSION,
                "kind": "problem",
                "body": {},
                "claim_boundary": STABLE_CORE_PRODUCT_CLAIM_BOUNDARY,
            }
        )


def test_round_trip_detects_field_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail closed when a rebuilt model loses fields during round-trip."""
    from scpn_quantum_control import stable_core_product as mod

    problem = build_problem(
        problem_id="p",
        coupling_matrix=((0.0,),),
        omega=(0.0,),
    )
    experiment = build_demo_experiment()

    def broken_problem_deserialise(envelope: Any) -> Any:
        # Return a different problem so field-loss detection fires.
        return build_problem(
            problem_id="other",
            coupling_matrix=((0.0,),),
            omega=(1.0,),
        )

    monkeypatch.setattr(mod, "deserialise_problem", broken_problem_deserialise)
    with pytest.raises(ValueError, match="round-trip lost or altered"):
        mod.round_trip_problem(problem)

    from scpn_quantum_control.stable_core import build_experiment as be

    def broken_exp(envelope: Any) -> Any:
        return be(
            experiment_id="different-id",
            problem=experiment.problem,
            backend=experiment.backend,
            objective=experiment.objective,
            seed=experiment.seed,
        )

    monkeypatch.setattr(mod, "deserialise_experiment", broken_exp)
    with pytest.raises(ValueError, match="round-trip lost or altered"):
        mod.round_trip_experiment(experiment)


def test_iter_stable_core_contracts_without_kind_returns_full_catalogue() -> None:
    """Unfiltered iter returns every catalogue row (covers kind is None branch)."""
    rows = iter_stable_core_contracts()
    assert len(rows) == len(list_stable_core_contract_ids())
    assert {row.contract_id for row in rows} == set(list_stable_core_contract_ids())


def test_problem_from_dict_rejects_empty_omega() -> None:
    """Empty omega sequence is refused with a clear ValueError."""
    with pytest.raises(ValueError, match="omega must be a non-empty sequence"):
        problem_from_dict(
            {
                "problem_id": "p",
                "coupling_matrix": [[0.0]],
                "omega": [],
            }
        )
