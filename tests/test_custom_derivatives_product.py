# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for custom derivatives product
"""Real-surface tests for ``scpn_quantum_control.custom_derivatives_product``."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control.custom_derivatives_product as custom_derivatives_product
from scpn_quantum_control.custom_derivatives_product import (
    CUSTOM_DERIVATIVES_CLAIM_BOUNDARY,
    CUSTOM_DERIVATIVES_PRODUCT_SCHEMA,
    DEFAULT_PRODUCT_NAMESPACE,
    CustomDerivativeContractRow,
    RegistrationResult,
    assert_custom_derivatives_product_integrity,
    build_custom_derivatives_product_registry,
    build_example_scaled_linear_rule,
    get_custom_derivative_contract,
    iter_custom_derivative_contracts,
    list_custom_derivative_contract_ids,
    list_product_registered_identities,
    map_custom_derivatives_public_surfaces,
    new_product_registry,
    parse_product_identity,
    probe_example_rule_round_trip,
    register_product_custom_rule,
    registration_contract_policy,
    require_product_custom_rule,
)
from scpn_quantum_control.program_ad_registry import (
    CustomDerivativeRule,
    PrimitiveIdentity,
)


def test_list_contracts_and_filters() -> None:
    """List stable contract identifiers and filter rows by contract kind."""
    ids = list_custom_derivative_contract_ids()
    assert "registration_contract" in ids
    assert "example_linear_rule" in ids
    assert ids == list_custom_derivative_contract_ids()
    examples = iter_custom_derivative_contracts(kind="example_rule")
    assert len(examples) == 1


def test_get_known_and_unknown_fail_closed() -> None:
    """Resolve known contracts and reject blank or unknown identifiers."""
    row = get_custom_derivative_contract("registration_contract")
    assert row.claim_boundary == CUSTOM_DERIVATIVES_CLAIM_BOUNDARY
    assert row.metamorphic_verification_pointer
    with pytest.raises(ValueError, match="non-empty"):
        get_custom_derivative_contract("  ")
    with pytest.raises(ValueError, match="unknown contract_id"):
        get_custom_derivative_contract("not_a_contract")


def test_registration_policy() -> None:
    """Expose the fail-closed registration and residual-work policy."""
    policy = registration_contract_policy()
    assert CUSTOM_DERIVATIVES_PRODUCT_SCHEMA == "custom_derivatives_product.v2"
    assert policy["fail_closed_duplicate_without_overwrite"] is True
    assert policy["require_jvp_or_vjp"] is True
    assert policy["transform_algebra_ci_residual"] == "transform-algebra-interaction-coverage"
    assert policy["metamorphic_verification_residual"] == "custom-rule-metamorphic-verification"


def test_register_query_list_fail_closed_duplicate() -> None:
    """Register, query, list, reject duplicates, and permit explicit overwrite."""
    registry = new_product_registry()
    rule = build_example_scaled_linear_rule(scale=3.0)
    identity = f"{DEFAULT_PRODUCT_NAMESPACE}:demo@1"
    result = register_product_custom_rule(identity, rule, registry=registry)
    assert result.registered is True
    assert result.identity_key == parse_product_identity(identity).key
    assert result.rule_name == rule.name

    keys = list_product_registered_identities(registry=registry)
    assert result.identity_key in keys

    required = require_product_custom_rule(identity, registry=registry)
    assert required.name == rule.name

    # duplicate without overwrite fails closed
    other = build_example_scaled_linear_rule(scale=4.0, name="other")
    with pytest.raises(ValueError, match="already registered"):
        register_product_custom_rule(identity, other, registry=registry)

    # overwrite allowed
    overwritten = register_product_custom_rule(
        identity,
        other,
        registry=registry,
        overwrite=True,
    )
    assert overwritten.overwrite is True
    assert require_product_custom_rule(identity, registry=registry).name == "other"


def test_register_isolated_registry_default() -> None:
    """Omitting registry uses a fresh isolated registry (does not raise)."""
    rule = build_example_scaled_linear_rule()
    result = register_product_custom_rule(
        f"{DEFAULT_PRODUCT_NAMESPACE}:isolated@1",
        rule,
    )
    assert result.registered is True


def test_register_invalid_rule_and_registry() -> None:
    """Reject invalid rules, identities, and explicit registry objects."""
    registry = new_product_registry()
    with pytest.raises(ValueError, match="CustomDerivativeRule"):
        register_product_custom_rule(
            f"{DEFAULT_PRODUCT_NAMESPACE}:x@1",
            cast(Any, "nope"),
            registry=registry,
        )
    with pytest.raises(ValueError, match="registry must be"):
        register_product_custom_rule(
            f"{DEFAULT_PRODUCT_NAMESPACE}:x@1",
            build_example_scaled_linear_rule(),
            registry=cast(Any, "nope"),
        )
    with pytest.raises(ValueError, match="non-empty"):
        parse_product_identity("  ")
    with pytest.raises(ValueError, match="registry must be"):
        require_product_custom_rule("x:y@1", registry=cast(Any, None))
    with pytest.raises(ValueError, match="registry must be"):
        list_product_registered_identities(registry=cast(Any, "nope"))


def test_unknown_require_fail_closed() -> None:
    """Refuse lookup when the requested identity has no registered rule."""
    registry = new_product_registry()
    with pytest.raises(ValueError):
        require_product_custom_rule(
            f"{DEFAULT_PRODUCT_NAMESPACE}:missing@1",
            registry=registry,
        )


def test_example_rule_and_probe() -> None:
    """Validate the scaled-linear rule and its deterministic round-trip probe."""
    rule = build_example_scaled_linear_rule(scale=2.5)
    assert isinstance(rule, CustomDerivativeRule)
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None

    with pytest.raises(ValueError, match="non-empty"):
        build_example_scaled_linear_rule(name="")
    with pytest.raises(ValueError, match="finite"):
        build_example_scaled_linear_rule(scale=float("nan"))
    with pytest.raises(ValueError, match="non-zero"):
        build_example_scaled_linear_rule(scale=0.0)

    probe = probe_example_rule_round_trip(scale=2.0)
    assert probe["identity_key"]
    assert probe["value"] == [2.0, 4.0]
    assert probe["jvp"] == [2.0, 2.0]
    assert probe["registered_identities"]
    assert probe["claim_boundary"] == CUSTOM_DERIVATIVES_CLAIM_BOUNDARY

    with pytest.raises(ValueError, match="same shape"):
        probe_example_rule_round_trip(
            values=np.array([1.0, 2.0]),
            tangent=np.array([1.0]),
        )


def test_public_surfaces_and_registry() -> None:
    """Publish complete deterministic surface and product registry catalogues."""
    surfaces = map_custom_derivatives_public_surfaces()
    assert surfaces
    paths = {row["module_path"] for row in surfaces}
    assert "scpn_quantum_control.program_ad_registry" in paths
    assert "scpn_quantum_control.custom_derivatives_product" in paths

    registry = build_custom_derivatives_product_registry()
    assert registry["schema"] == CUSTOM_DERIVATIVES_PRODUCT_SCHEMA
    assert registry["blank_entry_count"] == 0
    assert registry["default_contract_id"] == "registration_contract"
    validated = assert_custom_derivatives_product_integrity(registry)
    assert validated["contract_count"] == len(list_custom_derivative_contract_ids())
    assert assert_custom_derivatives_product_integrity()["blank_entry_count"] == 0


def test_integrity_rejects_drift() -> None:
    """Reject contract-set drift, empty catalogues, and permissive policy."""
    registry = build_custom_derivatives_product_registry()
    contracts = cast(list[dict[str, object]], registry["contracts"])
    broken = dict(registry)
    broken["contracts"] = contracts + [
        {
            "contract_id": "ghost",
            "kind": "example_rule",
            "title": "t",
            "summary": "s",
            "module_path": "m",
            "symbol_name": "X",
            "metamorphic_verification_pointer": "a",
            "api_stability_class": "experimental_workbench",
            "as_of": "2026-07-24",
            "claim_boundary": CUSTOM_DERIVATIVES_CLAIM_BOUNDARY,
        }
    ]
    broken["contract_count"] = len(cast(list[object], broken["contracts"]))
    with pytest.raises(ValueError, match="drift"):
        assert_custom_derivatives_product_integrity(broken)

    empty: dict[str, object] = {"contracts": [], "blank_entry_count": 0, "contract_count": 0}
    with pytest.raises(ValueError, match="non-empty contracts"):
        assert_custom_derivatives_product_integrity(empty)

    bad_policy = dict(registry)
    bad_policy["registration_policy"] = {
        **cast(dict[str, object], registry["registration_policy"]),
        "fail_closed_duplicate_without_overwrite": False,
    }
    with pytest.raises(ValueError, match="fail closed on duplicates"):
        assert_custom_derivatives_product_integrity(bad_policy)


def test_integrity_rejects_blank_invalid() -> None:
    """Reject malformed, blank, duplicate, and count-drifted registry rows."""
    registry = build_custom_derivatives_product_registry()
    contracts = cast(list[dict[str, object]], registry["contracts"])

    non_map = dict(registry)
    non_map["contracts"] = [cast(Any, "nope")]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_custom_derivatives_product_integrity(non_map)

    blank_id = dict(registry)
    rows = [dict(row) for row in contracts]
    rows[0]["contract_id"] = "  "
    blank_id["contracts"] = rows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_custom_derivatives_product_integrity(blank_id)

    bad_kind = dict(registry)
    krows = [dict(row) for row in contracts]
    krows[1]["kind"] = "nope"
    bad_kind["contracts"] = krows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_custom_derivatives_product_integrity(bad_kind)

    no_symbol = dict(registry)
    srows = [dict(row) for row in contracts]
    srows[0]["symbol_name"] = ""
    no_symbol["contracts"] = srows
    with pytest.raises(ValueError, match="symbol_name"):
        assert_custom_derivatives_product_integrity(no_symbol)

    no_default = dict(registry)
    renamed = [dict(row) for row in contracts]
    for row in renamed:
        if row.get("contract_id") == "registration_contract":
            row["contract_id"] = "renamed"
    no_default["contracts"] = renamed
    with pytest.raises(ValueError, match="missing registration_contract|drift"):
        assert_custom_derivatives_product_integrity(no_default)

    dup = dict(registry)
    drows = [dict(row) for row in contracts]
    drows.append(dict(drows[0]))
    dup["contracts"] = drows
    dup["contract_count"] = len(drows)
    with pytest.raises(ValueError, match="duplicate contract_id"):
        assert_custom_derivatives_product_integrity(dup)

    blank_count = dict(registry)
    blank_count["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_custom_derivatives_product_integrity(blank_count)

    count_mismatch = dict(registry)
    count_mismatch["contract_count"] = 0
    with pytest.raises(ValueError, match="contract_count"):
        assert_custom_derivatives_product_integrity(count_mismatch)

    no_policy = dict(registry)
    no_policy["registration_policy"] = "nope"
    with pytest.raises(ValueError, match="registration_policy must be a mapping"):
        assert_custom_derivatives_product_integrity(no_policy)


def test_module_exports() -> None:
    """Keep the documented product functions publicly exported."""
    assert "register_product_custom_rule" in custom_derivatives_product.__all__
    assert "build_example_scaled_linear_rule" in custom_derivatives_product.__all__
    assert "probe_example_rule_round_trip" in custom_derivatives_product.__all__


def test_contract_row_validation() -> None:
    """Validate every custom-derivative contract-row invariant."""
    base: dict[str, Any] = {
        "contract_id": "x",
        "kind": "example_rule",
        "title": "t",
        "summary": "s",
        "module_path": "m",
        "symbol_name": "S",
    }
    assert CustomDerivativeContractRow(**base).contract_id == "x"
    with pytest.raises(ValueError, match="contract_id"):
        CustomDerivativeContractRow(**{**base, "contract_id": ""})
    with pytest.raises(ValueError, match="kind"):
        CustomDerivativeContractRow(**{**base, "kind": cast(Any, "nope")})
    with pytest.raises(ValueError, match="title"):
        CustomDerivativeContractRow(**{**base, "title": ""})
    with pytest.raises(ValueError, match="summary"):
        CustomDerivativeContractRow(**{**base, "summary": ""})
    with pytest.raises(ValueError, match="module_path"):
        CustomDerivativeContractRow(**{**base, "module_path": ""})
    with pytest.raises(ValueError, match="symbol_name"):
        CustomDerivativeContractRow(**{**base, "symbol_name": ""})
    with pytest.raises(ValueError, match="api_stability_class"):
        CustomDerivativeContractRow(**{**base, "api_stability_class": ""})
    with pytest.raises(ValueError, match="as_of"):
        CustomDerivativeContractRow(**{**base, "as_of": ""})


def test_registration_result_validation() -> None:
    """Validate successful registration-result identity and rule metadata."""
    ok = RegistrationResult(
        identity_key="ns:n@1",
        rule_name="r",
        registered=True,
        overwrite=False,
    )
    assert ok.to_dict()["registered"] is True
    with pytest.raises(ValueError, match="identity_key"):
        RegistrationResult(
            identity_key="",
            rule_name="r",
            registered=True,
            overwrite=False,
        )
    with pytest.raises(ValueError, match="rule_name"):
        RegistrationResult(
            identity_key="ns:n@1",
            rule_name="",
            registered=True,
            overwrite=False,
        )
    with pytest.raises(ValueError, match="registered must be True"):
        RegistrationResult(
            identity_key="ns:n@1",
            rule_name="r",
            registered=False,
            overwrite=False,
        )


def test_catalogue_map_runtime_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject blank, duplicate, and empty internal contract catalogues."""
    from scpn_quantum_control import custom_derivatives_product as mod

    good = get_custom_derivative_contract("registration_contract")
    blank = CustomDerivativeContractRow(
        contract_id="tmp",
        kind="example_rule",
        title="t",
        summary="s",
        module_path="m",
        symbol_name="S",
    )
    object.__setattr__(blank, "contract_id", "  ")
    monkeypatch.setattr(mod, "_CANONICAL_CONTRACTS", (blank,))
    with pytest.raises(RuntimeError, match="blank contract_id"):
        mod._catalogue_map()

    a = get_custom_derivative_contract("registration_contract")
    monkeypatch.setattr(mod, "_CANONICAL_CONTRACTS", (a, a))
    with pytest.raises(RuntimeError, match="duplicate contract_id"):
        mod._catalogue_map()

    monkeypatch.setattr(mod, "_CANONICAL_CONTRACTS", ())
    with pytest.raises(RuntimeError, match="non-empty"):
        mod._catalogue_map()

    monkeypatch.setattr(mod, "_CANONICAL_CONTRACTS", (good,))
    assert mod._catalogue_map()[good.contract_id].contract_id == good.contract_id


def test_contract_to_dict() -> None:
    """Serialize a contract row into its complete JSON-ready mapping."""
    row = get_custom_derivative_contract("metamorphic_boundary")
    payload = row.to_dict()
    assert payload["kind"] == "metamorphic_boundary"
    assert "metamorphic" in str(payload["metamorphic_verification_pointer"])
    assert payload["contract_id"] == "metamorphic_boundary"


def test_parse_identity_none_and_probe_mismatch_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a missing identity and inconsistent ambient probe values."""
    with pytest.raises(ValueError, match="non-empty string or PrimitiveIdentity"):
        parse_product_identity(cast(Any, None))

    # Force value mismatch path in probe by monkeypatching value_and_custom_jvp
    from scpn_quantum_control import custom_derivatives_product as mod
    from scpn_quantum_control.differentiable_result_contracts import JVPResult

    def broken_jvp(rule: Any, values: Any, tangent: Any) -> Any:
        del rule, values, tangent
        return JVPResult(
            value=np.array([0.0, 0.0]),
            jvp=np.array([2.0, 2.0]),
            tangent=np.array([1.0, 1.0]),
            method="custom",
            step=0.0,
            evaluations=1,
            parameter_names=("p0", "p1"),
            trainable=(True, True),
        )

    monkeypatch.setattr(
        "scpn_quantum_control.differentiable_custom_derivatives.value_and_custom_jvp",
        broken_jvp,
    )
    with pytest.raises(ValueError, match="value does not match"):
        mod.probe_example_rule_round_trip(scale=2.0)


def test_parse_identity_object() -> None:
    """Preserve an already structured primitive identity."""
    ident = PrimitiveIdentity(namespace="a", name="b", version="2")
    assert parse_product_identity(ident).key == "a:b@2"


def test_iter_contracts_without_kind_filter() -> None:
    """Unfiltered iter returns the full catalogue (kind is None branch)."""
    all_rows = iter_custom_derivative_contracts()
    assert len(all_rows) == len(list_custom_derivative_contract_ids())
    assert {row.contract_id for row in all_rows} == set(list_custom_derivative_contract_ids())


def test_example_rule_vjp_is_linear_scale() -> None:
    """Example scaled-linear VJP is the same linear map as the forward/JVP."""
    scale = 2.5
    rule = build_example_scaled_linear_rule(scale=scale)
    assert rule.vjp_rule is not None
    cotangent = np.array([0.5, -1.0], dtype=np.float64)
    # Position is unused for this linear map; pass a non-matching shape to prove it.
    out = np.asarray(rule.vjp_rule(np.array([9.0, 9.0]), cotangent), dtype=np.float64)
    np.testing.assert_allclose(out, scale * cotangent)


def test_probe_refuses_jvp_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Round-trip probe fails closed when ambient JVP disagrees with scale * tangent."""
    from scpn_quantum_control.differentiable_result_contracts import JVPResult

    def broken_jvp(rule: Any, values: Any, tangent: Any) -> Any:
        arr = np.asarray(values, dtype=np.float64)
        t = np.asarray(tangent, dtype=np.float64)
        # Correct value (scale * values with scale=2) but wrong JVP.
        return JVPResult(
            value=2.0 * arr,
            jvp=np.zeros_like(t),
            tangent=t,
            method="custom",
            step=0.0,
            evaluations=1,
            parameter_names=tuple(f"p{i}" for i in range(arr.size)),
            trainable=tuple(True for _ in range(arr.size)),
        )

    monkeypatch.setattr(
        "scpn_quantum_control.differentiable_custom_derivatives.value_and_custom_jvp",
        broken_jvp,
    )
    with pytest.raises(ValueError, match="JVP does not match"):
        custom_derivatives_product.probe_example_rule_round_trip(scale=2.0)
