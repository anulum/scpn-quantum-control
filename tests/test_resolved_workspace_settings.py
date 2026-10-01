# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — resolved workspace settings tests
"""Exercise policy and semantic identity through the original public Studio API."""

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from scpn_quantum_control.hardware.hal import BackendProfile, HardwareAbstractionLayer
from scpn_quantum_control.studio.workspace import (
    ResolvedSettings,
    SettingsPolicy,
    SettingsRefused,
    resolve_settings,
    settings_plan_digest,
)

REF: dict[str, object] = {
    "schema": "policy.v1",
    "sha256": "a" * 64,
    "media_type": "application/json",
}
ENV: dict[str, object] = {**REF, "schema": "environment.v1", "sha256": "b" * 64}
PROFILE = HardwareAbstractionLayer.with_builtin_profiles().profile("local_statevector")


def resolve(run: Mapping[str, object], *, profile: BackendProfile = PROFILE) -> ResolvedSettings:
    """Resolve explicit settings using real built-in HAL declaration and injected caps."""
    return resolve_settings(
        {"precision": "float64", "shots": 16, "theme": "dark"},
        {},
        {},
        run,
        policy=SettingsPolicy(
            REF,
            {"shots": 100, "memory_budget_bytes": 1024, "n_qubits": 50},
            {"device": ("local",), "precision": ("float32", "float64"), "units": ("rad",)},
        ),
        environment_ref=ENV,
        profile=profile,
    )


def test_resolved_workspace_settings_01() -> None:
    """Policy overrides refuse and name the exact governing policy without clamping."""
    for values in (
        {"shots": 101},
        {"memory_budget_bytes": 1025},
        {"device": "remote"},
        {"units": "degrees"},
    ):
        with pytest.raises(SettingsRefused, match="policy " + "a" * 64):
            resolve(values)
    effective = resolve({"shots": 100, "memory_budget_bytes": 1024}).body["effective"]
    assert isinstance(effective, Mapping)
    assert effective["shots"] == 100


def test_resolved_workspace_settings_02() -> None:
    """Every winning value identifies its actual precedence layer and snapshots inputs."""
    defaults: dict[str, object] = {"seed": 1, "shots": 1}
    project: dict[str, object] = {"shots": 2, "theme": "dark"}
    experiment: dict[str, object] = {"shots": 3, "precision": "float64"}
    run: dict[str, object] = {"shots": 4, "parameters": {"angle": 0.5}}
    result = resolve_settings(
        defaults,
        project,
        experiment,
        run,
        policy=SettingsPolicy(REF, {"shots": 4}, {}),
        environment_ref=ENV,
        profile=PROFILE,
    )
    assert result.body["origins"] == {
        "seed": "defaults",
        "theme": "project",
        "precision": "experiment",
        "shots": "run",
        "parameters": "run",
    }
    before = result.digest
    run["parameters"] = {"angle": 2.0}
    REF["sha256"] = "c" * 64
    try:
        assert result.digest == before
    finally:
        REF["sha256"] = "a" * 64


def test_resolved_workspace_settings_03() -> None:
    """Theme, notation and layout change full identity but leave semantic identity alone."""
    first = resolve({})
    second = resolve({"theme": "light", "notation": "dirac", "layout": "wide", "plot_rounding": 3})
    assert first.digest != second.digest
    assert settings_plan_digest(first) == settings_plan_digest(second)


def test_resolved_workspace_settings_04() -> None:
    """Precision, seed, units and numeric parameters each invalidate semantic identity."""
    first = settings_plan_digest(resolve({}))
    for change in (
        {"precision": "float32"},
        {"seed": 2},
        {"units": "rad"},
        {"parameters": {"x": 1.0}},
    ):
        assert settings_plan_digest(resolve(change)) != first


@pytest.mark.parametrize(
    "values",
    [
        {"token": "secret"},
        {"shots": True},
        {"seed": -1},
        {"plot_rounding": -1},
        {"parameters": []},
        {"parameters": {"x": True}},
        {"parameters": {"": 1}},
        {"parameters": {"x": float("inf")}},
        {"theme": ""},
        {"method": 1},
    ],
)
def test_malformed_settings_refuse(values: dict[str, object]) -> None:
    """Malformed and credential fields refuse through production resolution."""
    with pytest.raises(SettingsRefused):
        resolve(values)


@pytest.mark.parametrize(
    "ceilings,choices",
    [
        ({"theme": 3}, {}),
        ({"shots": True}, {}),
        ({"shots": 0}, {}),
        ({}, {"theme": ("dark",)}),
        ({}, {"device": ()}),
        ({}, {"precision": ("",)}),
    ],
)
def test_malformed_policy_refuses(
    ceilings: dict[str, int], choices: dict[str, tuple[str, ...]]
) -> None:
    """An invalid policy never becomes usable for admission."""
    with pytest.raises(SettingsRefused):
        SettingsPolicy(REF, ceilings, choices)


def test_route_capacity_and_unknown_capacity_refuse() -> None:
    """Existing HAL capability declarations remain authoritative without submission."""
    with pytest.raises(SettingsRefused, match="Backend differs"):
        resolve({"backend": "cloud"})
    with pytest.raises(SettingsRefused, match="Shots are unsupported"):
        resolve(
            {},
            profile=replace(
                PROFILE, capabilities=replace(PROFILE.capabilities, supports_shots=False)
            ),
        )
    with pytest.raises(SettingsRefused, match="Qubit request"):
        resolve(
            {"n_qubits": 5},
            profile=replace(PROFILE, capabilities=replace(PROFILE.capabilities, max_qubits=4)),
        )
    with pytest.raises(SettingsRefused, match="lacks capacity"):
        resolve_settings(
            {},
            {},
            {},
            {"memory_budget_bytes": 1},
            policy=SettingsPolicy(REF, {}, {}),
            environment_ref=ENV,
            profile=PROFILE,
        )
    result = resolve(
        {"backend": PROFILE.backend_id, "n_qubits": 2, "parameters": {"x": 1, "y": 0.5}}
    )
    assert result.body["requested"] == result.body["effective"]


def test_policy_snapshots_caller_maps_and_preserves_empty_resolution() -> None:
    """Later caller mutations cannot raise the admitted ceiling or replace policy identity."""
    caps = {"shots": 10}
    choices = {"device": ("local",)}
    policy = SettingsPolicy(REF, caps, choices)
    caps["shots"] = 1000
    choices["device"] = ("remote",)
    assert policy.ceilings["shots"] == 10
    assert policy.choices["device"] == ("local",)
    empty = resolve_settings({}, {}, {}, {}, policy=policy, environment_ref=ENV, profile=PROFILE)
    assert empty.body["origins"] == {}


def test_shadowed_forbidden_layer_still_refuses() -> None:
    """A later valid override cannot erase a policy-violating request in an earlier layer."""
    with pytest.raises(SettingsRefused, match="policy"):
        resolve_settings(
            {"shots": 200},
            {},
            {},
            {"shots": 1},
            policy=SettingsPolicy(REF, {"shots": 100}, {}),
            environment_ref=ENV,
            profile=PROFILE,
        )


def test_semantic_hash_refuses_unknown_effective_settings() -> None:
    """A generic recorded settings document cannot conceal unsupported semantic fields."""
    from scpn_quantum_control.studio.workspace import parse_resolved_settings

    recorded = parse_resolved_settings(
        {
            "schema": "resolved_settings.v1",
            "extensions": {},
            "body": {
                "requested": {},
                "effective": {"unknown_solver_option": 1},
                "origins": {"unknown_solver_option": "run"},
                "policy_ref": REF,
                "environment_ref": ENV,
                "rejected_fields": [],
            },
        }
    )
    with pytest.raises(SettingsRefused):
        settings_plan_digest(recorded)


def test_shared_inspector_fixture_matches_real_source_owned_resolution() -> None:
    """Resolve literal shared layers and compare the exact browser inspector record."""
    from scpn_quantum_control.studio.workspace import parse_resolved_settings, read_json

    fixture = cast(
        dict[str, object],
        read_json((Path(__file__).parent / "data/studio_workspace/settings.json").read_text()),
    )
    document = parse_resolved_settings(fixture["document"])
    layers = cast(dict[str, dict[str, object]], fixture["layers"])
    policy_ref = cast(Mapping[str, object], document.body["policy_ref"])
    environment_ref = cast(Mapping[str, object], document.body["environment_ref"])
    resolved = resolve_settings(
        layers["defaults"],
        layers["project"],
        layers["experiment"],
        layers["run"],
        policy=SettingsPolicy(policy_ref, {"shots": 100}, {}),
        environment_ref=environment_ref,
        profile=PROFILE,
    )
    assert resolved.to_dict() == document.to_dict()
    assert resolved.digest == document.digest
