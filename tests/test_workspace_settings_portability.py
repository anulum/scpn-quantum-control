# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — portable workspace settings tests
"""Exercise lossless non-secret import and explicit reset through public Studio APIs."""

import pytest

from scpn_quantum_control.hardware.hal import HardwareAbstractionLayer
from scpn_quantum_control.studio import (
    ResolvedSettings,
    SettingsPolicy,
    SettingsRefused,
    confirm_settings_reset,
    export_settings,
    import_settings,
    preview_settings_reset,
    resolve_settings,
    settings_plan_digest,
)

REF: dict[str, object] = {
    "schema": "policy.v1",
    "sha256": "a" * 64,
    "media_type": "application/json",
}
ENV: dict[str, object] = {**REF, "schema": "environment.v1"}
PROFILE = HardwareAbstractionLayer.with_builtin_profiles().profile("local_statevector")
POLICY = SettingsPolicy(REF, {"shots": 100}, {})


def test_resolved_workspace_settings_05() -> None:
    """Reset preview preserves the current record and confirmation refuses a stale source."""
    current = resolve_settings(
        {"shots": 1}, {}, {}, {"shots": 10}, policy=POLICY, environment_ref=ENV, profile=PROFILE
    )
    before = current.to_dict()
    preview = preview_settings_reset(
        current, {"shots": 1}, policy=POLICY, environment_ref=ENV, profile=PROFILE
    )
    assert current.to_dict() == before
    assert preview.candidate.body["origins"] == {"shots": "defaults"}
    assert confirm_settings_reset(current, preview).digest == preview.candidate.digest
    changed = resolve_settings(
        {}, {}, {}, {"shots": 2}, policy=POLICY, environment_ref=ENV, profile=PROFILE
    )
    with pytest.raises(SettingsRefused, match="changed after reset"):
        confirm_settings_reset(changed, preview)
    assert changed.body["requested"] == {"shots": 2}


def test_portable_values_reenter_current_policy_without_importing_authority() -> None:
    """Round-trip exact numeric values; apply the current policy before using imported requests."""
    values = {"seed": 9007199254740993, "parameters": {"x": -0.0}, "shots": 80, "theme": "light"}
    resolved = resolve_settings(
        {}, values, {}, {}, policy=POLICY, environment_ref=ENV, profile=PROFILE
    )
    wire = export_settings(resolved)
    imported = import_settings(wire)
    again = resolve_settings(
        {}, imported, {}, {}, policy=POLICY, environment_ref=ENV, profile=PROFILE
    )
    assert settings_plan_digest(again) == settings_plan_digest(resolved)
    assert "policy_ref" not in wire and "environment_ref" not in wire
    with pytest.raises(SettingsRefused, match="policy"):
        resolve_settings(
            {},
            imported,
            {},
            {},
            policy=SettingsPolicy(REF, {"shots": 50}, {}),
            environment_ref=ENV,
            profile=PROFILE,
        )


@pytest.mark.parametrize(
    "wire",
    [
        "{",
        "[]",
        '{"schema":"quantum_workspace_settings.v2","values":{}}',
        '{"schema":"quantum_workspace_settings.v1","values":[]}',
        '{"schema":"quantum_workspace_settings.v1","values":{"token":"secret"}}',
        '{"schema":"quantum_workspace_settings.v1","schema":"x","values":{}}',
        pytest.param(" " * 65537, id="65537-spaces"),
    ],
)
def test_portable_refusal_before_any_application(wire: str) -> None:
    """Malformed, oversized, secret and future-version input never yields applicable values."""
    with pytest.raises(SettingsRefused):
        import_settings(wire)


def test_export_refuses_unowned_or_oversized_record_values() -> None:
    """Legacy recorded metadata does not automatically qualify for portable settings export."""
    for values in ({"password": "secret"}, {"theme": "x" * 65537}):
        record = ResolvedSettings(
            {
                "requested": values,
                "effective": values,
                "origins": {k: "run" for k in values},
                "policy_ref": REF,
                "environment_ref": ENV,
                "rejected_fields": [],
            }
        )
        with pytest.raises(SettingsRefused):
            export_settings(record)


def test_portable_import_refuses_unpaired_unicode_before_returning_values() -> None:
    """Malformed Unicode yields the authored refusal through the public import API."""
    wire = '{"schema":"quantum_workspace_settings.v1","values":{"theme":"\ud800"}}'
    with pytest.raises(SettingsRefused, match="Malformed portable settings"):
        import_settings(wire)
