# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Dated Backend Profile Tests
"""Exercise offline profile projection through the original public facade."""

from __future__ import annotations

import json
import runpy
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from scpn_quantum_control.hardware.hal import built_in_backend_profiles
from scpn_quantum_control.hardware.provider_capability_discovery import (
    ProfileBinding,
    ProviderCapabilitySnapshot,
    build_backend_profiles,
    build_provider_route_catalogue,
)
from scpn_quantum_control.studio_workspace.canonical import canonical_digest

DAY = "2026-10-03"


def test_operator_backend_profiles_01() -> None:
    """The public offline adapter returns dated, digest-bound immutable-source metadata."""
    payload = build_backend_profiles(observed_at=DAY)
    assert payload["schema"] == "studio.backend-profiles.v1"
    assert payload["body"]["no_submit"] is True
    assert payload["body"]["observed_at"] == DAY
    assert len(payload["body"]["profiles"]) == len(build_provider_route_catalogue(observed_at=DAY))
    envelope = {key: payload[key] for key in ("schema", "body", "extensions")}
    assert canonical_digest(payload["schema"], envelope) == payload["sha256"]
    payload["body"]["profiles"][0]["body"]["device"] = "changed-copy"
    assert (
        build_backend_profiles(observed_at=DAY)["body"]["profiles"][0]["body"]["device"]
        != "changed-copy"
    )


def test_operator_backend_profiles_02() -> None:
    """Absent observations remain null instead of certifying availability or a zero limit."""
    payload = build_backend_profiles(observed_at=DAY)
    for row in payload["body"]["profiles"]:
        body = row["body"]
        assert body["observed"]["online"] is None
        assert body["observed"]["max_shots"] is None
        assert all(verb["declared"] is None and verb["observed"] is None for verb in body["verbs"])
        assert body["credential_refs"] is None or all(
            ref.startswith("credential-ref:") and len(ref) == 79 for ref in body["credential_refs"]
        )
        assert body["options"]["pulse"]["supported"] == body["declared"]["supports_pulse"]
        assert body["options"]["analog"]["supported"] == body["declared"]["supports_analog"]


def test_operator_backend_profiles_03() -> None:
    """Imported dependent references bind exactly one dated profile and no other row."""
    payload = build_backend_profiles(observed_at=DAY)
    row = payload["body"]["profiles"][0]
    binding = ProfileBinding(row["sha256"], "a" * 64, "b" * 64, "c" * 64)
    assert ProfileBinding(row["sha256"], None, "b" * 64, "c" * 64).plan_ref is None
    bound = build_backend_profiles(observed_at=DAY, binding=binding)
    assert bound["body"]["binding"]["profile_sha256"] == row["sha256"]
    with pytest.raises(ValueError, match="binding"):
        build_backend_profiles(observed_at="2026-10-04", binding=binding)


def test_operator_backend_profiles_04() -> None:
    """Even two broker aliases of the same backend retain separate profile identity."""
    payload = build_backend_profiles(observed_at=DAY)
    rows = {row["body"]["route_id"]: row for row in payload["body"]["profiles"]}
    one, two = rows["strangeworks/ibm_quantum"], rows["strangeworks/qiskit_runtime"]
    assert one["body"]["backend_id"] == two["body"]["backend_id"]
    assert one["sha256"] != two["sha256"]
    assert rows["direct/iqm"]["sha256"] != rows["qbraid/iqm"]["sha256"]


def test_operator_backend_profiles_05(monkeypatch: pytest.MonkeyPatch) -> None:
    """Public metadata load never opens network, reads environment or submits a job."""
    import os
    import socket

    from scpn_quantum_control.hardware.hal import HardwareAbstractionLayer

    def reject(*args: object, **kwargs: object) -> None:
        raise AssertionError("metadata crossed an execution/secret boundary")

    with monkeypatch.context() as guard:
        guard.setattr(HardwareAbstractionLayer, "submit", reject)
        guard.setattr(HardwareAbstractionLayer, "status", reject)
        guard.setattr(HardwareAbstractionLayer, "result", reject)
        guard.setattr(HardwareAbstractionLayer, "cancel", reject)
        guard.setattr(socket, "socket", reject)
        guard.setattr(os, "getenv", reject)
        guard.setattr(type(os.environ), "__getitem__", reject)
        payload = build_backend_profiles(observed_at=DAY)
    assert payload["body"]["profiles"]


def test_supplied_snapshot_projects_only_whitelisted_observations() -> None:
    """A supplied offline observation keeps device and calibration without arbitrary secrets."""
    snapshot = ProviderCapabilitySnapshot(
        route_id="direct/iqm",
        aggregator="direct",
        provider="iqm",
        backend_id="iqm_cloud",
        target_name="Garnet",
        n_qubits=20,
        supported_ir_formats=("openqasm3",),
        online=False,
        max_shots=4096,
        max_circuits=8,
        queue_depth=0,
        calibration_timestamp="2026-10-02T01:00:00Z",
        metadata={"api_key": "do-not-export", "region": "sensitive-arbitrary"},
    )
    payload = build_backend_profiles(observed_at=DAY, snapshots={snapshot.route_id: snapshot})
    row = next(
        row["body"]
        for row in payload["body"]["profiles"]
        if row["body"]["route_id"] == snapshot.route_id
    )
    assert row["device"] == "Garnet" and row["observed"]["online"] is False
    assert row["observed"]["queue_depth"] == 0 and row["observed"]["n_qubits"] == 20
    assert row["observed"]["calibration_ref"] is not None
    assert "do-not-export" not in json.dumps(payload) and "sensitive-arbitrary" not in json.dumps(
        payload
    )
    changed = build_backend_profiles(
        observed_at=DAY,
        snapshots={snapshot.route_id: replace(snapshot, target_name="Other-device")},
    )
    assert changed["sha256"] != payload["sha256"]


@pytest.mark.parametrize(
    "value", ["", "2026-02-29", "2026-10-03\n", "2026-10-3", "2026-10-03T00:00:00Z"]
)
def test_bad_dates_refuse_before_projection(value: str) -> None:
    """Impossible or ambiguous snapshot dates cannot mint a valid profile."""
    with pytest.raises(ValueError):
        build_backend_profiles(observed_at=value)


@pytest.mark.parametrize("value", ["", "x" * 64, "a" * 63, "A" * 64, "a" * 64 + "\n"])
def test_binding_requires_exact_opaque_digest(value: str) -> None:
    """Arbitrary credentials, URLs and malformed references cannot be stored as bindings."""
    with pytest.raises(ValueError):
        ProfileBinding(value, None, None, None)
    with pytest.raises(ValueError):
        ProfileBinding("a" * 64, value, None, None)


def test_empty_unknown_duplicate_and_mismatched_sources_are_refused() -> None:
    """No partial inventory or cross-route observation may enter a successful bundle."""
    rows = build_provider_route_catalogue(observed_at=DAY)
    with pytest.raises(ValueError):
        build_backend_profiles(observed_at=DAY, catalogue=())
    with pytest.raises(ValueError):
        build_backend_profiles(observed_at=DAY, catalogue=(rows[0], rows[0]))
    with pytest.raises(ValueError):
        build_backend_profiles(observed_at=DAY, profiles=())
    with pytest.raises(ValueError):
        build_backend_profiles(observed_at=DAY, profiles=(built_in_backend_profiles()[0],) * 2)
    with pytest.raises(ValueError):
        build_backend_profiles(
            observed_at=DAY, catalogue=build_provider_route_catalogue(observed_at="2026-10-04")
        )
    snapshot = ProviderCapabilitySnapshot(
        route_id="unknown",
        aggregator="direct",
        provider="iqm",
        backend_id="iqm_cloud",
        target_name="Garnet",
        n_qubits=20,
        supported_ir_formats=("openqasm3",),
    )
    with pytest.raises(ValueError):
        build_backend_profiles(observed_at=DAY, snapshots={"unknown": snapshot})
    with pytest.raises(ValueError):
        build_backend_profiles(observed_at=DAY, snapshots={"direct/iqm": snapshot})


def test_native_projection_refuses_invalid_observed_types_and_route_aliases() -> None:
    """Unchecked SDK types cannot become valid counts or bind a neighbouring route."""
    snapshot = ProviderCapabilitySnapshot(
        route_id="direct/iqm",
        aggregator="direct",
        provider="iqm",
        backend_id="iqm_cloud",
        target_name="Garnet",
        n_qubits=20,
        supported_ir_formats=("openqasm3",),
    )
    for changes in (
        {"aggregator": "qbraid"},
        {"provider": "other"},
        {"backend_id": "other"},
        {"online": 1},
        {"n_qubits": True},
        {"n_qubits": 1.5},
        {"max_shots": True},
        {"max_circuits": 1.5},
        {"n_qubits": 2**64},
        {"calibration_timestamp": "2026-10-03T01:00:00+00:00"},
        {"calibration_timestamp": "2026-02-29T01:00:00Z"},
        {"basis_gates": ("rz", "rz")},
        {"native_features": tuple(f"feature-{index}" for index in range(257))},
    ):
        changed = replace(snapshot, **changes)
        with pytest.raises(ValueError):
            build_backend_profiles(observed_at=DAY, snapshots={"direct/iqm": changed})
    accepted = build_backend_profiles(observed_at=DAY, snapshots={"direct/iqm": snapshot})
    assert accepted["body"]["profiles"]


def test_source_region_controls_and_capability_truthiness_are_refused() -> None:
    """The public projection rejects malformed declarations without coercion."""
    profile = next(p for p in built_in_backend_profiles() if p.backend_id == "iqm_cloud")
    for region in ("", " ", "bad\nregion", "x" * 513, "\ud800"):
        with pytest.raises(ValueError):
            build_backend_profiles(
                observed_at=DAY,
                profiles=(replace(profile, region=region),),
                catalogue=tuple(
                    row
                    for row in build_provider_route_catalogue(observed_at=DAY)
                    if row.route_id == "direct/iqm"
                ),
            )
    for capabilities in (
        replace(profile.capabilities, supports_pulse=cast(Any, 1)),
        replace(profile.capabilities, max_qubits=True),
    ):
        with pytest.raises(ValueError):
            build_backend_profiles(
                observed_at=DAY,
                profiles=(replace(profile, capabilities=capabilities),),
                catalogue=tuple(
                    row
                    for row in build_provider_route_catalogue(observed_at=DAY)
                    if row.route_id == "direct/iqm"
                ),
            )


def test_profile_count_and_encoded_byte_ceilings_refuse_whole_envelope() -> None:
    """A resource excess refuses before a partial catalogue can escape."""
    rows = build_provider_route_catalogue(observed_at=DAY)
    with pytest.raises(ValueError, match="ceiling"):
        build_backend_profiles(
            observed_at=DAY,
            catalogue=tuple(replace(rows[0], route_id=f"alias/{i}") for i in range(257)),
        )
    large_names = tuple(f"{i:03d}" + "x" * 509 for i in range(256))
    selected = tuple(
        row for row in rows if row.route_id in ("direct/iqm", "aws_braket/iqm", "qbraid/iqm")
    )
    snapshots = {
        row.route_id: ProviderCapabilitySnapshot(
            route_id=row.route_id,
            aggregator=row.broker or "direct",
            provider=row.provider,
            backend_id=row.device,
            target_name="opaque-target",
            n_qubits=20,
            supported_ir_formats=large_names,
            basis_gates=large_names,
            native_features=large_names,
        )
        for row in selected
    }
    with pytest.raises(ValueError, match="UTF-8 ceiling"):
        build_backend_profiles(observed_at=DAY, catalogue=selected, snapshots=snapshots)


def test_original_offline_export_cli_is_reproducible_and_checks_custody(
    tmp_path: Path,
) -> None:
    """The real CLI writes exact metadata and refuses stale checks without rewriting.

    Parameters
    ----------
    tmp_path
        Owned temporary export directory.

    """
    from tools.export_backend_profiles import main

    output = tmp_path / "profiles.json"
    argv = ["--observed-at", DAY, "--output", str(output)]
    assert main([*argv, "--check"]) == 1
    assert main(argv) == 0 and main([*argv, "--check"]) == 0
    output.write_text("invalid-prior")
    assert main([*argv, "--check"]) == 1 and output.read_text() == "invalid-prior"


def test_export_command_executes_the_original_public_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command entry point uses real process arguments and emits exact metadata.

    Parameters
    ----------
    tmp_path
        Owned new export directory.
    monkeypatch
        Isolated process argv for the actual public script entry point.

    """
    output = tmp_path / "cli" / "profiles.json"
    script = Path(__file__).resolve().parents[1] / "tools/export_backend_profiles.py"
    with monkeypatch.context() as args:
        args.setattr(sys, "argv", [str(script), "--observed-at", DAY, "--output", str(output)])
        with pytest.raises(SystemExit) as result:
            runpy.run_path(str(script), run_name="__main__")
    assert result.value.code == 0
    assert json.loads(output.read_text()) == build_backend_profiles(observed_at=DAY)
