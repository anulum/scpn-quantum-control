# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio executive replay handler tests
"""Tests for the hardware-result-pack re-verification ``replay`` handler."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")

from scpn_quantum_control.differentiable_claim_ledger import REPO_ROOT  # noqa: E402
from scpn_quantum_control.hardware_result_packs import (  # noqa: E402
    MANIFEST_RELATIVE_PATH,
    load_manifest,
)
from scpn_quantum_control.studio.executive import (  # noqa: E402
    ActionRegistry,
    ExecutiveRequest,
    preview_action,
    resolve_verb_contract,
    run_action,
)
from scpn_quantum_control.studio.executive_replay import (  # noqa: E402
    REPLAY_VERB,
    ReplayActionHandler,
    _as_pack_id,
    _normalise_replay,
    _safe_slug,
)

_COMMITTED_PACK_IDS = [
    str(pack["id"]) for pack in load_manifest(REPO_ROOT / MANIFEST_RELATIVE_PATH)["packs"]
]


def _registry() -> ActionRegistry:
    registry = ActionRegistry()
    registry.register(ReplayActionHandler())
    return registry


def _request(*, backend: str | None = None, **parameters: Any) -> ExecutiveRequest:
    return ExecutiveRequest(
        verb=REPLAY_VERB, action_id="replay-packs", parameters=parameters, backend=backend
    )


# --------------------------------------------------------------------------- #
# end-to-end
# --------------------------------------------------------------------------- #
def test_replay_reverifies_every_committed_pack() -> None:
    record = run_action(_request(), registry=_registry())
    assert record.result.status == "succeeded", record.result.error
    outputs = record.result.outputs
    assert outputs["replay_schema"] == "studio.evidence-replay.v1"
    assert outputs["replay_passed"] is True
    assert outputs["manifest"] == "data/hardware_result_packs/manifest.json"
    assert outputs["manifest_schema_version"] == 1
    assert outputs["pack_count"] == len(_COMMITTED_PACK_IDS)
    assert outputs["artifact_count"] >= outputs["pack_count"]
    assert [pack["id"] for pack in outputs["packs"]] == _COMMITTED_PACK_IDS
    assert all(pack["artifact_count"] >= 1 for pack in outputs["packs"])
    assert record.script is not None


def test_replay_selected_pack_only() -> None:
    record = run_action(_request(pack_ids=[_COMMITTED_PACK_IDS[0]]), registry=_registry())
    assert record.result.status == "succeeded", record.result.error
    outputs = record.result.outputs
    assert outputs["pack_count"] == 1
    assert outputs["packs"][0]["id"] == _COMMITTED_PACK_IDS[0]


def test_replay_fails_closed_on_unknown_pack_id() -> None:
    record = run_action(_request(pack_ids=["nonexistent-pack"]), registry=_registry())
    assert record.result.status == "failed"
    assert record.result.error is not None
    assert "unknown hardware result-pack IDs" in record.result.error
    assert record.script is None


def test_replay_fails_closed_on_artifact_drift(tmp_path: Any) -> None:
    import json
    import shutil

    from scpn_quantum_control.hardware_result_packs import verify_manifest

    manifest_path = REPO_ROOT / MANIFEST_RELATIVE_PATH
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pack = manifest["packs"][0]
    artifact = pack["artifacts"][0]
    source = REPO_ROOT / artifact["path"]
    target = tmp_path / artifact["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, target)
    target.write_bytes(target.read_bytes() + b"\n")
    drifted_manifest = tmp_path / "manifest.json"
    drifted_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_manifest(drifted_manifest, repo_root=tmp_path, pack_ids={str(pack["id"])})


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #
def test_replay_plan_defaults_python_backend_all_packs() -> None:
    plan = preview_action(_request(), registry=_registry())
    assert plan.backend == "python"
    assert plan.requires_approval is False
    assert len(plan.steps) == 5
    assert "all committed packs" in plan.steps[1]
    assert plan.parameters["pack_ids"] is None


def test_replay_plan_names_the_selection() -> None:
    plan = preview_action(_request(pack_ids=list(_COMMITTED_PACK_IDS[:2])), registry=_registry())
    assert "2 selected pack(s)" in plan.steps[1]
    assert plan.parameters["pack_ids"] == sorted(_COMMITTED_PACK_IDS[:2])


def test_replay_rejects_undeclared_backend() -> None:
    handler = ReplayActionHandler()
    contract = resolve_verb_contract(REPLAY_VERB)
    with pytest.raises(ValueError, match="is not declared for the replay verb"):
        handler.plan(_request(backend="abacus"), contract)


# --------------------------------------------------------------------------- #
# generated script
# --------------------------------------------------------------------------- #
def test_generated_replay_script_embeds_summary_and_compiles() -> None:
    record = run_action(_request(), registry=_registry())
    assert record.script is not None
    source = record.script.source
    compile(source, record.script.filename, "exec")
    assert record.script.filename == "replay_replay_packs.py"
    assert "verify_manifest" in source
    assert f"EXPECTED_PACK_COUNT = {record.result.outputs['pack_count']!r}" in source
    assert f"EXPECTED_ARTIFACT_COUNT = {record.result.outputs['artifact_count']!r}" in source
    assert "PACK_IDS = None" in source
    assert record.script.digest.startswith("sha256:")


def test_generated_selected_replay_script_embeds_pack_ids() -> None:
    record = run_action(_request(pack_ids=[_COMMITTED_PACK_IDS[0]]), registry=_registry())
    assert record.script is not None
    assert f"PACK_IDS = {[_COMMITTED_PACK_IDS[0]]!r}" in record.script.source


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [True, 1, "", "   ", None])
def test_as_pack_id_rejects_bad(bad: Any) -> None:
    with pytest.raises(ValueError):
        _as_pack_id(bad)


@pytest.mark.parametrize(
    "parameters",
    [
        {"unexpected": 1},
        {"pack_ids": "pack-a"},
        {"pack_ids": []},
        {"pack_ids": ["pack-a"] * 33},
        {"pack_ids": ["pack-a", "pack-a"]},
        {"pack_ids": ["pack-a", 2]},
    ],
)
def test_normalise_replay_rejects_invalid(parameters: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _normalise_replay(parameters)


def test_normalise_replay_defaults_and_sorts() -> None:
    assert _normalise_replay({}) == {"pack_ids": None}
    assert _normalise_replay({"pack_ids": ["b-pack", "a-pack"]}) == {
        "pack_ids": ["a-pack", "b-pack"]
    }


def test_safe_slug_normal_and_empty() -> None:
    assert _safe_slug("replay-packs.v1") == "replay_packs_v1"
    assert _safe_slug("!!!") == "action"
