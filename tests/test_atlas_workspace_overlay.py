# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — private Atlas overlay of an admitted workspace
"""Check the workspace overlay, its committed parity cases and its refusals."""

from __future__ import annotations

import json
import runpy
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from scpn_quantum_control.atlas import (
    AtlasContractError,
    AtlasSnapshot,
    parse_atlas_snapshot,
    snapshot_to_wire,
)
from scpn_quantum_control.atlas.extractors.workspace import (
    WorkspaceOverlayError,
    WorkspaceView,
    overlay_workspace,
    parse_workspace_view,
)
from tools.studio_atlas_workspace_overlay_cases import (
    DEFAULT_PATH,
    PUBLIC,
    WORKSPACES,
    build_overlay_cases,
    main,
)

Json = dict[str, object]


def _public(payload: Mapping[str, object] = PUBLIC) -> AtlasSnapshot:
    """Parse a public fixture snapshot.

    Parameters
    ----------
    payload : Mapping[str, object]
        The wire snapshot.

    Returns
    -------
    AtlasSnapshot
        The validated snapshot.

    """
    return parse_atlas_snapshot(payload)


def _view(**overrides: object) -> WorkspaceView:
    """Parse the first committed workspace with fields replaced.

    Parameters
    ----------
    **overrides : object
        Top-level fields to replace.

    Returns
    -------
    WorkspaceView
        The parsed view.

    """
    return parse_workspace_view({**WORKSPACES[0][2], **overrides})


def _revision(body: Json) -> WorkspaceView:
    """Return a view holding one experiment revision with ``body``.

    Parameters
    ----------
    body : Json
        The revision body.

    Returns
    -------
    WorkspaceView
        The parsed view.

    """
    return _view(documents={"3" * 64: {"schema": "experiment_revision.v1", "body": body}})


def test_the_committed_cases_are_current() -> None:
    """The committed cases equal a fresh build from the owner."""
    committed = json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(build_overlay_cases()))
    assert [case["name"] for case in committed["cases"]] == [name for name, _, _ in WORKSPACES]


def test_the_overlay_keeps_public_objects_and_projects_each_workspace_kind() -> None:
    """Every public object is unchanged; each workspace kind becomes a private node."""
    public = _public()
    overlay = overlay_workspace(public, _view())
    private = {node.id: node for node in overlay.nodes if node.access == "private"}

    assert overlay.access_scope == "private"
    assert set(public.nodes) <= set(overlay.nodes)
    assert set(public.edges) <= set(overlay.edges)
    assert {node.kind for node in private.values()} == {"parameter", "program", "revision", "run"}
    omega = private["parameter:workspace.0f0e0d0c-0b0a-4908-8706-050403020100/omega"]
    assert omega.source_digest is None
    assert dict(omega.null_reasons)["source_digest"] == (
        "the workspace holds several specifications of it"
    )
    assert all(edge.access == "private" for edge in overlay.edges if edge.source in private)


def test_a_replaced_workspace_buries_its_objects_and_carries_old_tombstones() -> None:
    """Gone objects are dated by the new root; earlier private tombstones are carried."""
    public = _public()
    first = overlay_workspace(public, _view())
    replaced = overlay_workspace(public, parse_workspace_view(WORKSPACES[2][2]), previous=first)
    emptied = overlay_workspace(public, parse_workspace_view(WORKSPACES[1][2]), previous=replaced)
    again = overlay_workspace(public, _view(), previous=replaced)

    runs = [tombstone for tombstone in emptied.tombstones if tombstone.id.startswith("run:")]
    assert {tombstone.removed_at for tombstone in runs} == {"2026-11-23T11:00:00Z"}
    assert {tombstone.reason for tombstone in runs} == {"no longer in the admitted workspace"}
    assert not any(tombstone.id.startswith("run:") for tombstone in again.tombstones)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (None, "$: must be an object"),
        ({**WORKSPACES[0][2], "project_id": "Project"}, "lowercase UUID"),
        ({**WORKSPACES[0][2], "project_id": 7}, "non-empty text"),
        ({**WORKSPACES[0][2], "workspace_hash": "a" * 63}, "lowercase sha256"),
        ({**WORKSPACES[0][2], "documents": []}, "$.documents: must be an object"),
        (
            {**WORKSPACES[0][2], "documents": {"1" * 64: {"body": {}}}},
            ".schema: must be non-empty",
        ),
    ],
)
def test_a_malformed_view_is_refused(payload: object, message: str) -> None:
    """Each malformed view field is refused with its address.

    Parameters
    ----------
    payload : object
        The view.
    message : str
        Expected part of the refusal.

    """
    with pytest.raises(WorkspaceOverlayError, match=message.replace("$", r"\$")):
        parse_workspace_view(payload)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"program_ref": {"sha256": "9" * 64}}, "parent_revision_hashes: must be an array"),
        (
            {"parent_revision_hashes": [3], "program_ref": {"sha256": "9" * 64}},
            r"parent_revision_hashes\[0\]: must be non-empty text",
        ),
        (
            {"parent_revision_hashes": [], "program_ref": {"sha256": "9" * 64}, "parameters": []},
            "parameters: must be an object",
        ),
    ],
)
def test_a_revision_without_what_the_projection_reads_is_refused(body: Json, message: str) -> None:
    """A revision missing a field the projection reads is refused.

    Parameters
    ----------
    body : Json
        The revision body.
    message : str
        Expected part of the refusal.

    """
    with pytest.raises(WorkspaceOverlayError, match=message):
        overlay_workspace(_public(), _revision(body))


def test_a_run_with_a_malformed_revision_is_refused() -> None:
    """A run record whose revision is not a digest is refused."""
    run = {"run_id": "r", "attempt_id": "a", "revision_hash": "x"}
    view = _view(documents={"8" * 64: {"schema": "local_run_record.v1", "body": run}})
    with pytest.raises(WorkspaceOverlayError, match="revision_hash: must be a lowercase sha256"):
        overlay_workspace(_public(), view)


def test_private_input_public_previous_collision_and_reuse_are_refused() -> None:
    """The inputs must be public then private, and identities never collide or revive."""
    public = _public()
    overlay = overlay_workspace(public, _view())
    with pytest.raises(WorkspaceOverlayError, match="public: must be a public snapshot"):
        overlay_workspace(overlay, _view())
    with pytest.raises(WorkspaceOverlayError, match="previous: must be an earlier private"):
        overlay_workspace(public, _view(), previous=public)

    wire = snapshot_to_wire(public)
    nodes = cast(list[Json], wire["nodes"])
    revision = f"revision:sha256:{'3' * 64}"
    clash = {
        **wire,
        "nodes": [
            *nodes,
            {**nodes[1], "id": revision, "kind": "revision", "source_digest": "3" * 64},
        ],
    }
    with pytest.raises(WorkspaceOverlayError, match="collides with a public node"):
        overlay_workspace(_public(clash), _view())

    tombstones = cast(list[Json], wire["tombstones"])
    buried = {
        **wire,
        "tombstones": [
            *tombstones,
            {
                "id": revision,
                "kind": "revision",
                "last_source_digest": "3" * 64,
                "removed_at": "2026-11-21T00:00:00Z",
                "reason": "fixture",
            },
        ],
    }
    with pytest.raises(AtlasContractError, match="is deleted and live"):
        overlay_workspace(_public(buried), _view())


def test_the_cases_cli_writes_and_checks(tmp_path: Path) -> None:
    """The committed-data CLI writes the cases and then finds them current.

    Parameters
    ----------
    tmp_path : Path
        Output directory.

    """
    output = tmp_path / "cases.json"
    assert main(["--write", "--json-path", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["cases"]
    assert main(["--check", "--json-path", str(output)]) == 0


def test_the_script_entry_point_checks_the_committed_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running the tool as a script exits with the check status.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces the process arguments.

    """
    monkeypatch.setattr(sys, "argv", ["studio_atlas_workspace_overlay_cases.py", "--check"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(
            str(DEFAULT_PATH.parents[2] / "tools/studio_atlas_workspace_overlay_cases.py"),
            run_name="__main__",
        )
    assert stopped.value.code == 0
