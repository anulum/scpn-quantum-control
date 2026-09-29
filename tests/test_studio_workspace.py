# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio workspace facade tests
"""Exercise the existing Studio facade with a complete empty local project."""

import scpn_quantum_control.studio as studio
from scpn_quantum_control.studio.workspace import (
    admit_workspace,
    parse_workspace_manifest,
    read_json,
    write_json,
)


def test_studio_entry_admits_empty_project_without_execution() -> None:
    """Create, export and admit a new project through the public Studio surface."""
    payload = {
        "schema": "quantum_workspace.v1",
        "body": {
            "project_id": "10000000-0000-4000-8000-000000000001",
            "revision_refs": [],
            "draft_ref": None,
            "created_at": "2026-09-29T00:00:00Z",
            "updated_at": "2026-09-29T00:00:00Z",
            "artefact_refs": [],
        },
        "extensions": {"title": "Synthetic local workspace"},
    }
    workspace = parse_workspace_manifest(payload)
    assert read_json(write_json(workspace.to_dict())) == payload
    receipt = admit_workspace(workspace, {}, {}, {}, {}, {})
    assert receipt.project_id == workspace.body["project_id"]
    assert receipt.document_hashes == receipt.raw_hashes == ()
    assert studio.parse_workspace_manifest(payload).digest == workspace.digest
    assert "parse_workspace_manifest" in dir(studio)
