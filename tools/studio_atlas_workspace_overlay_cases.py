# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — committed parity cases of the Atlas workspace overlay
"""Build, write and check the cases that bind the browser overlay to its owner.

Each case states a workspace view, optionally the case whose overlay precedes
it, and the overlay the Python owner
(:mod:`scpn_quantum_control.atlas.extractors.workspace`) computes from it: the
snapshot digest, the private node identifiers, the private edges, the
tombstones and the exclusions. The browser twin must reproduce every digest.

Run ``python tools/studio_atlas_workspace_overlay_cases.py --write`` after the
projection rules change, and ``--check`` to compare the committed file with a
fresh build.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from scpn_quantum_control.atlas import AtlasSnapshot, atlas_snapshot_digest, parse_atlas_snapshot
from scpn_quantum_control.atlas.extractors.workspace import (
    WORKSPACE_OVERLAY_VERSION,
    overlay_workspace,
    parse_workspace_view,
)
from scpn_quantum_control.studio.committed_data import run_committed_data_cli

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_PATH: Final[Path] = ROOT / "data/studio/atlas_workspace_overlay_cases.json"
CASES_SCHEMA: Final[str] = "scpn_qc_studio_atlas_workspace_overlay_cases_v1"

_PROJECT: Final[str] = "0f0e0d0c-0b0a-4908-8706-050403020100"
_OTHER_PROJECT: Final[str] = "1e2d3c4b-5a69-4788-9a6b-5c4d3e2f1a0b"
_RUN: Final[str] = "11111111-2222-4333-8444-555555555555"
_ATTEMPT: Final[str] = "66666666-7777-4888-8999-aaaaaaaaaaaa"


def _hex(character: str) -> str:
    """Return a 64-character digest made of one repeated hex digit.

    Parameters
    ----------
    character
        One lowercase hex digit.

    Returns
    -------
    str
        The repeated digit.

    """
    return character * 64


PUBLIC: Final[Mapping[str, object]] = {
    "schema": "quantum_atlas_snapshot.v1",
    "extractor_version": "atlas-extractor.2",
    "source_commit": "0123456789abcdef0123456789abcdef01234567",
    "evidence_set_hash": _hex("e"),
    "access_scope": "public",
    "nodes": [
        {
            "id": "capability:atlas.fixture/overlay",
            "kind": "capability",
            "label": "Overlay fixture capability",
            "source_ref": "data/fixture.json",
            "source_digest": None,
            "validity_domain": None,
            "as_of": None,
            "access": "public",
            "status": "current",
            "evidence_refs": [_hex("f")],
            "null_reasons": {
                "as_of": "committed declarations carry no observation time",
                "source_digest": "a declaration inside its input, without its own content",
                "validity_domain": "the input states no validity scope",
            },
        },
        {
            "id": "definition:repo/src:overlay.py",
            "kind": "definition",
            "label": "src/overlay.py",
            "source_ref": "src/overlay.py",
            "source_digest": _hex("d"),
            "validity_domain": None,
            "as_of": None,
            "access": "public",
            "status": "current",
            "evidence_refs": [_hex("f")],
            "null_reasons": {
                "as_of": "committed declarations carry no observation time",
                "validity_domain": "the input states no validity scope",
            },
        },
    ],
    "edges": [
        {
            "source": "definition:repo/src:overlay.py",
            "kind": "implements",
            "target": "capability:atlas.fixture/overlay",
            "relation_version": "atlas_relation.v1",
            "derivation_method": "declared",
            "source_ref": "data/fixture.json",
            "evidence_digest": _hex("f"),
            "assumptions": [],
            "mapping": None,
            "as_of": None,
            "access": "public",
        }
    ],
    "tombstones": [
        {
            "id": "definition:repo/src:retired.py",
            "kind": "definition",
            "last_source_digest": _hex("c"),
            "removed_at": "2026-11-20T00:00:00Z",
            "reason": "no longer extracted from the admitted inputs",
        }
    ],
    "exclusions": [{"source_ref": "data/fixture.json:missing.py", "reason": "absent"}],
}
"""A small valid public snapshot, so the cases do not follow every pin refresh."""


def _revision(parents: list[str], program: str, keys: list[str]) -> dict[str, object]:
    """Return an experiment revision document with the fields the overlay reads.

    Parameters
    ----------
    parents
        Parent revision hashes.
    program
        Program digest.
    keys
        Bound parameter keys.

    Returns
    -------
    dict[str, object]
        The document.

    """
    return {
        "schema": "experiment_revision.v1",
        "body": {
            "parent_revision_hashes": parents,
            "program_ref": {"schema": "program.v1", "sha256": program},
            "parameters": {key: {"dtype": "float64"} for key in keys},
        },
    }


WORKSPACES: Final[tuple[tuple[str, str | None, Mapping[str, object]], ...]] = (
    (
        "revisions_parameters_programs_and_runs",
        None,
        {
            "project_id": _PROJECT,
            "workspace_hash": _hex("a"),
            "updated_at": "2026-11-23T10:00:00Z",
            "documents": {
                _hex("1"): {"schema": "parameter_spec.v1", "body": {"key": "coupling.K"}},
                _hex("2"): {"schema": "parameter_spec.v1", "body": {"key": "omega"}},
                _hex("5"): {"schema": "parameter_spec.v1", "body": {"key": "omega"}},
                _hex("6"): {"schema": "parameter_spec.v1", "body": {"key": "bad key"}},
                _hex("3"): _revision([], _hex("9"), ["coupling.K", "omega", "bad key"]),
                _hex("4"): _revision([_hex("3"), _hex("7")], _hex("9"), ["coupling.K"]),
                _hex("8"): {
                    "schema": "local_run_record.v1",
                    "body": {"run_id": _RUN, "attempt_id": _ATTEMPT, "revision_hash": _hex("4")},
                },
                _hex("b"): {
                    "schema": "local_run_record.v1",
                    "body": {
                        "run_id": "run-1",
                        "attempt_id": _ATTEMPT,
                        "revision_hash": _hex("4"),
                    },
                },
                _hex("c"): {"schema": "resolved_settings.v1", "body": {}},
                _hex("e"): {
                    "schema": "local_run_record.v1",
                    "body": {
                        "run_id": "22222222-3333-4444-8555-666666666666",
                        "attempt_id": _ATTEMPT,
                        "revision_hash": _hex("7"),
                    },
                },
            },
        },
    ),
    (
        "empty_workspace",
        None,
        {
            "project_id": _PROJECT,
            "workspace_hash": _hex("0"),
            "updated_at": "2026-11-23T09:00:00Z",
            "documents": {},
        },
    ),
    (
        "replaced_workspace",
        "revisions_parameters_programs_and_runs",
        {
            "project_id": _OTHER_PROJECT,
            "workspace_hash": _hex("b"),
            "updated_at": "2026-11-23T11:00:00Z",
            "documents": {_hex("3"): _revision([], _hex("9"), [])},
        },
    ),
)
"""Named views and the case each one follows."""


def _expected(snapshot: AtlasSnapshot) -> dict[str, object]:
    """Return the facts of an overlay that the browser must reproduce.

    Parameters
    ----------
    snapshot
        The overlay.

    Returns
    -------
    dict[str, object]
        Its digest, private nodes and edges, tombstones and exclusions.

    """
    return {
        "snapshot_digest": atlas_snapshot_digest(snapshot),
        "private_node_ids": [node.id for node in snapshot.nodes if node.access == "private"],
        "private_edges": [
            [edge.source, edge.kind, edge.target]
            for edge in snapshot.edges
            if edge.access == "private"
        ],
        "tombstone_ids": [tombstone.id for tombstone in snapshot.tombstones],
        "exclusions": [
            [exclusion.source_ref, exclusion.reason] for exclusion in snapshot.exclusions
        ],
    }


def build_overlay_cases() -> dict[str, object]:
    """Build every committed overlay case with the owner's expected result.

    Returns
    -------
    dict[str, object]
        The cases document.

    """
    public = parse_atlas_snapshot(PUBLIC)
    overlays: dict[str, AtlasSnapshot] = {}
    cases: list[dict[str, object]] = []
    for name, previous, workspace in WORKSPACES:
        overlay = overlay_workspace(
            public,
            parse_workspace_view(workspace),
            previous=None if previous is None else overlays[previous],
        )
        overlays[name] = overlay
        cases.append(
            {
                "name": name,
                "previous": previous,
                "workspace": workspace,
                "expected": _expected(overlay),
            }
        )
    return {
        "schema": CASES_SCHEMA,
        "overlay_version": WORKSPACE_OVERLAY_VERSION,
        "public": PUBLIC,
        "public_digest": atlas_snapshot_digest(public),
        "cases": cases,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print, write or check the committed overlay cases.

    Parameters
    ----------
    argv
        CLI arguments, or the process arguments when omitted.

    Returns
    -------
    int
        The shared committed-data writer's exit status.

    """
    return run_committed_data_cli(
        argv,
        build=build_overlay_cases,
        default_path=DEFAULT_PATH,
        label="Atlas workspace overlay cases",
        description=__doc__,
    )


if __name__ == "__main__":
    sys.exit(main())
