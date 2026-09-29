# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — workspace reference admission
"""Admit complete local reference graphs without fetching or rewriting evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

from .contracts import (
    ExperimentRevision,
    LocalRunRecord,
    ParameterSpec,
    ResolvedSettings,
    WorkspaceDocument,
    WorkspaceManifest,
    parse_document,
    parse_parameter_spec,
    parse_workspace_manifest,
    validate_parameter_binding,
)


@dataclass(frozen=True)
class RawIdentity:
    """Original producer's verified identity, without workspace re-encoding.

    Parameters
    ----------
    schema
        Schema verified by the registered producer codec.
    kind
        Producer-owned record role, such as problem, program or plan.
    digest
        Digest verified using that producer's original encoding.

    """

    schema: str
    kind: str
    digest: str


@dataclass(frozen=True)
class RawArtifact:
    """Immutable source bytes that require a registered owning codec.

    Parameters
    ----------
    schema
        Claimed schema used to select the verifier, never proof of support.
    content
        Original bytes; no path, URL or executable input is loaded.

    """

    schema: str
    content: bytes

    def __post_init__(self) -> None:
        """Refuse mutable raw storage or an empty producer schema."""
        if (
            not isinstance(self.content, bytes)
            or not isinstance(self.schema, str)
            or not self.schema
        ):
            raise ValueError("raw artifact: immutable bytes and schema required")


@dataclass(frozen=True)
class WorkspaceAdmission:
    """Structural reference receipt, never permission to execute a run.

    Parameters
    ----------
    project_id
        Admitted project UUID.
    workspace_hash
        Exact admitted root identity, including its extensions and reference lists.
    document_hashes
        Sorted exact workspace document identities.
    raw_hashes
        Sorted original producer identities actually verified.

    """

    project_id: str
    workspace_hash: str
    document_hashes: tuple[str, ...]
    raw_hashes: tuple[str, ...]


def _acyclic(edges: Mapping[str, tuple[str, ...]], path: str) -> None:
    pending = {node: len(set(parents)) for node, parents in edges.items()}
    children: dict[str, list[str]] = {node: [] for node in edges}
    for node, parents in edges.items():
        for parent in set(parents):
            if parent not in edges:
                raise ValueError(f"{path}: dangling dependency {parent}")
            children[parent].append(node)
    ready = [node for node, degree in pending.items() if degree == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for child in children[node]:
            pending[child] -= 1
            if pending[child] == 0:
                ready.append(child)
    if visited != len(edges):
        raise ValueError(f"{path}: dependency cycle")


def admit_workspace(
    manifest: WorkspaceManifest,
    documents: Mapping[str, WorkspaceDocument],
    raw_records: Mapping[str, RawArtifact],
    raw_codecs: Mapping[str, Callable[[bytes], RawIdentity]],
    parameter_specs: Mapping[str, ParameterSpec],
    parameter_units: Mapping[str, str],
) -> WorkspaceAdmission:
    """Validate complete immutable indexes before a caller persists a workspace.

    Parameters
    ----------
    manifest
        Project root to admit, revalidated rather than trusted by type alone.
    documents
        Content-addressed workspace records, including parameter specifications.
    raw_records
        Original producer bytes indexed by their original digests.
    raw_codecs
        Explicit trusted offline, side-effect-free owning verifiers; never loaded
        from imported data. Unknown schemas receive no fallback.
    parameter_specs
        Parameter-key index, bound to immutable revision input references.
    parameter_units
        Explicit caller unit labels for the parameter-key index.

    Returns
    -------
    WorkspaceAdmission
        Exact reference identities; no filesystem, provider or worker action.

    Raises
    ------
    ValueError
        Identity, kind, parent, project, codec, parameter or domain is invalid.

    """
    root = parse_workspace_manifest(manifest.to_dict())
    project_id = cast(str, root.body["project_id"])
    index: dict[str, WorkspaceDocument] = {}
    for digest, candidate in tuple(documents.items()):
        record = parse_document(candidate.to_dict())
        if record.digest != digest:
            raise ValueError("$.documents: digest mismatch")
        if isinstance(record, WorkspaceManifest) and record.digest != root.digest:
            raise ValueError("$.documents: unexpected workspace root")
        if isinstance(record, ExperimentRevision) and record.body["project_id"] != project_id:
            raise ValueError("$.documents: cross-project revision")
        index[digest] = record
    if set(index) & set(raw_records):
        raise ValueError("$.documents: ambiguous raw/document identity")
    specs = {
        key: parse_parameter_spec(spec.to_dict()) for key, spec in tuple(parameter_specs.items())
    }
    units = dict(parameter_units)
    if specs.keys() != units.keys():
        raise ValueError("$.parameter_units: keys must match specifications")
    for key, spec in specs.items():
        if key != spec.body["key"] or not isinstance(index.get(spec.digest), ParameterSpec):
            raise ValueError("$.parameter_specs: key or indexed identity mismatch")
    _acyclic(
        {key: cast(tuple[str, ...], spec.body["dependency_keys"]) for key, spec in specs.items()},
        "$.parameter_specs",
    )
    raw_snapshot = dict(raw_records)
    codecs = dict(raw_codecs)
    verified: dict[str, RawIdentity] = {}

    def raw_identity(digest: str) -> RawIdentity:
        if digest in verified:
            return verified[digest]
        artifact = raw_snapshot.get(digest)
        if artifact is None:
            raise ValueError("$.references: dangling raw reference")
        codec = codecs.get(artifact.schema)
        if codec is None:
            raise ValueError("$.references: unsupported raw producer/schema")
        identity = codec(artifact.content)
        if (
            identity.schema != artifact.schema
            or identity.digest != digest
            or not isinstance(identity.kind, str)
            or not identity.kind
        ):
            raise ValueError("$.references: raw producer identity mismatch")
        verified[digest] = identity
        return identity

    workspace_schemas = {
        cls.schema
        for cls in (
            WorkspaceManifest,
            ExperimentRevision,
            ParameterSpec,
            ResolvedSettings,
            LocalRunRecord,
        )
    }

    def resolve(ref: Mapping[str, object], expected: str | None = None) -> None:
        digest = cast(str, ref["sha256"])
        record = index.get(digest)
        if record is None:
            if ref["schema"] in workspace_schemas:
                raise ValueError("$.references: workspace reference requires indexed document")
            raw = raw_identity(digest)
            schema, kind = raw.schema, raw.kind
        else:
            schema, kind = record.schema, record.schema
        if schema != ref["schema"] or (expected is not None and kind != expected):
            raise ValueError("$.references: schema or kind mismatch")

    for ref in cast(tuple[Mapping[str, object], ...], root.body["revision_refs"]):
        resolve(ref, ExperimentRevision.schema)
    for ref in cast(tuple[Mapping[str, object], ...], root.body["artefact_refs"]):
        resolve(ref)
    if root.body["draft_ref"] is not None:
        resolve(cast(Mapping[str, object], root.body["draft_ref"]), ExperimentRevision.schema)
    parents: dict[str, tuple[str, ...]] = {}
    for digest, record in index.items():
        body = record.body
        if isinstance(record, ExperimentRevision):
            parents[digest] = cast(tuple[str, ...], body["parent_revision_hashes"])
            resolve(cast(Mapping[str, object], body["problem_ref"]), "problem")
            resolve(cast(Mapping[str, object], body["program_ref"]), "program")
            resolve(
                cast(Mapping[str, object], body["semantic_settings_ref"]), ResolvedSettings.schema
            )
            inputs = cast(tuple[Mapping[str, object], ...], body["input_refs"])
            for ref in inputs:
                resolve(ref)
            input_hashes = {ref["sha256"] for ref in inputs}
            for key, value in cast(Mapping[str, Mapping[str, object]], body["parameters"]).items():
                binding_spec = specs.get(key)
                if binding_spec is None or binding_spec.digest not in input_hashes:
                    raise ValueError("$.parameters: missing immutable specification reference")
                validate_parameter_binding(binding_spec, value, units[key])
        elif isinstance(record, ResolvedSettings):
            resolve(cast(Mapping[str, object], body["policy_ref"]), "policy")
            resolve(cast(Mapping[str, object], body["environment_ref"]), "environment")
        elif isinstance(record, LocalRunRecord):
            if not isinstance(index.get(cast(str, body["revision_hash"])), ExperimentRevision):
                raise ValueError("$.run.revision_hash: missing revision")
            if raw_identity(cast(str, body["plan_hash"])).kind != "plan":
                raise ValueError("$.run.plan_hash: wrong raw kind")
            for ref in cast(tuple[Mapping[str, object], ...], body["output_refs"]):
                resolve(ref)
    _acyclic(parents, "$.revisions")
    return WorkspaceAdmission(
        project_id, root.digest, tuple(sorted(index)), tuple(sorted(verified))
    )
