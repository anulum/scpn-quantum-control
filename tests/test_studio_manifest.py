# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio schema-A manifest tests
"""Tests for the QUANTUM studio federation manifest (schema A + extension)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")

from scpn_quantum_control.studio import federation, manifest, verbs  # noqa: E402


def test_manifest_builds_with_studio_identity() -> None:
    """The schema-A manifest carries the studio identity and all verbs."""
    data = manifest.build_manifest().to_dict()
    assert data["studio"] == "scpn-quantum-control"
    assert len(data["verbs"]) == len(verbs.QUANTUM_VERBS) == 8
    assert data["content_digest"].startswith("sha256:")


def test_content_digest_is_reproducible() -> None:
    """The content digest depends on the declared surface, not git state."""
    first = manifest.build_manifest().to_dict()["content_digest"]
    second = manifest.build_manifest().to_dict()["content_digest"]
    assert first == second


def test_evidence_schemas_match_verb_outputs() -> None:
    """Every evidence schema a verb produces is declared in evidence_schemas()."""
    declared = set(verbs.evidence_schemas())
    produced = {schema for verb in verbs.QUANTUM_VERBS for schema in verb.produces}
    assert produced == declared


def test_verb_vocabulary_uses_locked_enums() -> None:
    """Verb attributes serialise to the locked SDK enum value strings."""
    data = manifest.build_manifest().to_dict()
    by_name = {v["verb"]: v for v in data["verbs"]}
    assert by_name["compile"]["side_effect"] == "read-only"
    assert by_name["compile"]["fidelity"] == "first-principles"
    assert by_name["simulate"]["side_effect"] == "simulated"


def test_execute_is_the_only_live_hardware_verb() -> None:
    """Only `execute` drives a QPU; it is certified-tier and live-hardware."""
    data = manifest.build_manifest().to_dict()
    live = [v["verb"] for v in data["verbs"] if v["side_effect"] == "live-hardware"]
    assert live == ["execute"]
    execute = next(v for v in data["verbs"] if v["verb"] == "execute")
    assert execute["safety_tier"] == "certified"


def test_federation_document_has_both_blocks() -> None:
    """The federation document is schema_a core + architecture_map extension."""
    doc = federation.build_federation_document()
    assert set(doc) == {"schema_a", "architecture_map"}
    assert doc["schema_a"]["studio"] == "scpn-quantum-control"


def test_architecture_map_extension_shape() -> None:
    """The v2 extension block carries the fleet-aligned field set (peer-aligned with SC-NEUROCORE)."""
    ext = federation.build_architecture_map_extension()
    assert ext["version"] == "architecture-map.v2"
    assert {
        "pipeline_stages",
        "capabilities",
        "backends",
        "interfaces",
        "wire_formats",
        "cross_repo",
        "boundaries",
    } <= set(ext)
    stages = [s["stage"] for s in ext["pipeline_stages"]]
    assert stages == [
        "problem",
        "hamiltonian",
        "circuit",
        "execution",
        "mitigation",
        "analysis",
        "ledger",
    ]
    assert {b["name"] for b in ext["backends"]} >= {"rust", "julia", "python"}
    assert {b["status"] for b in ext["backends"]} <= {
        "runtime-active",
        "build-available",
        "declared",
    }
    assert {c["status"] for c in ext["capabilities"]} <= {
        "wired",
        "library-only",
        "stub",
        "feasibility-only",
    }
    assert all({"kind", "entry"} <= set(i) for i in ext["interfaces"])
    assert all({"name", "schema_ref"} <= set(w) for w in ext["wire_formats"])


def test_federation_document_is_json_serialisable() -> None:
    """The whole document round-trips through JSON."""
    doc = federation.build_federation_document()
    assert json.loads(json.dumps(doc)) == doc


def test_write_federation_document(tmp_path: Path) -> None:
    """The emitter writes a parseable federation document to the studio path."""
    out = federation.write_federation_document(tmp_path)
    assert out == tmp_path / federation.STUDIO_MANIFEST_PATH
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["schema_a"]["studio"] == "scpn-quantum-control"


def test_manifest_passes_studio_conformance_gate() -> None:
    """The CapabilityManifest is admitted by the platform ``validate_studio_manifest`` gate.

    Keeps the federation contract honest in CI: any schema-A drift (bad digest form,
    duplicate verb, unversioned evidence schema, unknown contract era) reds the build.
    """
    from scpn_studio_platform import manifest as platform_manifest

    validate = getattr(platform_manifest, "validate_studio_manifest", None)
    if validate is None:  # pragma: no cover - only on SDK < 0.8
        pytest.skip("validate_studio_manifest unavailable (scpn-studio-platform < 0.8)")
    verdict = validate(manifest.build_manifest().to_dict())
    assert verdict.admitted, f"manifest rejected: {verdict.rejections}"
    assert verdict.rejections == ()
    assert verdict.warnings == ()
