# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio capability manifest (schema A)
"""The QUANTUM studio's capability manifest (schema A) on the platform contract.

Authors QUANTUM's :class:`scpn_studio_platform.manifest.CapabilityManifest`: the
verbs it advertises (:mod:`scpn_quantum_control.studio.verbs`), the evidence schemas
they emit, and a content-addressed digest of that declared surface. The digest is
computed with :func:`scpn_studio_platform.manifest.content_digest` over the canonical
JSON of the declared verbs and evidence schemas, so it is reproducible across
checkouts and independent of git state — the cross-repo ``capability_manifest`` drift
gotcha does not apply to the federation contract block.

The federated Studio UI panel is DEPLOYED: ``ui_module`` points at the live
Module Federation remote under the Hub origin
(``https://www.anulum.org/studios/scpn-quantum-control/``), exposing
``./QuantumStudioPanel``. The values were probe-verified live by the platform
keeper (container init/get, all chunks under the studio path, zero page
errors) before they landed here.
"""

from __future__ import annotations

import json
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from scpn_studio_platform.manifest import (
    CapabilityManifest,
    TransportProfile,
    UiModule,
    content_digest,
)

from .verbs import QUANTUM_VERBS, STUDIO_ID, evidence_schemas, verb_substrates

PLATFORM_SDK_RANGE = ">=0.9,<0.12"
"""The platform SDK SemVer range the studio builds on (matches the ``studio`` extra)."""

STUDIO_REMOTE_ENTRY = "https://www.anulum.org/studios/scpn-quantum-control/remoteEntry.js"
"""The deployed Module Federation remote entry (platform pull-deploy target)."""

STUDIO_EXPOSED_MODULE = "./QuantumStudioPanel"
"""The exposed MF module the Hub mounts (must match module-federation.config.ts)."""

STUDIO_FEDERATION_NAME = "scpn_quantum_control"
"""The MF container name (must match ``name`` in module-federation.config.ts)."""

PROTOCOL_VERSION = "1"
"""The SYNAPSE wire protocol version the studio pins."""


def _resolve_studio_version() -> str:
    """Return the source-tree or installed ``scpn-quantum-control`` version.

    Returns
    -------
    str
        The source-tree ``pyproject.toml`` version when available, then the
        installed distribution version, otherwise ``"0+unknown"`` so the manifest
        never carries a fabricated version.

    """
    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    if pyproject.exists():
        metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project = metadata.get("project", {})
        source_version = project.get("version")
        if isinstance(source_version, str) and source_version:
            return source_version
    try:
        return version("scpn-quantum-control")
    except PackageNotFoundError:  # pragma: no cover - only in a non-installed tree
        return "0+unknown"


STUDIO_VERSION = _resolve_studio_version()
"""The QUANTUM studio version this manifest stamps (the installed package version)."""


def declared_surface() -> dict[str, bytes]:
    """Return the content-addressable declared surface of the QUANTUM studio.

    The surface is the canonical JSON of each advertised verb plus the evidence
    schema list, keyed by a stable logical path. Hashing the declared *content*
    (not git state) is what makes the digest reproducible across checkouts.

    Returns
    -------
    dict[str, bytes]
        Mapping of logical path to canonical-JSON bytes, suitable for
        :func:`scpn_studio_platform.manifest.content_digest`.

    """
    surface: dict[str, bytes] = {
        f"verb/{verb.name}": json.dumps(
            verb.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        for verb in QUANTUM_VERBS
    }
    surface["evidence/schemas"] = json.dumps(
        list(evidence_schemas()), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    surface["evidence/substrates"] = json.dumps(
        {key: list(value) for key, value in verb_substrates().items()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return surface


def build_manifest(*, studio_version: str = STUDIO_VERSION) -> CapabilityManifest:
    """Build the QUANTUM studio's capability manifest (schema A).

    Parameters
    ----------
    studio_version
        The studio version to stamp; defaults to :data:`STUDIO_VERSION`.

    Returns
    -------
    CapabilityManifest
        The schema-A manifest, with a content digest over :func:`declared_surface`.

    """
    return CapabilityManifest(
        studio=STUDIO_ID,
        studio_version=studio_version,
        platform_sdk=PLATFORM_SDK_RANGE,
        content_digest=content_digest(declared_surface()),
        protocol_version=PROTOCOL_VERSION,
        transport_profile=TransportProfile.LOCAL_FIRST,
        verbs=QUANTUM_VERBS,
        evidence_types=evidence_schemas(),
        ui_module=UiModule(
            remote_entry=STUDIO_REMOTE_ENTRY,
            exposes=(STUDIO_EXPOSED_MODULE,),
            federation=STUDIO_FEDERATION_NAME,
        ),
    )


def build_catalogue(*, studio_version: str = STUDIO_VERSION) -> dict[str, object]:
    """Project declared verbs onto the existing bounded browser instruments.

    Parameters
    ----------
    studio_version
        Source or installed distribution version, bound into catalogue identity.

    Returns
    -------
    dict[str, object]
        Deterministically ordered metadata, source identity and local fragment
        routes. Advertised backends are declarations, never runtime probes.
        Browser availability must be measured by the consuming panel.

    """
    instruments = {
        "compile": (
            "#/build/compile-recompute",
            "XY compile recomputation",
            "Replays the committed XY input only; no general circuit editor.",
            "Committed K_nm, omega and compile settings; immutable replay input.",
        ),
        "differentiate": (
            "#/results/program-ad-replay",
            "program-AD gradient replay",
            "Replays the committed rational program only; no arbitrary Python execution.",
            "Committed rational program, parameter targets and float64 inputs.",
        ),
    }
    rows: list[dict[str, object]] = []
    for verb in sorted(QUANTUM_VERBS, key=lambda item: item.name):
        instrument = instruments.get(verb.name)
        rows.append(
            {
                "verb": verb.name,
                "api": f"scpn-studio-run {verb.name}",
                "runtime": "browser-wasm" if instrument else "local-python",
                "backends": list(verb.backends),
                "evidence": list(verb.produces),
                "route": instrument[0] if instrument else None,
                "label": instrument[1] if instrument else verb.name,
                "reason": instrument[2]
                if instrument
                else "Library-only: no browser dispatch route; optional backends require local checks.",
                "settings": instrument[3]
                if instrument
                else "Use the local CLI handler's parameters and policy; this catalogue submits nothing.",
            }
        )
    manifest = build_manifest(studio_version=studio_version)
    body: dict[str, object] = {
        "schema": "studio-capability-catalogue.v1",
        "source_studio": STUDIO_ID,
        "source_version": studio_version,
        "source_digest": manifest.content_digest,
        "rows": rows,
    }
    body["identity"] = content_digest(
        {"catalogue": json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")}
    )
    return body
