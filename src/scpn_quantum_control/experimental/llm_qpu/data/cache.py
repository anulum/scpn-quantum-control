# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — exact LLM-QPU cache identity
"""Fail-closed cache references across model, compressor, and SDK revisions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..contracts import canonical_bytes
from ..contracts.wire import _digest, _text
from .artifact_store import ArtifactStore, MissingArtifactError
from .lineage import load_manifest


class StaleCacheError(ValueError):
    """A source/stage has cache entries, but none match the exact revisions."""


def _check_manifest_key(store: ArtifactStore, key: CacheKey, digest: str) -> None:
    """Keep cache identity aligned with the admitted manifest contents."""
    manifest = load_manifest(store, digest)
    expected_kind = "latent_batch" if key.stage == "latent" else "compressed_latent_batch"
    if (
        manifest.artifact_kind != expected_kind
        or manifest.model_digest != key.model_digest
        or manifest.tokenizer_digest != key.tokenizer_digest
        or manifest.tap_id != key.tap_id
        or manifest.context_mode != key.context_mode
        or manifest.compressor_digest != key.compressor_digest
    ):
        raise ValueError("cache key disagrees with admitted manifest")


@dataclass(frozen=True, slots=True)
class CacheKey:
    """Complete identity of one reusable private research artifact."""

    source_digest: str
    stage: str
    model_digest: str
    tokenizer_digest: str
    tap_id: str
    context_mode: str
    compressor_digest: str | None
    sdk_lock_digest: str | None
    circuit_map_digest: str | None

    def __post_init__(self) -> None:
        """Reject shape-only and unversioned cache identities."""
        for name in ("source_digest", "model_digest", "tokenizer_digest"):
            _digest(getattr(self, name), name=name)
        if self.stage not in ("latent", "compressed", "compiled", "features"):
            raise ValueError("unsupported cache stage")
        _text(self.tap_id, name="tap ID")
        if self.context_mode not in ("contextual", "chunk_isolated"):
            raise ValueError("unsupported cache context mode")
        for name in ("compressor_digest", "sdk_lock_digest", "circuit_map_digest"):
            value = getattr(self, name)
            if value is not None:
                _digest(value, name=name)
        if self.stage == "latent" and any(
            value is not None
            for value in (self.compressor_digest, self.sdk_lock_digest, self.circuit_map_digest)
        ):
            raise ValueError("latent cache cannot claim downstream revisions")
        if self.stage in ("compressed", "compiled", "features") and self.compressor_digest is None:
            raise ValueError("derived cache requires a compressor digest")
        if self.stage in ("compiled", "features") and (
            self.sdk_lock_digest is None or self.circuit_map_digest is None
        ):
            raise ValueError("provider-derived cache requires SDK and circuit revisions")

    @property
    def family_digest(self) -> str:
        """Identify a source/stage independently of revision-specific values."""
        return hashlib.sha256(
            canonical_bytes({"source_digest": self.source_digest, "stage": self.stage})
        ).hexdigest()

    @property
    def digest(self) -> str:
        """Bind every model, context, compressor, SDK and circuit input."""
        return hashlib.sha256(
            canonical_bytes(
                {
                    "schema": "scpn.experimental.llm_qpu.cache_key.v1",
                    "source_digest": self.source_digest,
                    "stage": self.stage,
                    "model_digest": self.model_digest,
                    "tokenizer_digest": self.tokenizer_digest,
                    "tap_id": self.tap_id,
                    "context_mode": self.context_mode,
                    "compressor_digest": self.compressor_digest,
                    "sdk_lock_digest": self.sdk_lock_digest,
                    "circuit_map_digest": self.circuit_map_digest,
                }
            )
        ).hexdigest()


class ArtifactCache:
    """SQLite cache index that never substitutes a different revision."""

    def __init__(self, store: ArtifactStore) -> None:
        if type(store) is not ArtifactStore:
            raise ValueError("cache requires the private artifact store")
        self.store = store
        with store._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key_sha256 TEXT PRIMARY KEY,
                    family_sha256 TEXT NOT NULL,
                    object_sha256 TEXT NOT NULL,
                    FOREIGN KEY (object_sha256) REFERENCES objects(sha256)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS cache_by_family ON cache_entries(family_sha256)"
            )

    def bind(self, key: CacheKey, object_digest: str) -> None:
        """Bind an exact key once to a verified object, never overwrite it."""
        if type(key) is not CacheKey:
            raise ValueError("cache bind requires a CacheKey")
        record, _ = self.store.get(object_digest)
        if record.egress != "private_derived":
            raise ValueError("cache object must preserve private-derived egress")
        if key.stage in ("latent", "compressed"):
            _check_manifest_key(self.store, key, object_digest)
        with self.store._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                "SELECT object_sha256 FROM cache_entries WHERE key_sha256=?", (key.digest,)
            ).fetchone()
            if previous is not None and previous[0] != object_digest:
                raise StaleCacheError("exact cache key already binds different content")
            connection.execute(
                "INSERT OR IGNORE INTO cache_entries VALUES (?, ?, ?)",
                (key.digest, key.family_digest, object_digest),
            )

    def lookup(self, key: CacheKey) -> str:
        """Return only the exact verified revision, distinguishing stale from absent."""
        if type(key) is not CacheKey:
            raise ValueError("cache lookup requires a CacheKey")
        with self.store._connection() as connection:
            row = connection.execute(
                "SELECT object_sha256 FROM cache_entries WHERE key_sha256=?", (key.digest,)
            ).fetchone()
            older = connection.execute(
                "SELECT 1 FROM cache_entries WHERE family_sha256=? LIMIT 1",
                (key.family_digest,),
            ).fetchone()
        if row is None:
            if older is not None:
                raise StaleCacheError("cache source exists under a different revision")
            raise MissingArtifactError("cache source has no entry")
        digest = row[0]
        if type(digest) is not str:
            raise ValueError("indexed cache digest is invalid")
        if key.stage in ("latent", "compressed"):
            _check_manifest_key(self.store, key, digest)
        else:
            self.store.get(digest)
        return digest
