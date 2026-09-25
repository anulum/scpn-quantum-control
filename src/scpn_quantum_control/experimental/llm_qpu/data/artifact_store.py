# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — private LLM-QPU artifact custody
"""Bounded immutable CAS for private LLM-QPU shards and manifests."""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import stat
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..contracts import ArrayDescriptor

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_KINDS = frozenset({"array_shard", "manifest", "evidence"})
_RETENTION = frozenset({"pinned", "cache"})
_EGRESS = frozenset({"private_derived", "reviewed_synthetic", "value_free"})
_CHUNK = 1_048_576


class MissingArtifactError(FileNotFoundError):
    """An indexed or referenced content object is absent or unverifiable."""


class CorruptArtifactError(ValueError):
    """Existing bytes conflict with their immutable content address."""


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Bounded SQLite reference to one immutable on-disk payload."""

    sha256: str
    byte_length: int
    kind: Literal["array_shard", "manifest", "evidence"]
    retention: Literal["pinned", "cache"]
    egress: Literal["private_derived", "reviewed_synthetic", "value_free"]


def _reject_symlink_components(path: Path) -> None:
    """Refuse a pre-existing symlink in the custody path."""
    for component in (path, *path.parents):
        try:
            mode = component.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError("artifact path contains a symlink")


def _sync_directory(path: Path) -> None:
    """Persist directory-entry changes after atomic publish or quarantine."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_private_directory(path: Path) -> None:
    """Refuse a custody directory readable by other local accounts."""
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("artifact directory must be owned and private (0700)")


class ArtifactStore:
    """Private size-limited content store; SQLite contains references only."""

    def __init__(
        self,
        root: Path,
        *,
        max_total_bytes: int = 1_073_741_824,
        max_object_bytes: int = 16_777_216,
    ) -> None:
        if type(max_total_bytes) is not int or not 0 < max_total_bytes <= 2**42:
            raise ValueError("invalid total-byte capacity")
        if type(max_object_bytes) is not int or not 0 < max_object_bytes <= max_total_bytes:
            raise ValueError("invalid object-byte capacity")
        if not isinstance(root, Path):
            raise ValueError("artifact root must be a Path")
        _reject_symlink_components(root)
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        _reject_symlink_components(root)
        _require_private_directory(root)
        self.root = root.resolve(strict=True)
        self.max_total_bytes = max_total_bytes
        self.max_object_bytes = max_object_bytes
        self.objects = self.root / "objects"
        self.quarantine = self.root / "quarantine"
        for directory in (self.objects, self.quarantine):
            _reject_symlink_components(directory)
            directory.mkdir(mode=0o700, exist_ok=True)
            _require_private_directory(directory)
        self.database = self.root / "index.sqlite3"
        _reject_symlink_components(self.database)
        descriptor = os.open(self.database, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
            ):
                raise ValueError("artifact index must be owned and private (0600)")
        finally:
            os.close(descriptor)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS objects (
                    sha256 TEXT PRIMARY KEY,
                    byte_length INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    retention TEXT NOT NULL,
                    egress TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS manifest_shards (
                    manifest_sha256 TEXT NOT NULL,
                    shard_sha256 TEXT NOT NULL,
                    PRIMARY KEY (manifest_sha256, shard_sha256),
                    FOREIGN KEY (manifest_sha256) REFERENCES objects(sha256)
                );
                CREATE TABLE IF NOT EXISTS admitted_manifests (
                    sha256 TEXT PRIMARY KEY,
                    FOREIGN KEY (sha256) REFERENCES objects(sha256)
                );
                """
            )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        _reject_symlink_components(self.database)
        connection = sqlite3.connect(self.database, timeout=30.0)
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            with connection:
                yield connection
        finally:
            connection.close()

    def _path(self, digest: str) -> Path:
        if type(digest) is not str or _DIGEST.fullmatch(digest) is None:
            raise ValueError("artifact ID must be lowercase SHA-256")
        path = self.objects / digest[:2] / digest
        _reject_symlink_components(path)
        return path

    def _checked_bytes(self, path: Path, digest: str, *, maximum: int) -> bytes:
        _reject_symlink_components(path)
        if not path.is_file():
            raise MissingArtifactError(digest)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            size = os.fstat(descriptor).st_size
            if size < 0 or size > maximum:
                raise CorruptArtifactError("stored artifact exceeds declared limit")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, min(_CHUNK, maximum + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum:
                    raise CorruptArtifactError("stored artifact exceeds declared limit")
                chunks.append(chunk)
            payload = b"".join(chunks)
        finally:
            os.close(descriptor)
        if hashlib.sha256(payload).hexdigest() != digest:
            raise CorruptArtifactError("stored artifact digest mismatch")
        return payload

    def _quarantine_corrupt(self, path: Path, digest: str) -> None:
        target = self.quarantine / f"{digest}.{uuid.uuid4().hex}"
        _reject_symlink_components(path)
        os.replace(path, target)
        _sync_directory(path.parent)
        _sync_directory(self.quarantine)

    def put(
        self,
        payload: bytes,
        *,
        kind: Literal["array_shard", "manifest", "evidence"],
        retention: Literal["pinned", "cache"],
        egress: Literal["private_derived", "reviewed_synthetic", "value_free"],
    ) -> ArtifactRecord:
        """Atomically publish bounded bytes or verify an identical existing object."""
        if type(payload) is not bytes or not 0 < len(payload) <= self.max_object_bytes:
            raise ValueError("artifact payload must be bounded nonempty bytes")
        if kind not in _KINDS or retention not in _RETENTION or egress not in _EGRESS:
            raise ValueError("unknown artifact custody label")
        digest = hashlib.sha256(payload).hexdigest()
        target = self._path(digest)
        target.parent.mkdir(mode=0o700, exist_ok=True)
        _reject_symlink_components(target)
        record = ArtifactRecord(digest, len(payload), kind, retention, egress)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            indexed = connection.execute(
                "SELECT byte_length, kind, retention, egress FROM objects WHERE sha256=?",
                (digest,),
            ).fetchone()
            if indexed is not None and indexed != (
                record.byte_length,
                record.kind,
                record.retention,
                record.egress,
            ):
                raise CorruptArtifactError("existing content ID has different custody metadata")
            if indexed is None:
                used = connection.execute(
                    "SELECT COALESCE(SUM(byte_length), 0) FROM objects"
                ).fetchone()[0]
                if type(used) is not int or used + len(payload) > self.max_total_bytes:
                    raise ValueError("artifact store capacity exceeded")
            if target.exists():
                try:
                    current = self._checked_bytes(target, digest, maximum=self.max_object_bytes)
                except CorruptArtifactError:
                    self._quarantine_corrupt(target, digest)
                    raise
                if current != payload:
                    self._quarantine_corrupt(target, digest)
                    raise CorruptArtifactError("existing content ID has different bytes")
            else:
                temporary = target.parent / f".{digest}.{uuid.uuid4().hex}.tmp"
                try:
                    descriptor = os.open(
                        temporary,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                    )
                    try:
                        for offset in range(0, len(payload), _CHUNK):
                            chunk = memoryview(payload)[offset : offset + _CHUNK]
                            while chunk:
                                written = os.write(descriptor, chunk)
                                if written <= 0:
                                    raise OSError("short artifact write")
                                chunk = chunk[written:]
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                    os.link(temporary, target, follow_symlinks=False)
                    _sync_directory(target.parent)
                except FileExistsError:
                    if (
                        self._checked_bytes(target, digest, maximum=self.max_object_bytes)
                        != payload
                    ):
                        self._quarantine_corrupt(target, digest)
                        raise CorruptArtifactError("concurrent content ID collision") from None
                finally:
                    temporary.unlink(missing_ok=True)
            connection.execute(
                "INSERT OR IGNORE INTO objects VALUES (?, ?, ?, ?, ?)",
                (digest, len(payload), kind, retention, egress),
            )
        return record

    def put_array(
        self,
        descriptor: ArrayDescriptor,
        payload: bytes,
        *,
        retention: Literal["pinned", "cache"] = "pinned",
        egress: Literal["private_derived", "reviewed_synthetic"] = "private_derived",
    ) -> ArtifactRecord:
        """Store only raw descriptor-verified little-endian arrays, never pickle/zip."""
        if type(descriptor) is not ArrayDescriptor:
            raise ValueError("array storage requires an ArrayDescriptor")
        if payload.startswith((b"PK\x03\x04", b"\x93NUMPY")):
            raise ValueError("container formats are not raw array shards")
        descriptor.validate_payload(payload)
        return self.put(payload, kind="array_shard", retention=retention, egress=egress)

    def get(self, digest: str) -> tuple[ArtifactRecord, bytes]:
        """Read one indexed object after exact size and digest verification."""
        path = self._path(digest)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT byte_length, kind, retention, egress FROM objects WHERE sha256=?",
                (digest,),
            ).fetchone()
        if row is None:
            raise MissingArtifactError(digest)
        length, kind, retention, egress = row
        if type(length) is not int or not 0 < length <= self.max_object_bytes:
            raise CorruptArtifactError("indexed artifact length is invalid")
        payload = self._checked_bytes(path, digest, maximum=length)
        if len(payload) != length:
            raise CorruptArtifactError("indexed artifact length mismatch")
        return ArtifactRecord(digest, length, kind, retention, egress), payload

    def has(self, digest: str) -> bool:
        """Return true only for an indexed, readable object."""
        try:
            self.get(digest)
        except MissingArtifactError:
            return False
        return True


def open_repository_store(repo_root: Path) -> ArtifactStore:
    """Open the lane's private results root for real derived payloads."""
    if not isinstance(repo_root, Path) or not repo_root.is_dir():
        raise ValueError("repository root must be an existing directory")
    return ArtifactStore(repo_root / "results" / "experimental" / "llm_qpu" / "private")
