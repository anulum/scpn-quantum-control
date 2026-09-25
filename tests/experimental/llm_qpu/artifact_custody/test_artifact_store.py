# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — private artifact custody acceptance
"""Exercise immutable raw-shard custody against real disk failures."""

from __future__ import annotations

import hashlib
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from scpn_quantum_control.experimental.llm_qpu.contracts import ArrayDescriptor
from scpn_quantum_control.experimental.llm_qpu.data.artifact_store import (
    ArtifactStore,
    CorruptArtifactError,
    MissingArtifactError,
    open_repository_store,
)


def _array(values: tuple[float, ...]) -> tuple[ArrayDescriptor, bytes]:
    payload = struct.pack(f"<{len(values)}f", *values)
    return (
        ArrayDescriptor(
            "<f4", (len(values), 1), len(payload), hashlib.sha256(payload).hexdigest()
        ),
        payload,
    )


def test_t03a_conflicting_disk_bytes_are_quarantined_without_overwrite(tmp_path: Path) -> None:
    """A content address cannot silently replace conflicting bytes."""
    store = ArtifactStore(tmp_path / "private")
    descriptor, payload = _array((1.0, 2.0))
    record = store.put_array(descriptor, payload)
    object_path = store.objects / record.sha256[:2] / record.sha256
    object_path.write_bytes(b"conflicting bytes")

    with pytest.raises(CorruptArtifactError, match="digest mismatch"):
        store.put_array(descriptor, payload)

    assert not object_path.exists()
    assert [path.read_bytes() for path in store.quarantine.iterdir()] == [b"conflicting bytes"]
    with pytest.raises(MissingArtifactError):
        store.get(record.sha256)


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux RLIMIT_FSIZE")
def test_t03b_interrupted_write_leaves_no_valid_partial(tmp_path: Path) -> None:
    """A real kernel file-size limit leaves no admitted partial object."""
    root = tmp_path / "private"
    store = ArtifactStore(root)
    payload = b"x" * 8192
    digest = hashlib.sha256(payload).hexdigest()
    code = """
import resource
import signal
import sys
from pathlib import Path
from scpn_quantum_control.experimental.llm_qpu.data.artifact_store import ArtifactStore
store = ArtifactStore(Path(sys.argv[1]))
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
resource.setrlimit(resource.RLIMIT_FSIZE, (1024, hard))
try:
    store.put(b'x' * 8192, kind='evidence', retention='pinned', egress='private_derived')
except OSError:
    sys.exit(0)
sys.exit(1)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(root)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    with pytest.raises(MissingArtifactError):
        store.get(digest)
    assert list(store.objects.rglob("*")) == [store.objects / digest[:2]]
    assert not list(store.objects.rglob("*.tmp"))


def test_t03d_rejects_escape_symlink_and_archive_before_decode(tmp_path: Path) -> None:
    """Only digest paths and raw descriptor bytes enter the shard store."""
    store = ArtifactStore(tmp_path / "private")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        store.get("../outside")

    outside = tmp_path / "outside"
    outside.mkdir()
    (store.objects / "ff").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        store.get("f" * 64)

    payload = b"PK\x03\x04" + b"\0" * 4
    descriptor = ArrayDescriptor("<u4", (2, 1), 8, hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError, match="container formats"):
        store.put_array(descriptor, payload)
    assert not list(store.objects.rglob("*.tmp"))


def test_private_repo_root_and_index_permissions(tmp_path: Path) -> None:
    """Derived data lands under the dedicated results root with private index."""
    store = open_repository_store(tmp_path)
    assert store.root == tmp_path / "results" / "experimental" / "llm_qpu" / "private"
    assert store.root.stat().st_mode & 0o777 == 0o700
    assert store.database.stat().st_mode & 0o777 == 0o600

    store.root.chmod(0o755)
    with pytest.raises(ValueError, match="owned and private"):
        open_repository_store(tmp_path)
