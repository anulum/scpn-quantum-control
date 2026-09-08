# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch-hook installation
"""Real-repository tests for no-loss branch-hook installation."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tools.enforce_main_branch_policy import install_reference_transaction_hook


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Create an empty synthetic main repository without committing objects."""
    subprocess.run(
        ["git", "init", "--initial-branch=main", str(tmp_path)],
        check=True,
        capture_output=True,
    )
    return tmp_path


def test_existing_chain_is_not_replaced(repository: Path) -> None:
    """An executable owner chain remains byte-for-byte intact on refusal."""
    hook = repository / ".git/hooks/reference-transaction"
    original = b"#!/bin/sh\n# existing owner chain\nexit 7\n"
    hook.write_bytes(original)
    hook.chmod(0o751)
    before = hook.stat()
    with pytest.raises(FileExistsError, match="existing"):
        install_reference_transaction_hook(repository)
    assert hook.read_bytes() == original
    assert hook.stat().st_mode == before.st_mode
    assert hook.stat().st_mtime_ns == before.st_mtime_ns


@pytest.mark.parametrize("dangling", [False, True])
def test_symlink_hook_and_target_are_preserved(repository: Path, dangling: bool) -> None:
    """Neither a symlink nor its external target is replaced or chmodded."""
    target = repository / "owner-hook"
    if not dangling:
        target.write_text("#!/bin/sh\nexit 9\n", encoding="utf-8")
        target.chmod(0o640)
    hook = repository / ".git/hooks/reference-transaction"
    hook.symlink_to(target)
    with pytest.raises(FileExistsError):
        install_reference_transaction_hook(repository)
    assert hook.is_symlink() and hook.readlink() == target
    if dangling:
        assert not target.exists()
    else:
        assert target.read_text(encoding="utf-8") == "#!/bin/sh\nexit 9\n"
        assert target.stat().st_mode & 0o777 == 0o640


def test_reinstall_is_an_unchanged_noop(repository: Path) -> None:
    """An identical executable hook is returned without rewriting its inode."""
    hook = install_reference_transaction_hook(repository)
    before = hook.stat()
    assert install_reference_transaction_hook(repository) == hook
    after = hook.stat()
    assert (after.st_ino, after.st_mtime_ns, after.st_mode) == (
        before.st_ino,
        before.st_mtime_ns,
        before.st_mode,
    )


def test_nonexecutable_installation_is_not_silently_enabled(repository: Path) -> None:
    """An owner-disabled hook remains disabled until explicitly reviewed."""
    hook = install_reference_transaction_hook(repository)
    hook.chmod(0o600)
    before = hook.read_bytes()
    with pytest.raises(FileExistsError, match="existing"):
        install_reference_transaction_hook(repository)
    assert hook.read_bytes() == before and hook.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("competing_target", [False, True])
def test_atomic_publication_failure_preserves_target_and_cleans_staging(
    repository: Path, monkeypatch: pytest.MonkeyPatch, competing_target: bool
) -> None:
    """A failed atomic publish leaves no partial hook or private staging debris."""
    hook = repository / ".git/hooks/reference-transaction"
    owner_content = b"#!/bin/sh\nexit 17\n"

    def fail_publish(source: str, destination: Path) -> None:
        """Inspect the complete staged executable before modelling publication failure."""
        staged = Path(source)
        assert b"refusing transaction" in staged.read_bytes()
        assert staged.stat().st_mode & 0o777 == 0o755
        assert destination == hook
        if competing_target:
            hook.write_bytes(owner_content)
            raise FileExistsError("competing installer won")
        raise OSError("storage does not support atomic hard-link publication")

    monkeypatch.setattr(os, "link", fail_publish)
    with pytest.raises(OSError):
        install_reference_transaction_hook(repository)
    if competing_target:
        assert hook.read_bytes() == owner_content
    else:
        assert not hook.exists()
    assert list(hook.parent.glob(".reference-hook-*")) == []


def test_configured_hook_directory_is_respected(repository: Path) -> None:
    """Git's configured hook directory is used without populating the default."""
    directory = repository / "custom hooks"
    subprocess.run(
        ["git", "config", "core.hooksPath", str(directory)],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    hook = install_reference_transaction_hook(repository)
    assert hook == directory / "reference-transaction"
    assert hook.is_file()
    assert not (repository / ".git/hooks/reference-transaction").exists()


def test_missing_policy_script_blocks_real_reference_update(repository: Path) -> None:
    """An installed shim without either policy owner refuses Git's transaction."""
    install_reference_transaction_hook(repository)
    tree = subprocess.run(
        ["git", "mktree"],
        cwd=repository,
        input="",
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    result = subprocess.run(
        ["git", "update-ref", "refs/tags/fixture", tree],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "not found" in result.stderr and "refusing" in result.stderr
    assert (
        subprocess.run(
            ["git", "show-ref"],
            cwd=repository,
            capture_output=True,
            check=False,
        ).stdout
        == b""
    )
