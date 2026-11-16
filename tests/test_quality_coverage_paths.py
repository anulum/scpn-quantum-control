# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — quality-gate coverage database paths tests
"""The quality-gate coverage databases live in a private per-process directory."""

from __future__ import annotations

import stat
import subprocess
import sys
from pathlib import Path

import pytest

from tools.quality_coverage_paths import (
    QUALITY_COVERAGE_PREFIX,
    quality_coverage_dir,
    quality_coverage_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_directory_is_private_and_reused() -> None:
    """The directory is owner-only, carries the prefix and is the same on every call."""
    directory = quality_coverage_dir()
    assert directory.is_dir()
    assert directory.name.startswith(QUALITY_COVERAGE_PREFIX)
    assert len(directory.name) > len(QUALITY_COVERAGE_PREFIX)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert quality_coverage_dir() == directory


def test_named_database_lies_in_the_directory() -> None:
    """A bare name becomes a path directly inside the private directory."""
    path = Path(quality_coverage_file("scpn-qc-example-quality.coverage"))
    assert path.parent == quality_coverage_dir()
    assert path.name == "scpn-qc-example-quality.coverage"


@pytest.mark.parametrize("name", ["", ".", "..", "nested/name.coverage", "/tmp/name.coverage"])
def test_names_that_are_not_bare_file_names_are_refused(name: str) -> None:
    """Empty names, directory entries and anything with a separator are refused."""
    with pytest.raises(ValueError, match="bare file name"):
        quality_coverage_file(name)


def test_directory_is_removed_when_the_process_exits() -> None:
    """A separate process gets its own directory, which is gone after it exits."""
    code = (
        "from tools.quality_coverage_paths import quality_coverage_file\n"
        "path = quality_coverage_file('probe.coverage')\n"
        "open(path, 'w').close()\n"
        "print(path)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    probe = Path(result.stdout.strip())
    assert probe.name == "probe.coverage"
    assert probe.parent != quality_coverage_dir()
    assert not probe.parent.exists()
