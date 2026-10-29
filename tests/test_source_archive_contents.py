# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Source archive privacy regression
"""Exercise the actual package build policy against local private project records."""

from __future__ import annotations

import subprocess
import sys
import tarfile
from pathlib import Path


def test_source_build_excludes_private_memory_and_retains_public_namesakes(
    tmp_path: Path,
) -> None:
    """Build the real source configuration without exporting private project records.

    Parameters
    ----------
    tmp_path
        Isolated source and output directory for the real Hatch build.

    """
    root = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    project.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        (project / name).write_bytes((root / name).read_bytes())
    package = project / "src/scpn_quantum_control"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(
        (root / "src/scpn_quantum_control/__init__.py").read_bytes()
    )
    private = project / "agentic_project_memory"
    private.mkdir()
    (private / "memory_index.md").write_text("Private project navigation.\n")
    (private / "memory_manifest.json").write_text('{"private": true}\n')
    public = project / "docs"
    public.mkdir()
    (public / "memory_index.md").write_text("Public memory API documentation.\n")
    output = tmp_path / "dist"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--no-isolation",
            "--outdir",
            str(output),
            str(project),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    archives = list(output.glob("*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0]) as archive:
        members = {name.split("/", 1)[1] for name in archive.getnames()}
    assert "src/scpn_quantum_control/__init__.py" in members
    assert "docs/memory_index.md" in members
    assert "pyproject.toml" in members
    assert not any(name.startswith("agentic_project_memory/") for name in members)
