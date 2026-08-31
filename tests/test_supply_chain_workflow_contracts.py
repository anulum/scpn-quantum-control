# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — supply-chain workflow contracts
"""Pin CI dependency acquisition to immutable, hash-verified inputs."""

from pathlib import Path


def _workflow(path: str) -> str:
    """Return one tracked workflow as UTF-8 text."""
    return Path(path).read_text(encoding="utf-8")


def test_minimal_install_uses_locked_dependencies_and_hashed_local_wheels() -> None:
    """Keep the no-extra install real without unverified package acquisition."""
    workflow = _workflow(".github/workflows/ci-runtime-package.yml")
    start = workflow.index("  minimal-install:")
    block = workflow[start:]

    assert "requirements-ci-minimal-install-py312-linux.txt" in block
    assert block.count("python -m build --wheel --no-isolation") == 2
    assert "sha256(wheel.read_bytes()).hexdigest()" in block
    assert "python -m pip install --no-deps --require-hashes" in block
    assert "pip install ./oscillatools" not in block
    assert "pip install pytest hypothesis" not in block


def test_dependency_update_actions_use_the_reviewed_release_commits() -> None:
    """Keep all Dependabot-proposed action upgrades synchronized on exact SHAs."""
    codeql = _workflow(".github/workflows/codeql.yml")
    scorecard = _workflow(".github/workflows/scorecard.yml")
    pnpm_workflows = "\n".join(
        _workflow(path)
        for path in (
            ".github/workflows/ci-studio.yml",
            ".github/workflows/docs-strict.yml",
            ".github/workflows/studio-remote-release.yml",
        )
    )
    stale = _workflow(".github/workflows/stale.yml")

    codeql_sha = "cdf488f595d80d6e07e03d4674febd5ab45fa938"
    pnpm_sha = "0977fd99725f1db4007ccb2928dbb4e90d06cc86"
    assert codeql.count(codeql_sha) == 2
    assert scorecard.count(codeql_sha) == 1
    assert pnpm_workflows.count(pnpm_sha) == 3
    assert "actions/stale@4391f3da665fdf50b6810c1a66712fb9ba21aa93" in stale


def test_ci_lock_inputs_cover_the_new_security_and_typing_surfaces() -> None:
    """Bind optional runtimes and YAML typing to generated hash closures."""
    minimal_input = Path("requirements-ci-minimal-install-py312-linux.in").read_text(
        encoding="utf-8"
    )
    quimb_input = Path("requirements-ci-quimb-py312-linux.in").read_text(encoding="utf-8")
    assert "pytest==9.1.1" in minimal_input
    assert "hypothesis==6.165.10" in minimal_input
    assert "quimb==1.13.0" in quimb_input

    for path in (
        "requirements-ci-minimal-install-py312-linux.txt",
        "requirements-ci-quimb-py312-linux.txt",
    ):
        lock = Path(path).read_text(encoding="utf-8")
        assert "--hash=sha256:" in lock

    for path in (
        "requirements-ci-py311-linux.txt",
        "requirements-ci-py312-linux.txt",
        "requirements-ci-py313-linux.txt",
    ):
        lock = Path(path).read_text(encoding="utf-8")
        assert "types-pyyaml==6.0.12.20260815" in lock
