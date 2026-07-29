"""Lock every shipped language into the strict API-reference workflow."""

from __future__ import annotations

import json
from pathlib import Path


def test_rustdoc_covers_every_documentable_manifest() -> None:
    """Require Cargo doc for each package that exposes a documentable target."""
    workflow = Path(".github/workflows/docs-strict.yml").read_text()
    manifests = {
        "scpn_quantum_engine/Cargo.toml",
        "scpn_quantum_engine/program_ad_replay/Cargo.toml",
        "scpn_quantum_engine/studio_program_ad_wasm/Cargo.toml",
        "scpn_quantum_engine/studio_wasm_kernel/Cargo.toml",
    }
    assert all(f"cargo doc --manifest-path {path} --no-deps" in workflow for path in manifests)
    assert 'RUSTDOCFLAGS: "-D warnings"' in workflow


def test_typedoc_is_pinned_and_aggregated() -> None:
    """Require a pinned TypeDoc build and aggregate workflow dependency."""
    package = json.loads(Path("studio-web/package.json").read_text())
    assert package["scripts"]["docs:api"] == "typedoc --options typedoc.json"
    assert package["devDependencies"]["typedoc"] == "0.28.20"
    workflow = Path(".github/workflows/docs-strict.yml").read_text()
    assert "pnpm --dir studio-web docs:api" in workflow
    assert "needs: [build-strict, rust-reference, typescript-reference]" in workflow
    assert "name: rustdoc-reference" in workflow
    assert "name: typedoc-reference" in workflow


def test_reference_page_is_in_existing_api_navigation() -> None:
    """Expose the additive reference-build contract without replacing docs."""
    navigation = Path("mkdocs.yml").read_text()
    assert "Cross-Language Reference Builds: cross_language_api_reference.md" in navigation
    page = Path("docs/cross_language_api_reference.md").read_text()
    assert "cargo-fuzz" in page and "TypeDoc" in page and "MkDocstrings" in page
