# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Documentation Claim Boundary Tests
"""Regression tests for public scientific claim-boundary wording."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_phase1_short_paper_names_nearest_neighbour_truncation() -> None:
    """The hardware DLA paper must not imply a full all-to-all K circuit."""
    text = _read(
        "paper/submissions/submission_002_phase1_dla_parity/phase1_dla_parity_short_paper.md"
    )

    assert "nearest-neighbour truncation" in text
    assert "nearest-neighbour Hamiltonian" in text


def test_benchmarks_api_does_not_overclaim_hardware_only_full_dynamics() -> None:
    """Classical crossover docs must not turn estimates into a hardness claim."""
    text = _read("docs/benchmarks_api.md")
    collapsed = " ".join(text.split())

    assert "only quantum hardware" not in text
    assert "full Kuramoto-XY dynamics" not in text
    assert "no broad quantum-advantage claim follows" in collapsed


def test_legacy_preprint_marks_ibm_fez_scope_as_artifact_backed() -> None:
    """The legacy preprint must not overpromote early ibm_fez material."""
    text = _read("docs/preprint.md")
    collapsed = " ".join(text.split())

    assert "first quantum hardware demonstration" not in collapsed
    assert "artifact-backed legacy hardware evidence" in collapsed
    assert "not promoted as broad hardware validation" in collapsed


def test_legacy_preprint_no_survival_or_outperformance_overclaims() -> None:
    """Legacy hardware snapshots must stay descriptive and comparator-bounded."""
    text = _read("docs/preprint.md")
    collapsed = " ".join(text.split())

    assert "demonstrating that the Kuramoto coupling structure survives" not in collapsed
    assert "physics-informed circuit design outperforms" not in collapsed
    assert "descriptive hardware snapshot" in collapsed


def test_overview_paper_has_bounded_hardware_novelty_and_noise_language() -> None:
    """The overview manuscript must avoid broad hardware-novelty overclaims."""
    text = _read("paper/submissions/submission_001_ibm_fez_synchronisation/main.tex")
    collapsed = " ".join(text.split())

    assert "no prior hardware demonstration" not in collapsed
    assert "Superconducting qubits are native simulators of this physics" not in collapsed
    assert "has sufficient coherence for Kuramoto-XY simulation" not in collapsed
    assert "non-ergodic protection strengthens with size" not in collapsed
    assert "legacy artefact-backed campaign" in collapsed
    assert "not a broad quantum-advantage claim" in collapsed


def test_fim_manuscript_preserves_negative_hardware_boundary() -> None:
    """The FIM manuscript should stay bounded to exact structure plus negative IBM evidence."""
    text = _read("paper/submissions/submission_004_scpn_fim_hamiltonian/scpn_fim_hamiltonian.tex")
    collapsed = " ".join(text.split())

    assert "backend/circuit-specific negative hardware result" in collapsed
    assert (
        "No hardware coherence improvement, quantum advantage, or universal protection claim is made"
        in collapsed
    )
    assert "not a hardware robustness claim" in collapsed


def test_legacy_preprint_dtc_caption_is_not_hardware_first_claim() -> None:
    """Figure captions must not preserve stronger claims than the abstract."""
    text = _read("docs/preprint.md")
    collapsed = " ".join(text.split())

    assert "first such measurement" not in collapsed
    assert "not a promoted hardware DTC measurement" in collapsed


def test_public_claim_inventory_is_marked_as_legacy_triage() -> None:
    """The older claim inventory must not read as a submission-ready paper source."""
    text = _read("docs/PAPER_CLAIMS.md")
    collapsed = " ".join(text.split())

    assert "Legacy claim triage" in collapsed
    assert "not a submission-ready claim source" in collapsed
    assert "outperforming generic ansatze" not in collapsed
    assert "proves readout is clean" not in collapsed
    assert "The extremes follow coupling" not in collapsed
    assert "more robust to decoherence" not in collapsed


# --------------------------------------------------------------------------------------------------
# BP3 — Phase-7 vendor self-benchmark ratio provenance guard.
#
# The Phase-7 backend roadmap collects competitor GPU/backend performance ratios that originate as
# *vendor self-benchmarks* (documented conflict of interest + version staleness): the Julia
# GPU-kernel large-N throughput figure and the DiffMPC time-parallel-PCG-kernel-over-trajax figure.
# The roadmap keeps them strictly as "re-measure before quoting" planning inputs — we have not
# reproduced them (the accelerated GPU tier is owner-gated cloud work). This guard enforces that
# discipline on every public / citable surface: an unreproduced vendor ratio may never reach the
# README, the docs site or the JOSS paper presented as our own result. A qualified mention (carrying a
# vendor-reported caveat in the same document) is allowed; an unqualified one fails the gate.

#: Distinctive competitor tool names and ratio strings whose only source is an unreproduced vendor
#: self-benchmark. Named products (DiffEqGPU, DiffMPC, trajax, torchdiffeq) are unambiguous
#: competitor references; the two ratio strings are the exact figures quoted in the roadmap.
_UNREPRODUCED_VENDOR_CLAIM_TOKENS = (
    "diffeqgpu",  # Utkarsh et al., CMAME 419 (2024) — Julia GPU-kernel large-N throughput
    "diffmpc",  # Toyota (2025) — time-parallel PCG GPU kernel over the trajax baseline
    "trajax",  # JAX trajectory-optimisation baseline used in the vendor GPU comparison
    "torchdiffeq",  # PyTorch ODE baseline used in vendor differentiable-solver comparisons
    "20–100×",  # vendor large-N GPU throughput ratio (en-dash / multiplication sign)
    "20-100x",  # ascii variant of the same ratio
    "4–7×",  # DiffMPC PCG-kernel-over-trajax ratio
    "4-7x",  # ascii variant of the same ratio
)

#: Any one of these, present in the same document, marks a vendor figure as vendor-reported rather
#: than our own measurement, which satisfies the "re-measure before quoting" discipline.
_VENDOR_REPORTED_QUALIFIERS = (
    "vendor self-benchmark",
    "vendor-reported",
    "not our measurement",
    "under favourable conditions",
    "conflict of interest",
)


def _tracked_public_docs() -> list[Path]:
    """Return the public / citable Markdown surfaces (README, docs site, paper).

    Internal planning docs under ``docs/internal`` are gitignored and never shipped, so they are
    excluded — the roadmap is where these vendor figures legitimately live as planning inputs.
    """
    docs: list[Path] = [REPO_ROOT / "README.md"]
    for directory in (
        REPO_ROOT / "docs",
        REPO_ROOT / "oscillatools" / "docs",
        REPO_ROOT / "paper",
    ):
        docs.extend(sorted(directory.rglob("*.md")))
    return [
        path
        for path in docs
        if path.exists() and "internal" not in path.relative_to(REPO_ROOT).parts
    ]


def test_phase7_vendor_self_benchmark_ratios_not_quoted_unqualified_in_public_docs() -> None:
    """No unreproduced competitor GPU/backend vendor ratio may appear unqualified in a public doc.

    Phase-7 backend performance ratios are vendor self-benchmarks (documented conflict of interest +
    version staleness); the roadmap records them only as "re-measure before quoting" planning inputs.
    Until the owner-gated GPU tier reproduces them they must not leak into any public / citable surface
    as if they were our measurement. A qualified mention (with a vendor-reported caveat in the same
    document) is allowed; an unqualified one is not — this is the enforcement of BP3.
    """
    offenders: list[str] = []
    for doc in _tracked_public_docs():
        lowered = doc.read_text(encoding="utf-8").lower()
        if any(qualifier in lowered for qualifier in _VENDOR_REPORTED_QUALIFIERS):
            continue
        relative = doc.relative_to(REPO_ROOT)
        offenders.extend(
            f"{relative} quotes '{token}' with no vendor-reported qualifier"
            for token in _UNREPRODUCED_VENDOR_CLAIM_TOKENS
            if token in lowered
        )
    assert not offenders, (
        "Unqualified Phase-7 vendor self-benchmark ratios found in public docs "
        "(re-measure or add a vendor-reported caveat):\n" + "\n".join(offenders)
    )


def test_competitive_benchmark_reports_our_own_measurement_not_vendor_ratios() -> None:
    """The public competitive evidence must be our own honest measurement, not vendor self-benchmarks.

    The correct alternative to quoting a vendor ratio is to measure it ourselves: the competitive
    benchmark doc reports live rows for present competitors and honest ``unavailable`` rows for absent
    ones, and reports honestly where a competitor is faster — the discipline BP3 protects.
    """
    text = _read("docs/kuramoto_competitive_benchmark.md")
    lowered = text.lower()
    assert "unavailable" in lowered
    assert "honest" in lowered
    for token in _UNREPRODUCED_VENDOR_CLAIM_TOKENS:
        assert token not in lowered, (
            f"competitive evidence must not carry the unreproduced vendor token '{token}'"
        )
