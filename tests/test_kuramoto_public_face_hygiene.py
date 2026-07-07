# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Kuramoto public-face language hygiene tests
"""Regression guard against self-promotional framing on the Kuramoto-moat surfaces.

The quantum/differentiable lane already governs its own public wording through
``audit_differentiable_promotion_language`` and the claim-boundary tests in
``test_doc_claim_boundaries``. The Kuramoto toolkit's own public surfaces — the
``docs/kuramoto_*.md`` pages (including the generated handbook) and the toolkit's
JOSS paper — had no equivalent guard, so nothing prevented a future edit (a paper
finalisation, a handbook regeneration, a docstring change) from introducing the
comparative/superlative self-framing the project bars from every public surface:
"ahead of the competition / orders of magnitude / world-class / SOTA / definitive /
dominance". This module supplies that guard.

The matcher is subject-aware on purpose. A scientific paper's "State of the field"
section legitimately describes competitors ("DynamicalSystems.jl is a state-of-the-art
library"), and physics prose legitimately uses words like "dominant" ("the dominant
cluster locks first") and "orders of magnitude" ("differ by orders of magnitude in
flops"). None of those may trip the guard. So the forbidden patterns split into:

* Tier 1 — marketing puffery with no neutral scientific use (``world-class``,
  ``cutting-edge``, ``unrivalled`` …): forbidden bare.
* Tier 2 — subject-aware constructions: uniqueness superlatives (``the definitive
  toolkit``) that can only describe oneself; category descriptors (``state-of-the-art``,
  ``SOTA``, ``dominant`` …) forbidden only when bound to a first-person referent
  (``our``/``this``); and explicit comparison against the field.

The final three tests are the guard's own witnesses: it is proven to bite on
representative violations, proven not to over-fire on neutral/bounded language, and
proven to still cover the barred vocabulary — so a hollowed-out pattern set fails too.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Tier 1 — marketing puffery with no neutral scientific use: forbidden bare, in any
# context. ("gold-standard" is deliberately excluded — it has genuine neutral use in
# "the gold-standard method for X".)
_TIER1_PATTERNS: tuple[str, ...] = (
    r"world[\s-]?class",
    r"best[\s-]in[\s-]class",
    r"cutting[\s-]edge",
    r"industry[\s-]leading",
    r"world[\s-]?leading",
    r"unrivall?ed",
    r"unmatched",
    r"unparalleled",
    r"revolutionary",
    r"game[\s-]?chang(?:ing|er)",
    r"groundbreaking",
    r"second[\s-]to[\s-]none",
    r"\bdominance\b",
    r"bar none",
)

_PRODUCT_NOUNS = r"(?:toolkit|package|library|framework|suite|implementation)"

# Tier 2 — subject-aware, so neutral descriptions of OTHER tools never trip the guard:
#   * uniqueness superlatives (definitive/premier/foremost/preeminent — only one thing
#     can be "the definitive X") qualifying a product noun are self-promotional even
#     without an explicit "our";
#   * category descriptors (state-of-the-art/SOTA/superior/dominant/world-leading) are
#     forbidden only when bound to a first-person referent ("our"/"this");
#   * explicit comparative framing against the field.
_TIER2_PATTERNS: tuple[str, ...] = (
    rf"\bthe\s+(?:[\w-]+\s+){{0,3}}?(?:definitive|premier|foremost|preeminent)"
    rf"\s+(?:[\w-]+\s+){{0,3}}?{_PRODUCT_NOUNS}",
    r"(?:\bour\b|\bthis\b)\s+(?:[\w-]+\s+){0,3}?(?:state[\s-]of[\s-]the[\s-]art|"
    r"\bSOTA\b|definitive|premier|foremost|preeminent|superior|dominant|world[\s-]?leading)\b",
    r"ahead of (?:the )?(?:field|competition|pack|state[\s-]of[\s-]the[\s-]art)",
    r"orders of magnitude (?:faster|better|more accurate|ahead|superior|beyond)",
)

_FORBIDDEN = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in (_TIER1_PATTERNS + _TIER2_PATTERNS)
)

# The JOSS paper for the differentiable Kuramoto toolkit (distinct from the
# quantum-control paper submission_joss_001).
_JOSS_KURAMOTO_PAPER = "paper/submissions_joss/submission_joss_002_kuramoto_toolkit/paper.md"

# The docs the enumeration must always find, so the net cannot silently shrink (a
# page renamed out of the ``kuramoto_`` prefix would otherwise escape the guard).
_REQUIRED_DOC_SURFACES: frozenset[str] = frozenset(
    {
        "docs/kuramoto_variants.md",
        "docs/kuramoto_competitive_benchmark.md",
        "docs/kuramoto_standalone_package_decision.md",
        "docs/kuramoto_core_facade.md",
        "docs/kuramoto_handbook.md",
    }
)


def scan_for_self_promotion(text: str) -> list[str]:
    """Return every forbidden self-promotional match found in ``text``.

    Each entry is ``"<matched phrase>" (pattern <index>)`` for a legible failure.
    An empty list means the text is clean.
    """
    hits: list[str] = []
    for index, pattern in enumerate(_FORBIDDEN):
        for match in pattern.finditer(text):
            hits.append(f"{match.group(0)!r} (pattern {index})")
    return hits


def kuramoto_moat_surfaces() -> list[Path]:
    """Enumerate the Kuramoto-moat public surfaces present in the tree.

    ``docs/kuramoto_*.md`` is globbed so a surface added later (for example the
    visualisation page) is guarded automatically, plus the toolkit's JOSS paper.
    """
    surfaces = sorted((REPO_ROOT / "docs").glob("kuramoto_*.md"))
    paper = REPO_ROOT / _JOSS_KURAMOTO_PAPER
    if paper.exists():
        surfaces.append(paper)
    return surfaces


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


@pytest.mark.parametrize("surface", kuramoto_moat_surfaces(), ids=_relative)
def test_kuramoto_moat_surface_is_free_of_self_promotional_framing(surface: Path) -> None:
    """No Kuramoto-moat public surface may carry comparative/superlative self-framing."""
    hits = scan_for_self_promotion(surface.read_text(encoding="utf-8"))
    assert not hits, f"{_relative(surface)} carries barred self-promotional framing: {hits}"


def test_surface_enumeration_covers_the_known_kuramoto_moat_docs() -> None:
    """The net must still cover every known doc surface plus the JOSS paper.

    Guards against a page being renamed out of the ``kuramoto_`` prefix, or the JOSS
    paper being moved, which would silently drop it from the guard.
    """
    found = {_relative(path) for path in kuramoto_moat_surfaces()}
    missing = _REQUIRED_DOC_SURFACES - found
    assert not missing, f"Kuramoto-moat doc surfaces dropped from the guard: {sorted(missing)}"
    assert _JOSS_KURAMOTO_PAPER in found, "the Kuramoto JOSS paper is no longer guarded"


def test_joss_paper_state_of_the_field_heading_is_permitted() -> None:
    """The JOSS paper is guarded, yet its legitimate section headings stay clean.

    The paper is the highest-risk finalisation surface; its required "State of the
    field" section (which describes competitors neutrally) must never be mistaken for
    self-promotional "state-of-the-art" framing.
    """
    paper = REPO_ROOT / _JOSS_KURAMOTO_PAPER
    text = paper.read_text(encoding="utf-8")
    assert "# State of the field" in text
    assert not scan_for_self_promotion(text)


# Representative barred framing — one per HG1-enumerated failure mode. A guard that
# never fires is worthless, so each of these must be caught.
_SELF_PROMOTIONAL_SAMPLES: tuple[str, ...] = (
    "the definitive Kuramoto toolkit for research",  # uniqueness superlative + product
    "the premier oscillator-dynamics package",  # uniqueness, hyphenated filler
    "the definitive open-source implementation",  # uniqueness, hyphenated filler
    "our state-of-the-art toolkit",  # category descriptor + first-person referent
    "this world-leading framework",  # category descriptor + first-person referent
    "delivers world-class performance",  # bare puffery
    "a cutting-edge package",  # bare puffery
    "our dominance in oscillator simulation",  # bare puffery noun
    "orders of magnitude faster than SciMLSensitivity",  # quantitative comparison
    "puts us ahead of the competition",  # explicit comparison against the field
)

# Neutral or bounded language that MUST stay clean — the false-positive analysis
# encoded as a regression. Covers competitor description, physics idiom, honest
# self-deprecation, mathematics, and boundary-guarded benchmark prose.
_NEUTRAL_SAMPLES: tuple[str, ...] = (
    "State of the field",  # required JOSS section heading
    "DynamicalSystems.jl is a state-of-the-art dynamical-systems library",  # competitor
    "a state-of-the-art solver from the SciML ecosystem",  # competitor
    "the dominant cluster locks first",  # physics idiom
    "the dominant oscillator in the array",  # physics idiom
    "the dominant frequency component",  # physics idiom
    "differ by orders of magnitude in flops",  # neutral quantitative
    "classical solvers outperform quantum simulation here",  # honest self-deprecation
    "leading eigenvalue of the Jacobian",  # mathematics
    "the leading-order correction",  # mathematics
    "definitive proof of phase locking",  # mathematics, not a product claim
    "our adaptive DOPRI5 was the fastest available solver on this loaded host, "
    "but that ordering must be re-measured on an isolated host",  # bounded benchmark prose
)


@pytest.mark.parametrize("sample", _SELF_PROMOTIONAL_SAMPLES)
def test_guard_detects_representative_self_promotional_framing(sample: str) -> None:
    """The guard must fire on every representative barred construction."""
    assert scan_for_self_promotion(sample), f"barred framing slipped past the guard: {sample!r}"


@pytest.mark.parametrize("sample", _NEUTRAL_SAMPLES)
def test_guard_permits_neutral_and_bounded_language(sample: str) -> None:
    """Neutral competitor description, physics idiom and bounded prose must stay clean."""
    hits = scan_for_self_promotion(sample)
    assert not hits, f"the guard over-fired on legitimate language {sample!r}: {hits}"


def test_forbidden_pattern_set_covers_the_hg1_enumerated_vocabulary() -> None:
    """The pattern set must still cover every HG1-enumerated failure mode.

    Guards against the vocabulary being quietly gutted, which would leave a guard
    that passes while enforcing nothing. Each probe is a minimal string that only the
    intended failure mode can match.
    """
    coverage_probes = {
        "world-class": "world-class",
        "cutting-edge": "cutting-edge",
        "dominance": "our dominance here",
        "SOTA": "our SOTA package",
        "state-of-the-art (self)": "our state-of-the-art toolkit",
        "definitive (uniqueness)": "the definitive toolkit",
        "orders-of-magnitude": "orders of magnitude faster",
        "ahead-of-the-field": "ahead of the competition",
    }
    uncovered = [
        name for name, probe in coverage_probes.items() if not scan_for_self_promotion(probe)
    ]
    assert not uncovered, f"HG1-enumerated vocabulary no longer guarded: {uncovered}"
