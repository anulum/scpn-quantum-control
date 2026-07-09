# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — public-face language hygiene tests
"""Regression guard against self-promotional framing on the oscillatools surfaces.

Nothing otherwise prevents a future edit — a handbook regeneration, a docstring
change, a README rewrite — from introducing the comparative/superlative
self-framing the project bars from every public surface: "ahead of the
competition / orders of magnitude / world-class / SOTA / definitive / dominance".
This module supplies that guard for the standalone distribution's docs and READMEs.

The matcher is subject-aware. Neutral competitor description ("DynamicalSystems.jl
is a state-of-the-art library"), physics idiom ("the dominant cluster locks
first"), and mathematics ("leading eigenvalue of the Jacobian") must never trip
it. So the forbidden patterns split into:

* Tier 1 — marketing puffery with no neutral scientific use (``world-class``,
  ``cutting-edge`` …): forbidden bare.
* Tier 2 — subject-aware constructions: uniqueness superlatives (``the definitive
  toolkit``) that can only describe oneself; category descriptors
  (``state-of-the-art``, ``SOTA``, ``dominant`` …) forbidden only when bound to a
  first-person referent (``our``/``this``); and explicit comparison against the
  field.

The final three tests are the guard's own witnesses: it is proven to bite on
representative violations, proven not to over-fire on neutral/bounded language,
and proven to still cover the barred vocabulary.
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

# The docs pages the enumeration must always find, so the net cannot silently shrink
# (a page renamed or dropped would otherwise escape the guard).
_REQUIRED_DOC_SURFACES: frozenset[str] = frozenset(
    {
        "docs/index.md",
        "docs/handbook.md",
        "docs/capabilities.md",
        "docs/gradient_coverage_matrix.md",
        "docs/why_oscillatools.md",
        "docs/gallery.md",
        "docs/tier_benchmarks.md",
        "docs/competitive_benchmark.md",
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


def public_surfaces() -> list[Path]:
    """Enumerate the oscillatools public prose surfaces present in the tree.

    ``docs/*.md`` is globbed so a page added later is guarded automatically, plus
    the distribution README and the examples README.
    """
    surfaces = sorted((REPO_ROOT / "docs").glob("*.md"))
    for readme in (REPO_ROOT / "README.md", REPO_ROOT / "examples" / "README.md"):
        if readme.exists():
            surfaces.append(readme)
    return surfaces


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


@pytest.mark.parametrize("surface", public_surfaces(), ids=_relative)
def test_public_surface_is_free_of_self_promotional_framing(surface: Path) -> None:
    """No oscillatools public surface may carry comparative/superlative self-framing."""
    hits = scan_for_self_promotion(surface.read_text(encoding="utf-8"))
    assert not hits, f"{_relative(surface)} carries barred self-promotional framing: {hits}"


def test_surface_enumeration_covers_the_known_docs() -> None:
    """The net must still cover every known docs page.

    Guards against a page being renamed or dropped, which would silently remove it
    from the guard.
    """
    found = {_relative(path) for path in public_surfaces()}
    missing = _REQUIRED_DOC_SURFACES - found
    assert not missing, f"oscillatools doc surfaces dropped from the guard: {sorted(missing)}"


# Representative barred framing — one per enumerated failure mode. A guard that never
# fires is worthless, so each of these must be caught.
_SELF_PROMOTIONAL_SAMPLES: tuple[str, ...] = (
    "the definitive Kuramoto toolkit for research",  # uniqueness superlative + product
    "the premier oscillator-dynamics package",  # uniqueness, hyphenated filler
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
# self-deprecation, and mathematics (including this package's own "leading" usage).
_NEUTRAL_SAMPLES: tuple[str, ...] = (
    "DynamicalSystems.jl is a state-of-the-art dynamical-systems library",  # competitor
    "a state-of-the-art solver from the SciML ecosystem",  # competitor
    "the dominant cluster locks first",  # physics idiom
    "the dominant frequency component",  # physics idiom
    "differ by orders of magnitude in flops",  # neutral quantitative
    "classical solvers outperform this toolkit here",  # honest self-deprecation
    "leading eigenvalue of the Jacobian",  # mathematics
    "leading coherence eigenvectors, and partial-synchronisation",  # this package's own prose
    "the leading-order correction",  # mathematics
)


@pytest.mark.parametrize("sample", _SELF_PROMOTIONAL_SAMPLES)
def test_guard_detects_representative_self_promotional_framing(sample: str) -> None:
    """The guard must fire on every representative barred construction."""
    assert scan_for_self_promotion(sample), f"barred framing slipped past the guard: {sample!r}"


@pytest.mark.parametrize("sample", _NEUTRAL_SAMPLES)
def test_guard_permits_neutral_and_bounded_language(sample: str) -> None:
    """Neutral competitor description, physics idiom and mathematics must stay clean."""
    hits = scan_for_self_promotion(sample)
    assert not hits, f"the guard over-fired on legitimate language {sample!r}: {hits}"


def test_forbidden_pattern_set_covers_the_enumerated_vocabulary() -> None:
    """The pattern set must still cover every enumerated failure mode.

    Guards against the vocabulary being quietly gutted, which would leave a guard
    that passes while enforcing nothing.
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
    assert not uncovered, f"enumerated vocabulary no longer guarded: {uncovered}"
