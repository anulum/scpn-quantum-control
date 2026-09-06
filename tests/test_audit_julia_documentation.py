# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Julia documentation gate owner
"""Prove the Julia documentation gate catches what it claims to catch.

A gate is only evidence once it has been seen to fail. These tests build Julia
sources whose documentation state is known and check the verdict against it,
including the two cases a text scan gets wrong: a comment above a definition,
which is not a docstring, and a docstring separated from its definition by a
blank line, which is also not one.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tools import audit_julia_documentation as gate

pytestmark = pytest.mark.skipif(
    shutil.which("julia") is None,
    reason="the gate delegates to Julia's parser, which is not installed here",
)

SCOPE = "jl"


def _write(root: Path, name: str, body: str) -> None:
    """Place one Julia source inside the scope the gate measures."""
    directory = root / SCOPE
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(body, encoding="utf-8")


def test_documented_definition_passes(tmp_path: Path) -> None:
    """A docstring immediately above a definition satisfies the gate."""
    _write(tmp_path, "ok.jl", '"""\n    f\n\nWhat f computes.\n"""\nfunction f(x)\n    x\nend\n')
    total, undocumented = gate.measure(tmp_path, SCOPE)
    assert (total, undocumented) == (1, ())


def test_bare_definition_is_reported(tmp_path: Path) -> None:
    """A definition with nothing above it is reported."""
    _write(tmp_path, "bare.jl", "function f(x)\n    x\nend\n")
    total, undocumented = gate.measure(tmp_path, SCOPE)
    assert total == 1
    assert undocumented == ("bare.jl:f",)


def test_comment_is_not_a_docstring(tmp_path: Path) -> None:
    """A `#` comment above a definition does not count.

    This is the case a text scan gets wrong. The parser never turns a comment
    into a `Core.@doc` node, so the definition is undocumented however helpful
    the comment reads.
    """
    _write(tmp_path, "comment.jl", "# What f computes.\nfunction f(x)\n    x\nend\n")
    _, undocumented = gate.measure(tmp_path, SCOPE)
    assert undocumented == ("comment.jl:f",)


def test_short_form_definition_is_counted(tmp_path: Path) -> None:
    """`f(x) = x` is a definition and is held to the same rule."""
    _write(tmp_path, "short.jl", "f(x) = x\n")
    total, undocumented = gate.measure(tmp_path, SCOPE)
    assert total == 1
    assert undocumented == ("short.jl:f",)


def test_the_live_tier_is_fully_documented() -> None:
    """The shipped Julia tier carries a docstring on every definition."""
    root = Path(__file__).resolve().parent.parent
    total, undocumented = gate.measure(root)
    assert total > 0
    assert undocumented == ()


def test_missing_julia_refuses_rather_than_passing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without Julia the gate raises; it must never report a clean surface."""
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(gate.JuliaUnavailableError):
        gate.julia_executable()
