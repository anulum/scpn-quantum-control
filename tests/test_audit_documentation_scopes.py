# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — documentation scope gate owner
"""Prove the scope gate fails on new debt and passes on the live tree.

The gate exists to stop a scope that reached zero from quietly re-opening, so
the case that matters is a newly added file with an undocumented function. That
is what these tests build and check, rather than asserting only that today's
tree is clean — a gate never seen to fail is not evidence.
"""

from __future__ import annotations

from pathlib import Path

from tools import audit_documentation_scopes as gate


def _write_module(root: Path, scope: str, name: str, body: str) -> None:
    """Place one module inside a scanned scope."""
    directory = root / scope
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(body, encoding="utf-8")


def test_documented_module_passes(tmp_path: Path) -> None:
    """A fully documented module leaves the scope clean."""
    _write_module(
        tmp_path,
        "src",
        "ok.py",
        '"""A module that documents itself."""\n\n\ndef f() -> int:\n    """Return one."""\n    return 1\n',
    )
    assert gate.unexempt(gate.scan(tmp_path, ("src",)), tmp_path) == []


def test_undocumented_function_is_reported(tmp_path: Path) -> None:
    """An undocumented public function re-opens the scope and is caught."""
    _write_module(
        tmp_path,
        "src",
        "bare.py",
        '"""A module whose function is undocumented."""\n\n\ndef f() -> int:\n    return 1\n',
    )
    offenders = gate.unexempt(gate.scan(tmp_path, ("src",)), tmp_path)
    assert len(offenders) == 1
    assert "bare.py" in offenders[0]


def test_exempt_file_is_not_reported(tmp_path: Path) -> None:
    """A recorded exemption suppresses its own findings and nothing else."""
    exempt_path = next(iter(gate.EXEMPT))
    _write_module(
        tmp_path,
        str(Path(exempt_path).parent),
        Path(exempt_path).name,
        '"""Frozen runner."""\n\n\ndef f() -> int:\n    return 1\n',
    )
    findings = gate.scan(tmp_path, ("data",))
    assert findings, "the scan must still see the file; the exemption filters, it does not hide"
    assert gate.unexempt(findings, tmp_path) == []


def test_every_exemption_states_a_reason() -> None:
    """An exemption without a reason is an exclusion, and is not allowed."""
    for path, reason in gate.EXEMPT.items():
        assert reason.strip(), path
        assert len(reason.split()) >= 4, f"{path}: the reason is too thin to review"


def test_the_live_tree_holds_every_scope_at_zero() -> None:
    """The repository itself satisfies the rule the gate enforces."""
    root = Path(__file__).resolve().parent.parent
    assert gate.unexempt(gate.scan(root, gate.ENFORCED_SCOPES), root) == []
