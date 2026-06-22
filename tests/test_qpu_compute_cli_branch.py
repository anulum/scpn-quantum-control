# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch test for the QPU compute CLI dispatch
"""Fail-closed test for the QPU compute CLI command dispatcher."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from scpn_quantum_control import qpu_compute


def test_main_rejects_unsupported_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unrecognised parsed command fails closed.

    Argparse normally rejects unknown subcommands, so the dispatcher's final
    guard is exercised through a parser stand-in that yields a foreign command.
    """

    def _fake_parser() -> Any:
        return SimpleNamespace(parse_args=lambda _argv: SimpleNamespace(command="bogus"))

    monkeypatch.setattr(qpu_compute, "_build_parser", _fake_parser)
    with pytest.raises(ValueError, match="unsupported command"):
        qpu_compute.main([])
