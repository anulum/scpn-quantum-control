# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch test for the UltraScale HLS quantiser fallback
"""Native-engine fallback test for the Q-format quantiser."""

from __future__ import annotations

import sys

import pytest

from scpn_quantum_control.codegen.ultrascale_hls import _python_quantise, quantise_q_format


def test_quantise_falls_back_to_python(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the native engine the quantiser uses the Python implementation."""
    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", None)
    values = [0.5, -0.25, 0.125]
    assert quantise_q_format(values, 4, 8) == _python_quantise(values, 4, 8)
