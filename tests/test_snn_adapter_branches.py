# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch test for the SNN adapter dependency guard
"""Dependency-guard test for the Arcane neuron bridge."""

from __future__ import annotations

import sys

import pytest

from scpn_quantum_control.bridge.snn_adapter import ArcaneNeuronBridge


def test_arcane_bridge_requires_sc_neurocore(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing the bridge without sc-neurocore raises a clear ImportError."""
    monkeypatch.setitem(sys.modules, "sc_neurocore.neurons.models", None)
    with pytest.raises(ImportError, match="sc-neurocore required"):
        ArcaneNeuronBridge(n_neurons=2, n_inputs=2)
