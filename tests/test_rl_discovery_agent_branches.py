# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the RL discovery agent
"""Pre-discovery branch tests for the RL witness-discovery agent.

Covers the empty next-parameters result and the empty-evaluations save path
when no discovery run has populated a result yet.
"""

from __future__ import annotations

import json
from pathlib import Path

from scpn_quantum_control.analysis.rl_discovery_agent import RLDiscoveryAgent


def test_get_next_params_empty_before_discovery() -> None:
    """Before any discovery run the next-parameters lookup is empty."""
    assert RLDiscoveryAgent().get_next_params() == {}


def test_save_discovered_phases_empty_before_discovery(tmp_path: Path) -> None:
    """Before any discovery run the saved payload records empty evaluations."""
    path = tmp_path / "phases.json"
    RLDiscoveryAgent().save_discovered_phases(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"evaluations": []}
