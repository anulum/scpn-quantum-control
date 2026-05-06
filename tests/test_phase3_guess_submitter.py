# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- tests for Phase 3 GUESS submitter
"""Tests for the Phase 3 GUESS submitter circuit matrix and gates."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts" / "phase3_guess_dla_ibm.py"
    spec = importlib.util.spec_from_file_location("phase3_guess_dla_ibm", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Phase 3 GUESS submitter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_guess_circuit_matrix_matches_preregistration() -> None:
    module = _load_module()

    main, readout = module.build_guess_circuits()

    assert len(main) == 192
    assert len(readout) == 4
    assert {meta["noise_scale"] for meta, _ in main} == {1, 3, 5}
    assert {meta["depth"] for meta, _ in main} == {6, 8, 10, 14}
    assert {meta["initial"] for meta, _ in main} == {"0011", "0001"}
    assert {meta["initial"] for meta, _ in readout} == {"0011", "0001", "0000", "1111"}
    assert all(meta["physical_qubits"] == [5, 6, 7, 8] for meta, _ in main + readout)


def test_folded_circuit_depth_increases_with_noise_scale() -> None:
    module = _load_module()

    main, _ = module.build_guess_circuits()
    depth6_even = {
        meta["noise_scale"]: circuit.depth()
        for meta, circuit in main
        if meta["depth"] == 6 and meta["sector"] == "even" and meta["rep"] == 0
    }

    assert depth6_even[1] < depth6_even[3] < depth6_even[5]


def test_submit_requires_budget_confirmation(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(sys, "argv", ["phase3_guess_dla_ibm.py", "--submit"])

    assert module.main() == 2
