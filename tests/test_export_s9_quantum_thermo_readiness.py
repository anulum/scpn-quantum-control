# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — export s9 quantum thermo readiness tests
# scpn-quantum-control -- S9 readiness export tests
"""Tests for the S9 quantum-thermodynamics readiness export script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _load_export_module() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "export_s9_quantum_thermo_readiness.py"
    )
    spec = importlib.util.spec_from_file_location(
        "export_s9_quantum_thermo_readiness", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load S9 export script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export_module = _load_export_module()


def test_s9_export_writes_json_and_markdown(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "data"
    doc_path = tmp_path / "quantum_thermo.md"

    monkeypatch.setattr(
        export_module,
        "parse_args",
        lambda: export_module.argparse.Namespace(out_dir=out_dir, doc_path=doc_path),
    )

    assert export_module.main() == 0
    json_files = list(out_dir.glob("quantum_thermo_readiness_*.json"))
    assert len(json_files) == 1
    payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert payload["hardware_submission_allowed"] is False
    assert payload["thermodynamic_peak_claim_allowed"] is False
    assert "scpn-bench s9-quantum-thermo-readiness" in doc_path.read_text(encoding="utf-8")
