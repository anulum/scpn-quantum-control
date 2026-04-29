# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for p_h1 Open Question Audit Runner
"""Tests for the p_h1 open-question audit runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_p_h1_open_question_audit.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "_run_p_h1_open_question_audit",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit_module = _load_script_module()
_scientific_decision = audit_module._scientific_decision
build_audit_payload = audit_module.build_audit_payload
main = audit_module.main


def test_scientific_decision_keeps_p_h1_open_for_knm_graph_mismatch():
    decision = _scientific_decision(
        graph_p_h1=0.9689,
        target=0.72,
        graph_relative_deviation_pct=34.5,
        square_relative_deviation_pct=0.45,
    )

    assert decision["derived_from_first_principles"] is False
    assert decision["requires_ibm_hardware"] is False
    assert decision["current_label"] == "open_empirical_theoretical_parameter"
    assert decision["knm_graph_candidate"]["status"] == "rejects_0.72_as_graph_derivation"


def test_build_audit_payload_has_preregistered_gates_and_provenance():
    payload = build_audit_payload(
        n_values=[4],
        n_seeds=1,
        n_thermalize=20,
        n_measure=20,
        n_temps=4,
        base_seed=7,
        command=["python", "scripts/run_p_h1_open_question_audit.py"],
    )

    assert payload["audit"] == "p_h1_open_question"
    assert payload["decision"]["current_label"] == "open_empirical_theoretical_parameter"
    assert payload["derivation_audit"]["is_derivable"] is False
    assert payload["finite_size_probe"]["n_values"] == [4]
    assert payload["provenance"]["git_commit"]
    assert "ibm_hardware" in payload["preregistered_gates"]


def test_main_writes_json_payload(tmp_path):
    output = tmp_path / "p_h1_audit.json"
    code = main(
        [
            "--output",
            str(output),
            "--n-values",
            "4",
            "--n-seeds",
            "1",
            "--n-thermalize",
            "20",
            "--n-measure",
            "20",
            "--n-temps",
            "4",
        ]
    )

    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["decision"]["requires_ibm_hardware"] is False
    assert payload["finite_size_probe"]["n_seeds"] == 1
