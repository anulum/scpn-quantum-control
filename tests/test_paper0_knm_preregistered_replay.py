# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 K_nm preregistered replay tests
"""Tests for the first Paper 0 K_nm preregistered downstream replay."""

from __future__ import annotations

import json

from scripts.run_paper0_knm_preregistered_replay import (
    SCHEMA,
    build_replay_payload,
    render_markdown,
    write_replay_artifacts,
)


def test_replay_payload_is_non_closing_and_claim_bounded() -> None:
    payload = build_replay_payload()

    assert payload["schema"] == SCHEMA
    assert payload["status"] == "blocked_non_closing_preregistered_replay"
    assert "no-QPU preregistration" in payload["claim_boundary"]
    assert "does not authorise hardware submission" in payload["claim_boundary"]
    assert payload["gates"]["qpu_submission"] == "blocked_no_qpu_preregistration_lane"
    assert payload["gates"]["claim_promotion"] == "blocked_measured_system_gate_open"
    decision = payload["promotion_decision"]
    assert decision["decision"] == "do_not_promote"
    assert decision["hardware_submission_authorised"] is False
    assert decision["claim_promotion_authorised"] is False
    assert "qpu_submission" in decision["blocking_gates"]
    assert len(decision["required_evidence_before_reconsideration"]) >= 4
    assert len(decision["falsifiers"]) >= 4
    manifest = payload["reproducibility"]["input_manifest"]
    assert set(manifest) == {
        "primary_candidate",
        "negative_control",
        "negative_measured_couplings",
    }
    assert all(len(entry["sha256"]) == 64 for entry in manifest.values())
    assert payload["reproducibility"]["randomness_policy"].startswith("fixed local")


def test_replay_payload_preserves_primary_and_negative_control_diagnostics() -> None:
    payload = build_replay_payload()

    primary = payload["primary_candidate"]
    negative = payload["negative_control"]
    assert primary["source_name"] == "eeg_alpha_plv_8ch"
    assert primary["domain"] == "eeg"
    assert primary["matrix_shape"] == [8, 8]
    assert "No per-edge uncertainty model" in primary["blockers"][1]
    assert primary["diagnostics"]["candidate_edge_count"] == 28
    assert primary["null_model"]["seed"] == 2701
    assert primary["null_model"]["permutations"] == 512
    assert 0.0 <= primary["null_model"]["two_sided_empirical_p"] <= 1.0
    assert negative["source_name"] == "ieee5bus_power_grid"
    assert negative["domain"] == "power-grid"
    assert negative["matrix_shape"] == [5, 5]
    assert negative["null_model"]["seed"] == 2702
    assert negative["null_model"]["permutations"] == 512
    assert 0.0 <= negative["null_model"]["two_sided_empirical_p"] <= 1.0
    assert negative["measured_couplings"]["normalisation_locked"] is True
    assert negative["measured_couplings"]["entries_with_uncertainty"] > 0


def test_replay_markdown_exposes_inputs_gates_and_no_qpu_boundary() -> None:
    markdown = render_markdown(build_replay_payload())

    assert "# GOTM-SCPN Paper 0 K_nm preregistered replay" in markdown
    assert "docs/paper0/paper0_first_preregistered_downstream_experiment.md" in markdown
    assert "## Reproducibility manifest" in markdown
    assert "Input digests:" in markdown
    assert "## Promotion decision" in markdown
    assert "Hardware submission authorised: `False`" in markdown
    assert "Falsifiers:" in markdown
    assert "offline deterministic replay; no QPU submission" in markdown
    assert "blocked_primary_missing_per_edge_uncertainty" in markdown
    assert "Empirical two-sided null p-value" in markdown
    assert "Observed-vs-null z-score" in markdown
    assert "IEEE 5-bus power grid" in markdown


def test_replay_writer_is_deterministic(tmp_path) -> None:
    output_json = tmp_path / "paper0_knm_preregistered_replay.json"
    output_doc = tmp_path / "paper0_knm_preregistered_replay.md"

    first_payload = write_replay_artifacts(output_json=output_json, output_doc=output_doc)
    first_json = output_json.read_text(encoding="utf-8")
    first_doc = output_doc.read_text(encoding="utf-8")
    second_payload = write_replay_artifacts(output_json=output_json, output_doc=output_doc)

    assert first_payload == second_payload
    assert first_json == output_json.read_text(encoding="utf-8")
    assert first_doc == output_doc.read_text(encoding="utf-8")
    assert json.loads(first_json)["schema"] == SCHEMA
