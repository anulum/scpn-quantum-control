# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Biological QEC CLI Tests
"""Tests for biological QEC report CLI generation."""

from __future__ import annotations

import json

import numpy as np

from scpn_quantum_control.qec.biological_cli import main


def test_biological_qec_cli_generates_json_report(tmp_path):
    """CLI emits a complete biological QEC execution payload."""
    k_path = tmp_path / "K.npy"
    z_path = tmp_path / "z.npy"
    d_path = tmp_path / "domains.json"
    m_path = tmp_path / "metadata.json"
    out_path = tmp_path / "result.json"

    K = np.array(
        [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=float,
    )
    z_errors = np.array([0, 1, 0], dtype=np.int8)

    np.save(k_path, K)
    np.save(z_path, z_errors)
    d_path.write_text(json.dumps({"0": "L1", "1": "L1", "2": "L2", "3": "L2"}), encoding="utf-8")
    m_path.write_text(json.dumps({"campaign": "cli"}), encoding="utf-8")

    rc = main(
        [
            "--k",
            str(k_path),
            "--z-errors",
            str(z_path),
            "--domains",
            str(d_path),
            "--metadata",
            str(m_path),
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["success"] is True
    assert payload["metadata"]["campaign"] == "cli"
    assert payload["diagnostics"]["n_nodes"] == 4


def test_biological_qec_cli_generates_batch_report(tmp_path):
    """CLI accepts 2D z-error matrix and emits aggregate batch payload."""
    k_path = tmp_path / "K.npy"
    z_path = tmp_path / "z_batch.npy"
    out_path = tmp_path / "result_batch.json"

    K = np.array(
        [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=float,
    )
    z_batch = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int8)
    np.save(k_path, K)
    np.save(z_path, z_batch)

    rc = main(
        [
            "--k",
            str(k_path),
            "--z-errors",
            str(z_path),
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["n_runs"] == 2
    assert len(payload["runs"]) == 2
    assert 0.0 <= payload["success_rate"] <= 1.0
