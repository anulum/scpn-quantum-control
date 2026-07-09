# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for UltraScale+ HLS artifact export runner
"""Tests for ``scripts/export_ultrascale_hls_artifact.py``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from scpn_quantum_control.codegen.ultrascale_hls import (
    HLS_ARTIFACT_SCHEMA_VERSION,
    HLS_CONSUMER_CONTRACT_VERSION,
    verify_hls_artifact_manifest,
)


def test_export_ultrascale_hls_artifact_cli(tmp_path: Path) -> None:
    """The production runner emits and verifies a manifest-bound HLS artifact."""
    repo = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "artifacts"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo / "src"), str(repo / "oscillatools" / "src"), str(repo)]
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "export_ultrascale_hls_artifact.py"),
            "--output-dir",
            str(output_dir),
            "--artifact-id",
            "cli-hls-v1",
            "--target-sku",
            "zu3eg",
            "--sample-rate-hz",
            "100000000",
            "--n-samples",
            "32",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "consumer_contract_version: sc-neurocore.hdl_gen.hls_ingest.v1" in proc.stdout

    manifest_path = output_dir / "cli-hls-v1" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == HLS_ARTIFACT_SCHEMA_VERSION
    assert payload["consumer_contract_version"] == HLS_CONSUMER_CONTRACT_VERSION
    assert payload["pulse"]["sample_count"] == 32
    assert verify_hls_artifact_manifest(manifest_path).valid
    assert (output_dir / "cli-hls-v1" / "pulse_axi_stream.hpp").is_file()
