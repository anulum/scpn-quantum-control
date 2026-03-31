# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Phase Artifact
"""Tests for shared phase artifact schema — elite multi-angle coverage."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_quantum_control.bridge.phase_artifact import (
    LayerStateArtifact,
    LockSignatureArtifact,
    UPDEPhaseArtifact,
)

# ---------------------------------------------------------------------------
# LockSignatureArtifact
# ---------------------------------------------------------------------------


class TestLockSignature:
    def test_to_dict(self):
        lock = LockSignatureArtifact(source_layer=0, target_layer=1, plv=0.91, mean_lag=0.2)
        d = lock.to_dict()
        assert d == {"source_layer": 0, "target_layer": 1, "plv": 0.91, "mean_lag": 0.2}

    def test_from_dict_roundtrip(self):
        lock = LockSignatureArtifact(source_layer=2, target_layer=5, plv=0.75, mean_lag=0.05)
        restored = LockSignatureArtifact.from_dict(lock.to_dict())
        assert restored == lock

    def test_frozen(self):
        lock = LockSignatureArtifact(source_layer=0, target_layer=1, plv=0.5, mean_lag=0.1)
        with pytest.raises(AttributeError):
            lock.plv = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LayerStateArtifact
# ---------------------------------------------------------------------------


class TestLayerState:
    def test_defaults_empty_locks(self):
        layer = LayerStateArtifact(R=0.5, psi=1.0)
        assert layer.lock_signatures == {}

    def test_to_dict(self):
        lock = LockSignatureArtifact(source_layer=0, target_layer=1, plv=0.8, mean_lag=0.1)
        layer = LayerStateArtifact(R=0.7, psi=1.5, lock_signatures={"0_1": lock})
        d = layer.to_dict()
        assert d["R"] == 0.7
        assert "0_1" in d["lock_signatures"]

    def test_from_dict_roundtrip(self):
        lock = LockSignatureArtifact(source_layer=0, target_layer=1, plv=0.8, mean_lag=0.1)
        layer = LayerStateArtifact(R=0.65, psi=2.0, lock_signatures={"0_1": lock})
        restored = LayerStateArtifact.from_dict(layer.to_dict())
        assert pytest.approx(0.65) == restored.R
        assert restored.lock_signatures["0_1"].plv == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# UPDEPhaseArtifact — dict roundtrip
# ---------------------------------------------------------------------------


class TestUPDEDictRoundtrip:
    def test_full_roundtrip(self):
        layer0 = LayerStateArtifact(
            R=0.72,
            psi=1.1,
            lock_signatures={"0_1": LockSignatureArtifact(0, 1, 0.91, 0.2)},
        )
        layer1 = LayerStateArtifact(R=0.48, psi=2.4)
        artifact = UPDEPhaseArtifact(
            layers=[layer0, layer1],
            cross_layer_alignment=np.array([[1.0, 0.9], [0.9, 1.0]]),
            stability_proxy=-0.37,
            regime_id="DEGRADED",
            metadata={"origin": "unit-test"},
        )

        payload = artifact.to_dict()
        restored = UPDEPhaseArtifact.from_dict(payload)
        assert restored.regime_id == "DEGRADED"
        assert restored.layers[0].lock_signatures["0_1"].plv == pytest.approx(0.91)
        np.testing.assert_allclose(restored.cross_layer_alignment, artifact.cross_layer_alignment)

    def test_empty_metadata(self):
        layer = LayerStateArtifact(R=0.5, psi=0.0)
        artifact = UPDEPhaseArtifact(
            layers=[layer],
            cross_layer_alignment=np.eye(1),
            stability_proxy=0.0,
            regime_id="X",
        )
        d = artifact.to_dict()
        assert d["metadata"] == {}


# ---------------------------------------------------------------------------
# UPDEPhaseArtifact — JSON roundtrip
# ---------------------------------------------------------------------------


class TestUPDEJSONRoundtrip:
    def test_json_roundtrip(self):
        layer = LayerStateArtifact(R=0.6, psi=0.5)
        artifact = UPDEPhaseArtifact(
            layers=[layer],
            cross_layer_alignment=np.eye(1),
            stability_proxy=-0.1,
            regime_id="NOMINAL",
            metadata={"origin": "test"},
        )
        json_str = artifact.to_json(indent=None)
        restored = UPDEPhaseArtifact.from_json(json_str)
        assert restored.metadata["origin"] == "test"
        assert restored.regime_id == "NOMINAL"

    def test_json_is_valid(self):
        layer = LayerStateArtifact(R=0.5, psi=0.0)
        artifact = UPDEPhaseArtifact(
            layers=[layer],
            cross_layer_alignment=np.eye(1),
            stability_proxy=0.0,
            regime_id="X",
        )
        parsed = json.loads(artifact.to_json())
        assert "regime_id" in parsed

    def test_json_indent(self):
        layer = LayerStateArtifact(R=0.5, psi=0.0)
        artifact = UPDEPhaseArtifact(
            layers=[layer],
            cross_layer_alignment=np.eye(1),
            stability_proxy=0.0,
            regime_id="X",
        )
        indented = artifact.to_json(indent=2)
        assert "\n" in indented
        compact = artifact.to_json(indent=None)
        assert "\n" not in compact


# ---------------------------------------------------------------------------
# UPDEPhaseArtifact — validation
# ---------------------------------------------------------------------------


class TestUPDEValidation:
    def test_rejects_mismatched_alignment_shape(self):
        with pytest.raises(ValueError, match="shape must match"):
            UPDEPhaseArtifact(
                layers=[LayerStateArtifact(R=0.5, psi=0.0)],
                cross_layer_alignment=np.eye(2),
                stability_proxy=0.0,
                regime_id="X",
            )

    def test_rejects_nonfinite_alignment(self):
        with pytest.raises(ValueError, match="must contain only finite"):
            UPDEPhaseArtifact(
                layers=[LayerStateArtifact(R=0.5, psi=0.0)],
                cross_layer_alignment=np.array([[np.nan]]),
                stability_proxy=0.0,
                regime_id="X",
            )

    def test_rejects_R_out_of_bounds(self):
        with pytest.raises(ValueError, match="R must be in"):
            LayerStateArtifact(R=1.2, psi=0.0)

    def test_multiple_layers_alignment(self):
        layers = [LayerStateArtifact(R=0.5, psi=i * 0.1) for i in range(3)]
        alignment = np.eye(3)
        artifact = UPDEPhaseArtifact(
            layers=layers,
            cross_layer_alignment=alignment,
            stability_proxy=0.0,
            regime_id="X",
        )
        assert len(artifact.layers) == 3
        assert artifact.cross_layer_alignment.shape == (3, 3)


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------


class TestExports:
    def test_all_exports(self):
        from scpn_quantum_control.bridge import phase_artifact

        assert set(phase_artifact.__all__) == {
            "LockSignatureArtifact",
            "LayerStateArtifact",
            "UPDEPhaseArtifact",
        }
