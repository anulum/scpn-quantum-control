"""Tests for shared phase artifact schema."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.phase_artifact import (
    LayerStateArtifact,
    LockSignatureArtifact,
    UPDEPhaseArtifact,
)


def test_phase_artifact_roundtrip_dict_and_json() -> None:
    layer0 = LayerStateArtifact(
        R=0.72,
        psi=1.1,
        lock_signatures={
            "0_1": LockSignatureArtifact(source_layer=0, target_layer=1, plv=0.91, mean_lag=0.2)
        },
    )
    layer1 = LayerStateArtifact(R=0.48, psi=2.4)
    artifact = UPDEPhaseArtifact(
        layers=[layer0, layer1],
        cross_layer_alignment=np.array([[1.0, 0.9], [0.9, 1.0]], dtype=np.float64),
        stability_proxy=-0.37,
        regime_id="DEGRADED",
        metadata={"origin": "unit-test"},
    )

    payload = artifact.to_dict()
    restored = UPDEPhaseArtifact.from_dict(payload)
    assert restored.regime_id == "DEGRADED"
    assert restored.layers[0].lock_signatures["0_1"].plv == pytest.approx(0.91)
    np.testing.assert_allclose(restored.cross_layer_alignment, artifact.cross_layer_alignment)

    restored_json = UPDEPhaseArtifact.from_json(artifact.to_json(indent=None))
    assert restored_json.metadata["origin"] == "unit-test"


def test_phase_artifact_rejects_mismatched_alignment_shape() -> None:
    with pytest.raises(ValueError, match="shape must match number of layers"):
        UPDEPhaseArtifact(
            layers=[LayerStateArtifact(R=0.5, psi=0.0)],
            cross_layer_alignment=np.array([[1.0, 0.2], [0.2, 1.0]], dtype=np.float64),
            stability_proxy=0.0,
            regime_id="NOMINAL",
        )


def test_phase_artifact_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="must contain only finite values"):
        UPDEPhaseArtifact(
            layers=[LayerStateArtifact(R=0.5, psi=0.0)],
            cross_layer_alignment=np.array([[np.nan]], dtype=np.float64),
            stability_proxy=0.0,
            regime_id="NOMINAL",
        )

    with pytest.raises(ValueError, match="must be finite"):
        LockSignatureArtifact(source_layer=0, target_layer=1, plv=np.inf, mean_lag=0.1)


def test_layer_state_bounds_are_enforced() -> None:
    with pytest.raises(ValueError, match="R must be in \\[0, 1\\]"):
        LayerStateArtifact(R=1.2, psi=0.0)
