# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Phase Artifact Errors
"""Error-path tests for phase_artifact dataclass validation — elite coverage."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.phase_artifact import (
    LayerStateArtifact,
    LockSignatureArtifact,
    UPDEPhaseArtifact,
)

# ---------------------------------------------------------------------------
# LockSignatureArtifact validation
# ---------------------------------------------------------------------------


class TestLockSignatureErrors:
    def test_negative_source_layer(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            LockSignatureArtifact(source_layer=-1, target_layer=0, plv=0.5, mean_lag=0.1)

    def test_negative_target_layer(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            LockSignatureArtifact(source_layer=0, target_layer=-1, plv=0.5, mean_lag=0.1)

    def test_inf_plv(self):
        with pytest.raises(ValueError, match="must be finite"):
            LockSignatureArtifact(source_layer=0, target_layer=1, plv=np.inf, mean_lag=0.1)

    def test_nan_plv(self):
        with pytest.raises(ValueError, match="must be finite"):
            LockSignatureArtifact(source_layer=0, target_layer=1, plv=np.nan, mean_lag=0.1)

    def test_inf_mean_lag(self):
        with pytest.raises(ValueError, match="must be finite"):
            LockSignatureArtifact(source_layer=0, target_layer=1, plv=0.5, mean_lag=np.inf)

    def test_nan_mean_lag(self):
        with pytest.raises(ValueError, match="must be finite"):
            LockSignatureArtifact(source_layer=0, target_layer=1, plv=0.5, mean_lag=np.nan)

    def test_valid_construction(self):
        lock = LockSignatureArtifact(source_layer=0, target_layer=1, plv=0.9, mean_lag=0.2)
        assert lock.source_layer == 0
        assert lock.plv == 0.9


# ---------------------------------------------------------------------------
# LayerStateArtifact validation
# ---------------------------------------------------------------------------


class TestLayerStateErrors:
    def test_nonstring_lock_key(self):
        lock = LockSignatureArtifact(source_layer=0, target_layer=1, plv=0.5, mean_lag=0.1)
        with pytest.raises(TypeError, match="keys must be strings"):
            LayerStateArtifact(R=0.5, psi=0.0, lock_signatures={1: lock})

    def test_R_above_one(self):
        with pytest.raises(ValueError, match="R must be in \\[0, 1\\]"):
            LayerStateArtifact(R=1.2, psi=0.0)

    def test_R_below_zero(self):
        with pytest.raises(ValueError, match="R must be in \\[0, 1\\]"):
            LayerStateArtifact(R=-0.1, psi=0.0)

    def test_nan_R(self):
        with pytest.raises(ValueError, match="must be finite"):
            LayerStateArtifact(R=np.nan, psi=0.0)

    def test_nan_psi(self):
        with pytest.raises(ValueError, match="must be finite"):
            LayerStateArtifact(R=0.5, psi=np.nan)

    def test_valid_boundary_R_zero(self):
        layer = LayerStateArtifact(R=0.0, psi=0.0)
        assert layer.R == 0.0

    def test_valid_boundary_R_one(self):
        layer = LayerStateArtifact(R=1.0, psi=0.0)
        assert layer.R == 1.0


# ---------------------------------------------------------------------------
# UPDEPhaseArtifact validation
# ---------------------------------------------------------------------------


class TestUPDEErrors:
    def test_empty_regime_id(self):
        layer = LayerStateArtifact(R=0.5, psi=0.0)
        with pytest.raises(ValueError, match="non-empty"):
            UPDEPhaseArtifact(
                layers=[layer], cross_layer_alignment=np.eye(1), stability_proxy=0.5, regime_id=""
            )

    def test_whitespace_only_regime_id(self):
        layer = LayerStateArtifact(R=0.5, psi=0.0)
        with pytest.raises(ValueError, match="non-empty"):
            UPDEPhaseArtifact(
                layers=[layer],
                cross_layer_alignment=np.eye(1),
                stability_proxy=0.5,
                regime_id="   ",
            )

    def test_1d_alignment(self):
        layer = LayerStateArtifact(R=0.5, psi=0.0)
        with pytest.raises(ValueError, match="2-D"):
            UPDEPhaseArtifact(
                layers=[layer],
                cross_layer_alignment=np.array([1.0]),
                stability_proxy=0.5,
                regime_id="test",
            )

    def test_mismatched_alignment_shape(self):
        with pytest.raises(ValueError, match="shape must match"):
            UPDEPhaseArtifact(
                layers=[LayerStateArtifact(R=0.5, psi=0.0)],
                cross_layer_alignment=np.eye(2),
                stability_proxy=0.0,
                regime_id="X",
            )

    def test_nan_alignment(self):
        with pytest.raises(ValueError, match="must contain only finite"):
            UPDEPhaseArtifact(
                layers=[LayerStateArtifact(R=0.5, psi=0.0)],
                cross_layer_alignment=np.array([[np.nan]]),
                stability_proxy=0.0,
                regime_id="X",
            )

    def test_inf_stability_proxy(self):
        with pytest.raises(ValueError, match="must be finite"):
            UPDEPhaseArtifact(
                layers=[LayerStateArtifact(R=0.5, psi=0.0)],
                cross_layer_alignment=np.eye(1),
                stability_proxy=np.inf,
                regime_id="X",
            )

    def test_valid_construction(self):
        layer = LayerStateArtifact(R=0.5, psi=1.0)
        artifact = UPDEPhaseArtifact(
            layers=[layer], cross_layer_alignment=np.eye(1), stability_proxy=-0.5, regime_id="OK"
        )
        assert artifact.regime_id == "OK"
        assert artifact.stability_proxy == -0.5
