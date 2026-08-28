# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase Artifact Errors
"""Error-path tests for phase_artifact dataclass validation."""

from __future__ import annotations

from typing import Any, cast

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
    """Exercise lock-signature validation failures and valid construction."""

    def test_negative_source_layer(self) -> None:
        """Negative source-layer indices fail closed."""
        with pytest.raises(ValueError, match="must be >= 0"):
            LockSignatureArtifact(source_layer=-1, target_layer=0, plv=0.5, mean_lag=0.1)

    def test_negative_target_layer(self) -> None:
        """Negative target-layer indices fail closed."""
        with pytest.raises(ValueError, match="must be >= 0"):
            LockSignatureArtifact(source_layer=0, target_layer=-1, plv=0.5, mean_lag=0.1)

    def test_inf_plv(self) -> None:
        """Infinite phase-locking values are rejected."""
        with pytest.raises(ValueError, match="must be finite"):
            LockSignatureArtifact(source_layer=0, target_layer=1, plv=np.inf, mean_lag=0.1)

    def test_nan_plv(self) -> None:
        """NaN phase-locking values are rejected."""
        with pytest.raises(ValueError, match="must be finite"):
            LockSignatureArtifact(source_layer=0, target_layer=1, plv=np.nan, mean_lag=0.1)

    def test_inf_mean_lag(self) -> None:
        """Infinite mean phase lags are rejected."""
        with pytest.raises(ValueError, match="must be finite"):
            LockSignatureArtifact(source_layer=0, target_layer=1, plv=0.5, mean_lag=np.inf)

    def test_nan_mean_lag(self) -> None:
        """NaN mean phase lags are rejected."""
        with pytest.raises(ValueError, match="must be finite"):
            LockSignatureArtifact(source_layer=0, target_layer=1, plv=0.5, mean_lag=np.nan)

    def test_valid_construction(self) -> None:
        """Finite lock metrics and non-negative indices construct successfully."""
        lock = LockSignatureArtifact(source_layer=0, target_layer=1, plv=0.9, mean_lag=0.2)
        assert lock.source_layer == 0
        assert lock.plv == 0.9


# ---------------------------------------------------------------------------
# LayerStateArtifact validation
# ---------------------------------------------------------------------------


class TestLayerStateErrors:
    """Exercise layer-state validation failures and interval boundaries."""

    def test_nonstring_lock_key(self) -> None:
        """Lock-signature mappings reject non-string keys."""
        lock = LockSignatureArtifact(source_layer=0, target_layer=1, plv=0.5, mean_lag=0.1)
        invalid_locks: dict[Any, LockSignatureArtifact] = {1: lock}
        with pytest.raises(TypeError, match="keys must be strings"):
            LayerStateArtifact(
                R=0.5,
                psi=0.0,
                lock_signatures=cast(dict[str, LockSignatureArtifact], invalid_locks),
            )

    def test_R_above_one(self) -> None:
        """Coherence above one is rejected."""
        with pytest.raises(ValueError, match="R must be in \\[0, 1\\]"):
            LayerStateArtifact(R=1.2, psi=0.0)

    def test_R_below_zero(self) -> None:
        """Negative coherence is rejected."""
        with pytest.raises(ValueError, match="R must be in \\[0, 1\\]"):
            LayerStateArtifact(R=-0.1, psi=0.0)

    def test_nan_R(self) -> None:
        """NaN coherence is rejected."""
        with pytest.raises(ValueError, match="must be finite"):
            LayerStateArtifact(R=np.nan, psi=0.0)

    def test_nan_psi(self) -> None:
        """NaN mean phase is rejected."""
        with pytest.raises(ValueError, match="must be finite"):
            LayerStateArtifact(R=0.5, psi=np.nan)

    def test_valid_boundary_R_zero(self) -> None:
        """Zero coherence is an inclusive valid boundary."""
        layer = LayerStateArtifact(R=0.0, psi=0.0)
        assert layer.R == 0.0

    def test_valid_boundary_R_one(self) -> None:
        """Unit coherence is an inclusive valid boundary."""
        layer = LayerStateArtifact(R=1.0, psi=0.0)
        assert layer.R == 1.0


# ---------------------------------------------------------------------------
# UPDEPhaseArtifact validation
# ---------------------------------------------------------------------------


class TestUPDEErrors:
    """Exercise complete UPDE artifact validation boundaries."""

    def test_empty_regime_id(self) -> None:
        """An empty regime identifier is rejected."""
        layer = LayerStateArtifact(R=0.5, psi=0.0)
        with pytest.raises(ValueError, match="non-empty"):
            UPDEPhaseArtifact(
                layers=[layer], cross_layer_alignment=np.eye(1), stability_proxy=0.5, regime_id=""
            )

    def test_whitespace_only_regime_id(self) -> None:
        """A whitespace-only regime identifier is rejected."""
        layer = LayerStateArtifact(R=0.5, psi=0.0)
        with pytest.raises(ValueError, match="non-empty"):
            UPDEPhaseArtifact(
                layers=[layer],
                cross_layer_alignment=np.eye(1),
                stability_proxy=0.5,
                regime_id="   ",
            )

    def test_1d_alignment(self) -> None:
        """One-dimensional alignment data is rejected."""
        layer = LayerStateArtifact(R=0.5, psi=0.0)
        with pytest.raises(ValueError, match="2-D"):
            UPDEPhaseArtifact(
                layers=[layer],
                cross_layer_alignment=np.array([1.0]),
                stability_proxy=0.5,
                regime_id="test",
            )

    def test_mismatched_alignment_shape(self) -> None:
        """Alignment shape must match the layer count."""
        with pytest.raises(ValueError, match="shape must match"):
            UPDEPhaseArtifact(
                layers=[LayerStateArtifact(R=0.5, psi=0.0)],
                cross_layer_alignment=np.eye(2),
                stability_proxy=0.0,
                regime_id="X",
            )

    def test_nan_alignment(self) -> None:
        """Alignment matrices reject NaN entries."""
        with pytest.raises(ValueError, match="must contain only finite"):
            UPDEPhaseArtifact(
                layers=[LayerStateArtifact(R=0.5, psi=0.0)],
                cross_layer_alignment=np.array([[np.nan]]),
                stability_proxy=0.0,
                regime_id="X",
            )

    def test_inf_stability_proxy(self) -> None:
        """Infinite stability proxies are rejected."""
        with pytest.raises(ValueError, match="must be finite"):
            UPDEPhaseArtifact(
                layers=[LayerStateArtifact(R=0.5, psi=0.0)],
                cross_layer_alignment=np.eye(1),
                stability_proxy=np.inf,
                regime_id="X",
            )

    def test_valid_construction(self) -> None:
        """A finite square artifact constructs successfully."""
        layer = LayerStateArtifact(R=0.5, psi=1.0)
        artifact = UPDEPhaseArtifact(
            layers=[layer], cross_layer_alignment=np.eye(1), stability_proxy=-0.5, regime_id="OK"
        )
        assert artifact.regime_id == "OK"
        assert artifact.stability_proxy == -0.5
