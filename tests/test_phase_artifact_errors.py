# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Error-path tests for phase_artifact dataclass validation."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.phase_artifact import (
    LayerStateArtifact,
    LockSignatureArtifact,
    UPDEPhaseArtifact,
)


def test_lock_signature_negative_layer():
    with pytest.raises(ValueError, match="must be >= 0"):
        LockSignatureArtifact(source_layer=-1, target_layer=0, plv=0.5, mean_lag=0.1)


def test_layer_state_nonstring_lock_key():
    lock = LockSignatureArtifact(source_layer=0, target_layer=1, plv=0.5, mean_lag=0.1)
    with pytest.raises(TypeError, match="keys must be strings"):
        LayerStateArtifact(R=0.5, psi=0.0, lock_signatures={1: lock})


def test_upde_empty_regime_id():
    layer = LayerStateArtifact(R=0.5, psi=0.0)
    alignment = np.eye(1)
    with pytest.raises(ValueError, match="non-empty"):
        UPDEPhaseArtifact(
            layers=[layer], cross_layer_alignment=alignment, stability_proxy=0.5, regime_id=""
        )


def test_upde_1d_alignment():
    layer = LayerStateArtifact(R=0.5, psi=0.0)
    with pytest.raises(ValueError, match="2-D"):
        UPDEPhaseArtifact(
            layers=[layer],
            cross_layer_alignment=np.array([1.0]),
            stability_proxy=0.5,
            regime_id="test",
        )
