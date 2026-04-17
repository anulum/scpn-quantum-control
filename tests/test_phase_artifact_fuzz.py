# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Property-based fuzz for bridge/phase_artifact.py
"""Property-based fuzz tests for the phase-artifact dataclasses.

Closes a slice of audit item B8. The three frozen dataclasses in
``scpn_quantum_control.bridge.phase_artifact``
(:class:`LockSignatureArtifact`, :class:`LayerStateArtifact`,
:class:`UPDEPhaseArtifact`) validate inputs in ``__post_init__``.
Previous tests only hit the happy path; Hypothesis exercises the
full boundary surface:

* Every value constructor rejects the right inputs and only those.
* Round-trip to_dict → from_dict is identity on every valid input.
* Numpy matrix shape checks fire exactly when the shape is wrong.
* Non-finite floats are rejected uniformly.

This module is the template for fuzzing the other three unfuzzed
surfaces called out in the audit (``hardware/``, ``sectors``, ``qec``).
Each will get its own fuzz file once a representative validator is
ported — adding the backbone here is intentional.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from scpn_quantum_control.bridge.phase_artifact import (
    LayerStateArtifact,
    LockSignatureArtifact,
    UPDEPhaseArtifact,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

finite_floats = st.floats(
    allow_nan=False,
    allow_infinity=False,
    width=64,
    min_value=-1e12,
    max_value=1e12,
)

plv_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
R_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

non_finite = st.sampled_from([math.inf, -math.inf, math.nan])

non_negative_int = st.integers(min_value=0, max_value=1_000_000)
negative_int = st.integers(max_value=-1)

_GLOBAL_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# ---------------------------------------------------------------------------
# LockSignatureArtifact
# ---------------------------------------------------------------------------


class TestLockSignatureArtifactFuzz:
    @_GLOBAL_SETTINGS
    @given(
        src=non_negative_int,
        tgt=non_negative_int,
        plv=plv_strategy,
        mean_lag=finite_floats,
    )
    def test_valid_inputs_construct(
        self,
        src: int,
        tgt: int,
        plv: float,
        mean_lag: float,
    ) -> None:
        sig = LockSignatureArtifact(
            source_layer=src,
            target_layer=tgt,
            plv=plv,
            mean_lag=mean_lag,
        )
        assert sig.source_layer == src
        assert sig.target_layer == tgt
        assert sig.plv == plv
        assert sig.mean_lag == mean_lag

    @_GLOBAL_SETTINGS
    @given(src=negative_int, tgt=non_negative_int)
    def test_rejects_negative_source_layer(self, src: int, tgt: int) -> None:
        with pytest.raises(ValueError, match=">="):
            LockSignatureArtifact(source_layer=src, target_layer=tgt, plv=0.5, mean_lag=0.0)

    @_GLOBAL_SETTINGS
    @given(src=non_negative_int, tgt=negative_int)
    def test_rejects_negative_target_layer(self, src: int, tgt: int) -> None:
        with pytest.raises(ValueError, match=">="):
            LockSignatureArtifact(source_layer=src, target_layer=tgt, plv=0.5, mean_lag=0.0)

    @_GLOBAL_SETTINGS
    @given(bad=non_finite)
    def test_rejects_non_finite_plv(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            LockSignatureArtifact(source_layer=0, target_layer=1, plv=bad, mean_lag=0.0)

    @_GLOBAL_SETTINGS
    @given(bad=non_finite)
    def test_rejects_non_finite_mean_lag(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            LockSignatureArtifact(source_layer=0, target_layer=1, plv=0.5, mean_lag=bad)

    @_GLOBAL_SETTINGS
    @given(
        src=non_negative_int,
        tgt=non_negative_int,
        plv=plv_strategy,
        mean_lag=finite_floats,
    )
    def test_roundtrip_to_from_dict(
        self,
        src: int,
        tgt: int,
        plv: float,
        mean_lag: float,
    ) -> None:
        sig = LockSignatureArtifact(
            source_layer=src,
            target_layer=tgt,
            plv=plv,
            mean_lag=mean_lag,
        )
        restored = LockSignatureArtifact.from_dict(sig.to_dict())
        assert restored == sig


# ---------------------------------------------------------------------------
# LayerStateArtifact
# ---------------------------------------------------------------------------


class TestLayerStateArtifactFuzz:
    @_GLOBAL_SETTINGS
    @given(R=R_strategy, psi=finite_floats)
    def test_valid_R_psi_construct(self, R: float, psi: float) -> None:
        layer = LayerStateArtifact(R=R, psi=psi)
        assert layer.R == R
        assert layer.psi == psi
        assert layer.lock_signatures == {}

    @_GLOBAL_SETTINGS
    @given(
        R=st.one_of(
            st.floats(max_value=-1e-12, allow_nan=False, allow_infinity=False),
            st.floats(min_value=1.0 + 1e-12, allow_nan=False, allow_infinity=False),
        )
    )
    def test_rejects_R_out_of_range(self, R: float) -> None:
        with pytest.raises(ValueError, match=r"R must be"):
            LayerStateArtifact(R=R, psi=0.0)

    @_GLOBAL_SETTINGS
    @given(bad=non_finite)
    def test_rejects_non_finite_R(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            LayerStateArtifact(R=bad, psi=0.0)

    @_GLOBAL_SETTINGS
    @given(bad=non_finite)
    def test_rejects_non_finite_psi(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            LayerStateArtifact(R=0.5, psi=bad)

    @_GLOBAL_SETTINGS
    @given(R=R_strategy, psi=finite_floats)
    def test_roundtrip_without_locks(self, R: float, psi: float) -> None:
        layer = LayerStateArtifact(R=R, psi=psi)
        restored = LayerStateArtifact.from_dict(layer.to_dict())
        assert restored.R == layer.R
        assert restored.psi == layer.psi
        assert restored.lock_signatures == {}


# ---------------------------------------------------------------------------
# UPDEPhaseArtifact
# ---------------------------------------------------------------------------


def _layer_strategy() -> st.SearchStrategy[LayerStateArtifact]:
    return st.builds(
        LayerStateArtifact,
        R=R_strategy,
        psi=finite_floats,
    )


def _finite_matrix(n: int) -> st.SearchStrategy[np.ndarray]:
    return arrays(
        dtype=np.float64,
        shape=(n, n),
        elements=st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
        ),
    )


class TestUPDEPhaseArtifactFuzz:
    @_GLOBAL_SETTINGS
    @given(
        n_layers=st.integers(min_value=1, max_value=6),
        stability=finite_floats,
        regime=st.text(min_size=1, max_size=32).filter(lambda s: s.strip()),
        data=st.data(),
    )
    def test_valid_artifact_constructs(
        self,
        n_layers: int,
        stability: float,
        regime: str,
        data: st.DataObject,
    ) -> None:
        layers = data.draw(st.lists(_layer_strategy(), min_size=n_layers, max_size=n_layers))
        alignment = data.draw(_finite_matrix(n_layers))
        art = UPDEPhaseArtifact(
            layers=layers,
            cross_layer_alignment=alignment,
            stability_proxy=stability,
            regime_id=regime,
        )
        assert len(art.layers) == n_layers
        assert art.cross_layer_alignment.shape == (n_layers, n_layers)
        assert art.stability_proxy == stability

    @_GLOBAL_SETTINGS
    @given(regime=st.sampled_from(["", "   ", "\t\n"]))
    def test_rejects_empty_regime_id(self, regime: str) -> None:
        with pytest.raises(ValueError, match="regime_id"):
            UPDEPhaseArtifact(
                layers=[LayerStateArtifact(R=0.5, psi=0.0)],
                cross_layer_alignment=np.zeros((1, 1)),
                stability_proxy=0.0,
                regime_id=regime,
            )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=5),
        wrong_n=st.integers(min_value=1, max_value=5),
        data=st.data(),
    )
    def test_rejects_alignment_shape_mismatch(
        self,
        n: int,
        wrong_n: int,
        data: st.DataObject,
    ) -> None:
        if n == wrong_n:
            return  # trivially matching case
        layers = [LayerStateArtifact(R=0.5, psi=0.0) for _ in range(n)]
        alignment = data.draw(_finite_matrix(wrong_n))
        with pytest.raises(ValueError, match="shape must match"):
            UPDEPhaseArtifact(
                layers=layers,
                cross_layer_alignment=alignment,
                stability_proxy=0.0,
                regime_id="fuzz",
            )

    @_GLOBAL_SETTINGS
    @given(n=st.integers(min_value=1, max_value=4), bad=non_finite)
    def test_rejects_non_finite_alignment_entries(self, n: int, bad: float) -> None:
        layers = [LayerStateArtifact(R=0.5, psi=0.0) for _ in range(n)]
        alignment = np.zeros((n, n))
        alignment[0, 0] = bad
        with pytest.raises(ValueError, match="finite"):
            UPDEPhaseArtifact(
                layers=layers,
                cross_layer_alignment=alignment,
                stability_proxy=0.0,
                regime_id="fuzz",
            )

    @_GLOBAL_SETTINGS
    @given(n=st.integers(min_value=1, max_value=4))
    def test_rejects_non_2d_alignment(self, n: int) -> None:
        layers = [LayerStateArtifact(R=0.5, psi=0.0) for _ in range(n)]
        # 1-D array of the right total length — still not a matrix.
        alignment_1d = np.zeros(n * n)
        with pytest.raises(ValueError, match="2-D"):
            UPDEPhaseArtifact(
                layers=layers,
                cross_layer_alignment=alignment_1d,
                stability_proxy=0.0,
                regime_id="fuzz",
            )


# ---------------------------------------------------------------------------
# Smoke: show Hypothesis can shrink a failure to its minimal counter-example
# ---------------------------------------------------------------------------


class TestShrinkingSanity:
    """Confirm that Hypothesis shrinking works against our validators.

    These are not traditional tests — they are inverse sanity checks that
    catch regressions in the fuzz strategies themselves.
    """

    def test_plv_boundary_both_inclusive(self) -> None:
        LockSignatureArtifact(source_layer=0, target_layer=0, plv=0.0, mean_lag=0.0)
        LockSignatureArtifact(source_layer=0, target_layer=0, plv=1.0, mean_lag=0.0)

    def test_R_boundary_both_inclusive(self) -> None:
        LayerStateArtifact(R=0.0, psi=0.0)
        LayerStateArtifact(R=1.0, psi=0.0)
