# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the KYMA v3 non-oscillator baselines
"""Tests for the KYMA v3 non-oscillator baselines."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")

import numpy as np

from scpn_quantum_control.benchmarks.kyma_v3 import baselines, task

_INITS = {
    "mlp": (baselines.mlp_init, baselines.mlp_param_count),
    "gnn": (baselines.gnn_init, baselines.gnn_param_count),
    "transformer": (baselines.transformer_init, baselines.transformer_param_count),
    "sequential": (baselines.sequential_init, baselines.sequential_param_count),
}


@pytest.mark.parametrize("name", sorted(_INITS))
@pytest.mark.parametrize("width", [2, 3, 5])
def test_param_count_formula_matches_the_initialised_tensors(name: str, width: int) -> None:
    """Check that param count formula matches the initialised tensors."""
    init, count = _INITS[name]
    params = init(0, width)
    assert sum(int(np.prod(value.shape)) for value in params.values()) == count(width)


def test_features_encode_phases_and_one_hot_codes() -> None:
    """Check that features encode phases and one hot codes."""
    dataset = task.build_dataset()
    mask = np.zeros(dataset.size, dtype=bool)
    mask[[0, dataset.size - 1]] = True
    feats = baselines.features(dataset, mask)
    assert feats.shape == (2, baselines.FEATURE_DIM)
    assert baselines.FEATURE_DIM == 16
    last = feats[1]
    state = dataset.states[-1] * np.pi / 2
    assert np.allclose(last[:3], np.sin(state))
    assert np.allclose(last[3:6], np.cos(state))
    assert last[6:9].tolist() == [1.0, 0.0, 0.0]  # first op R0
    assert last[9:13].tolist() == [0.0, 1.0, 0.0, 0.0]  # second op R1
    assert last[13:16].tolist() == [1.0, 0.0, 0.0]  # query a


def test_closest_width_prefers_the_smaller_width_on_ties() -> None:
    """Check that closest width prefers the smaller width on ties."""
    assert baselines.closest_width(lambda width: 10 * width, 25) == 2
    assert baselines.closest_width(baselines.mlp_param_count, 108) == 5
    assert baselines.closest_width(baselines.gnn_param_count, 108) == 4
    assert baselines.closest_width(baselines.transformer_param_count, 108) == 3


@pytest.mark.parametrize("name", ["mlp", "gnn", "transformer", "sequential"])
def test_train_and_predict_returns_labels_and_timing(name: str) -> None:
    """Check that train and predict returns labels and timing."""
    dataset = task.build_dataset()
    predicted_train, predicted_test, seconds = baselines.train_and_predict(
        name, 3, dataset, seed=1, epochs=2, learning_rate=0.01
    )
    assert predicted_train.shape == (2112,)
    assert predicted_test.shape == (64,)
    assert set(np.unique(predicted_test)) <= {0, 1, 2, 3}
    assert seconds > 0.0


def test_unknown_baseline_is_refused() -> None:
    """Check that unknown baseline is refused."""
    with pytest.raises(ValueError, match="unknown baseline"):
        baselines.train_and_predict(
            "lstm", 3, task.build_dataset(), seed=0, epochs=1, learning_rate=0.1
        )


def test_sequential_angles_apply_the_second_map_only_for_pairs() -> None:
    """Check that sequential angles apply the second map only for pairs."""
    import jax.numpy as jnp

    params = baselines.sequential_init(0, 2)
    states = jnp.asarray([[1, 2, 3], [1, 2, 3]])
    single = baselines.sequential_angles(
        params, states, jnp.asarray([0, 0]), jnp.asarray([task.NO_OPERATION, 1])
    )
    assert single.shape == (2, 3)
    assert not np.allclose(np.asarray(single[0]), np.asarray(single[1]))
