# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — neural-operator caller RNG custody qualification
"""Run explicitly in the actual Torch-enabled learning quality profile."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import numpy as np
import pytest

from oscillatools.neural_operator import (
    KuramotoOperatorDataset,
    TrainedKuramotoOperator,
    simulate_operator_dataset,
    train_kuramoto_neural_operator,
)


def _dataset() -> KuramotoOperatorDataset:
    return simulate_operator_dataset(
        np.array([1.0, 1.2]),
        np.array([[0.0, 0.1], [0.1, 0.0]]),
        n_trajectories=2,
        dt=0.01,
        n_steps=2,
        seed=73,
    )


@pytest.mark.parametrize("caller_seed", [137, 911])
def test_training_preserves_caller_stream_and_seeded_replay(caller_seed: int) -> None:
    """Train real models without consuming the caller stream or losing replay.

    Parameters
    ----------
    caller_seed
        Independent caller stream whose next draws must remain untouched.

    """
    import torch

    saved = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(caller_seed)
    torch.set_rng_state(generator.get_state())
    before = torch.get_rng_state().clone()
    dataset = _dataset()
    try:
        first = train_kuramoto_neural_operator(
            dataset, latent_dim=2, hidden_dim=4, epochs=2, seed=7
        )
        assert torch.equal(torch.get_rng_state(), before)
        assert torch.equal(torch.rand(5), torch.rand(5, generator=generator))
        advanced = torch.get_rng_state().clone()
        replay = train_kuramoto_neural_operator(
            dataset, latent_dim=2, hidden_dim=4, epochs=2, seed=7
        )
        other = train_kuramoto_neural_operator(
            dataset, latent_dim=2, hidden_dim=4, epochs=2, seed=8
        )
        assert torch.equal(torch.get_rng_state(), advanced)
        np.testing.assert_array_equal(first.loss_history, replay.loss_history)
        for name, weights in first.model.state_dict().items():
            assert torch.equal(weights, replay.model.state_dict()[name])
        assert any(
            not torch.equal(weights, other.model.state_dict()[name])
            for name, weights in first.model.state_dict().items()
        )
        forecast = first.forecast(np.array([0.2, 0.4]), np.array([0.0, 0.01, 0.02]))
        assert forecast.shape == (3, 2) and np.all(np.isfinite(forecast))
        assert torch.equal(torch.get_rng_state(), advanced)
    finally:
        torch.set_rng_state(saved)


def test_real_training_failure_restores_the_caller_stream() -> None:
    """Preserve caller randomness when real loss evaluation rejects target shape."""
    import torch

    dataset = _dataset()
    invalid = replace(dataset, targets=dataset.targets[:, :3])
    before = torch.get_rng_state().clone()

    try:
        with (
            pytest.warns(UserWarning, match="target size"),
            pytest.raises(RuntimeError, match="size of tensor"),
        ):
            train_kuramoto_neural_operator(invalid, latent_dim=2, hidden_dim=4, epochs=2, seed=7)
        assert torch.equal(torch.get_rng_state(), before)
    finally:
        torch.set_rng_state(before)


def test_concurrent_training_owns_independent_rng_streams() -> None:
    """Keep real concurrent models and intervening caller draws independent."""
    import torch

    saved = torch.get_rng_state().clone()
    dataset = _dataset()
    seeds = (7, 8)
    reference = [
        train_kuramoto_neural_operator(dataset, latent_dim=8, hidden_dim=16, epochs=3, seed=seed)
        for seed in seeds
    ]
    generator = torch.Generator().manual_seed(137)
    torch.set_rng_state(generator.get_state())
    barrier = Barrier(3)

    def train(seed: int) -> TrainedKuramotoOperator:
        barrier.wait()
        return train_kuramoto_neural_operator(
            dataset, latent_dim=8, hidden_dim=16, epochs=3, seed=seed
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(train, seed) for seed in seeds]
            barrier.wait()
            actual = torch.stack([torch.rand(1) for _ in range(32)])
            concurrent = [future.result() for future in futures]
        expected = torch.stack([torch.rand(1, generator=generator) for _ in range(32)])
        assert torch.equal(actual, expected)
        assert torch.equal(torch.get_rng_state(), generator.get_state())
        for trained, control in zip(concurrent, reference, strict=True):
            np.testing.assert_array_equal(trained.loss_history, control.loss_history)
            for name, weights in control.model.state_dict().items():
                assert torch.equal(weights, trained.model.state_dict()[name])
    finally:
        torch.set_rng_state(saved)
