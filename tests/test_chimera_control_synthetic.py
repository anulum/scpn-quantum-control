# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Synthetic chimera tests
"""Real production-force tests for the finite synthetic chimera generator."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest

from scpn_quantum_control.chimera_control.schema import SyntheticRegime
from scpn_quantum_control.chimera_control.synthetic import (
    SYNTHETIC_CHIMERA_SOURCE,
    SyntheticChimeraConfig,
    SyntheticChimeraRun,
    build_two_population_coupling,
    generate_two_population_chimera,
)


def test_reference_regime_factories_freeze_published_couplings() -> None:
    """Freeze the finite reference parameters for both synthetic regimes."""
    chimera = SyntheticChimeraConfig.for_regime(
        SyntheticRegime.CHIMERA_TRANSIENT,
        population_size=8,
        seed=7,
    )
    synchronised = SyntheticChimeraConfig.for_regime(
        SyntheticRegime.SYNCHRONISED_CONTROL,
        population_size=8,
        seed=9,
    )

    assert (chimera.intra_coupling, chimera.inter_coupling, chimera.steps) == (
        0.75,
        0.25,
        1200,
    )
    assert (synchronised.intra_coupling, synchronised.inter_coupling) == (0.6, 0.4)
    assert synchronised.steps == 700
    assert chimera.frustration == pytest.approx(math.pi / 2.0 - 0.1)
    with pytest.raises(ValueError, match="unsupported synthetic regime"):
        SyntheticChimeraConfig.for_regime(cast(SyntheticRegime, "unknown"))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"population_size": 1}, "population_size"),
        ({"population_size": 2.5}, "population_size"),
        ({"population_size": True}, "population_size"),
        ({"steps": 0}, "steps"),
        ({"steps": 2.5}, "steps"),
        ({"steps": True}, "steps"),
        ({"settle_steps": -1}, "settle_steps"),
        ({"settle_steps": 1.5}, "settle_steps"),
        ({"settle_steps": 1200}, "settle_steps"),
        ({"settle_steps": True}, "settle_steps"),
        ({"dt": 0.0}, "dt"),
        ({"dt": float("nan")}, "dt"),
        ({"beta": 0.0}, "beta"),
        ({"beta": math.pi / 2.0}, "beta"),
        ({"intra_coupling": -0.1}, "intra_coupling"),
        ({"inter_coupling": float("inf")}, "inter_coupling"),
        ({"seed": True}, "seed"),
        ({"seed": 1.5}, "seed"),
        ({"regime": "chimera_transient"}, "regime"),
    ],
)
def test_synthetic_config_rejects_invalid_values(kwargs: dict[str, object], message: str) -> None:
    """Reject invalid configuration values at the public construction boundary."""
    with pytest.raises(ValueError, match=message):
        SyntheticChimeraConfig(**kwargs)


def test_block_coupling_is_symmetric_normalised_and_read_only() -> None:
    """Build an immutable symmetric two-population coupling matrix."""
    config = SyntheticChimeraConfig(population_size=3, steps=2, settle_steps=0)
    coupling = build_two_population_coupling(config)

    assert coupling.shape == (6, 6)
    np.testing.assert_allclose(coupling, coupling.T)
    np.testing.assert_allclose(np.diag(coupling), 0.0)
    assert coupling[0, 1] == pytest.approx(0.75 / 3)
    assert coupling[0, 3] == pytest.approx(0.25 / 3)
    assert not coupling.flags.writeable


def test_generator_is_deterministic_immutable_and_digest_bound() -> None:
    """Bind deterministic trajectory custody to immutable arrays and a digest."""
    config = SyntheticChimeraConfig(
        population_size=4,
        steps=12,
        settle_steps=2,
        seed=41,
    )
    first = generate_two_population_chimera(config)
    second = generate_two_population_chimera(config)

    assert first.content_digest == second.content_digest
    np.testing.assert_array_equal(first.phases, second.phases)
    np.testing.assert_array_equal(first.coupling, second.coupling)
    assert first.phases.shape == (13, 8)
    assert first.times.shape == (13,)
    assert first.settled_phases.shape == (11, 8)
    assert not first.phases.flags.writeable
    assert not first.times.flags.writeable
    assert not first.coupling.flags.writeable
    assert not first.diagnostics.community_order_parameters.flags.writeable
    assert not first.settled_phases.flags.writeable
    assert first.source == SYNTHETIC_CHIMERA_SOURCE
    assert len(first.content_digest) == 64


def test_default_generator_uses_reference_chimera_configuration() -> None:
    """Use the bounded chimera transient as the explicit default regime."""
    run = generate_two_population_chimera(
        SyntheticChimeraConfig(population_size=2, steps=1, settle_steps=0)
    )
    assert run.config.regime is SyntheticRegime.CHIMERA_TRANSIENT


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"phases": np.zeros((1, 1))}, "phases shape"),
        ({"times": np.zeros(1)}, "times shape"),
        ({"coupling": np.zeros((1, 1))}, "coupling shape"),
        ({"source": " "}, "source"),
        ({"content_digest": "bad"}, "content_digest"),
        ({"content_digest": "z" * 64}, "content_digest"),
    ],
)
def test_run_contract_rejects_inconsistent_custody(
    changes: dict[str, object], message: str
) -> None:
    """Reject custody objects whose arrays or digest contradict the configuration."""
    valid = generate_two_population_chimera(
        SyntheticChimeraConfig(population_size=2, steps=2, settle_steps=0)
    )
    values: dict[str, object] = {
        "config": valid.config,
        "hierarchy": valid.hierarchy,
        "phases": valid.phases,
        "times": valid.times,
        "coupling": valid.coupling,
        "diagnostics": valid.diagnostics,
        "source": valid.source,
        "content_digest": valid.content_digest,
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        SyntheticChimeraRun(**values)


def test_run_contract_rejects_non_finite_or_wrong_rank_arrays() -> None:
    """Reject malformed trajectory arrays and inconsistent diagnostic custody."""
    valid = generate_two_population_chimera(
        SyntheticChimeraConfig(population_size=2, steps=2, settle_steps=0)
    )
    with pytest.raises(ValueError, match="2-dimensional"):
        SyntheticChimeraRun(
            valid.config,
            valid.hierarchy,
            np.array([0.0]),
            valid.times,
            valid.coupling,
            valid.diagnostics,
            valid.source,
            valid.content_digest,
        )
    diagnostics = valid.diagnostics.__class__(
        community_order_parameters=np.ones((1, 2)),
        chimera_index=0.0,
        metastability_index=0.0,
        community_metastability=0.0,
    )
    with pytest.raises(ValueError, match="diagnostics shape"):
        SyntheticChimeraRun(
            valid.config,
            valid.hierarchy,
            valid.phases,
            valid.times,
            valid.coupling,
            diagnostics,
            valid.source,
            valid.content_digest,
        )
    bad_diagnostics = valid.diagnostics.__class__(
        community_order_parameters=valid.diagnostics.community_order_parameters,
        chimera_index=-1.0,
        metastability_index=0.0,
        community_metastability=0.0,
    )
    with pytest.raises(ValueError, match="diagnostics.chimera_index"):
        SyntheticChimeraRun(
            valid.config,
            valid.hierarchy,
            valid.phases,
            valid.times,
            valid.coupling,
            bad_diagnostics,
            valid.source,
            valid.content_digest,
        )
    bad = np.array(valid.phases, copy=True)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        SyntheticChimeraRun(
            valid.config,
            valid.hierarchy,
            bad,
            valid.times,
            valid.coupling,
            valid.diagnostics,
            valid.source,
            valid.content_digest,
        )
