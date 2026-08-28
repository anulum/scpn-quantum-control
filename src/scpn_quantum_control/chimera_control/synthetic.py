# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Synthetic chimera generator
r"""Exact finite-N two-population Kuramoto-Sakaguchi trajectory generation.

The generator integrates

.. math::

   \dot\theta_j = \sum_{k\ne j} K_{jk}
   \sin(\theta_k-\theta_j-\alpha)

with the production :func:`oscillatools.accel.sakaguchi_force` and classical
RK4. The chimera configuration follows Abrams, Mirollo, Strogatz, and Wiley,
PRL 101, 084103 (2008), DOI ``10.1103/PhysRevLett.101.084103``. A finite-N run
is labelled a transient configuration, never a proof of an infinite-population
attractor.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from oscillatools.accel.kuramoto_chimera import (
    ChimeraDiagnostics,
    chimera_diagnostics,
)
from oscillatools.accel.sakaguchi_kuramoto import sakaguchi_force

from .schema import FloatArray, MultiscaleHierarchy, SyntheticRegime, two_population_hierarchy

SYNTHETIC_CHIMERA_SOURCE = (
    "Abrams-Mirollo-Strogatz-Wiley two-population Kuramoto-Sakaguchi "
    "finite-N synthetic configuration"
)


def _immutable_float(values: object, *, name: str, ndim: int) -> FloatArray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != ndim or array.size == 0:
        raise ValueError(f"{name} must be a non-empty {ndim}-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True, slots=True)
class SyntheticChimeraConfig:
    """Configuration for an exact two-population synthetic trajectory.

    Parameters
    ----------
    regime
        Explicit synthetic regime label.
    population_size
        Oscillators per population; the total node count is twice this value.
    dt
        Positive RK4 integration step.
    steps
        Number of RK4 steps; output includes the initial state.
    settle_steps
        Initial samples excluded from diagnostics.
    beta
        ``beta = pi/2 - alpha``; ``alpha`` is the Sakaguchi phase lag.
    intra_coupling, inter_coupling
        Population-normalised coupling strengths before division by
        ``population_size``.
    seed
        NumPy generator seed for the publication-style initial condition.

    """

    regime: SyntheticRegime = SyntheticRegime.CHIMERA_TRANSIENT
    population_size: int = 64
    dt: float = 0.05
    steps: int = 1200
    settle_steps: int = 200
    beta: float = 0.1
    intra_coupling: float = 0.75
    inter_coupling: float = 0.25
    seed: int = 20260702

    def __post_init__(self) -> None:
        """Validate the finite synthetic integration configuration."""
        if not isinstance(self.regime, SyntheticRegime):
            raise ValueError("regime must be a SyntheticRegime value")
        if (
            isinstance(self.population_size, bool)
            or not isinstance(self.population_size, int)
            or self.population_size < 2
        ):
            raise ValueError("population_size must be an integer greater than one")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        if (
            isinstance(self.settle_steps, bool)
            or not isinstance(self.settle_steps, int)
            or self.settle_steps < 0
            or self.settle_steps >= self.steps
        ):
            raise ValueError("settle_steps must satisfy 0 <= settle_steps < steps")
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be a finite positive value")
        if not np.isfinite(self.beta) or not 0.0 < self.beta < math.pi / 2.0:
            raise ValueError("beta must be finite and lie in (0, pi/2)")
        for name, value in (
            ("intra_coupling", self.intra_coupling),
            ("inter_coupling", self.inter_coupling),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative value")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")

    @classmethod
    def for_regime(
        cls,
        regime: SyntheticRegime,
        *,
        population_size: int = 64,
        seed: int = 20260702,
    ) -> SyntheticChimeraConfig:
        """Return the frozen reference configuration for ``regime``.

        The chimera row uses ``mu=0.75``, ``nu=0.25`` and 1200 steps. The
        synchronised contrast uses ``mu=0.6``, ``nu=0.4`` and 700 steps. Both
        use ``dt=0.05``, ``beta=0.1``, and 200 settle steps.
        """
        if regime is SyntheticRegime.CHIMERA_TRANSIENT:
            return cls(regime=regime, population_size=population_size, seed=seed)
        if regime is SyntheticRegime.SYNCHRONISED_CONTROL:
            return cls(
                regime=regime,
                population_size=population_size,
                steps=700,
                intra_coupling=0.6,
                inter_coupling=0.4,
                seed=seed,
            )
        raise ValueError(f"unsupported synthetic regime: {regime!r}")

    @property
    def frustration(self) -> float:
        """Return Sakaguchi phase lag ``alpha = pi/2 - beta``."""
        return float(math.pi / 2.0 - self.beta)


@dataclass(frozen=True, slots=True)
class SyntheticChimeraRun:
    """Immutable trajectory, coupling, hierarchy, and settled diagnostics.

    ``phases`` has shape ``(steps + 1, 2 * population_size)`` and ``times`` has
    shape ``(steps + 1,)``. ``diagnostics`` is evaluated over
    ``phases[settle_steps:]`` at the population scale. Arrays are copied and
    read-only; ``content_digest`` binds configuration and numerical custody.
    """

    config: SyntheticChimeraConfig
    hierarchy: MultiscaleHierarchy
    phases: FloatArray
    times: FloatArray
    coupling: FloatArray
    diagnostics: ChimeraDiagnostics
    source: str
    content_digest: str

    def __post_init__(self) -> None:
        """Validate and freeze trajectory, coupling, and diagnostic custody."""
        phases = _immutable_float(self.phases, name="phases", ndim=2)
        times = _immutable_float(self.times, name="times", ndim=1)
        coupling = _immutable_float(self.coupling, name="coupling", ndim=2)
        node_count = self.hierarchy.node_count
        if phases.shape != (self.config.steps + 1, node_count):
            raise ValueError("phases shape does not match config and hierarchy")
        if times.shape != (self.config.steps + 1,):
            raise ValueError("times shape does not match config steps")
        if coupling.shape != (node_count, node_count):
            raise ValueError("coupling shape does not match hierarchy")
        if not self.source.strip():
            raise ValueError("source must be non-empty")
        if not _is_sha256(self.content_digest):
            raise ValueError("content_digest must be a SHA-256 hexadecimal digest")
        diagnostic_values = _immutable_float(
            self.diagnostics.community_order_parameters,
            name="diagnostics.community_order_parameters",
            ndim=2,
        )
        if diagnostic_values.shape != (
            self.config.steps - self.config.settle_steps + 1,
            2,
        ):
            raise ValueError("diagnostics shape does not match settled population trajectory")
        diagnostics = ChimeraDiagnostics(
            community_order_parameters=diagnostic_values,
            chimera_index=float(self.diagnostics.chimera_index),
            metastability_index=float(self.diagnostics.metastability_index),
            community_metastability=float(self.diagnostics.community_metastability),
        )
        for name, value in (
            ("chimera_index", diagnostics.chimera_index),
            ("metastability_index", diagnostics.metastability_index),
            ("community_metastability", diagnostics.community_metastability),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"diagnostics.{name} must be finite and non-negative")
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "coupling", coupling)
        object.__setattr__(self, "diagnostics", diagnostics)

    @property
    def settled_phases(self) -> FloatArray:
        """Return a read-only view of the post-settle trajectory."""
        settled = self.phases[self.config.settle_steps :]
        settled.setflags(write=False)
        return settled


def build_two_population_coupling(config: SyntheticChimeraConfig) -> FloatArray:
    """Return the symmetric population-normalised block coupling matrix.

    Off-diagonal within-population entries are ``intra_coupling / N`` and
    between-population entries are ``inter_coupling / N``. The diagonal is
    exactly zero because self-coupling is excluded by the force definition.
    """
    size = config.population_size
    total = 2 * size
    coupling = np.empty((total, total), dtype=np.float64)
    coupling[:size, :size] = coupling[size:, size:] = config.intra_coupling / size
    coupling[:size, size:] = coupling[size:, :size] = config.inter_coupling / size
    np.fill_diagonal(coupling, 0.0)
    coupling.setflags(write=False)
    return coupling


def generate_two_population_chimera(
    config: SyntheticChimeraConfig | None = None,
) -> SyntheticChimeraRun:
    """Generate one deterministic finite-N two-population trajectory.

    Parameters
    ----------
    config
        Validated generator configuration. ``None`` selects the reference
        chimera-transient configuration.

    Returns
    -------
    SyntheticChimeraRun
        Immutable full trajectory, settled population diagnostics, hierarchy,
        block coupling, source label, and SHA-256 custody digest.

    Notes
    -----
    The first population starts near coherence and the second uniformly on the
    circle. Natural frequencies are identical and zero. A generated row is a
    finite synthetic regression fixture, not a physical or biological model.

    """
    resolved = config or SyntheticChimeraConfig()
    hierarchy = two_population_hierarchy(resolved.population_size)
    coupling = build_two_population_coupling(resolved)
    rng = np.random.default_rng(resolved.seed)
    current = np.empty(hierarchy.node_count, dtype=np.float64)
    current[: resolved.population_size] = 0.01 * rng.standard_normal(resolved.population_size)
    current[resolved.population_size :] = rng.uniform(
        -math.pi,
        math.pi,
        resolved.population_size,
    )
    trajectory = np.empty((resolved.steps + 1, hierarchy.node_count), dtype=np.float64)
    trajectory[0] = current
    dt = resolved.dt
    alpha = resolved.frustration
    mutable_coupling = np.asarray(coupling)
    for step in range(resolved.steps):
        k1 = sakaguchi_force(current, mutable_coupling, alpha)
        k2 = sakaguchi_force(current + 0.5 * dt * k1, mutable_coupling, alpha)
        k3 = sakaguchi_force(current + 0.5 * dt * k2, mutable_coupling, alpha)
        k4 = sakaguchi_force(current + dt * k3, mutable_coupling, alpha)
        current += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory[step + 1] = current
    times = np.arange(resolved.steps + 1, dtype=np.float64) * dt
    communities = hierarchy.level("population").communities
    settled = trajectory[resolved.settle_steps :]
    diagnostics = chimera_diagnostics(settled, communities)
    digest = hashlib.sha256()
    digest.update(repr(resolved).encode("utf-8"))
    digest.update(trajectory.tobytes(order="C"))
    digest.update(np.asarray(coupling).tobytes(order="C"))
    return SyntheticChimeraRun(
        config=resolved,
        hierarchy=hierarchy,
        phases=trajectory,
        times=times,
        coupling=coupling,
        diagnostics=diagnostics,
        source=SYNTHETIC_CHIMERA_SOURCE,
        content_digest=digest.hexdigest(),
    )


__all__ = [
    "SYNTHETIC_CHIMERA_SOURCE",
    "SyntheticChimeraConfig",
    "SyntheticChimeraRun",
    "build_two_population_coupling",
    "generate_two_population_chimera",
]
