# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable DLA-parity projection
"""Linear parity-sector projection with exact JVP, VJP, and leakage gradient."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_quantum_control.analysis.dla_parity_theorem import project_to_parity_sector

from .schema import DLA_TOPOLOGY_CLAIM_BOUNDARY, ParitySector

ComplexArray: TypeAlias = NDArray[np.complex128]
BoolArray: TypeAlias = NDArray[np.bool_]


def _read_only_complex(value: NDArray[np.complex128]) -> ComplexArray:
    out = np.array(value, dtype=np.complex128, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, slots=True)
class ParityLeakageEvaluation:
    """Parity leakage value and Euclidean complex gradient.

    ``gradient`` represents derivatives with respect to real and imaginary
    components under the real inner product. For absolute leakage mass it is
    ``2 * outside_sector(state)``.

    Parameters
    ----------
    value:
        Absolute outside-sector squared norm or its normalised fraction.
    gradient:
        Read-only complex gradient with the same shape as the evaluated state.
    normalised:
        Whether ``value`` is divided by total state norm squared.
    state_norm_squared:
        Positive total squared norm used by the evaluation.

    """

    value: float
    gradient: ComplexArray
    normalised: bool
    state_norm_squared: float

    def __post_init__(self) -> None:
        """Validate scalar values and take immutable gradient custody."""
        if not np.isfinite(self.value) or self.value < 0.0:
            raise ValueError("value must be finite and non-negative")
        if not np.isfinite(self.state_norm_squared) or self.state_norm_squared <= 0.0:
            raise ValueError("state_norm_squared must be finite and positive")
        gradient = np.asarray(self.gradient, dtype=np.complex128)
        if gradient.ndim != 1 or not np.all(np.isfinite(gradient)):
            raise ValueError("gradient must be a finite vector")
        object.__setattr__(self, "gradient", _read_only_complex(gradient))


@dataclass(frozen=True, slots=True)
class ParitySectorProjector:
    """Project finite state vectors into one computational-basis parity sector.

    Parameters
    ----------
    n_qubits:
        Positive number of qubits. The dense local API is capped at 20 qubits
        to prevent accidental exponential allocation.
    sector:
        Even or odd Hamming-weight sector.

    Notes
    -----
    The forward map delegates to the existing DLA-parity projector. It is a
    fixed self-adjoint linear map, so its JVP and VJP are the same projection.
    This does not prove that an arbitrary Hamiltonian or ansatz preserves the
    selected sector.

    """

    n_qubits: int
    sector: ParitySector
    claim_boundary: str = DLA_TOPOLOGY_CLAIM_BOUNDARY
    _mask: BoolArray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the sector contract and construct its immutable mask."""
        if isinstance(self.n_qubits, bool) or not isinstance(self.n_qubits, int):
            raise ValueError("n_qubits must be an integer")
        if self.n_qubits < 1 or self.n_qubits > 20:
            raise ValueError("n_qubits must lie in [1, 20] for dense local projection")
        if not isinstance(self.sector, ParitySector):
            raise ValueError("sector must be a ParitySector")
        if not isinstance(self.claim_boundary, str) or not self.claim_boundary.strip():
            raise ValueError("claim_boundary must be a non-empty string")
        indices = np.arange(1 << self.n_qubits, dtype=np.uint64)
        parity = np.fromiter(
            (int(index).bit_count() % 2 for index in indices),
            dtype=np.int8,
            count=indices.size,
        )
        mask = np.asarray(parity == self.sector.value, dtype=np.bool_)
        mask.setflags(write=False)
        object.__setattr__(self, "_mask", mask)
        object.__setattr__(self, "claim_boundary", self.claim_boundary.strip())

    @property
    def dimension(self) -> int:
        """Dense Hilbert-space dimension ``2**n_qubits``."""
        return 1 << self.n_qubits

    @property
    def mask(self) -> BoolArray:
        """Read-only basis mask for the selected parity sector."""
        return self._mask

    def as_state(self, state: NDArray[np.complex128], *, name: str = "state") -> ComplexArray:
        """Validate and copy a dense vector into immutable complex custody.

        Parameters
        ----------
        state:
            State-like vector with shape ``(2**n_qubits,)``.
        name:
            Field name included in validation errors.

        Returns
        -------
        numpy.ndarray
            Read-only ``complex128`` copy; caller memory is never aliased.

        Raises
        ------
        ValueError
            If rank, length, or finiteness violates the dense-state contract.

        """
        value = np.asarray(state, dtype=np.complex128)
        if value.ndim != 1 or value.shape != (self.dimension,):
            raise ValueError(f"{name} must have shape ({self.dimension},)")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain only finite values")
        return _read_only_complex(value)

    def project(self, state: NDArray[np.complex128]) -> ComplexArray:
        """Project a dense vector into this computational-basis parity sector.

        The forward calculation delegates to the existing
        ``analysis.dla_parity_theorem.project_to_parity_sector`` owner.

        Parameters
        ----------
        state:
            Finite complex vector with shape ``(2**n_qubits,)``.

        Returns
        -------
        numpy.ndarray
            Read-only projected complex vector of the same shape.

        """
        value = self.as_state(state)
        projected = project_to_parity_sector(value, self.sector.value, self.n_qubits)
        return _read_only_complex(projected)

    def jvp(self, tangent: NDArray[np.complex128]) -> ComplexArray:
        """Apply the exact projector Jacobian to a tangent vector.

        Because the parity projector is fixed and linear, the JVP is the same
        parity projection. The input and read-only output both have shape
        ``(2**n_qubits,)``.
        """
        return self.project(self.as_state(tangent, name="tangent"))

    def vjp(self, cotangent: NDArray[np.complex128]) -> ComplexArray:
        """Apply the exact adjoint projector Jacobian to a cotangent.

        The projector is self-adjoint, so this VJP uses the same projection as
        ``project`` and ``jvp``. The returned complex vector is read-only.
        """
        return self.project(self.as_state(cotangent, name="cotangent"))

    def leakage_value_and_gradient(
        self,
        state: NDArray[np.complex128],
        *,
        normalised: bool = False,
    ) -> ParityLeakageEvaluation:
        """Return outside-sector mass or fraction and its analytic gradient.

        Parameters
        ----------
        state:
            Finite dense complex vector of shape ``(2**n_qubits,)``.
        normalised:
            If false, return absolute outside-sector squared norm. If true,
            divide by total squared norm and differentiate that quotient.

        Returns
        -------
        ParityLeakageEvaluation
            Leakage value, exact Euclidean complex gradient, normalisation
            mode, and the positive total norm used by the calculation.

        Raises
        ------
        ValueError
            If the state is zero, malformed, or non-finite.

        """
        value = self.as_state(state)
        norm_squared = float(np.vdot(value, value).real)
        if norm_squared <= 0.0:
            raise ValueError("state must have positive norm")
        outside = np.array(value, copy=True)
        outside[self._mask] = 0.0
        outside_mass = float(np.vdot(outside, outside).real)
        if not normalised:
            gradient = 2.0 * outside
            leakage = outside_mass
        else:
            leakage = outside_mass / norm_squared
            gradient = 2.0 * (outside / norm_squared - (outside_mass / norm_squared**2) * value)
        return ParityLeakageEvaluation(
            value=float(leakage),
            gradient=np.asarray(gradient, dtype=np.complex128),
            normalised=normalised,
            state_norm_squared=norm_squared,
        )


__all__ = [
    "BoolArray",
    "ComplexArray",
    "ParityLeakageEvaluation",
    "ParitySectorProjector",
]
