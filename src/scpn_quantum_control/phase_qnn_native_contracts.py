# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native bounded phase model tensor contracts
"""Numerical tensor operations required by the bounded phase classifier model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class PhaseTensor(Protocol):
    """Native model values, including tensors produced from trainable variables."""

    @property
    def shape(self) -> Sequence[int]:
        """Return native dimensions without materialising tensor values."""
        ...

    def __add__(self, other: PhaseTensor, /) -> PhaseTensor:
        """Add native tensors with backend broadcasting."""
        ...

    def __sub__(self, other: PhaseTensor, /) -> PhaseTensor:
        """Subtract a native tensor."""
        ...

    def __rsub__(self, other: float, /) -> PhaseTensor:
        """Subtract a native tensor from a real scalar."""
        ...

    def __mul__(self, other: PhaseTensor | float, /) -> PhaseTensor:
        """Multiply native tensors or a real scalar."""
        ...

    def __rmul__(self, other: float, /) -> PhaseTensor:
        """Scale a native tensor by a real scalar."""
        ...

    def __truediv__(self, other: int | float, /) -> PhaseTensor:
        """Divide by a dimension or a real scalar without host conversion."""
        ...

    def __getitem__(self, key: tuple[None, slice] | tuple[slice, None], /) -> PhaseTensor:
        """Insert the sample or parameter broadcast axis."""
        ...


class TorchPhaseTensor(PhaseTensor, Protocol):
    """Native PyTorch tensor operations used by this model."""

    def unsqueeze(self, dim: int, /) -> PhaseTensor:
        """Insert the explicit parameter broadcast axis."""
        ...


class JAXPhaseNamespace(Protocol):
    """The model's JAX cosine, sine and mean operations."""

    def cos(self, values: PhaseTensor, /) -> PhaseTensor:
        """Evaluate the native elementwise cosine."""
        ...

    def sin(self, values: PhaseTensor, /) -> PhaseTensor:
        """Evaluate the native elementwise sine."""
        ...

    def mean(self, values: PhaseTensor, *, axis: int | None = None) -> PhaseTensor:
        """Reduce native values over one axis or all entries."""
        ...


class TorchPhaseNamespace(Protocol):
    """The model's native PyTorch cosine and reduction operations."""

    def cos(self, values: PhaseTensor, /) -> PhaseTensor:
        """Evaluate the native elementwise cosine."""
        ...

    def mean(self, values: PhaseTensor, *, dim: int | None = None) -> PhaseTensor:
        """Reduce native values over one dimension or all entries."""
        ...


class TensorMean(Protocol):
    """An already resolved native TensorFlow mean operation."""

    def __call__(self, values: PhaseTensor, *, axis: int | None = None) -> PhaseTensor:
        """Reduce a native tensor without changing its differentiation context."""
        ...
