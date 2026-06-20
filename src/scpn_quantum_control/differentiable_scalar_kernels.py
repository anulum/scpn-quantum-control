# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- scalar differentiable programming kernels
"""Scalar forward- and reverse-mode automatic differentiation kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .differentiable_parameter_contracts import _as_real_scalar


@dataclass(frozen=True)
class DualNumber:
    """Forward-mode automatic differentiation scalar with one tangent lane."""

    primal: float
    tangent: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "primal", _as_real_scalar("dual primal", self.primal))
        object.__setattr__(self, "tangent", _as_real_scalar("dual tangent", self.tangent))

    @staticmethod
    def coerce(value: object) -> DualNumber:
        """Return a dual number, treating real scalars as zero-tangent constants."""

        if isinstance(value, DualNumber):
            return value
        return DualNumber(_as_real_scalar("dual operand", value), 0.0)

    def __add__(self, other: object) -> DualNumber:
        rhs = DualNumber.coerce(other)
        return DualNumber(self.primal + rhs.primal, self.tangent + rhs.tangent)

    def __radd__(self, other: object) -> DualNumber:
        return self.__add__(other)

    def __sub__(self, other: object) -> DualNumber:
        rhs = DualNumber.coerce(other)
        return DualNumber(self.primal - rhs.primal, self.tangent - rhs.tangent)

    def __rsub__(self, other: object) -> DualNumber:
        lhs = DualNumber.coerce(other)
        return lhs.__sub__(self)

    def __mul__(self, other: object) -> DualNumber:
        rhs = DualNumber.coerce(other)
        return DualNumber(
            self.primal * rhs.primal,
            self.tangent * rhs.primal + self.primal * rhs.tangent,
        )

    def __rmul__(self, other: object) -> DualNumber:
        return self.__mul__(other)

    def __truediv__(self, other: object) -> DualNumber:
        rhs = DualNumber.coerce(other)
        if rhs.primal == 0.0:
            raise ValueError("dual division denominator must be non-zero")
        return DualNumber(
            self.primal / rhs.primal,
            (self.tangent * rhs.primal - self.primal * rhs.tangent) / rhs.primal**2,
        )

    def __rtruediv__(self, other: object) -> DualNumber:
        lhs = DualNumber.coerce(other)
        return lhs.__truediv__(self)

    def __neg__(self) -> DualNumber:
        return DualNumber(-self.primal, -self.tangent)

    def __pow__(self, other: object) -> DualNumber:
        rhs = DualNumber.coerce(other)
        if self.primal <= 0.0 and rhs.tangent != 0.0:
            raise ValueError("dual variable exponent requires positive base")
        primal = self.primal**rhs.primal
        if rhs.tangent == 0.0:
            tangent = rhs.primal * self.primal ** (rhs.primal - 1.0) * self.tangent
        else:
            tangent = primal * (
                rhs.tangent * float(np.log(self.primal)) + rhs.primal * self.tangent / self.primal
            )
        return DualNumber(primal, tangent)

    def __rpow__(self, other: object) -> DualNumber:
        lhs = DualNumber.coerce(other)
        return lhs.__pow__(self)


def dual_sin(value: object) -> DualNumber:
    """Evaluate the forward-mode sine primitive."""

    arg = DualNumber.coerce(value)
    return DualNumber(float(np.sin(arg.primal)), float(np.cos(arg.primal)) * arg.tangent)


def dual_cos(value: object) -> DualNumber:
    """Evaluate the forward-mode cosine primitive."""

    arg = DualNumber.coerce(value)
    return DualNumber(float(np.cos(arg.primal)), -float(np.sin(arg.primal)) * arg.tangent)


def dual_exp(value: object) -> DualNumber:
    """Evaluate the forward-mode exponential primitive."""

    arg = DualNumber.coerce(value)
    primal = float(np.exp(arg.primal))
    return DualNumber(primal, primal * arg.tangent)


def dual_log(value: object) -> DualNumber:
    """Evaluate the forward-mode natural-log primitive."""

    arg = DualNumber.coerce(value)
    if arg.primal <= 0.0:
        raise ValueError("dual log input must be positive")
    return DualNumber(float(np.log(arg.primal)), arg.tangent / arg.primal)


class ReverseNode:
    """Reverse-mode automatic differentiation scalar with local pullbacks."""

    __slots__ = ("adjoint", "parents", "primal")

    def __init__(
        self,
        primal: float,
        parents: tuple[tuple[ReverseNode, float], ...] = (),
    ) -> None:
        self.primal = _as_real_scalar("reverse primal", primal)
        self.parents = parents
        self.adjoint = 0.0

    @staticmethod
    def coerce(value: object) -> ReverseNode:
        """Return a reverse node, treating real scalars as constants."""

        if isinstance(value, ReverseNode):
            return value
        return ReverseNode(_as_real_scalar("reverse operand", value))

    def __add__(self, other: object) -> ReverseNode:
        rhs = ReverseNode.coerce(other)
        return ReverseNode(self.primal + rhs.primal, ((self, 1.0), (rhs, 1.0)))

    def __radd__(self, other: object) -> ReverseNode:
        return self.__add__(other)

    def __sub__(self, other: object) -> ReverseNode:
        rhs = ReverseNode.coerce(other)
        return ReverseNode(self.primal - rhs.primal, ((self, 1.0), (rhs, -1.0)))

    def __rsub__(self, other: object) -> ReverseNode:
        lhs = ReverseNode.coerce(other)
        return lhs.__sub__(self)

    def __mul__(self, other: object) -> ReverseNode:
        rhs = ReverseNode.coerce(other)
        return ReverseNode(
            self.primal * rhs.primal,
            ((self, rhs.primal), (rhs, self.primal)),
        )

    def __rmul__(self, other: object) -> ReverseNode:
        return self.__mul__(other)

    def __truediv__(self, other: object) -> ReverseNode:
        rhs = ReverseNode.coerce(other)
        if rhs.primal == 0.0:
            raise ValueError("reverse division denominator must be non-zero")
        return ReverseNode(
            self.primal / rhs.primal,
            ((self, 1.0 / rhs.primal), (rhs, -self.primal / rhs.primal**2)),
        )

    def __rtruediv__(self, other: object) -> ReverseNode:
        lhs = ReverseNode.coerce(other)
        return lhs.__truediv__(self)

    def __neg__(self) -> ReverseNode:
        return ReverseNode(-self.primal, ((self, -1.0),))

    def __pow__(self, other: object) -> ReverseNode:
        rhs = ReverseNode.coerce(other)
        if self.primal <= 0.0 and isinstance(other, ReverseNode):
            raise ValueError("reverse variable exponent requires positive base")
        primal = self.primal**rhs.primal
        parents: list[tuple[ReverseNode, float]] = []
        parents.append((self, rhs.primal * self.primal ** (rhs.primal - 1.0)))
        if isinstance(other, ReverseNode):
            parents.append((rhs, primal * float(np.log(self.primal))))
        return ReverseNode(primal, tuple(parents))

    def __rpow__(self, other: object) -> ReverseNode:
        lhs = ReverseNode.coerce(other)
        return lhs.__pow__(self)


def reverse_sin(value: object) -> ReverseNode:
    """Evaluate the reverse-mode sine primitive."""

    arg = ReverseNode.coerce(value)
    return ReverseNode(float(np.sin(arg.primal)), ((arg, float(np.cos(arg.primal))),))


def reverse_cos(value: object) -> ReverseNode:
    """Evaluate the reverse-mode cosine primitive."""

    arg = ReverseNode.coerce(value)
    return ReverseNode(float(np.cos(arg.primal)), ((arg, -float(np.sin(arg.primal))),))


def reverse_exp(value: object) -> ReverseNode:
    """Evaluate the reverse-mode exponential primitive."""

    arg = ReverseNode.coerce(value)
    primal = float(np.exp(arg.primal))
    return ReverseNode(primal, ((arg, primal),))


def reverse_log(value: object) -> ReverseNode:
    """Evaluate the reverse-mode natural-log primitive."""

    arg = ReverseNode.coerce(value)
    if arg.primal <= 0.0:
        raise ValueError("reverse log input must be positive")
    return ReverseNode(float(np.log(arg.primal)), ((arg, 1.0 / arg.primal),))


__all__ = [
    "DualNumber",
    "ReverseNode",
    "dual_cos",
    "dual_exp",
    "dual_log",
    "dual_sin",
    "reverse_cos",
    "reverse_exp",
    "reverse_log",
    "reverse_sin",
]
