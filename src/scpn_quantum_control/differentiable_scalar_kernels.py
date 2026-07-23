# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable scalar kernels module
# scpn-quantum-control -- scalar differentiable programming kernels
"""Scalar forward- and reverse-mode automatic differentiation kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .differentiable_parameter_contracts import _as_real_scalar


@dataclass(frozen=True)
class DualNumber:
    """Forward-mode automatic differentiation scalar with one tangent lane.

    Parameters
    ----------
    primal
        Real primal scalar carried by the differentiable expression.
    tangent
        Real derivative with respect to the active scalar seed.

    Attributes
    ----------
    primal
        Validated real primal scalar.
    tangent
        Validated real tangent scalar for the active derivative lane.

    """

    primal: float
    tangent: float = 0.0

    def __post_init__(self) -> None:
        """Validate the stored primal and tangent as real scalar values."""
        object.__setattr__(self, "primal", _as_real_scalar("dual primal", self.primal))
        object.__setattr__(self, "tangent", _as_real_scalar("dual tangent", self.tangent))

    @staticmethod
    def coerce(value: object) -> DualNumber:
        """Convert an operand into a forward-mode scalar.

        Parameters
        ----------
        value
            Existing :class:`DualNumber` or real scalar operand.

        Returns
        -------
        DualNumber
            The original dual value, or a zero-tangent constant dual value.

        Raises
        ------
        ValueError
            If ``value`` is not a finite real scalar or dual value.

        """
        if isinstance(value, DualNumber):
            return value
        return DualNumber(_as_real_scalar("dual operand", value), 0.0)

    def __add__(self, other: object) -> DualNumber:
        """Return the forward-mode addition rule."""
        rhs = DualNumber.coerce(other)
        return DualNumber(self.primal + rhs.primal, self.tangent + rhs.tangent)

    def __radd__(self, other: object) -> DualNumber:
        """Return reflected forward-mode addition."""
        return self.__add__(other)

    def __sub__(self, other: object) -> DualNumber:
        """Return the forward-mode subtraction rule."""
        rhs = DualNumber.coerce(other)
        return DualNumber(self.primal - rhs.primal, self.tangent - rhs.tangent)

    def __rsub__(self, other: object) -> DualNumber:
        """Return reflected forward-mode subtraction."""
        lhs = DualNumber.coerce(other)
        return lhs.__sub__(self)

    def __mul__(self, other: object) -> DualNumber:
        """Return the forward-mode product rule."""
        rhs = DualNumber.coerce(other)
        return DualNumber(
            self.primal * rhs.primal,
            self.tangent * rhs.primal + self.primal * rhs.tangent,
        )

    def __rmul__(self, other: object) -> DualNumber:
        """Return reflected forward-mode multiplication."""
        return self.__mul__(other)

    def __truediv__(self, other: object) -> DualNumber:
        """Return the forward-mode quotient rule."""
        rhs = DualNumber.coerce(other)
        if rhs.primal == 0.0:
            raise ValueError("dual division denominator must be non-zero")
        return DualNumber(
            self.primal / rhs.primal,
            (self.tangent * rhs.primal - self.primal * rhs.tangent) / rhs.primal**2,
        )

    def __rtruediv__(self, other: object) -> DualNumber:
        """Return reflected forward-mode division."""
        lhs = DualNumber.coerce(other)
        return lhs.__truediv__(self)

    def __neg__(self) -> DualNumber:
        """Return the negated primal and tangent."""
        return DualNumber(-self.primal, -self.tangent)

    def __pow__(self, other: object) -> DualNumber:
        """Return the forward-mode scalar power rule."""
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
        """Return reflected forward-mode scalar exponentiation."""
        lhs = DualNumber.coerce(other)
        return lhs.__pow__(self)


def dual_sin(value: object) -> DualNumber:
    """Evaluate the forward-mode sine primitive.

    Parameters
    ----------
    value
        Existing :class:`DualNumber` or real scalar input.

    Returns
    -------
    DualNumber
        Sine value with tangent multiplied by ``cos(value)``.

    Raises
    ------
    ValueError
        If ``value`` cannot be represented as a real scalar dual value.

    """
    arg = DualNumber.coerce(value)
    return DualNumber(float(np.sin(arg.primal)), float(np.cos(arg.primal)) * arg.tangent)


def dual_cos(value: object) -> DualNumber:
    """Evaluate the forward-mode cosine primitive.

    Parameters
    ----------
    value
        Existing :class:`DualNumber` or real scalar input.

    Returns
    -------
    DualNumber
        Cosine value with tangent multiplied by ``-sin(value)``.

    Raises
    ------
    ValueError
        If ``value`` cannot be represented as a real scalar dual value.

    """
    arg = DualNumber.coerce(value)
    return DualNumber(float(np.cos(arg.primal)), -float(np.sin(arg.primal)) * arg.tangent)


def dual_exp(value: object) -> DualNumber:
    """Evaluate the forward-mode exponential primitive.

    Parameters
    ----------
    value
        Existing :class:`DualNumber` or real scalar input.

    Returns
    -------
    DualNumber
        Exponential value with tangent multiplied by ``exp(value)``.

    Raises
    ------
    ValueError
        If ``value`` cannot be represented as a real scalar dual value.

    """
    arg = DualNumber.coerce(value)
    primal = float(np.exp(arg.primal))
    return DualNumber(primal, primal * arg.tangent)


def dual_log(value: object) -> DualNumber:
    """Evaluate the forward-mode natural-log primitive.

    Parameters
    ----------
    value
        Existing :class:`DualNumber` or positive real scalar input.

    Returns
    -------
    DualNumber
        Natural-log value with tangent divided by the positive primal.

    Raises
    ------
    ValueError
        If ``value`` is not real or its primal is not strictly positive.

    """
    arg = DualNumber.coerce(value)
    if arg.primal <= 0.0:
        raise ValueError("dual log input must be positive")
    return DualNumber(float(np.log(arg.primal)), arg.tangent / arg.primal)


class ReverseNode:
    """Reverse-mode automatic differentiation scalar with local pullbacks.

    Parameters
    ----------
    primal
        Real primal scalar represented by the node.
    parents
        Parent nodes paired with local derivative coefficients.

    Attributes
    ----------
    primal
        Validated real primal scalar.
    parents
        Tuple of upstream nodes and pullback coefficients.
    adjoint
        Reverse accumulation slot seeded by a downstream traversal.

    """

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
        """Convert an operand into a reverse-mode node.

        Parameters
        ----------
        value
            Existing :class:`ReverseNode` or real scalar operand.

        Returns
        -------
        ReverseNode
            The original node, or a constant node with no parents.

        Raises
        ------
        ValueError
            If ``value`` is not a finite real scalar or reverse node.

        """
        if isinstance(value, ReverseNode):
            return value
        return ReverseNode(_as_real_scalar("reverse operand", value))

    def __add__(self, other: object) -> ReverseNode:
        """Return the reverse-mode addition pullback."""
        rhs = ReverseNode.coerce(other)
        return ReverseNode(self.primal + rhs.primal, ((self, 1.0), (rhs, 1.0)))

    def __radd__(self, other: object) -> ReverseNode:
        """Return reflected reverse-mode addition."""
        return self.__add__(other)

    def __sub__(self, other: object) -> ReverseNode:
        """Return the reverse-mode subtraction pullback."""
        rhs = ReverseNode.coerce(other)
        return ReverseNode(self.primal - rhs.primal, ((self, 1.0), (rhs, -1.0)))

    def __rsub__(self, other: object) -> ReverseNode:
        """Return reflected reverse-mode subtraction."""
        lhs = ReverseNode.coerce(other)
        return lhs.__sub__(self)

    def __mul__(self, other: object) -> ReverseNode:
        """Return the reverse-mode product pullback."""
        rhs = ReverseNode.coerce(other)
        return ReverseNode(
            self.primal * rhs.primal,
            ((self, rhs.primal), (rhs, self.primal)),
        )

    def __rmul__(self, other: object) -> ReverseNode:
        """Return reflected reverse-mode multiplication."""
        return self.__mul__(other)

    def __truediv__(self, other: object) -> ReverseNode:
        """Return the reverse-mode quotient pullback."""
        rhs = ReverseNode.coerce(other)
        if rhs.primal == 0.0:
            raise ValueError("reverse division denominator must be non-zero")
        return ReverseNode(
            self.primal / rhs.primal,
            ((self, 1.0 / rhs.primal), (rhs, -self.primal / rhs.primal**2)),
        )

    def __rtruediv__(self, other: object) -> ReverseNode:
        """Return reflected reverse-mode division."""
        lhs = ReverseNode.coerce(other)
        return lhs.__truediv__(self)

    def __neg__(self) -> ReverseNode:
        """Return the reverse-mode negation pullback."""
        return ReverseNode(-self.primal, ((self, -1.0),))

    def __pow__(self, other: object) -> ReverseNode:
        """Return the reverse-mode scalar power pullback."""
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
        """Return reflected reverse-mode scalar exponentiation."""
        lhs = ReverseNode.coerce(other)
        return lhs.__pow__(self)


def reverse_sin(value: object) -> ReverseNode:
    """Evaluate the reverse-mode sine primitive.

    Parameters
    ----------
    value
        Existing :class:`ReverseNode` or real scalar input.

    Returns
    -------
    ReverseNode
        Sine node with local pullback coefficient ``cos(value)``.

    Raises
    ------
    ValueError
        If ``value`` cannot be represented as a real scalar reverse node.

    """
    arg = ReverseNode.coerce(value)
    return ReverseNode(float(np.sin(arg.primal)), ((arg, float(np.cos(arg.primal))),))


def reverse_cos(value: object) -> ReverseNode:
    """Evaluate the reverse-mode cosine primitive.

    Parameters
    ----------
    value
        Existing :class:`ReverseNode` or real scalar input.

    Returns
    -------
    ReverseNode
        Cosine node with local pullback coefficient ``-sin(value)``.

    Raises
    ------
    ValueError
        If ``value`` cannot be represented as a real scalar reverse node.

    """
    arg = ReverseNode.coerce(value)
    return ReverseNode(float(np.cos(arg.primal)), ((arg, -float(np.sin(arg.primal))),))


def reverse_exp(value: object) -> ReverseNode:
    """Evaluate the reverse-mode exponential primitive.

    Parameters
    ----------
    value
        Existing :class:`ReverseNode` or real scalar input.

    Returns
    -------
    ReverseNode
        Exponential node with local pullback coefficient ``exp(value)``.

    Raises
    ------
    ValueError
        If ``value`` cannot be represented as a real scalar reverse node.

    """
    arg = ReverseNode.coerce(value)
    primal = float(np.exp(arg.primal))
    return ReverseNode(primal, ((arg, primal),))


def reverse_log(value: object) -> ReverseNode:
    """Evaluate the reverse-mode natural-log primitive.

    Parameters
    ----------
    value
        Existing :class:`ReverseNode` or positive real scalar input.

    Returns
    -------
    ReverseNode
        Natural-log node with local pullback coefficient ``1 / value``.

    Raises
    ------
    ValueError
        If ``value`` is not real or its primal is not strictly positive.

    """
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
