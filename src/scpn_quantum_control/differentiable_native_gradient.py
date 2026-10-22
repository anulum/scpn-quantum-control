# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native gradient construction and tape readout
"""Construct native gradients and read recorded derivatives without reporting policy."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def jax_gradient(jax: Any, objective: Callable[[Any], Any]) -> Any:
    """Construct the actual JAX gradient callable without wrapping its result.

    Parameters
    ----------
    jax
        Actual runtime within the caller's precision and device contexts.
    objective
        Original scalar numerical function, retaining its defining namespace,
        closure and any registered custom differentiation rule.

    Returns
    -------
    object
        The native gradient callable itself. Application, reuse and composition
        remain with the caller; no host conversion or case construction occurs.

    """
    return jax.grad(objective)


def torch_gradient(torch: Any, objective: Callable[[Any], Any]) -> Any:
    """Construct the actual torch.func gradient without a replacement callable.

    Parameters
    ----------
    torch
        Actual PyTorch runtime chosen by the caller.
    objective
        Original scalar numerical function with its existing registered rules.

    Returns
    -------
    object
        Native gradient function, retaining the runtime's callable metadata and
        parameter semantics. No application or host conversion is performed.

    """
    return torch.func.grad(objective)


def tensorflow_gradient(tape: Any, value: Any, parameters: Any) -> Any:
    """Read a gradient from the caller's actual TensorFlow recording tape.

    Parameters
    ----------
    tape
        Native tape whose creation, watch calls, recording context, persistence
        and lifetime remain with the original numerical caller.
    value
        Actual recorded scalar value, with no replacement expression or copy.
    parameters
        Original watched native inputs, including an existing nested container.

    Returns
    -------
    object
        Native derivative structure from tape.gradient, without a host copy,
        altered tape policy or report record.

    """
    return tape.gradient(value, parameters)


__all__ = ["jax_gradient", "torch_gradient", "tensorflow_gradient"]
