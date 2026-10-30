# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — AOTAutograd forward/backward graph replay
"""Replay captured AOTAutograd forward and backward graphs on supplied inputs.

The forward graph returns the scalar output followed by the tensors saved for
the backward pass; the backward graph consumes those tensors and the output
cotangent. The caller chooses the inputs and supplies tensor construction and
host conversion as resolvers; this module selects no model and loads no
framework.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
Resolver: TypeAlias = Callable[[], Callable[..., Any]]
"""Zero-argument callable returning the operation to call at that moment."""


def replay_aot_graphs(
    *,
    torch_module: Any,
    forward_graph: object,
    backward_graph: object,
    inputs: Sequence[object],
    tensor: Resolver,
    output_tuple: Resolver,
    scalar_to_float: Resolver,
    values_to_numpy: Resolver,
) -> tuple[float, FloatArray]:
    """Run the forward graph, then the backward graph with a unit cotangent.

    Parameters
    ----------
    torch_module
        Supplied PyTorch module providing ``ones_like``.
    forward_graph, backward_graph
        Loaded callable AOTAutograd graphs.
    inputs
        Host values converted, in order, to the forward graph's tensors.
    tensor
        Resolver of ``(torch, values) -> tensor``.
    output_tuple
        Resolver of ``(values, *, name) -> tuple`` validating graph outputs.
    scalar_to_float, values_to_numpy
        Resolvers of the scalar and host vector conversions.

    Returns
    -------
    tuple[float, FloatArray]
        Scalar forward output and the gradient of the first input.

    Raises
    ------
    RuntimeError
        If a graph returns no outputs or ``torch.ones_like`` is unavailable.

    """
    forward_call = cast(Callable[..., object], forward_graph)
    backward_call = cast(Callable[..., object], backward_graph)
    input_tensors = tuple(tensor()(torch_module, values) for values in inputs)
    forward_output = output_tuple()(
        forward_call(*input_tensors),
        name="loaded AOTAutograd forward output",
    )
    if not forward_output:
        raise RuntimeError("loaded AOTAutograd forward graph returned no outputs")
    loaded_loss_tensor = forward_output[0]
    saved_tensors = forward_output[1:]
    ones_like = getattr(torch_module, "ones_like", None)
    if not callable(ones_like):
        raise RuntimeError("PyTorch module does not expose torch.ones_like")
    backward_output = output_tuple()(
        backward_call(*saved_tensors, ones_like(loaded_loss_tensor)),
        name="loaded AOTAutograd backward output",
    )
    if not backward_output:
        raise RuntimeError("loaded AOTAutograd backward graph returned no outputs")
    return (
        scalar_to_float()(loaded_loss_tensor),
        values_to_numpy()(backward_output[0]),
    )


__all__ = ["replay_aot_graphs"]
