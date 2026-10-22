# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — PyTorch native transform composition qualification
"""Qualify CPU torch.func compositions, including registered custom derivatives."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .differentiable_structured_transform_execution import torch_structured_derivatives
from .differentiable_transform_algebra import (
    TransformAlgebraCase,
    _framework_quantum_bridge_boundary_case,
    _framework_transform_case,
)


def run_torch_transform_algebra_cases(
    matrix: NDArray[np.float64], *, batch_axis: int = 0
) -> tuple[TransformAlgebraCase, ...]:
    """Execute native CPU PyTorch derivatives and custom higher-order rules.

    Parameters
    ----------
    matrix:
        Validated finite dimensionless float64 samples, ordered sample first.
    batch_axis:
        Native ``torch.func.vmap`` input axis, zero or one.

    Returns
    -------
    tuple[TransformAlgebraCase, ...]
        Actual numeric comparisons and the distinct generic quantum-bridge
        refusal. Missing PyTorch or native transforms fail this qualification.

    """
    import torch

    return _execute_cases(torch, matrix, batch_axis)


def _execute_cases(
    torch: Any, matrix: NDArray[np.float64], batch_axis: int
) -> tuple[TransformAlgebraCase, ...]:
    """Evaluate torch.func on explicitly placed CPU float64 tensors."""
    x = torch.tensor(matrix[0], dtype=torch.float64, device="cpu")
    batch = torch.tensor(
        matrix if batch_axis == 0 else matrix.T, dtype=torch.float64, device="cpu"
    )
    tangent = torch.linspace(0.2, 0.7, matrix.shape[1], dtype=torch.float64, device="cpu")

    def array(value: Any) -> NDArray[np.float64]:
        return np.asarray(value.detach().cpu().numpy(), dtype=np.float64)

    def loss(params: Any) -> Any:
        return torch.sum(params**4)

    derivative = torch.func.grad(loss)
    curvature = np.diag(12.0 * matrix[0] ** 2)
    expected_gradient = 4.0 * matrix[0] ** 3
    mapped = torch.func.vmap(derivative, in_dims=batch_axis, out_dims=0)(batch)

    def aggregate(params: Any) -> Any:
        return torch.sum(torch.func.vmap(loss, in_dims=batch_axis)(params))

    aggregate_gradient = torch.func.grad(aggregate)(batch)
    if batch_axis == 1:
        aggregate_gradient = aggregate_gradient.T
    _, pushed = torch.func.jvp(derivative, (x,), (tangent,))
    _, pullback = torch.func.vjp(derivative, x)
    (pulled,) = pullback(tangent)

    structured = {
        "weights": x,
        "bias": torch.tensor(0.3, dtype=torch.float64, device="cpu"),
    }

    def structured_loss(params: Any) -> Any:
        return torch.sum((params["weights"] + params["bias"]) ** 4)

    structured_gradient, structured_hessian = torch_structured_derivatives(
        torch, structured_loss, structured
    )
    shifted = matrix[0] + 0.3
    diagonal = 12.0 * shifted**2
    structured_actual = np.concatenate(
        [
            array(structured_gradient["weights"]).ravel(),
            array(structured_gradient["bias"]).ravel(),
            array(structured_hessian["weights"]["weights"]).ravel(),
            array(structured_hessian["weights"]["bias"]).ravel(),
            array(structured_hessian["bias"]["weights"]).ravel(),
            array(structured_hessian["bias"]["bias"]).ravel(),
        ]
    )
    structured_expected = np.concatenate(
        [
            4.0 * shifted**3,
            [np.sum(4.0 * shifted**3)],
            np.diag(diagonal).ravel(),
            diagonal,
            diagonal,
            [np.sum(diagonal)],
        ]
    )

    def forward(params: Any) -> Any:
        return loss(params)

    def setup_context(ctx: Any, inputs: Any, output: Any) -> None:
        (params,) = inputs
        ctx.save_for_backward(params)
        ctx.save_for_forward(params)

    def backward(ctx: Any, cotangent: Any) -> Any:
        (params,) = ctx.saved_tensors
        return cotangent * 4.0 * params**3

    def jvp(ctx: Any, direction: Any) -> Any:
        (params,) = ctx.saved_tensors
        return torch.sum(4.0 * params**3 * direction)

    registered: Any = type(
        "QuarticTransformRule",
        (torch.autograd.Function,),
        {
            "forward": staticmethod(forward),
            "setup_context": staticmethod(setup_context),
            "backward": staticmethod(backward),
            "jvp": staticmethod(jvp),
            "generate_vmap_rule": True,
        },
    )
    registered_apply = registered.apply

    def custom_composed(params: Any) -> Any:
        return torch.sin(registered_apply(params))

    primal = np.sum(matrix[0] ** 4)
    custom_curvature = np.cos(primal) * curvature - np.sin(primal) * np.outer(
        expected_gradient, expected_gradient
    )
    custom_actual = np.concatenate(
        [
            array(torch.func.grad(custom_composed)(x)).ravel(),
            array(torch.func.jacfwd(torch.func.grad(custom_composed))(x)).ravel(),
            array(torch.func.jacrev(torch.func.grad(custom_composed))(x)).ravel(),
        ]
    )
    custom_expected = np.concatenate(
        [np.cos(primal) * expected_gradient, custom_curvature.ravel(), custom_curvature.ravel()]
    )

    structured_batch = {"weights": batch, "bias": structured["bias"]}
    structured_mapped = torch.func.vmap(
        torch.func.grad(structured_loss), in_dims=({"weights": batch_axis, "bias": None},)
    )(structured_batch)
    batched_custom = torch.func.vmap(torch.func.grad(custom_composed), in_dims=batch_axis)(batch)
    structured_batch_actual = np.concatenate(
        [array(structured_mapped["weights"]).ravel(), array(structured_mapped["bias"]).ravel()]
    )
    custom_batch_actual = array(batched_custom)
    custom_batch_expected = np.cos(np.sum(matrix**4, axis=1))[:, None] * 4.0 * matrix**3

    shifted_batch_gradient = 4.0 * (matrix + 0.3) ** 3
    structured_batch_expected = np.concatenate(
        [shifted_batch_gradient.ravel(), np.sum(shifted_batch_gradient, axis=1)]
    )

    def case(category: str, actual: Any, expected: Any, *transforms: str) -> TransformAlgebraCase:
        numeric = array(actual) if isinstance(actual, torch.Tensor) else actual
        return _framework_transform_case(
            "pytorch", str(torch.__version__), category, numeric, expected, transforms
        )

    return (
        case("framework_grad", derivative(x), expected_gradient, "torch.func.grad"),
        case("framework_hessian", torch.func.hessian(loss)(x), curvature, "torch.func.hessian"),
        case(
            "framework_vmap_grad",
            mapped,
            4.0 * matrix**3,
            "torch.func.vmap(torch.func.grad)",
            f"in_dims={batch_axis}",
        ),
        case(
            "framework_grad_vmap",
            aggregate_gradient,
            4.0 * matrix**3,
            "torch.func.grad(sum(torch.func.vmap))",
            f"in_dims={batch_axis}",
        ),
        case(
            "framework_jacfwd_jacrev",
            np.concatenate(
                [
                    array(torch.func.jacfwd(derivative)(x)).ravel(),
                    array(torch.func.jacrev(derivative)(x)).ravel(),
                ]
            ),
            np.tile(curvature.ravel(), 2),
            "torch.func.jacfwd(grad)",
            "torch.func.jacrev(grad)",
        ),
        case(
            "framework_jvp_vjp",
            np.concatenate([array(pushed).ravel(), array(pulled).ravel()]),
            np.concatenate([curvature @ array(tangent), curvature.T @ array(tangent)]),
            "torch.func.jvp(grad)",
            "torch.func.vjp(grad)",
        ),
        case(
            "framework_structured",
            structured_actual,
            structured_expected,
            "dict",
            "torch.func.grad",
            "torch.func.jacfwd(grad)",
        ),
        case(
            "framework_custom_rules",
            custom_actual,
            custom_expected,
            "torch.autograd.Function",
            "setup_context",
            "backward",
            "jvp",
            "torch.func.jacfwd(grad(sin(custom)))",
            "torch.func.jacrev(grad(sin(custom)))",
        ),
        case(
            "framework_structured_vmap",
            structured_batch_actual,
            structured_batch_expected,
            "torch.func.vmap(grad(dict),in_dims={weights:axis,bias:None})",
        ),
        case(
            "framework_custom_vmap",
            custom_batch_actual,
            custom_batch_expected,
            "torch.func.vmap(grad(sin(autograd.Function)))",
        ),
        _framework_quantum_bridge_boundary_case("pytorch"),
    )
