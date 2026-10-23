# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Derivative backend capability refusals
"""Exercise incompatible runtime inputs and late refusal construction."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from scpn_quantum_control.differentiable_backend_capabilities import (
    BackendCapabilityPrimitives,
    resolve_torch_autograd_function,
    resolve_torch_autograd_grad,
)


def _primitives(
    invalid: Callable[[], type[RuntimeError]] = lambda: RuntimeError,
) -> BackendCapabilityPrimitives:
    return BackendCapabilityPrimitives(
        attribute=lambda: getattr,
        predicate=lambda: callable,
        presence=lambda: hasattr,
        invalid=invalid,
    )


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            resolve_torch_autograd_function,
            "PyTorch module does not expose torch.autograd.Function.apply",
        ),
        (resolve_torch_autograd_grad, "PyTorch module does not expose torch.autograd.grad"),
    ],
)
def test_actual_incompatible_object_has_no_autograd_capability(
    operation: Callable[..., object], message: str
) -> None:
    """Reject a genuine Python object without inventing an SDK implementation.

    Parameters
    ----------
    operation
        Public native capability admission operation.
    message
        Exact existing adapter refusal.

    """
    with pytest.raises(RuntimeError) as caught:
        operation(object(), primitives=_primitives())
    assert str(caught.value) == message


def test_runtime_refusal_constructor_is_resolved_after_the_descriptor() -> None:
    """A real descriptor may change the constructor before admission fails."""
    constructors: list[type[RuntimeError]] = [RuntimeError]

    class CapabilityError(RuntimeError):
        """Refusal selected while resolving an unsupported runtime capability."""

    class MissingAutograd:
        """An incompatible input with an observable Python descriptor."""

        @property
        def autograd(self) -> None:
            """Select the refusal constructor during the actual attribute read."""
            constructors[0] = CapabilityError

    with pytest.raises(CapabilityError) as caught:
        resolve_torch_autograd_function(
            MissingAutograd(), primitives=_primitives(lambda: constructors[0])
        )
    assert str(caught.value) == "PyTorch module does not expose torch.autograd.Function.apply"
