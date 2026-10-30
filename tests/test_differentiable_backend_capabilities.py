# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Derivative backend capability refusals
"""Exercise incompatible runtime inputs and late refusal construction."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from scpn_quantum_control.differentiable_backend_capabilities import (
    BackendCapabilityPrimitives,
    jax_runtime_available,
    resolve_torch_autograd_function,
    resolve_torch_autograd_grad,
    tensorflow_runtime_available,
    torch_runtime_available,
)

_RUNTIMES: list[tuple[Callable[[], bool], str]] = [
    (torch_runtime_available, "torch"),
    (jax_runtime_available, "jax"),
    (tensorflow_runtime_available, "tensorflow"),
]


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


@pytest.mark.parametrize(("available", "package"), _RUNTIMES)
def test_installed_runtimes_are_reported_available(
    available: Callable[[], bool], package: str
) -> None:
    """Report each locked optional runtime that the environment can import.

    Parameters
    ----------
    available
        Public availability operation under test.
    package
        Top-level package the operation imports.

    """
    pytest.importorskip(package)
    assert available() is True


@pytest.mark.parametrize(("available", "package"), _RUNTIMES)
def test_runtime_that_cannot_be_imported_is_unavailable(
    available: Callable[[], bool], package: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report a runtime the import system refuses as unavailable.

    Parameters
    ----------
    available
        Public availability operation under test.
    package
        Top-level package blocked in ``sys.modules``.
    monkeypatch
        Restores the import table after the case.

    """
    monkeypatch.setitem(sys.modules, package, None)
    assert available() is False


@pytest.mark.parametrize("failure", ["AttributeError", "RuntimeError", "ValueError"])
def test_jax_that_fails_while_importing_is_unavailable(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Treat a JAX installation that raises at import time as unavailable.

    Parameters
    ----------
    failure
        Exception the installed package raises from its initialiser.
    tmp_path
        Directory holding the failing package.
    monkeypatch
        Restores the import table and search path after the case.

    """
    package = tmp_path / "jax"
    package.mkdir()
    (package / "__init__.py").write_text(f"raise {failure}('broken JAX installation')\n")
    for name in [name for name in sys.modules if name == "jax" or name.startswith("jax.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(tmp_path))
    assert jax_runtime_available() is False
