# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — complex real-block and spectral derivative conformance
"""Complex derivatives follow the real-block convention; spectra refuse rather than clip."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.wirtinger_calculus import (
    SpectralDegeneracyError,
    framework_complex_gradient,
    hermitian_eigenpair_derivative,
    real_block_gradient,
    real_block_jacobian,
    spectral_projector_derivative,
    wirtinger_partials,
)

POINTS = (
    np.array([0.3 - 0.7j]),
    np.array([-1.25 + 0.5j]),
    np.array([2.0 + 0.0j]),
)


def _within_budget(actual: NDArray[Any], reference: NDArray[Any], absolute: float = 1e-10) -> bool:
    """Apply the frozen float64 budget ``|a - r| <= abs + 1e-8 |r|`` elementwise."""
    return bool(np.all(np.abs(actual - reference) <= absolute + 1e-8 * np.abs(reference)))


def _unitary(size: int, seed: int) -> NDArray[np.complex128]:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size)))
    unitary: NDArray[np.complex128] = q.astype(np.complex128)
    return unitary


def _hermitian(eigenvalues: tuple[float, ...], seed: int) -> NDArray[np.complex128]:
    q = _unitary(len(eigenvalues), seed)
    return (q @ np.diag(eigenvalues) @ q.conj().T).astype(np.complex128)


def _direction(size: int, seed: int) -> NDArray[np.complex128]:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    return (0.5 * (raw + raw.conj().T)).astype(np.complex128)


def _projector(matrix: NDArray[np.complex128], indices: tuple[int, ...]) -> NDArray[np.complex128]:
    _, vectors = np.linalg.eigh(matrix)
    chosen = vectors[:, list(indices)]
    projector: NDArray[np.complex128] = (chosen @ chosen.conj().T).astype(np.complex128)
    return projector


def test_complex_spectral_conformance_01() -> None:
    """conj(x + iy) has real-block Jacobian diag(1, -1); other fixtures match by hand."""
    for z in POINTS:
        assert _within_budget(real_block_jacobian(np.conj, z), np.diag([1.0, -1.0]))
        x, y = float(z[0].real), float(z[0].imag)
        square = real_block_jacobian(lambda v: v * v, z)
        assert _within_budget(square, np.array([[2 * x, -2 * y], [2 * y, 2 * x]]), 1e-9)
        modulus = real_block_jacobian(lambda v: v * np.conj(v), z)
        assert _within_budget(modulus, np.array([[2 * x, 2 * y], [0.0, 0.0]]), 1e-9)
        partials = wirtinger_partials(lambda v: complex(np.conj(v[0])), z)
        assert _within_budget(partials.df_dconj_z, np.array([1.0 + 0.0j]))
    pair = real_block_jacobian(np.conj, np.array([0.1 + 0.2j, -0.3 + 0.4j]))
    assert pair.shape == (4, 4)
    assert _within_budget(pair, np.diag([1.0, 1.0, -1.0, -1.0]))


def test_complex_spectral_conformance_02() -> None:
    """|z|^2 has real-block gradient (2x, 2y); each framework convention is explicit."""
    for z in POINTS:
        x, y = float(z[0].real), float(z[0].imag)
        loss = lambda v: float(np.sum(np.abs(v) ** 2))  # noqa: E731
        assert _within_budget(real_block_gradient(loss, z), np.array([2 * x, 2 * y]), 1e-9)
        torch_style = framework_complex_gradient(loss, z, convention="torch")
        jax_style = framework_complex_gradient(loss, z, convention="jax")
        assert _within_budget(torch_style, 2.0 * z, 1e-9)
        assert _within_budget(jax_style, 2.0 * np.conj(z), 1e-9)
        assert _within_budget(jax_style, np.conj(torch_style), 1e-15)


def test_torch_autograd_returns_the_torch_convention() -> None:
    """PyTorch autograd of a real loss returns dL/dx + i dL/dy."""
    torch = pytest.importorskip("torch", reason="PyTorch is the reference for its own convention")
    for z in POINTS:
        tensor = torch.tensor(z, dtype=torch.complex128, requires_grad=True)
        (tensor.abs() ** 2 + tensor.real * tensor.imag).sum().backward()
        assert tensor.grad is not None
        expected = framework_complex_gradient(
            lambda v: float(np.sum(np.abs(v) ** 2 + v.real * v.imag)), z, convention="torch"
        )
        assert _within_budget(tensor.grad.numpy(), expected, 1e-9)


def test_jax_grad_returns_the_jax_convention() -> None:
    """jax.grad of a real loss returns dL/dx - i dL/dy."""
    jax = pytest.importorskip("jax", reason="JAX is the reference for its own convention")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    for z in POINTS:
        gradient = np.asarray(
            jax.grad(lambda v: jnp.sum(jnp.abs(v) ** 2 + v.real * v.imag))(jnp.asarray(z))
        )
        expected = framework_complex_gradient(
            lambda v: float(np.sum(np.abs(v) ** 2 + v.real * v.imag)), z, convention="jax"
        )
        assert _within_budget(gradient, expected, 1e-9)


def test_complex_spectral_conformance_03() -> None:
    """A repeated eigenvalue refuses its eigenvector derivative; a tiny gap is not clipped."""
    degenerate = _hermitian((1.0, 1.0, 3.0), seed=4)
    direction = _direction(3, seed=5)
    before = (degenerate.copy(), direction.copy())
    for index in (0, 1):
        with pytest.raises(SpectralDegeneracyError, match="undefined"):
            hermitian_eigenpair_derivative(degenerate, direction, index)
    simple = hermitian_eigenpair_derivative(degenerate, direction, 2)
    assert simple.minimum_gap == pytest.approx(2.0, rel=1e-12)
    h = 1e-6
    shifted = [np.linalg.eigvalsh(degenerate + sign * h * direction)[2] for sign in (1.0, -1.0)]
    assert simple.eigenvalue_derivative == pytest.approx(
        (shifted[0] - shifted[1]) / (2 * h), rel=1e-7
    )
    assert np.array_equal(degenerate, before[0]) and np.array_equal(direction, before[1])
    for gap in (1e-3, 1e-6, 1e-9):
        near = np.diag([0.0, gap]).astype(np.complex128)
        swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        result = hermitian_eigenpair_derivative(near, swap, 0)
        assert np.linalg.norm(result.eigenvector_derivative) == pytest.approx(1.0 / gap, rel=1e-12)
    with pytest.raises(SpectralDegeneracyError):
        hermitian_eigenpair_derivative(np.diag([0.0, 1e-9]), np.eye(2), 0, gap_tolerance=1e-8)


def test_simple_eigenvector_derivative_matches_a_refined_reference() -> None:
    """The simple eigenvector derivative matches Richardson-refined differences in its gauge."""
    matrix = _hermitian((-1.0, 0.5, 2.0), seed=8)
    direction = _direction(3, seed=9)
    result = hermitian_eigenpair_derivative(matrix, direction, 1)
    assert abs(np.vdot(result.eigenvector, result.eigenvector_derivative)) < 1e-12

    def aligned(step: float) -> NDArray[np.complex128]:
        _, vectors = np.linalg.eigh(matrix + step * direction)
        vector = vectors[:, 1]
        phase = np.exp(-1j * np.angle(np.vdot(result.eigenvector, vector)))
        gauged: NDArray[np.complex128] = (vector * phase).astype(np.complex128)
        return gauged

    def central(step: float) -> NDArray[np.complex128]:
        return (aligned(step) - aligned(-step)) / (2 * step)

    refined = (4.0 * central(5e-5) - central(1e-4)) / 3.0
    assert np.max(np.abs(refined - result.eigenvector_derivative)) < 1e-8


def test_complex_spectral_conformance_04() -> None:
    """The projector onto a degenerate cluster and its derivative ignore eigenvector phase."""
    matrix = _hermitian((1.0, 1.0, 3.0, 4.5), seed=12)
    direction = _direction(4, seed=13)
    result = spectral_projector_derivative(matrix, direction, (1, 0))
    assert result.indices == (0, 1)
    assert result.separating_gap == pytest.approx(2.0, rel=1e-12)
    _, vectors = np.linalg.eigh(matrix)
    phases = np.exp(1j * np.array([0.7, -2.1]))
    rotated = vectors[:, :2] * phases
    mixing = _unitary(2, seed=14)
    for basis in (rotated, vectors[:, :2] @ mixing):
        assert _within_budget(basis @ basis.conj().T, result.projector, 1e-12)
    derivative = result.projector_derivative
    identity = np.eye(4)
    projector = result.projector
    assert _within_budget(derivative, derivative.conj().T, 1e-12)
    assert np.max(np.abs(projector @ derivative @ projector)) < 1e-12
    assert np.max(np.abs((identity - projector) @ derivative @ (identity - projector))) < 1e-12

    def central(step: float) -> NDArray[np.complex128]:
        return (
            _projector(matrix + step * direction, (0, 1))
            - _projector(matrix - step * direction, (0, 1))
        ) / (2 * step)

    refined = (4.0 * central(5e-5) - central(1e-4)) / 3.0
    assert np.max(np.abs(refined - derivative)) < 1e-8
    whole = spectral_projector_derivative(matrix, direction, (0, 1, 2, 3))
    assert _within_budget(whole.projector, identity.astype(np.complex128), 1e-12)
    assert np.max(np.abs(whole.projector_derivative)) == 0.0


def test_unseparated_subspace_is_refused() -> None:
    """A subspace that splits a repeated eigenvalue is refused, not approximated."""
    matrix = _hermitian((1.0, 1.0, 3.0), seed=21)
    with pytest.raises(SpectralDegeneracyError, match="not separated"):
        spectral_projector_derivative(matrix, _direction(3, seed=22), (0,))


@pytest.mark.parametrize(
    ("matrix", "tangent", "message"),
    [
        (np.array([[1.0, 2.0], [0.0, 1.0]]), np.eye(2), "matrix must be Hermitian"),
        (np.eye(2), np.array([[0.0, 1.0], [0.0, 0.0]]), "tangent must be Hermitian"),
        (np.eye(2), np.eye(3), "tangent must have the shape"),
        (np.zeros((0, 0)), np.zeros((0, 0)), "non-empty square matrix"),
        (np.ones(3), np.ones(3), "non-empty square matrix"),
        (np.array([[np.nan, 0.0], [0.0, 1.0]]), np.eye(2), "matrix must be finite"),
    ],
)
def test_malformed_spectral_inputs_are_refused(
    matrix: NDArray[Any], tangent: NDArray[Any], message: str
) -> None:
    """Malformed spectral inputs are refused by both spectral entry points."""
    with pytest.raises(ValueError, match=message):
        hermitian_eigenpair_derivative(matrix, tangent, 0)
    with pytest.raises(ValueError, match=message):
        spectral_projector_derivative(matrix, tangent, (0,))


@pytest.mark.parametrize("index", [-1, 2, True, 1.0])
def test_malformed_eigenpair_index_is_refused(index: Any) -> None:
    """An index that does not address one eigenpair is refused."""
    with pytest.raises(ValueError, match="index must address one eigenpair"):
        hermitian_eigenpair_derivative(np.diag([0.0, 1.0]), np.eye(2), index)


@pytest.mark.parametrize("indices", [(), (0, 0), (2,), (True,)])
def test_malformed_projector_indices_are_refused(indices: tuple[Any, ...]) -> None:
    """Empty, repeated, out-of-range or Boolean indices are refused."""
    with pytest.raises(ValueError, match="indices must be distinct"):
        spectral_projector_derivative(np.diag([0.0, 1.0]), np.eye(2), indices)


@pytest.mark.parametrize("tolerance", [0.0, -1.0, float("inf")])
def test_malformed_gap_tolerance_is_refused(tolerance: float) -> None:
    """A gap tolerance must be positive and finite."""
    with pytest.raises(ValueError, match="gap_tolerance"):
        hermitian_eigenpair_derivative(np.diag([0.0, 1.0]), np.eye(2), 0, gap_tolerance=tolerance)


def test_malformed_complex_functions_are_refused() -> None:
    """Complex losses, unstable output sizes, empty or non-finite outputs are refused."""
    z = np.array([0.5 + 0.5j])
    with pytest.raises(ValueError, match="loss must be real-valued"):
        real_block_gradient(cast(Any, lambda v: complex(v[0])), z)
    with pytest.raises(ValueError, match="convention must be"):
        framework_complex_gradient(lambda v: 0.0, z, convention="tensorflow")  # type: ignore[arg-type]
    calls = iter([1.0, [1.0, 2.0], [1.0, 2.0]])
    with pytest.raises(ValueError, match="same output size"):
        real_block_jacobian(lambda v: next(calls), z)
    with pytest.raises(ValueError, match="non-empty complex vector"):
        real_block_jacobian(lambda v: np.zeros(0), z)
    with pytest.raises(ValueError, match="non-empty complex vector"):
        real_block_jacobian(lambda v: np.zeros((2, 2)), z)
    with pytest.raises(ValueError, match="finite values"):
        real_block_jacobian(lambda v: np.inf, z)
