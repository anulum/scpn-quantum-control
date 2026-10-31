# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — complex real-block and spectral derivative conformance
"""Complex derivatives follow the real-block convention; spectra refuse rather than clip."""

from __future__ import annotations

import importlib
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.wirtinger_calculus import (
    SpectralDegeneracyError,
    framework_complex_gradient,
    hermitian_eigenpair_derivative,
    holomorphic_gradient,
    is_holomorphic,
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


@pytest.mark.parametrize("magnitude", [1.0e2, 1.0e4, 1.0e6, 1.0e8])
def test_difference_step_scales_with_the_point(magnitude: float) -> None:
    """Far from the origin the derivatives keep the frozen relative budget.

    Parameters
    ----------
    magnitude
        Distance of the evaluation point from the origin.

    """
    z = (magnitude * np.array([0.6 + 0.8j, -0.28 + 0.96j])).astype(np.complex128)
    gradient = real_block_gradient(lambda v: float(np.sum(np.abs(v) ** 2)), z)
    assert _within_budget(gradient, np.concatenate((2.0 * z.real, 2.0 * z.imag)), absolute=0.0)
    jacobian = real_block_jacobian(lambda v: 3.0 * v, z)
    assert _within_budget(jacobian, 3.0 * np.eye(4), absolute=1e-12)
    partials = wirtinger_partials(lambda v: v[0] * v[1], z)
    assert _within_budget(partials.df_dz, np.array([z[1], z[0]]), absolute=0.0)


def test_step_below_floating_point_resolution_is_refused() -> None:
    """A step that rounds away at ``z`` raises instead of dividing by zero."""
    with pytest.raises(ValueError, match="floating-point resolution"):
        real_block_jacobian(lambda v: v, np.array([1.0 + 1.0j]), step=1e-20)


def test_hermitian_checks_use_the_matrix_own_scale() -> None:
    """Small physical units are admitted and small non-Hermitian inputs are refused."""
    small = 1e-12 * np.diag([1.0, 2.0, 3.0])
    tangent = 1e-12 * np.diag([1.0, 0.0, 0.0])
    derivative = hermitian_eigenpair_derivative(small, tangent, 0)
    assert _within_budget(np.array([derivative.eigenvalue_derivative]), np.array([1e-12]), 0.0)
    unit = np.diag([0.0, 1.0])
    for triangular in (np.array([[0.0, 1e-13], [0.0, 0.0]]), np.array([[0.0, 0.0], [1e-13, 0.0]])):
        with pytest.raises(ValueError, match="tangent must be Hermitian"):
            hermitian_eigenpair_derivative(unit, triangular, 1)
    with pytest.raises(ValueError, match="matrix must be Hermitian"):
        hermitian_eigenpair_derivative(
            1e-20 * np.array([[1.0, 5.0], [0.0, 3.0]]), np.zeros((2, 2)), 0, gap_tolerance=1e-30
        )
    zero = hermitian_eigenpair_derivative(np.diag([0.0, 1.0]), np.zeros((2, 2)), 0)
    assert zero.eigenvalue_derivative == 0.0


def test_real_loss_in_complex_arithmetic_is_admitted() -> None:
    """``vdot(v, H v)`` carries imaginary rounding; a genuinely complex loss refuses."""
    hamiltonian = _hermitian((-1.0, 0.25, 2.0), 3)
    state = _unitary(3, 4)[:, 0]

    def expectation(vector: NDArray[np.complex128]) -> complex:
        return complex(np.vdot(vector, hamiltonian @ vector))

    gradient = real_block_gradient(expectation, state)
    force = 2.0 * hamiltonian @ state
    assert _within_budget(gradient, np.concatenate((force.real, force.imag)), absolute=1e-8)
    torch_style = framework_complex_gradient(expectation, state, convention="torch")
    assert _within_budget(torch_style, force, absolute=1e-8)
    with pytest.raises(ValueError, match="real-valued"):
        real_block_gradient(lambda v: complex(float(np.sum(np.abs(v) ** 2)), 1e-6), state)


def _reference_projector_derivative(
    matrix: NDArray[np.complex128], tangent: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """Return ``d(v v^H)`` of the lowest eigenvector at 50 digits."""
    mpmath: Any = importlib.import_module("mpmath")
    mpmath.mp.dps = 50
    exact = mpmath.matrix(matrix.tolist())
    direction = mpmath.matrix(tangent.tolist())
    values, vectors = mpmath.eighe(exact)
    order = sorted(range(matrix.shape[0]), key=lambda i: values[i])
    lowest = vectors[:, order[0]]
    reference = mpmath.matrix(matrix.shape[0], matrix.shape[0])
    for j in order[1:]:
        other = vectors[:, j]
        coupling = (other.H * direction * lowest)[0] / (values[order[0]] - values[j])
        term = other * coupling * lowest.H
        reference += term + term.H
    return np.array(reference.tolist(), dtype=np.complex128)


@pytest.mark.parametrize(
    ("gap", "decoupled"),
    [(1e-3, False), (1e-6, False), (1e-4, True), (1e-6, True), (1e-8, True)],
    ids=["generic-1e-3", "generic-1e-6", "split-1e-4", "split-1e-6", "split-1e-8"],
)
def test_near_degenerate_derivative_error_stays_within_its_estimate(
    gap: float, decoupled: bool
) -> None:
    """Against a 50-digit reference, the error stays within the reported bound.

    The eigenvector derivative depends on the phase of ``v``, so the
    gauge-invariant ``d(v v^H) = dv v^H + v dv^H`` is compared; its error is at
    most twice that of ``dv``. A tangent that splits the near-degenerate pair
    without mixing it makes the error grow as ``1 / gap**2``; at a gap of
    ``1e-8`` the derivative is then wrong at order one, and the estimate must
    say so.

    Parameters
    ----------
    gap
        Distance between the differentiated eigenvalue and its neighbour.
    decoupled
        Whether the tangent splits the pair without coupling it.

    """
    if decoupled:
        rotation = _unitary(3, 21)
        matrix = (rotation @ np.diag([0.0, gap, 1.0]) @ rotation.conj().T).astype(np.complex128)
        split = np.array([[1.0, 0.0, 0.3], [0.0, -1.0, 0.0], [0.3, 0.0, 0.0]])
        tangent = (rotation @ split @ rotation.conj().T).astype(np.complex128)
    else:
        matrix = _hermitian((0.0, gap, 1.0, 2.0), 11)
        tangent = _direction(4, 12)
    result = hermitian_eigenpair_derivative(matrix, tangent, 0)
    vector = result.eigenvector
    derivative = np.outer(result.eigenvector_derivative, vector.conj()) + np.outer(
        vector, result.eigenvector_derivative.conj()
    )
    error = float(np.linalg.norm(derivative - _reference_projector_derivative(matrix, tangent)))
    assert error <= 2.0 * result.rounding_error_estimate + 1e-14
    if decoupled and gap == 1e-8:
        assert result.rounding_error_estimate > np.linalg.norm(result.eigenvector_derivative)


def test_small_complex_losses_are_judged_at_their_own_scale() -> None:
    """A tiny loss whose imaginary part is comparable to it is refused; zero is real."""
    z = np.array([0.3 - 0.4j, 0.1 + 0.2j])
    with pytest.raises(ValueError, match="real-valued"):
        real_block_gradient(lambda v: complex(1e-15 * float(np.sum(np.abs(v) ** 2)), 1e-13), z)
    with pytest.raises(ValueError, match="real-valued"):
        real_block_gradient(lambda v: complex(0.0, 5e-13 * float(np.real(v[0]))), z)
    assert np.all(real_block_gradient(lambda v: 0.0, z) == 0.0)


def test_holomorphy_tolerance_is_relative() -> None:
    """``z**2`` far out is holomorphic; a tiny ``conj`` is not."""
    far = np.array([1e8 + 1e8j])
    assert is_holomorphic(lambda v: v[0] ** 2, far)
    assert _within_budget(holomorphic_gradient(lambda v: v[0] ** 2, far), 2.0 * far)
    assert not is_holomorphic(lambda v: 1e-12 * np.conj(v[0]), np.array([0.3 + 0.1j]))
    assert is_holomorphic(lambda v: 2.0 + 0.0j, np.array([0.3 + 0.1j]))


def test_numpy_integer_indices_are_admitted() -> None:
    """``np.int64`` indices address eigenpairs like Python integers."""
    matrix = _hermitian((0.0, 1.0, 3.0), 5)
    tangent = _direction(3, 6)
    by_numpy = hermitian_eigenpair_derivative(matrix, tangent, cast(int, np.int64(1)))
    by_python = hermitian_eigenpair_derivative(matrix, tangent, 1)
    assert by_numpy.eigenvalue_derivative == by_python.eigenvalue_derivative
    projector = spectral_projector_derivative(matrix, tangent, (cast(int, np.int64(0)),))
    assert projector.indices == (0,)


@pytest.mark.parametrize("magnitude", [1e-200, 1e200])
def test_rounding_estimate_stays_finite_at_extreme_scales(magnitude: float) -> None:
    """Neither a tiny nor a huge matrix overflows the estimate, and a zero tangent gives zero.

    Parameters
    ----------
    magnitude
        Scale of the matrix and of its tangent.

    """
    matrix = magnitude * _hermitian((0.0, 1.0, 3.0), 31)
    result = hermitian_eigenpair_derivative(matrix, magnitude * _direction(3, 32), 0)
    assert np.isfinite(result.rounding_error_estimate)
    assert np.all(np.isfinite(result.eigenvector_derivative))
    still = hermitian_eigenpair_derivative(matrix, np.zeros((3, 3)), 0)
    assert still.rounding_error_estimate == 0.0


def test_offset_spectrum_error_stays_within_the_estimate() -> None:
    """A large common offset still leaves the error inside the estimate."""
    matrix = _hermitian((1e3, 1e3 + 1e-2, 1e3 + 1.0), 41)
    tangent = _direction(3, 42)
    result = hermitian_eigenpair_derivative(matrix, tangent, 0)
    vector = result.eigenvector
    derivative = np.outer(result.eigenvector_derivative, vector.conj()) + np.outer(
        vector, result.eigenvector_derivative.conj()
    )
    error = float(np.linalg.norm(derivative - _reference_projector_derivative(matrix, tangent)))
    assert error <= 2.0 * result.rounding_error_estimate


def test_real_loss_passing_near_zero_needs_its_term_scale() -> None:
    """``<psi|H|psi>`` near zero is refused by default and admitted with ``scale``."""
    hamiltonian = _hermitian((-2.0, -1.0, 1.0, 2.0), 51)
    _, vectors = np.linalg.eigh(hamiltonian)
    state = ((vectors[:, 1] + vectors[:, 2]) / np.sqrt(2.0)).astype(np.complex128)

    def expectation(vector: NDArray[np.complex128]) -> complex:
        return complex(np.vdot(vector, hamiltonian @ vector))

    force = 2.0 * hamiltonian @ state
    gradient = real_block_gradient(expectation, state, scale=2.0)
    assert _within_budget(gradient, np.concatenate((force.real, force.imag)), absolute=1e-8)
    with pytest.raises(ValueError, match="scale must be"):
        real_block_gradient(expectation, state, scale=-1.0)


def test_holomorphy_is_judged_per_coordinate_with_a_noise_floor() -> None:
    """A large holomorphic part cannot hide a conjugate coordinate; critical points pass."""
    point = np.array([0.4 + 0.2j, -0.3 + 0.5j])
    assert not is_holomorphic(lambda v: 1e8 * v[0] + np.conj(v[1]), point)
    assert not is_holomorphic(lambda v: v[0] + 1e-3 * np.conj(v[1]), point, tolerance=1e-2)
    critical = np.array([1.0 + 0.0j])
    assert is_holomorphic(lambda v: v[0] ** 2 - 2.0 * v[0], critical)
    assert np.all(np.abs(holomorphic_gradient(lambda v: v[0] ** 2 - 2.0 * v[0], critical)) < 1e-8)
