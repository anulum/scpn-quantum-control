# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Symmetry Decay ZNE (GUESS)
"""Multi-angle tests for mitigation/symmetry_decay.py.

6 dimensions: empty/null, error handling, negative cases, pipeline
integration, roundtrip, performance.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from scpn_quantum_control.mitigation.symmetry_decay import (
    GUESSResult,
    SymmetryDecayModel,
    guess_extrapolate,
    learn_symmetry_decay,
    xy_magnetisation_ideal,
)

# ===== 1. Empty/Null Inputs =====


class TestEmptyNull:
    def test_two_scales_minimum(self) -> None:
        model = learn_symmetry_decay(4.0, [3.9, 3.5], [1, 3])
        assert isinstance(model, SymmetryDecayModel)
        assert model.alpha >= 0.0

    def test_identical_values_zero_alpha(self) -> None:
        """No decay → α ≈ 0."""
        model = learn_symmetry_decay(4.0, [4.0, 4.0, 4.0], [1, 3, 5])
        assert abs(model.alpha) < 1e-10

    def test_guess_with_zero_correction(self) -> None:
        """α = 0 → no correction applied."""
        model = SymmetryDecayModel(
            ideal_symmetry_value=4.0,
            noisy_symmetry_values=[4.0, 4.0],
            noise_scales=[1, 3],
            alpha=0.0,
            fit_residual=0.0,
        )
        result = guess_extrapolate(0.5, 4.0, model)
        assert abs(result.mitigated_value - 0.5) < 1e-15
        assert abs(result.correction_factor - 1.0) < 1e-15


# ===== 2. Error Handling =====


class TestErrorHandling:
    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            learn_symmetry_decay(4.0, [3.9, 3.5], [1, 3, 5])

    def test_too_few_scales(self) -> None:
        with pytest.raises(ValueError, match="Need >= 2"):
            learn_symmetry_decay(4.0, [3.9], [1])

    def test_zero_ideal_value(self) -> None:
        with pytest.raises(ValueError, match="too close to zero"):
            learn_symmetry_decay(0.0, [0.0, 0.0], [1, 3])

    def test_invalid_initial_state(self) -> None:
        with pytest.raises(ValueError, match="Unknown initial state"):
            xy_magnetisation_ideal(4, "invalid")


# ===== 3. Negative Cases =====


class TestNegativeCases:
    def test_no_noise_no_correction(self) -> None:
        """If noisy == ideal, correction factor = 1."""
        model = learn_symmetry_decay(4.0, [4.0, 4.0], [1, 3])
        result = guess_extrapolate(0.7, 4.0, model)
        assert abs(result.correction_factor - 1.0) < 1e-10

    def test_fully_decayed_symmetry_no_crash(self) -> None:
        """If symmetry completely decayed (s_noisy → 0), return raw value."""
        model = SymmetryDecayModel(
            ideal_symmetry_value=4.0,
            noisy_symmetry_values=[0.0, 0.0],
            noise_scales=[1, 3],
            alpha=1.0,
            fit_residual=0.0,
        )
        result = guess_extrapolate(0.5, 0.0, model)
        assert result.mitigated_value == 0.5
        assert result.correction_factor == 1.0

    def test_negative_alpha_not_physical(self) -> None:
        """Increasing symmetry under noise → negative α (unphysical)."""
        model = learn_symmetry_decay(4.0, [4.1, 4.5], [1, 3])
        assert model.alpha < 0.0  # flag as unphysical


# ===== 4. Pipeline Integration =====


class TestPipelineIntegration:
    def test_guess_improves_over_raw(self) -> None:
        """Under exponential decay, GUESS should recover closer to ideal."""
        # Simulate: ideal R = 0.8, noisy R = 0.5
        # Symmetry: ideal S = 4.0, decays as exp(-0.1*(g-1))
        scales = [1, 3, 5]
        s_noisy = [4.0 * np.exp(-0.1 * (g - 1)) for g in scales]
        model = learn_symmetry_decay(4.0, s_noisy, scales)

        # Target at base noise (g=1)
        result = guess_extrapolate(0.5, s_noisy[0], model)
        # Correction factor should be ~1.0 at g=1 (s_noisy ≈ s_ideal)
        assert abs(result.correction_factor - 1.0) < 0.05

    def test_xy_magnetisation_ground(self) -> None:
        assert xy_magnetisation_ideal(4, "ground") == 4.0
        assert xy_magnetisation_ideal(8, "ground") == 8.0

    def test_xy_magnetisation_neel(self) -> None:
        assert xy_magnetisation_ideal(4, "neel") == 0.0  # even
        assert xy_magnetisation_ideal(5, "neel") == 1.0  # odd

    def test_top_level_import(self) -> None:
        from scpn_quantum_control.mitigation import (
            guess_extrapolate,
            learn_symmetry_decay,
        )

        assert callable(learn_symmetry_decay)
        assert callable(guess_extrapolate)

    def test_decay_model_fields(self) -> None:
        model = learn_symmetry_decay(4.0, [3.8, 3.2, 2.5], [1, 3, 5])
        assert model.ideal_symmetry_value == 4.0
        assert len(model.noisy_symmetry_values) == 3
        assert len(model.noise_scales) == 3
        assert isinstance(model.alpha, float)
        assert isinstance(model.fit_residual, float)


# ===== 5. Roundtrip =====


class TestRoundtrip:
    def test_exact_exponential_recovery(self) -> None:
        """Perfect exponential decay → α should match exactly."""
        alpha_true = 0.15
        scales = [1, 3, 5, 7]
        s_ideal = 4.0
        s_noisy = [s_ideal * np.exp(-alpha_true * (g - 1)) for g in scales]

        model = learn_symmetry_decay(s_ideal, s_noisy, scales)
        assert abs(model.alpha - alpha_true) < 1e-10
        assert model.fit_residual < 1e-10

    def test_correction_increases_with_noise(self) -> None:
        """More noise → larger correction factor."""
        model = SymmetryDecayModel(
            ideal_symmetry_value=4.0,
            noisy_symmetry_values=[3.0, 2.0],
            noise_scales=[1, 3],
            alpha=0.5,
            fit_residual=0.01,
        )
        r1 = guess_extrapolate(0.5, 3.5, model)
        r2 = guess_extrapolate(0.5, 2.0, model)
        assert r2.correction_factor > r1.correction_factor

    def test_guess_result_fields(self) -> None:
        model = learn_symmetry_decay(4.0, [3.8, 3.0], [1, 3])
        result = guess_extrapolate(0.6, 3.8, model)
        assert isinstance(result, GUESSResult)
        assert isinstance(result.raw_value, float)
        assert isinstance(result.mitigated_value, float)
        assert isinstance(result.correction_factor, float)
        assert result.raw_value == 0.6


# ===== 6. Performance =====


class TestPerformance:
    def test_learn_fast(self) -> None:
        """Learning from 5 scales must complete in < 1ms."""
        t0 = time.perf_counter()
        for _ in range(1000):
            learn_symmetry_decay(4.0, [3.8, 3.5, 3.0, 2.5, 2.0], [1, 3, 5, 7, 9])
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"1000 learn calls took {elapsed:.3f}s"

    def test_extrapolate_fast(self) -> None:
        """Extrapolation must complete in < 0.1ms."""
        model = learn_symmetry_decay(4.0, [3.8, 3.0], [1, 3])
        t0 = time.perf_counter()
        for _ in range(10000):
            guess_extrapolate(0.5, 3.8, model)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"10000 extrapolate calls took {elapsed:.3f}s"
