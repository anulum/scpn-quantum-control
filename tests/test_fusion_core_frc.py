# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the fusion-core FRC calibration bridge
"""Tests for bridge/fusion_core_frc.py.

The pure calibration core is exercised against a duck-typed equilibrium stand-in that
satisfies :class:`FRCEquilibriumLike` without importing scpn-fusion-core; the
fusion-core import path is exercised through a monkeypatched module, mirroring the
optional-dependency convention of ``test_control_plasma_knm_mock.py``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.bridge import fusion_core_frc as fcf
from scpn_quantum_control.bridge.fusion_core_frc import (
    FusionCoreFRCCalibration,
    calibrate_frc_surrogate_from_equilibrium,
    calibrate_frc_surrogate_from_inputs,
)
from scpn_quantum_control.control.qaoa_pulsed_cost import FRCPlasmaSurrogate

_B_EXT = 4.0
_S_PARAM = 2.6
_DENSITY_PEAK = 3.0e21
_R_NULL = 0.18
_R_S = 0.25


@dataclass(frozen=True)
class _FakeEquilibrium:
    """Minimal stand-in satisfying the FRCEquilibriumLike structural contract."""

    s_parameter: float = _S_PARAM
    density_peak_m3: float = _DENSITY_PEAK
    converged: bool = True
    R_null: float = _R_NULL
    target_separatrix_radius_m: float = _R_S


def _expected_mass_density(density_peak: float, mass_amu: float) -> float:
    return density_peak * mass_amu * fcf._ATOMIC_MASS_KG


# ---------------------------------------------------------------------------
# calibrate_frc_surrogate_from_equilibrium — derived quantities
# ---------------------------------------------------------------------------


class TestCalibrateFromEquilibrium:
    """Verify calibration from a supplied equilibrium contract."""

    def test_returns_surrogate_and_calibration(self) -> None:
        """Return both the scheduler surrogate and provenance record."""
        surrogate, calibration = calibrate_frc_surrogate_from_equilibrium(
            _FakeEquilibrium(), reference_field_T=_B_EXT
        )
        assert isinstance(surrogate, FRCPlasmaSurrogate)
        assert isinstance(calibration, FusionCoreFRCCalibration)

    def test_field_and_s_parameter_are_taken_from_equilibrium(self) -> None:
        """Derive the reference field and separation parameter exactly."""
        surrogate, _ = calibrate_frc_surrogate_from_equilibrium(
            _FakeEquilibrium(), reference_field_T=_B_EXT
        )
        assert surrogate.reference_field_T == pytest.approx(_B_EXT)
        assert surrogate.reference_s_parameter == pytest.approx(_S_PARAM)

    def test_mass_density_is_number_density_times_ion_mass(self) -> None:
        """Convert peak number density with the configured ion mass."""
        surrogate, _ = calibrate_frc_surrogate_from_equilibrium(
            _FakeEquilibrium(), reference_field_T=_B_EXT, mass_amu=fcf._DEUTERIUM_MASS_AMU
        )
        assert surrogate.plasma_mass_density_kg_per_m3 == pytest.approx(
            _expected_mass_density(_DENSITY_PEAK, fcf._DEUTERIUM_MASS_AMU)
        )

    def test_surrogate_is_self_consistent_at_reference_field(self) -> None:
        """Reproduce the equilibrium separation at the reference field."""
        # By construction s(reference_field_T) must reproduce the fusion-core s parameter.
        surrogate, _ = calibrate_frc_surrogate_from_equilibrium(
            _FakeEquilibrium(), reference_field_T=_B_EXT
        )
        assert surrogate.s_parameter(_B_EXT) == pytest.approx(_S_PARAM)

    def test_non_derived_fields_come_from_base_surrogate(self) -> None:
        """Retain every surrogate field not determined by equilibrium."""
        base = FRCPlasmaSurrogate(
            atwood_number=0.5, perturbation_wavelength_m=0.02, elongation=7.0
        )
        surrogate, _ = calibrate_frc_surrogate_from_equilibrium(
            _FakeEquilibrium(), reference_field_T=_B_EXT, base_surrogate=base
        )
        assert surrogate.atwood_number == pytest.approx(0.5)
        assert surrogate.perturbation_wavelength_m == pytest.approx(0.02)
        assert surrogate.s_compression_exponent == pytest.approx(base.s_compression_exponent)

    def test_elongation_defaults_to_base_when_not_supplied(self) -> None:
        """Retain base elongation when no override is supplied."""
        base = FRCPlasmaSurrogate(elongation=6.5)
        surrogate, calibration = calibrate_frc_surrogate_from_equilibrium(
            _FakeEquilibrium(), reference_field_T=_B_EXT, base_surrogate=base
        )
        assert surrogate.elongation == pytest.approx(6.5)
        assert "elongation" not in calibration.fusion_core_derived_fields

    def test_supplied_elongation_overrides_and_is_recorded(self) -> None:
        """Apply and record an explicit elongation override."""
        surrogate, calibration = calibrate_frc_surrogate_from_equilibrium(
            _FakeEquilibrium(), reference_field_T=_B_EXT, elongation=9.0
        )
        assert surrogate.elongation == pytest.approx(9.0)
        assert "elongation" in calibration.fusion_core_derived_fields


# ---------------------------------------------------------------------------
# calibrate_frc_surrogate_from_equilibrium — fail-closed and validation
# ---------------------------------------------------------------------------


class TestCalibrateFromEquilibriumGuards:
    """Verify fail-closed equilibrium and scalar validation."""

    def test_unconverged_equilibrium_is_rejected(self) -> None:
        """Reject unconverged equilibrium evidence by default."""
        with pytest.raises(ValueError, match="did not converge"):
            calibrate_frc_surrogate_from_equilibrium(
                _FakeEquilibrium(converged=False), reference_field_T=_B_EXT
            )

    def test_unconverged_allowed_when_not_required(self) -> None:
        """Permit explicitly diagnostic use of unconverged evidence."""
        surrogate, _ = calibrate_frc_surrogate_from_equilibrium(
            _FakeEquilibrium(converged=False), reference_field_T=_B_EXT, require_converged=False
        )
        assert surrogate.reference_field_T == pytest.approx(_B_EXT)

    @pytest.mark.parametrize("bad_field", [0.0, -1.0, float("nan"), float("inf")])
    def test_non_positive_reference_field_rejected(self, bad_field: float) -> None:
        """Reject non-positive and non-finite reference fields."""
        with pytest.raises(ValueError, match="reference_field_T"):
            calibrate_frc_surrogate_from_equilibrium(
                _FakeEquilibrium(), reference_field_T=bad_field
            )

    def test_non_positive_s_parameter_rejected(self) -> None:
        """Reject a non-positive equilibrium separation parameter."""
        with pytest.raises(ValueError, match="s_parameter"):
            calibrate_frc_surrogate_from_equilibrium(
                _FakeEquilibrium(s_parameter=0.0), reference_field_T=_B_EXT
            )

    def test_non_positive_density_rejected(self) -> None:
        """Reject a non-positive equilibrium peak density."""
        with pytest.raises(ValueError, match="density_peak_m3"):
            calibrate_frc_surrogate_from_equilibrium(
                _FakeEquilibrium(density_peak_m3=-1.0), reference_field_T=_B_EXT
            )

    def test_non_positive_mass_rejected(self) -> None:
        """Reject a non-positive configured ion mass."""
        with pytest.raises(ValueError, match="mass_amu"):
            calibrate_frc_surrogate_from_equilibrium(
                _FakeEquilibrium(), reference_field_T=_B_EXT, mass_amu=0.0
            )


# ---------------------------------------------------------------------------
# FusionCoreFRCCalibration — provenance artifact
# ---------------------------------------------------------------------------


class TestCalibrationArtifact:
    """Verify immutable calibration provenance and serialization."""

    def _make(self) -> FusionCoreFRCCalibration:
        _, calibration = calibrate_frc_surrogate_from_equilibrium(
            _FakeEquilibrium(), reference_field_T=_B_EXT, elongation=5.0
        )
        return calibration

    def test_records_source_and_derived_fields(self) -> None:
        """Record the source and every fusion-derived surrogate field."""
        calibration = self._make()
        assert calibration.source == "scpn-fusion-core"
        assert "reference_field_T" in calibration.fusion_core_derived_fields
        assert "plasma_mass_density_kg_per_m3" in calibration.fusion_core_derived_fields

    def test_dict_round_trip(self) -> None:
        """Round-trip calibration provenance through primitive fields."""
        calibration = self._make()
        restored = FusionCoreFRCCalibration.from_dict(calibration.to_dict())
        assert restored == calibration

    def test_json_round_trip(self) -> None:
        """Round-trip calibration provenance through canonical JSON."""
        calibration = self._make()
        restored = FusionCoreFRCCalibration.from_json(calibration.to_json())
        assert restored == calibration

    def test_metadata_is_copied_not_aliased(self) -> None:
        """Copy caller-owned metadata before retaining provenance."""
        meta: dict[str, Any] = {"run": "abc"}
        calibration = FusionCoreFRCCalibration(
            reference_field_T=_B_EXT,
            reference_s_parameter=_S_PARAM,
            plasma_mass_density_kg_per_m3=1.0,
            mass_amu=2.014,
            elongation=5.0,
            ion_density_peak_m3=_DENSITY_PEAK,
            null_radius_m=_R_NULL,
            separatrix_radius_m=_R_S,
            fusion_core_derived_fields=("reference_field_T",),
            metadata=meta,
        )
        meta["run"] = "mutated"
        assert calibration.metadata["run"] == "abc"

    def test_invalid_calibration_field_rejected(self) -> None:
        """Reject invalid derived scalar evidence in the record."""
        with pytest.raises(ValueError, match="reference_s_parameter"):
            FusionCoreFRCCalibration(
                reference_field_T=_B_EXT,
                reference_s_parameter=-1.0,
                plasma_mass_density_kg_per_m3=1.0,
                mass_amu=2.014,
                elongation=5.0,
                ion_density_peak_m3=_DENSITY_PEAK,
                null_radius_m=_R_NULL,
                separatrix_radius_m=_R_S,
                fusion_core_derived_fields=(),
            )

    def test_empty_source_rejected(self) -> None:
        """Reject an empty calibration provenance source."""
        with pytest.raises(ValueError, match="source"):
            FusionCoreFRCCalibration(
                reference_field_T=_B_EXT,
                reference_s_parameter=_S_PARAM,
                plasma_mass_density_kg_per_m3=1.0,
                mass_amu=2.014,
                elongation=5.0,
                ion_density_peak_m3=_DENSITY_PEAK,
                null_radius_m=_R_NULL,
                separatrix_radius_m=_R_S,
                fusion_core_derived_fields=(),
                source="   ",
            )


# ---------------------------------------------------------------------------
# _default_rho_grid
# ---------------------------------------------------------------------------


class TestDefaultRhoGrid:
    """Verify the deterministic fusion-core radial grid helper."""

    def test_grid_satisfies_fusion_core_constraints(self) -> None:
        """Start at the axis and extend monotonically past the separatrix."""
        grid = fcf._default_rho_grid(_R_S, 64)
        assert grid.size == 64
        assert grid[0] == 0.0
        assert np.all(np.diff(grid) > 0.0)
        assert grid[-1] > _R_S

    @pytest.mark.parametrize("n_points", [3, 0, -5])
    def test_too_few_points_rejected(self, n_points: int) -> None:
        """Reject grids below the fusion-core minimum width."""
        with pytest.raises(ValueError, match="n_grid_points"):
            fcf._default_rho_grid(_R_S, n_points)

    def test_non_positive_radius_rejected(self) -> None:
        """Reject a non-positive separatrix radius."""
        with pytest.raises(ValueError, match="separatrix_radius_m"):
            fcf._default_rho_grid(0.0, 64)


# ---------------------------------------------------------------------------
# calibrate_frc_surrogate_from_inputs — fusion-core import path (mocked)
# ---------------------------------------------------------------------------


def _make_mock_frc_module() -> tuple[ModuleType, dict[str, Any]]:
    """Build a fake scpn_fusion.core.frc_rigid_rotor module and a capture dict."""
    captured: dict[str, Any] = {}
    mod = ModuleType("scpn_fusion.core.frc_rigid_rotor")

    def _inputs(**kwargs: Any) -> SimpleNamespace:
        captured["inputs"] = kwargs
        return SimpleNamespace(**kwargs)

    def _solve(inputs: SimpleNamespace, grid: NDArray[np.float64]) -> _FakeEquilibrium:
        captured["grid"] = np.asarray(grid)
        captured["solved_with"] = inputs
        return _FakeEquilibrium()

    mod.RigidRotorFRCInputs = _inputs  # type: ignore[attr-defined]
    mod.solve_frc_equilibrium = _solve  # type: ignore[attr-defined]
    return mod, captured


@pytest.fixture
def mock_frc_module(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Inject an offline fusion-core module and return captured inputs."""
    mod, captured = _make_mock_frc_module()

    def _mock_import(name: str) -> ModuleType:
        if name == "scpn_fusion.core.frc_rigid_rotor":
            return mod
        raise ImportError(name)

    monkeypatch.setattr(fcf, "import_module", _mock_import)
    return captured


class TestCalibrateFromInputs:
    """Verify lazy fusion-core input solving and calibration."""

    def test_returns_calibrated_surrogate(self, mock_frc_module: dict[str, Any]) -> None:
        """Return calibrated values through the lazy solver entry point."""
        surrogate, calibration = calibrate_frc_surrogate_from_inputs(
            n0=_DENSITY_PEAK, T_i_eV=300.0, T_e_eV=250.0, R_s=_R_S, B_ext=_B_EXT
        )
        assert surrogate.reference_field_T == pytest.approx(_B_EXT)
        assert surrogate.reference_s_parameter == pytest.approx(_S_PARAM)
        assert calibration.source == "scpn-fusion-core"

    def test_builds_a_valid_default_grid(self, mock_frc_module: dict[str, Any]) -> None:
        """Build a valid default radial grid at the requested width."""
        calibrate_frc_surrogate_from_inputs(
            n0=_DENSITY_PEAK, T_i_eV=300.0, T_e_eV=250.0, R_s=_R_S, B_ext=_B_EXT, n_grid_points=80
        )
        grid = mock_frc_module["grid"]
        assert grid.size == 80
        assert grid[0] == 0.0
        assert grid[-1] > _R_S

    def test_passes_inputs_through(self, mock_frc_module: dict[str, Any]) -> None:
        """Forward physical inputs to the fusion-core constructor unchanged."""
        calibrate_frc_surrogate_from_inputs(
            n0=1.0e21, T_i_eV=320.0, T_e_eV=260.0, R_s=_R_S, B_ext=_B_EXT
        )
        passed = mock_frc_module["inputs"]
        assert passed["n0"] == pytest.approx(1.0e21)
        assert passed["B_ext"] == pytest.approx(_B_EXT)
        assert passed["theta_dot"] == 0.0

    def test_custom_grid_is_used(self, mock_frc_module: dict[str, Any]) -> None:
        """Forward a caller-supplied radial grid without replacement."""
        custom = np.linspace(0.0, 2.0 * _R_S, 16, dtype=np.float64)
        calibrate_frc_surrogate_from_inputs(
            n0=_DENSITY_PEAK, T_i_eV=300.0, T_e_eV=250.0, R_s=_R_S, B_ext=_B_EXT, rho_grid=custom
        )
        np.testing.assert_allclose(mock_frc_module["grid"], custom)

    def test_metadata_records_grid_and_radius(self, mock_frc_module: dict[str, Any]) -> None:
        """Record solved grid width and separatrix radius in metadata."""
        _, calibration = calibrate_frc_surrogate_from_inputs(
            n0=_DENSITY_PEAK, T_i_eV=300.0, T_e_eV=250.0, R_s=_R_S, B_ext=_B_EXT, n_grid_points=96
        )
        assert calibration.metadata["n_grid_points"] == 96
        assert calibration.metadata["solver_separatrix_radius_m"] == pytest.approx(_R_S)


# ---------------------------------------------------------------------------
# _import_frc_module — optional dependency import + sys.path management
# ---------------------------------------------------------------------------


class TestImportFrcModule:
    """Verify lazy optional import and temporary path custody."""

    def test_repo_src_inserted_and_cleaned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Insert a fallback source path temporarily and remove it."""
        mod, _ = _make_mock_frc_module()

        def _mock_import(name: str) -> ModuleType:
            if name == "scpn_fusion.core.frc_rigid_rotor":
                return mod
            raise ImportError(name)

        monkeypatch.setattr(fcf, "import_module", _mock_import)
        fake_src = "/tmp/fake_scpn_fusion/src"
        path_before = sys.path.copy()
        result = fcf._import_frc_module(repo_src=fake_src)
        assert result is mod
        assert fake_src not in sys.path or sys.path == path_before

    def test_without_repo_src_keeps_sys_path(self, mock_frc_module: dict[str, Any]) -> None:
        """Leave the interpreter path unchanged without a fallback."""
        path_before = sys.path.copy()
        fcf._import_frc_module()
        assert sys.path == path_before

    def test_missing_module_raises_with_guidance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raise actionable guidance when fusion-core is unavailable."""

        def _mock_import(name: str) -> ModuleType:
            raise ImportError(name)

        monkeypatch.setattr(fcf, "import_module", _mock_import)
        with pytest.raises(ImportError, match="scpn-fusion-core"):
            fcf._import_frc_module()


# ---------------------------------------------------------------------------
# package export surface
# ---------------------------------------------------------------------------


def test_symbols_exported_from_bridge_package() -> None:
    """Keep bridge-package exports identical to implementation symbols."""
    from scpn_quantum_control import bridge

    assert (
        bridge.calibrate_frc_surrogate_from_equilibrium is calibrate_frc_surrogate_from_equilibrium
    )
    assert bridge.FusionCoreFRCCalibration is FusionCoreFRCCalibration
