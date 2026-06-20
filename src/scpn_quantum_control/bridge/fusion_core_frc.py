# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Fusion-Core FRC surrogate calibration bridge
"""Calibrate the quantum FRC scheduler surrogate from a SCPN-FUSION-CORE run.

The pulsed-shot QAOA scheduler in
:mod:`scpn_quantum_control.control.qaoa_pulsed_cost` evaluates its cost through a
control-grade :class:`~scpn_quantum_control.control.qaoa_pulsed_cost.FRCPlasmaSurrogate`
whose defaults are deliberately documented as values that *should be matched to a
SCPN-FUSION-CORE run when one is available*. SCPN-FUSION-CORE is the canonical
field-reversed-configuration (FRC) physics laboratory (see the 2026-05-31 cross-repo
solver-ownership broadcast); this module is the missing consumption path.

The bridge reads a fusion-core rigid-rotor FRC equilibrium and overrides exactly the
three surrogate knobs that the equilibrium determines on physical grounds:

- ``reference_field_T`` — the external confining field at which the equilibrium was
  solved (``B_ext``). The surrogate compression law anchors ``s`` to this field.
- ``reference_s_parameter`` — the ion-gyroradii separation ``s = R_s / rho_i`` that
  fusion-core computed for the equilibrium (Steinhauer, *Review of field-reversed
  configurations*, Physics of Plasmas 18, 070501, 2011).
- ``plasma_mass_density_kg_per_m3`` — the peak ion number density converted to mass
  density, feeding the magnetic-tension term of the MRTI penalty.

Every other surrogate field (compression exponent, Atwood number, perturbation
wavelength, areal mass, initial perturbation, tilt threshold) stays at the supplied
base-surrogate value because the radial equilibrium does not determine it; the
:class:`FusionCoreFRCCalibration` provenance record states explicitly which fields are
fusion-core-derived and which are control-grade defaults.

The pure calibration core (:func:`calibrate_frc_surrogate_from_equilibrium`) takes any
object satisfying :class:`FRCEquilibriumLike` and needs no fusion-core import. The
convenience entry point (:func:`calibrate_frc_surrogate_from_inputs`) solves the
equilibrium through fusion-core, importing it lazily with an optional ``repo_src``
fallback in the style of :mod:`scpn_quantum_control.bridge.control_plasma_knm`.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..control.qaoa_pulsed_cost import FRCPlasmaSurrogate

# CODATA 2018 unified atomic mass unit [kg]; matches scpn-fusion-core's ATOMIC_MASS_KG.
_ATOMIC_MASS_KG = 1.66053906660e-27
# Deuteron mass in atomic mass units; matches scpn-fusion-core's DEUTERIUM_MASS_AMU.
_DEUTERIUM_MASS_AMU = 2.014


class FRCEquilibriumLike(Protocol):
    """Structural contract for a fusion-core rigid-rotor FRC equilibrium.

    Only the fields the calibration consumes are required, which decouples the bridge
    from the concrete ``scpn_fusion.core.frc_rigid_rotor.FRCEquilibriumState`` type and
    keeps the pure core testable without a fusion-core import.

    Attributes
    ----------
    s_parameter : float
        Ion-gyroradii separation ``s = R_s / rho_i`` of the equilibrium.
    density_peak_m3 : float
        Peak ion number density of the equilibrium [m^-3].
    converged : bool
        Whether the equilibrium solver reached its residual tolerance.
    R_null : float
        Field-null radius of the equilibrium [m].
    target_separatrix_radius_m : float
        Requested separatrix radius ``R_s`` of the equilibrium [m].
    """

    @property
    def s_parameter(self) -> float:
        """Ion-gyroradii separation ``s = R_s / rho_i`` of the equilibrium."""
        ...

    @property
    def density_peak_m3(self) -> float:
        """Peak ion number density of the equilibrium [m^-3]."""
        ...

    @property
    def converged(self) -> bool:
        """Whether the equilibrium solver reached its residual tolerance."""
        ...

    @property
    def R_null(self) -> float:
        """Field-null radius of the equilibrium [m]."""
        ...

    @property
    def target_separatrix_radius_m(self) -> float:
        """Requested separatrix radius ``R_s`` of the equilibrium [m]."""
        ...


def _finite_positive(name: str, value: float) -> float:
    """Return ``value`` as a float, raising if it is not finite and strictly positive."""
    x = float(value)
    if not math.isfinite(x) or x <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return x


@dataclass(frozen=True)
class FusionCoreFRCCalibration:
    """Provenance record for an FRC surrogate calibrated from fusion-core.

    The record makes the calibration auditable: it stores the fusion-core-derived
    quantities, the resolved elongation, the source equilibrium diagnostics, and the
    explicit list of surrogate fields that were overridden from fusion-core (as opposed
    to retained from the control-grade base surrogate).

    Attributes
    ----------
    reference_field_T : float
        External confining field the equilibrium was solved at [T].
    reference_s_parameter : float
        Fusion-core ``s`` parameter at ``reference_field_T``.
    plasma_mass_density_kg_per_m3 : float
        Peak ion mass density derived from ``density_peak_m3`` and ``mass_amu`` [kg/m^3].
    mass_amu : float
        Ion mass in atomic mass units used for the number-to-mass density conversion.
    elongation : float
        FRC elongation ``E`` applied to the surrogate (caller-supplied or base default).
    ion_density_peak_m3 : float
        Peak ion number density read from the equilibrium [m^-3].
    null_radius_m : float
        Field-null radius of the source equilibrium [m].
    separatrix_radius_m : float
        Requested separatrix radius of the source equilibrium [m].
    fusion_core_derived_fields : tuple[str, ...]
        Surrogate fields overridden from fusion-core physics.
    source : str
        Provenance tag; always ``"scpn-fusion-core"``.
    metadata : dict[str, Any]
        Free-form provenance metadata (e.g. fusion-core version, grid size).
    """

    reference_field_T: float
    reference_s_parameter: float
    plasma_mass_density_kg_per_m3: float
    mass_amu: float
    elongation: float
    ion_density_peak_m3: float
    null_radius_m: float
    separatrix_radius_m: float
    fusion_core_derived_fields: tuple[str, ...]
    source: str = "scpn-fusion-core"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_field_T",
            _finite_positive("reference_field_T", self.reference_field_T),
        )
        object.__setattr__(
            self,
            "reference_s_parameter",
            _finite_positive("reference_s_parameter", self.reference_s_parameter),
        )
        object.__setattr__(
            self,
            "plasma_mass_density_kg_per_m3",
            _finite_positive("plasma_mass_density_kg_per_m3", self.plasma_mass_density_kg_per_m3),
        )
        object.__setattr__(self, "mass_amu", _finite_positive("mass_amu", self.mass_amu))
        object.__setattr__(self, "elongation", _finite_positive("elongation", self.elongation))
        object.__setattr__(
            self,
            "ion_density_peak_m3",
            _finite_positive("ion_density_peak_m3", self.ion_density_peak_m3),
        )
        object.__setattr__(self, "null_radius_m", float(self.null_radius_m))
        object.__setattr__(
            self,
            "separatrix_radius_m",
            _finite_positive("separatrix_radius_m", self.separatrix_radius_m),
        )
        object.__setattr__(
            self,
            "fusion_core_derived_fields",
            tuple(str(name) for name in self.fusion_core_derived_fields),
        )
        source = str(self.source).strip()
        if not source:
            raise ValueError("source must be non-empty.")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Convert the calibration record to a JSON-serializable dictionary."""
        return {
            "reference_field_T": self.reference_field_T,
            "reference_s_parameter": self.reference_s_parameter,
            "plasma_mass_density_kg_per_m3": self.plasma_mass_density_kg_per_m3,
            "mass_amu": self.mass_amu,
            "elongation": self.elongation,
            "ion_density_peak_m3": self.ion_density_peak_m3,
            "null_radius_m": self.null_radius_m,
            "separatrix_radius_m": self.separatrix_radius_m,
            "fusion_core_derived_fields": list(self.fusion_core_derived_fields),
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the calibration record to canonical JSON text."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FusionCoreFRCCalibration:
        """Reconstruct a calibration record from a dictionary payload."""
        return cls(
            reference_field_T=float(data["reference_field_T"]),
            reference_s_parameter=float(data["reference_s_parameter"]),
            plasma_mass_density_kg_per_m3=float(data["plasma_mass_density_kg_per_m3"]),
            mass_amu=float(data["mass_amu"]),
            elongation=float(data["elongation"]),
            ion_density_peak_m3=float(data["ion_density_peak_m3"]),
            null_radius_m=float(data["null_radius_m"]),
            separatrix_radius_m=float(data["separatrix_radius_m"]),
            fusion_core_derived_fields=tuple(data.get("fusion_core_derived_fields", ())),
            source=str(data.get("source", "scpn-fusion-core")),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, payload: str) -> FusionCoreFRCCalibration:
        """Reconstruct a calibration record from JSON text."""
        return cls.from_dict(json.loads(payload))


def calibrate_frc_surrogate_from_equilibrium(
    equilibrium: FRCEquilibriumLike,
    *,
    reference_field_T: float,
    elongation: float | None = None,
    mass_amu: float = _DEUTERIUM_MASS_AMU,
    base_surrogate: FRCPlasmaSurrogate | None = None,
    require_converged: bool = True,
) -> tuple[FRCPlasmaSurrogate, FusionCoreFRCCalibration]:
    """Calibrate an :class:`FRCPlasmaSurrogate` from a fusion-core FRC equilibrium.

    Parameters
    ----------
    equilibrium : FRCEquilibriumLike
        Solved fusion-core rigid-rotor FRC equilibrium.
    reference_field_T : float
        External confining field ``B_ext`` the equilibrium was solved at [T].
    elongation : float or None, optional
        FRC elongation ``E``. When ``None`` the base surrogate's ``elongation`` is kept.
    mass_amu : float, optional
        Ion mass in atomic mass units for number-to-mass density conversion. Defaults to
        deuterium (2.014).
    base_surrogate : FRCPlasmaSurrogate or None, optional
        Surrogate whose non-derived fields are retained. A default
        :class:`FRCPlasmaSurrogate` is used when ``None``.
    require_converged : bool, optional
        When ``True`` (default), an unconverged equilibrium raises ``ValueError``
        (fail-closed); set ``False`` only for explicitly diagnostic calibrations.

    Returns
    -------
    tuple[FRCPlasmaSurrogate, FusionCoreFRCCalibration]
        The calibrated surrogate and its provenance record.

    Raises
    ------
    ValueError
        If the equilibrium is unconverged (and ``require_converged``), or a derived
        quantity is not finite and positive.
    """
    from ..control.qaoa_pulsed_cost import FRCPlasmaSurrogate

    if require_converged and not bool(equilibrium.converged):
        raise ValueError(
            "fusion-core equilibrium did not converge; refusing to calibrate a surrogate "
            "from it (pass require_converged=False only for diagnostic use)."
        )

    base = base_surrogate if base_surrogate is not None else FRCPlasmaSurrogate()
    mass = _finite_positive("mass_amu", mass_amu)
    field_t = _finite_positive("reference_field_T", reference_field_T)
    s_param = _finite_positive("equilibrium.s_parameter", equilibrium.s_parameter)
    density_peak = _finite_positive("equilibrium.density_peak_m3", equilibrium.density_peak_m3)
    mass_density = density_peak * mass * _ATOMIC_MASS_KG
    resolved_elongation = (
        base.elongation if elongation is None else _finite_positive("elongation", elongation)
    )

    surrogate = FRCPlasmaSurrogate(
        reference_field_T=field_t,
        reference_s_parameter=s_param,
        s_compression_exponent=base.s_compression_exponent,
        atwood_number=base.atwood_number,
        perturbation_wavelength_m=base.perturbation_wavelength_m,
        areal_mass_kg_per_m2=base.areal_mass_kg_per_m2,
        plasma_mass_density_kg_per_m3=mass_density,
        elongation=resolved_elongation,
        initial_perturbation_m=base.initial_perturbation_m,
        tilt_kinetic_threshold=base.tilt_kinetic_threshold,
    )

    derived: tuple[str, ...] = (
        "reference_field_T",
        "reference_s_parameter",
        "plasma_mass_density_kg_per_m3",
    )
    if elongation is not None:
        derived = (*derived, "elongation")

    calibration = FusionCoreFRCCalibration(
        reference_field_T=field_t,
        reference_s_parameter=s_param,
        plasma_mass_density_kg_per_m3=mass_density,
        mass_amu=mass,
        elongation=resolved_elongation,
        ion_density_peak_m3=density_peak,
        null_radius_m=float(equilibrium.R_null),
        separatrix_radius_m=_finite_positive(
            "equilibrium.target_separatrix_radius_m", equilibrium.target_separatrix_radius_m
        ),
        fusion_core_derived_fields=derived,
    )
    return surrogate, calibration


def _import_frc_module(*, repo_src: str | Path | None = None) -> ModuleType:
    """Import ``scpn_fusion.core.frc_rigid_rotor`` with an optional local src path."""
    inserted = False
    src = str(Path(repo_src).resolve()) if repo_src is not None else ""
    if src and src not in sys.path:
        sys.path.insert(0, src)
        inserted = True
    try:
        return import_module("scpn_fusion.core.frc_rigid_rotor")
    except (
        ImportError,
        ModuleNotFoundError,
    ) as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "Unable to import scpn_fusion.core.frc_rigid_rotor. Install scpn-fusion-core "
            "or pass repo_src='<path>/scpn-fusion-core/src' to bridge functions."
        ) from exc
    finally:
        if inserted and sys.path and sys.path[0] == src:
            sys.path.pop(0)


def _default_rho_grid(separatrix_radius_m: float, n_grid_points: int) -> NDArray[np.float64]:
    """Build a fusion-core-valid radial grid: 0 at the axis, extending past ``R_s``."""
    if not isinstance(n_grid_points, int) or n_grid_points < 4:
        raise ValueError("n_grid_points must be an integer >= 4")
    r_s = _finite_positive("separatrix_radius_m", separatrix_radius_m)
    return np.linspace(0.0, 1.5 * r_s, n_grid_points, dtype=np.float64)


def calibrate_frc_surrogate_from_inputs(
    *,
    n0: float,
    T_i_eV: float,
    T_e_eV: float,
    R_s: float,
    B_ext: float,
    theta_dot: float = 0.0,
    delta: float | None = None,
    rho_grid: NDArray[np.float64] | None = None,
    n_grid_points: int = 129,
    elongation: float | None = None,
    mass_amu: float = _DEUTERIUM_MASS_AMU,
    base_surrogate: FRCPlasmaSurrogate | None = None,
    repo_src: str | Path | None = None,
) -> tuple[FRCPlasmaSurrogate, FusionCoreFRCCalibration]:
    """Solve a fusion-core FRC equilibrium and calibrate the scheduler surrogate.

    Parameters
    ----------
    n0 : float
        Central ion number density [m^-3].
    T_i_eV, T_e_eV : float
        Ion and electron temperatures [eV].
    R_s : float
        Requested separatrix radius [m].
    B_ext : float
        External confining field [T]; becomes the surrogate ``reference_field_T``.
    theta_dot : float, optional
        Rigid-rotor rotation rate. Only the certified ``0.0`` no-rotation limit is
        supported by fusion-core; other values raise ``NotImplementedError`` there.
    delta : float or None, optional
        Steinhauer profile width; fusion-core uses the ion gyroradius when ``None``.
    rho_grid : numpy.ndarray or None, optional
        Radial grid passed to the solver. When ``None`` a grid of ``n_grid_points``
        from the axis to ``1.5 * R_s`` is built.
    n_grid_points : int, optional
        Number of points in the default grid (ignored when ``rho_grid`` is given).
    elongation, mass_amu, base_surrogate : see
        :func:`calibrate_frc_surrogate_from_equilibrium`.
    repo_src : str or pathlib.Path or None, optional
        Local ``scpn-fusion-core/src`` path used as an import fallback.

    Returns
    -------
    tuple[FRCPlasmaSurrogate, FusionCoreFRCCalibration]
        The calibrated surrogate and its provenance record (metadata records the
        solved separatrix radius and grid size).

    Raises
    ------
    ImportError
        If fusion-core cannot be imported and no usable ``repo_src`` is given.
    """
    module = _import_frc_module(repo_src=repo_src)
    grid = (
        _default_rho_grid(R_s, n_grid_points)
        if rho_grid is None
        else np.asarray(rho_grid, dtype=np.float64)
    )
    inputs = module.RigidRotorFRCInputs(
        n0=n0, T_i_eV=T_i_eV, T_e_eV=T_e_eV, theta_dot=theta_dot, R_s=R_s, B_ext=B_ext, delta=delta
    )
    equilibrium = module.solve_frc_equilibrium(inputs, grid)
    surrogate, calibration = calibrate_frc_surrogate_from_equilibrium(
        equilibrium,
        reference_field_T=B_ext,
        elongation=elongation,
        mass_amu=mass_amu,
        base_surrogate=base_surrogate,
    )
    enriched = FusionCoreFRCCalibration.from_dict(
        {
            **calibration.to_dict(),
            "metadata": {
                **calibration.metadata,
                "n_grid_points": int(grid.size),
                "solver_separatrix_radius_m": float(R_s),
            },
        }
    )
    return surrogate, enriched


__all__ = [
    "FRCEquilibriumLike",
    "FusionCoreFRCCalibration",
    "calibrate_frc_surrogate_from_equilibrium",
    "calibrate_frc_surrogate_from_inputs",
]
