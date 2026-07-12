# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — thermodynamic witness module
"""Thermodynamic witness observable for calibrated work samples."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np


class ThermodynamicWitness:
    """
    Analyse explicitly supplied thermodynamic work samples.

    The witness intentionally refuses to infer work from bitstring counts.
    Counts can identify the measurement batch, but thermodynamic work must
    come from a calibrated protocol with energy units.
    """

    def __call__(self, counts: Mapping[str, int] | None = None, **kwargs: Any) -> dict[str, float]:
        """Summarise calibrated work samples into a thermodynamic witness.

        Requires ``work_samples_joule`` (an iterable) or ``work_joule`` (a scalar)
        from a calibrated protocol; ``counts`` is ignored. Returns the mean and
        sample variance of the work, and — when the optional
        ``delta_free_energy_joule`` and/or ``beta_per_joule`` keywords are given —
        the dissipated work and the Jarzynski free-energy estimate with its
        residual against the supplied free-energy difference.
        """
        _ = counts
        samples_raw = kwargs.get("work_samples_joule")
        scalar_raw = kwargs.get("work_joule")
        if samples_raw is None and scalar_raw is None:
            raise ValueError(
                "ThermodynamicWitness requires work_samples_joule or work_joule from a "
                "calibrated work protocol; it will not invent a default work value."
            )

        if samples_raw is not None:
            samples = np.asarray(list(samples_raw), dtype=float)
        else:
            samples = np.asarray([float(cast(Any, scalar_raw))], dtype=float)

        if samples.size == 0:
            raise ValueError("work_samples_joule must contain at least one sample.")
        if not np.all(np.isfinite(samples)):
            raise ValueError("work_samples_joule must contain finite joule values.")

        mean_work = float(np.mean(samples))
        variance = float(np.var(samples, ddof=1)) if samples.size > 1 else 0.0
        result = {
            "mean_work_joule": mean_work,
            "work_variance_joule2": variance,
            "n_work_samples": float(samples.size),
        }

        if "delta_free_energy_joule" in kwargs:
            delta_f = float(kwargs["delta_free_energy_joule"])
            if not np.isfinite(delta_f):
                raise ValueError("delta_free_energy_joule must be finite.")
            result["delta_free_energy_joule"] = delta_f
            result["dissipated_work_joule"] = mean_work - delta_f

        if "beta_per_joule" in kwargs:
            beta = float(kwargs["beta_per_joule"])
            if not np.isfinite(beta) or beta <= 0.0:
                raise ValueError("beta_per_joule must be finite and positive.")
            exp_average = float(np.mean(np.exp(-beta * samples)))
            jarzynski_delta_f = -np.log(exp_average) / beta
            result["beta_per_joule"] = beta
            result["jarzynski_delta_free_energy_joule"] = float(jarzynski_delta_f)
            if "delta_free_energy_joule" in result:
                result["jarzynski_residual_joule"] = (
                    jarzynski_delta_f - result["delta_free_energy_joule"]
                )

        return result
