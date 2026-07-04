# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — Package root (facade placeholder)
"""oscillatools — a differentiable, control-oriented toolkit for coupled-phase-oscillator dynamics.

This module is the package root. Phase F2 of the extraction lifts the Kuramoto facade
(the public surface re-exported from :mod:`oscillatools.accel`) into this file; until
that lift lands, this placeholder carries only the distribution version.
"""

from __future__ import annotations

from ._version import __version__

__all__ = ["__version__"]
