# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — optional local job service
"""Durable local jobs around the local-only executive handlers.

Import from the submodules: :mod:`.service` (the service and its transport),
:mod:`.journal`, :mod:`.registry` and :mod:`.worker`. This package imports
nothing itself, so a worker process applies its resource limits before the
numerical stack loads.
"""

from __future__ import annotations

__all__: list[str] = []
