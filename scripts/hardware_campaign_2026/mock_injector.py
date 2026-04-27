#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — retired campaign injector guard
"""Retired local injector guard.

Campaign runs must use the real :class:`AsyncHardwareRunner` path and
explicitly labelled source artifacts. This module intentionally fails at
import time so legacy helper imports cannot monkeypatch hardware execution
or emit fabricated observables.
"""

raise RuntimeError(
    "Local campaign injectors are retired. Use AsyncHardwareRunner with "
    "source-backed or explicitly labelled smoke-test artifacts instead."
)
