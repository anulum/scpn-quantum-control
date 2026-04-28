#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Sophisticated campaign I/O helpers
"""Path helpers for sophisticated campaign scripts."""

from __future__ import annotations

from pathlib import Path

CAMPAIGN_DIR = Path(__file__).resolve().parent


def parameter_path(filename: str) -> Path:
    """Return a campaign-local parameter-cache path."""
    return CAMPAIGN_DIR / "params" / filename


def result_path(filename: str) -> Path:
    """Return a campaign-local result path and ensure its directory exists."""
    path = CAMPAIGN_DIR / "results" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
