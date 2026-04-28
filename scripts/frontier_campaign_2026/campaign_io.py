#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Frontier campaign I/O helpers
"""Path helpers for frontier campaign scripts."""

from __future__ import annotations

from pathlib import Path

CAMPAIGN_DIR = Path(__file__).resolve().parent


def campaign_path(*parts: str) -> Path:
    """Return a path anchored to the frontier campaign directory."""
    return CAMPAIGN_DIR.joinpath(*parts)


def parameter_path(filename: str) -> Path:
    """Return a parameter-cache path anchored to the campaign directory."""
    return campaign_path("params", filename)


def result_path(filename: str) -> Path:
    """Return a result path and ensure the result directory exists."""
    path = campaign_path("results", filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
