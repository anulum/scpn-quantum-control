#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — generate params script
"""Compatibility wrapper for the frontier campaign parameter generator.

The implementation lives in ``scripts/frontier_campaign_2026/generate_params.py``
and fails closed unless synthetic smoke-test parameters are explicitly
requested.
"""

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "frontier_campaign_2026" / "generate_params.py"
    runpy.run_path(str(target), run_name="__main__")
