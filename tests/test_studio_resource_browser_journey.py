# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — test studio browser journey
"""Public resource journey refuses external URLs before constructing a browser."""

from __future__ import annotations

import pytest

from tools.studio_resource_browser_journey import run_resource_journey


@pytest.mark.parametrize(
    "url", ["http://example.com:4173/", "https://127.0.0.1:4173/", "http://127.0.0.1:4173/?q=1"]
)
def test_public_resource_journey_refuses_external_preview(url: str) -> None:
    """Preserve the original URL-owner refusal at the resource entry point."""
    with pytest.raises(ValueError):
        run_resource_journey(url)
