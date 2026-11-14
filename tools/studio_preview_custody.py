# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — owned-preview request custody for Studio browser journeys
"""Keep a Studio browser journey on its owned preview, with no submission.

Every instrument journey routes all page requests through one guard: a request
outside the owned literal-loopback preview is recorded and aborted, a request
with any method other than GET is recorded and aborted, and every other
request continues. Journeys that qualify the WebAssembly kernels also record
each ``.wasm`` path they load.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Callable

    from playwright.sync_api import Route


def preview_route_guard(
    origin: str,
    external: list[str],
    submissions: list[str],
    kernels: list[str] | None = None,
) -> Callable[[Route], None]:
    """Build the route handler that keeps a page on its owned preview.

    Parameters
    ----------
    origin
        Host and port of the owned literal-loopback preview.
    external
        Caller-owned record; receives ``"outside-preview"`` per refused request.
    submissions
        Caller-owned record; receives the method of each refused non-GET request.
    kernels
        Optional caller-owned record of the ``.wasm`` paths the page loads.

    Returns
    -------
    Callable
        Handler that aborts outside requests and submissions and lets the rest through.

    """

    def bound(route: Route) -> None:
        target = urlsplit(route.request.url)
        if target.scheme != "http" or target.netloc != origin:
            external.append("outside-preview")
            route.abort()
        elif route.request.method != "GET":
            submissions.append(route.request.method)
            route.abort()
        else:
            if kernels is not None and target.path.endswith(".wasm"):
                kernels.append(target.path)
            route.continue_()

    return bound
