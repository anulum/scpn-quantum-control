# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Scoped PyTorch integer specialisation
"""Keep Python integers static while ``torch.compile`` traces dynamic tensors.

With ``dynamic=True`` Dynamo treats integers read from object attributes as
symbolic. When those integers describe a fixed structure, symbolic tracing only
fails, and the failed attempt can mark shared frames to be skipped for the rest
of the process. PyTorch exposes integer specialisation only through
``torch._dynamo.config``, so the setting is patched for a scope and the caller's
value is restored on exit. The adapter is generic: callers state what is being
compiled and which releases they qualified.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from contextlib import AbstractContextManager, nullcontext
from typing import Any, cast


def torch_minor_release(torch_module: Any) -> tuple[int, int] | None:
    """Return the ``(major, minor)`` release of a PyTorch module, if readable.

    Parameters
    ----------
    torch_module
        Supplied PyTorch module.

    Returns
    -------
    tuple[int, int] | None
        Major and minor release numbers, or ``None`` when ``__version__`` does
        not start with them.

    """
    match = re.match(r"(\d+)\.(\d+)", str(getattr(torch_module, "__version__", "")))
    return (int(match.group(1)), int(match.group(2))) if match else None


def torch_static_integer_specialisation(
    torch_module: Any,
    dynamic: bool,
    *,
    purpose: str,
    qualified_releases: Collection[tuple[int, int]],
) -> AbstractContextManager[object]:
    """Return a scope that keeps integers static during dynamic compilation.

    Parameters
    ----------
    torch_module
        Supplied PyTorch module.
    dynamic
        Whether the caller compiles with ``dynamic=True``; otherwise the scope
        does nothing.
    purpose
        Name of the compiled route, used in the refusal message.
    qualified_releases
        ``(major, minor)`` releases on which the caller qualified the scope.

    Returns
    -------
    AbstractContextManager[object]
        ``torch._dynamo.config.patch(specialize_int=True)`` for a qualified
        dynamic compilation, or a null context when ``dynamic`` is false.

    Raises
    ------
    RuntimeError
        If ``dynamic`` is requested on an unqualified release, or the runtime
        does not expose integer specialisation. Both refusals happen before
        anything is compiled.

    """
    if not dynamic:
        return nullcontext()
    release = torch_minor_release(torch_module)
    if release not in qualified_releases:
        raise RuntimeError(
            f"{purpose} is qualified only on PyTorch "
            + ", ".join(f"{major}.{minor}" for major, minor in sorted(qualified_releases))
            + f"; found {getattr(torch_module, '__version__', 'unknown')}"
        )
    config = getattr(getattr(torch_module, "_dynamo", None), "config", None)
    if config is None or not hasattr(config, "specialize_int") or not hasattr(config, "patch"):
        raise RuntimeError(
            "PyTorch module does not expose static integer specialisation for dynamic compilation"
        )
    return cast(AbstractContextManager[object], config.patch(specialize_int=True))


__all__ = ["torch_minor_release", "torch_static_integer_specialisation"]
