# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Shared pickle identity fixture
"""Share the original nominal-identity pickle consumer across owned fixtures."""

from __future__ import annotations

import importlib
import io
import pickle
from collections.abc import Mapping


class IdentityUnpickler(pickle.Unpickler):
    """Resolve only exact known globals through their genuine public import paths."""

    def __init__(self, data: bytes, expected: Mapping[tuple[str, str], object]) -> None:
        """Bind an in-memory pickle to the explicitly expected runtime objects.

        Parameters
        ----------
        data
            Bytes produced locally by the compatibility test.
        expected
            Exact module/name bindings permitted for this round trip.

        """
        self._expected = dict(expected)
        super().__init__(io.BytesIO(data))

    def find_class(self, module: str, name: str) -> object:
        """Reject unknown globals and changed bindings before reconstruction.

        Parameters
        ----------
        module
            Defining module requested by the pickle stream.
        name
            Global name requested in that module.

        Returns
        -------
        object
            The expected object, resolved through its actual import path.

        Raises
        ------
        pickle.UnpicklingError
            If the global is unregistered or its public binding has changed.

        """
        try:
            expected = self._expected[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(f"unregistered pickle global {module}.{name}") from exc
        actual = getattr(importlib.import_module(module), name)
        if actual is not expected:
            raise pickle.UnpicklingError(f"changed pickle binding {module}.{name}")
        return expected
