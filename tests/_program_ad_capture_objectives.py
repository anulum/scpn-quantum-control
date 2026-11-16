# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — capture objectives with a fixed source charge
"""Objectives whose charged source is this small module, not a growing test file.

Whole-program AD charges the source of the module that defines an objective
against the execution memory cap. A test whose cap must separate two nearby
totals defines its objective here, so its margin does not shrink as the test
module grows.
"""

from __future__ import annotations

from collections.abc import Callable

from scpn_quantum_control import TraceADArray


def string_capture_objective(text: str) -> Callable[[TraceADArray], object]:
    """Return an objective that captures ``text`` and reads only whether it is bound.

    Parameters
    ----------
    text
        String whose captured snapshot is charged against the memory cap.

    Returns
    -------
    Callable[[TraceADArray], object]
        Objective returning twice its first parameter while ``text`` is bound.

    """

    def objective(values: TraceADArray) -> object:
        return values[0] * (2.0 if text is not None else 5.0)

    return objective
