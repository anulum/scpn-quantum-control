# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — typed phase result objects
"""Typed result objects for phase-dynamics APIs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TrajectoryResult(Mapping[str, NDArray[np.float64]]):
    """Immutable Kuramoto trajectory with legacy mapping compatibility."""

    times: NDArray[np.float64]
    R: NDArray[np.float64]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=np.float64)
        order_parameter = np.asarray(self.R, dtype=np.float64)
        if times.ndim != 1:
            raise ValueError(f"times must be one-dimensional, got shape {times.shape}")
        if order_parameter.ndim != 1:
            raise ValueError(f"R must be one-dimensional, got shape {order_parameter.shape}")
        if times.shape != order_parameter.shape:
            raise ValueError(
                f"times and R must have identical shape, got {times.shape} and {order_parameter.shape}"
            )
        if not np.all(np.isfinite(times)):
            raise ValueError("times must contain only finite values")
        if not np.all(np.isfinite(order_parameter)):
            raise ValueError("R must contain only finite values")
        times_copy = np.array(times, dtype=np.float64, copy=True)
        order_copy = np.array(order_parameter, dtype=np.float64, copy=True)
        times_copy.setflags(write=False)
        order_copy.setflags(write=False)
        object.__setattr__(self, "times", times_copy)
        object.__setattr__(self, "R", order_copy)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def __getitem__(self, key: str) -> NDArray[np.float64]:
        if key == "times":
            return self.times
        if key == "R":
            return self.R
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        yield "times"
        yield "R"

    def __len__(self) -> int:
        return 2

    def to_dict(self) -> dict[str, NDArray[np.float64]]:
        """Return the legacy dictionary representation."""
        return {"times": self.times, "R": self.R}
