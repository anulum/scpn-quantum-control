# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable API test helpers tests
# scpn-quantum-control -- differentiable API test helpers
"""Shared optional-runtime helpers for differentiable API and dashboard tests."""

from __future__ import annotations

import pytest


def _require_torch_backend() -> None:
    pytest.importorskip("torch", reason="native Torch differentiable rows require PyTorch")
