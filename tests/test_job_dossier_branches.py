# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the job-dossier validators
"""Guard tests for the job-dossier text and mapping validators."""

from __future__ import annotations

import pytest

from scpn_quantum_control.hardware.job_dossier import (
    _require_non_empty_mapping,
    _require_text,
)


def test_require_text_rejects_blank() -> None:
    """A blank text value is rejected."""
    with pytest.raises(ValueError, match="label must be non-empty text"):
        _require_text("   ", "label")


def test_require_non_empty_mapping_rejects_empty() -> None:
    """An empty mapping is rejected."""
    with pytest.raises(ValueError, match="metadata must be non-empty"):
        _require_non_empty_mapping({}, "metadata")
