# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch test for the identity binding spec default
"""Default-spec branch test for the identity attractor builder."""

from __future__ import annotations

from scpn_quantum_control.identity.binding_spec import build_identity_attractor


def test_build_identity_attractor_uses_default_spec() -> None:
    """Omitting the spec falls back to the Arcane Sapience default spec."""
    attractor = build_identity_attractor()
    assert attractor is not None
