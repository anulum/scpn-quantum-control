# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — QEC Decoder Boundary Documentation Tests
"""Public documentation guards for the QEC decoder boundary."""

from __future__ import annotations

import re
from pathlib import Path

from scpn_quantum_control.qec import (
    BiologicalMWPMDecoder,
    BiologicalSurfaceCode,
    ControlQEC,
    MWPMDecoder,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BOUNDARY_DOC = _REPO_ROOT / "docs" / "qec_decoder_boundary.md"
_API_DOC = _REPO_ROOT / "docs" / "api.md"
_LINK_SURFACES = (
    _REPO_ROOT / "docs" / "api.md",
    _REPO_ROOT / "docs" / "multiscale_qec.md",
    _REPO_ROOT / "docs" / "dla_protected_subspace.md",
    _REPO_ROOT / "docs" / "architecture_map.md",
    _REPO_ROOT / "mkdocs.yml",
)


def test_qec_decoder_boundary_names_live_decoder_surfaces() -> None:
    """The boundary page documents only decoder classes exported by the package."""
    text = _BOUNDARY_DOC.read_text(encoding="utf-8")
    collapsed = re.sub(r"\s+", " ", text.lower())

    assert ControlQEC.__name__ in text
    assert MWPMDecoder.__name__ in text
    assert BiologicalSurfaceCode.__name__ in text
    assert BiologicalMWPMDecoder.__name__ in text
    assert "minimum-weight perfect matching" in collapsed


def test_qec_decoder_boundary_records_absent_decoder_families() -> None:
    """The page keeps absent QEC decoder families out of public claims."""
    collapsed = re.sub(r"\s+", " ", _BOUNDARY_DOC.read_text(encoding="utf-8"))

    required_phrases = (
        "No union-find decoder is implemented or exported.",
        "No MWPM decoder with explicit rough-boundary absorption is implemented",
        "No hardware syndrome-extraction controller",
        "fails closed when any connected component has odd syndrome parity",
    )
    missing = [phrase for phrase in required_phrases if phrase not in collapsed]
    assert not missing, f"QEC decoder boundary lost required non-claims: {missing}"


def test_qec_decoder_boundary_is_linked_from_public_surfaces() -> None:
    """Public QEC pages and MkDocs navigation link the decoder boundary."""
    missing = [
        str(path.relative_to(_REPO_ROOT))
        for path in _LINK_SURFACES
        if "qec_decoder_boundary.md" not in path.read_text(encoding="utf-8")
    ]
    assert not missing, f"QEC decoder boundary is not linked from: {missing}"


def test_control_qec_api_snippet_matches_live_public_methods() -> None:
    """The API page documents the public ControlQEC methods that actually exist."""
    api_text = _API_DOC.read_text(encoding="utf-8")
    control_section = api_text.split("### `control_qec.ControlQEC`", maxsplit=1)[1].split(
        "## config", maxsplit=1
    )[0]

    expected_methods = ("simulate_errors", "get_syndrome", "decode_and_correct")
    for method_name in expected_methods:
        assert hasattr(ControlQEC, method_name)
        assert f".{method_name}(" in control_section

    stale_methods = ("protect_signal", "decode_syndrome")
    for method_name in stale_methods:
        assert not hasattr(ControlQEC, method_name)
        assert method_name not in control_section
