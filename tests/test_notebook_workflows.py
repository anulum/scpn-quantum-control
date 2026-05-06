# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — notebook workflow contracts
"""Static contracts for notebook workflow artefacts.

The notebook boundary is intentionally not executed in unit tests:
many notebooks are exploratory, long-running, or require optional
external datasets.  This test constrains repository hygiene instead:
notebooks must be valid nbformat JSON, contain cells, keep recognised
cell types, and store cell sources in normal notebook-compatible forms.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
VALID_CELL_TYPES = {"code", "markdown", "raw"}


def _notebook_paths() -> tuple[Path, ...]:
    """Return committed notebooks outside virtual environments."""
    return tuple(sorted(NOTEBOOKS.rglob("*.ipynb")))


def test_notebook_workflows_are_valid_static_artefacts() -> None:
    notebooks = _notebook_paths()

    assert len(notebooks) >= 40
    for notebook in notebooks:
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        assert payload.get("nbformat") == 4, f"{notebook} must use nbformat 4"
        assert isinstance(payload.get("metadata"), dict), f"{notebook} metadata must be a dict"
        cells = payload.get("cells")
        assert isinstance(cells, list), f"{notebook} cells must be a list"
        assert cells, f"{notebook} must contain at least one cell"
        for cell in cells:
            assert isinstance(cell, dict), f"{notebook} cells must be objects"
            assert cell.get("cell_type") in VALID_CELL_TYPES, f"{notebook} has invalid cell type"
            source = cell.get("source")
            assert isinstance(source, str | list), f"{notebook} cell source must be text or lines"
