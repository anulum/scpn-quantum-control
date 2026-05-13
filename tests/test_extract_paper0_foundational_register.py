# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Paper 0 exhaustive extraction tests
"""Tests for exhaustive Paper 0 register extraction."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "extract_paper0_foundational_register.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("extract_paper0_register", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 extraction script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _str(text: str) -> dict[str, str]:
    return {"t": "Str", "c": text}


def _space() -> dict[str, str]:
    return {"t": "Space"}


def test_extract_register_records_every_top_level_block() -> None:
    module = _load_module()
    ast = {
        "blocks": [
            {"t": "Header", "c": [1, ["", [], []], [_str("Paper"), _space(), _str("0")]]},
            {
                "t": "Para",
                "c": [
                    _str("We"),
                    _space(),
                    _str("posit"),
                    _space(),
                    _str("UPDE"),
                    _space(),
                    _str("dynamics."),
                ],
            },
            {"t": "Para", "c": [{"t": "Math", "c": [{"t": "DisplayMath"}, "x=y"]}]},
            {"t": "HorizontalRule"},
        ]
    }

    result = module.extract_register(ast)

    assert result.summary["top_level_ast_blocks"] == 4
    assert result.summary["exhaustive_register_records"] == 4
    assert [record["record_id"] for record in result.records] == [
        "P0B00001",
        "P0B00002",
        "P0B00003",
        "P0B00004",
    ]
    assert result.records[3]["has_text"] is False
    assert result.records[3]["review_status"] == "unreviewed"


def test_extract_register_links_equations_and_claim_classification() -> None:
    module = _load_module()
    ast = {
        "blocks": [
            {
                "t": "Header",
                "c": [2, ["", [], []], [_str("UPDE"), _space(), _str("Section")]],
            },
            {
                "t": "Para",
                "c": [
                    _str("The"),
                    _space(),
                    _str("mechanism"),
                    _space(),
                    _str("requires"),
                    _space(),
                    {"t": "Math", "c": [{"t": "InlineMath"}, "K_{ij}"]},
                    _space(),
                    _str("coupling."),
                ],
            },
        ]
    }

    result = module.extract_register(ast)
    claim = result.records[1]

    assert claim["section_path"] == "UPDE Section"
    assert claim["math_ids"] == ["EQ0001"]
    assert claim["is_claim_candidate"] is True
    assert claim["is_mechanism_candidate"] is True
    assert claim["requires_canonical_review"] is True
    assert {"claim_language", "equation", "mechanism", "topology_or_coupling"}.issubset(
        set(claim["semantic_tags"])
    )


def test_build_coverage_report_marks_exhaustive_block_match() -> None:
    module = _load_module()
    ast = {
        "blocks": [
            {"t": "Header", "c": [1, ["", [], []], [_str("Root")]]},
            {"t": "Para", "c": [{"t": "Math", "c": [{"t": "DisplayMath"}, "a=b"]}]},
        ]
    }

    result = module.extract_register(ast)
    report = module.build_coverage_report(result)

    assert "Top-level AST blocks: `2`" in report
    assert "Exhaustive register records: `2`" in report
    assert "Block coverage status: `match`" in report
    assert "Math nodes: `1`" in report
