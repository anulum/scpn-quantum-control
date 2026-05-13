# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Paper 0 canonical review ledger tests
"""Tests for Paper 0 canonical review ledger generation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "canonicalise_paper0_review_ledger.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("canonicalise_paper0_ledger", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 canonical ledger script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_review_ledger_preserves_every_exhaustive_record() -> None:
    module = _load_module()
    records = [
        {
            "record_id": "P0B00001",
            "block_index": 1,
            "block_type": "Header",
            "section_path": "Root",
            "has_text": True,
            "text": "Root",
            "math_ids": [],
            "image_ids": [],
            "table_id": None,
            "semantic_tags": ["context"],
            "is_claim_candidate": False,
            "is_mechanism_candidate": False,
        },
        {
            "record_id": "P0B00002",
            "block_index": 2,
            "block_type": "HorizontalRule",
            "section_path": "Root",
            "has_text": False,
            "text": "",
            "math_ids": [],
            "image_ids": [],
            "table_id": None,
            "semantic_tags": ["structural"],
            "is_claim_candidate": False,
            "is_mechanism_candidate": False,
        },
    ]

    result = module.build_review_ledger(records)

    assert result.summary["source_record_count"] == 2
    assert result.summary["ledger_record_count"] == 2
    assert result.summary["coverage_match"] is True
    assert [entry["source_record_id"] for entry in result.entries] == [
        "P0B00001",
        "P0B00002",
    ]


def test_review_ledger_assigns_controlled_categories_and_next_actions() -> None:
    module = _load_module()
    records = [
        {
            "record_id": "P0B00010",
            "block_index": 10,
            "block_type": "Para",
            "section_path": "UPDE",
            "has_text": True,
            "text": "The mechanism requires validation by experiment.",
            "math_ids": [],
            "image_ids": [],
            "table_id": None,
            "semantic_tags": ["claim_language", "mechanism", "validation"],
            "is_claim_candidate": True,
            "is_mechanism_candidate": True,
        },
        {
            "record_id": "P0B00011",
            "block_index": 11,
            "block_type": "Para",
            "section_path": "UPDE",
            "has_text": True,
            "text": "$d\\theta/dt=\\omega$",
            "math_ids": ["EQ0003"],
            "image_ids": [],
            "table_id": None,
            "semantic_tags": ["equation", "topology_or_coupling"],
            "is_claim_candidate": True,
            "is_mechanism_candidate": True,
        },
    ]

    result = module.build_review_ledger(records)

    assert result.entries[0]["canonical_category"] == "validation_target"
    assert result.entries[0]["promotion_state"] == "requires_domain_review"
    assert result.entries[0]["next_action"] == "map_to_executable_validation_protocol"
    assert result.entries[1]["canonical_category"] == "equation"
    assert result.entries[1]["paper0_equation_record_keys"] == ["upde.base_phase"]
    assert result.entries[1]["next_action"] == "canonicalise_latex_variables_and_units"


def test_review_ledger_summary_counts_categories_and_open_scientific_reviews() -> None:
    module = _load_module()
    records = [
        {
            "record_id": "P0B00001",
            "block_index": 1,
            "block_type": "Para",
            "section_path": "A",
            "has_text": True,
            "text": "plain context",
            "math_ids": [],
            "image_ids": [],
            "table_id": None,
            "semantic_tags": ["context"],
            "is_claim_candidate": False,
            "is_mechanism_candidate": False,
        },
        {
            "record_id": "P0B00002",
            "block_index": 2,
            "block_type": "Para",
            "section_path": "A",
            "has_text": True,
            "text": "claim",
            "math_ids": [],
            "image_ids": [],
            "table_id": None,
            "semantic_tags": ["claim_language"],
            "is_claim_candidate": True,
            "is_mechanism_candidate": False,
        },
    ]

    result = module.build_review_ledger(records)
    report = module.build_review_report(result)

    assert result.summary["category_counts"] == {"claim": 1, "context": 1}
    assert result.summary["requires_domain_review_count"] == 1
    assert "Coverage status: `match`" in report
    assert "- claim: `1`" in report
