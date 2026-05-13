#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Paper 0 exhaustive register extractor
"""Build an exhaustive block-level register from the Paper 0 Pandoc AST."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_AST_PATH = DEFAULT_EXTRACTION_DIR / "paper0_pandoc_ast.json"

CLAIM_LANGUAGE_RE = re.compile(
    r"\b("
    r"posits?|asserts?|claims?|predicts?|requires?|implies?|therefore|must|"
    r"defines?|formalises?|governs?|is governed by|is equivalent to|"
    r"is proportional to|leads to|results in|stabili[sz]es?|couples?|"
    r"modulates?|optimises?|minimi[sz]es?|maximi[sz]es?|falsifiable|"
    r"testable|prediction|theorem|lemma|proof|hypothesis|postulate|axiom"
    r")\b",
    re.IGNORECASE,
)
MECHANISM_RE = re.compile(
    r"\b("
    r"mechanism|coupling|transduction|feedback|projection|aggregation|"
    r"synchroni[sz]ation|phase-lock|phase lock|controller|stabilisation|"
    r"stabilization|modulation|Hamiltonian|Lagrangian|UPDE|FIM|QEC|SOC"
    r")\b",
    re.IGNORECASE,
)
SYMBOLIC_RE = re.compile(
    r"(=|\\frac|\\sum|\\int|\\dot|\\nabla|\\propto|\\approx|"
    r"\bsin\b|\bcos\b|\bUPDE\b|\bFIM\b|K_\{?ij|Kij|K_nm|"
    r"\bsigma\b|\btheta\b|Psi|Phi|SEC|QEC)",
    re.IGNORECASE,
)
VALIDATION_RE = re.compile(
    r"\b(falsifi|validat|experiment|prediction|test|observable|measure|"
    r"protocol|null control|control)\b",
    re.IGNORECASE,
)
TOPOLOGY_RE = re.compile(
    r"\b(topolog|graph|matrix|K_nm|Kij|K_\{?ij|coupling|edge|layer|"
    r"inter-layer|intra-layer)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class RegisterExtraction:
    """In-memory result of exhaustive register extraction."""

    records: list[dict[str, Any]]
    summary: dict[str, Any]


def _inline_text(inlines: list[Any]) -> str:
    parts: list[str] = []
    for item in inlines:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        node_type = item.get("t")
        content = item.get("c")
        if node_type == "Str":
            parts.append(str(content))
        elif node_type in {"Space", "SoftBreak", "LineBreak"}:
            parts.append(" ")
        elif node_type == "Math":
            parts.append(f"${content[1]}$")
        elif node_type == "Image":
            parts.append(f"[IMAGE:{_inline_text(content[1])}]")
        elif node_type == "Link":
            parts.append(_inline_text(content[1] if len(content) > 1 else []))
        elif isinstance(content, list):
            parts.append(_inline_text(content))
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _block_text(block: dict[str, Any]) -> str:
    block_type = block.get("t")
    content = block.get("c")
    if block_type in {"Para", "Plain"}:
        return _inline_text(content)
    if block_type == "Header":
        return _inline_text(content[2])
    if block_type == "BlockQuote":
        return " ".join(_block_text(child) for child in content).strip()
    if block_type in {"BulletList", "OrderedList"}:
        items = content[1] if block_type == "OrderedList" else content
        rendered_items = [" ".join(_block_text(child) for child in item).strip() for item in items]
        return " | ".join(item for item in rendered_items if item)
    if block_type == "Table":
        return "[TABLE]"
    if block_type == "HorizontalRule":
        return ""
    return ""


def _walk_node(node: Any, visitor: Any) -> None:
    if isinstance(node, dict):
        visitor(node)
        for value in node.values():
            _walk_node(value, visitor)
    elif isinstance(node, list):
        for value in node:
            _walk_node(value, visitor)


def _collect_block_inline_ids(
    block: dict[str, Any],
    *,
    equation_start: int,
    image_start: int,
) -> tuple[list[str], list[str]]:
    """Collect local equation and image ids from a block."""
    math_ids: list[str] = []
    image_ids: list[str] = []

    def visitor(node: dict[str, Any]) -> None:
        node_type = node.get("t")
        if node_type == "Math":
            math_ids.append(f"EQ{equation_start + len(math_ids) + 1:04d}")
        elif node_type == "Image":
            image_ids.append(f"IMG{image_start + len(image_ids) + 1:04d}")

    _walk_node(block, visitor)
    return math_ids, image_ids


def _semantic_tags(text: str, *, math_count: int, image_count: int, table_count: int) -> list[str]:
    tags: set[str] = set()
    if CLAIM_LANGUAGE_RE.search(text):
        tags.add("claim_language")
    if MECHANISM_RE.search(text):
        tags.add("mechanism")
    if SYMBOLIC_RE.search(text) or math_count:
        tags.add("equation")
    if VALIDATION_RE.search(text):
        tags.add("validation")
    if TOPOLOGY_RE.search(text):
        tags.add("topology_or_coupling")
    if image_count:
        tags.add("figure_or_media")
    if table_count:
        tags.add("table")
    if not tags and text:
        tags.add("context")
    if not tags:
        tags.add("structural")
    return sorted(tags)


def extract_register(ast: dict[str, Any]) -> RegisterExtraction:
    """Extract one reviewable register record for every top-level AST block."""
    blocks = ast.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError("Pandoc AST must contain a top-level blocks list")

    records: list[dict[str, Any]] = []
    section_stack: list[str] = []
    equation_counter = 0
    image_counter = 0
    table_counter = 0
    heading_count = 0

    for block_index, block in enumerate(blocks, start=1):
        block_type = str(block.get("t", "Unknown"))
        if block_type == "Header":
            level = int(block["c"][0])
            title = _inline_text(block["c"][2])
            section_stack = section_stack[: level - 1] + [title]
            heading_count += 1

        block_math_ids, block_image_ids = _collect_block_inline_ids(
            block,
            equation_start=equation_counter,
            image_start=image_counter,
        )
        equation_counter += len(block_math_ids)
        image_counter += len(block_image_ids)
        if block_type == "Table":
            table_counter += 1

        text = _block_text(block)
        tags = _semantic_tags(
            text,
            math_count=len(block_math_ids),
            image_count=len(block_image_ids),
            table_count=1 if block_type == "Table" else 0,
        )
        is_claim_candidate = bool({"claim_language", "equation"} & set(tags))
        is_mechanism_candidate = "mechanism" in tags or "topology_or_coupling" in tags
        records.append(
            {
                "record_id": f"P0B{block_index:05d}",
                "block_index": block_index,
                "block_type": block_type,
                "section_path": " > ".join(section_stack),
                "has_text": bool(text),
                "text": text,
                "math_ids": block_math_ids,
                "image_ids": block_image_ids,
                "table_id": f"TBL{table_counter:03d}" if block_type == "Table" else None,
                "semantic_tags": tags,
                "is_claim_candidate": is_claim_candidate,
                "is_mechanism_candidate": is_mechanism_candidate,
                "requires_canonical_review": is_claim_candidate or is_mechanism_candidate,
                "review_status": "unreviewed",
            }
        )

    summary = {
        "top_level_ast_blocks": len(blocks),
        "exhaustive_register_records": len(records),
        "block_coverage_match": len(blocks) == len(records),
        "heading_records": heading_count,
        "math_nodes": equation_counter,
        "image_nodes": image_counter,
        "table_nodes": table_counter,
        "textual_records": sum(1 for record in records if record["has_text"]),
        "claim_candidate_records": sum(1 for record in records if record["is_claim_candidate"]),
        "mechanism_candidate_records": sum(
            1 for record in records if record["is_mechanism_candidate"]
        ),
        "canonical_review_status": "unreviewed",
    }
    return RegisterExtraction(records=records, summary=summary)


def build_coverage_report(result: RegisterExtraction) -> str:
    """Render a concise extraction coverage report."""
    status = "match" if result.summary["block_coverage_match"] else "mismatch"
    return "\n".join(
        [
            "# Paper 0 Exhaustive Register Coverage",
            "",
            f"- Top-level AST blocks: `{result.summary['top_level_ast_blocks']}`",
            f"- Exhaustive register records: `{result.summary['exhaustive_register_records']}`",
            f"- Block coverage status: `{status}`",
            f"- Textual records: `{result.summary['textual_records']}`",
            f"- Heading records: `{result.summary['heading_records']}`",
            f"- Math nodes: `{result.summary['math_nodes']}`",
            f"- Image nodes: `{result.summary['image_nodes']}`",
            f"- Table nodes: `{result.summary['table_nodes']}`",
            f"- Claim-candidate records: `{result.summary['claim_candidate_records']}`",
            f"- Mechanism-candidate records: `{result.summary['mechanism_candidate_records']}`",
            f"- Canonical review status: `{result.summary['canonical_review_status']}`",
            "",
            "Every top-level Pandoc AST block has a register record. Candidate",
            "labels are triage labels only; they do not promote a block to a",
            "validated claim or theorem until canonical review is complete.",
            "",
        ]
    )


def write_outputs(
    result: RegisterExtraction,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write exhaustive register, summary, and coverage report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    register_path = output_dir / f"paper0_exhaustive_register_{date_tag}.jsonl"
    summary_path = output_dir / f"paper0_exhaustive_register_summary_{date_tag}.json"
    report_path = output_dir / f"paper0_exhaustive_register_coverage_{date_tag}.md"
    register_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in result.records) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(result.summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(build_coverage_report(result), encoding="utf-8")
    return {
        "register": register_path,
        "summary": summary_path,
        "coverage_report": report_path,
    }


def main() -> int:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ast", type=Path, default=DEFAULT_AST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args()

    ast = json.loads(args.ast.read_text(encoding="utf-8"))
    result = extract_register(ast)
    paths = write_outputs(result, output_dir=args.output_dir, date_tag=args.date_tag)
    for key, path in paths.items():
        print(f"wrote_{key}={path}")
    print(f"exhaustive_register_records={result.summary['exhaustive_register_records']}")
    print(f"block_coverage_match={result.summary['block_coverage_match']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
