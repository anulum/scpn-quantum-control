# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable notebook curriculum tests
"""Real-surface tests for the differentiable notebook curriculum registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

import scpn_quantum_control.differentiable_notebook_curriculum as curriculum
from scpn_quantum_control.differentiable_notebook_curriculum import (
    DIFFERENTIABLE_NOTEBOOK_CURRICULUM_CLAIM_BOUNDARY,
    DIFFERENTIABLE_NOTEBOOK_CURRICULUM_SCHEMA,
    CurriculumNotebookRow,
    MaterialisedCurriculumProbe,
    PathEligibilityDecision,
    assert_differentiable_curriculum_integrity,
    build_differentiable_curriculum_registry,
    decide_differentiable_curriculum_path,
    get_curriculum_notebook,
    iter_curriculum_notebooks,
    list_curriculum_notebook_ids,
    map_differentiable_curriculum_public_surfaces,
    materialise_curriculum_probe,
    resolve_curriculum_directory,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _registry_notebooks(registry: dict[str, object]) -> list[dict[str, object]]:
    """Narrow a validated registry notebook collection for drift fixtures."""
    raw = registry["notebooks"]
    assert isinstance(raw, list)
    return cast(list[dict[str, object]], raw)


def test_list_and_filters() -> None:
    """Expose the ordered core-six catalogue and runtime-class filters."""
    ids = list_curriculum_notebook_ids()
    assert len(ids) == 6
    assert ids[0] == "01_parameter_shift_kuramoto_xy"
    assert "05_fail_closed_boundaries" in ids
    assert ids == list_curriculum_notebook_ids()
    fast = iter_curriculum_notebooks(runtime_class="cpu_local_fast")
    assert fast
    assert all(row.runtime_class == "cpu_local_fast" for row in fast)
    optional = iter_curriculum_notebooks(runtime_class="cpu_optional_framework")
    assert len(optional) == 2


def test_get_known_and_unknown_fail_closed() -> None:
    """Resolve known notebooks while rejecting blank and unknown ids."""
    row = get_curriculum_notebook("01_parameter_shift_kuramoto_xy")
    assert row.hardware_execution is False
    assert row.claim_boundary == DIFFERENTIABLE_NOTEBOOK_CURRICULUM_CLAIM_BOUNDARY
    with pytest.raises(ValueError, match="non-empty"):
        get_curriculum_notebook("  ")
    with pytest.raises(ValueError, match="unknown notebook_id"):
        get_curriculum_notebook("not_a_notebook")


def test_path_eligibility_refuse_and_allow() -> None:
    """Allow the bounded curriculum and refuse hardware or archive expansion."""
    allowed = decide_differentiable_curriculum_path()
    assert allowed.allowed is True
    assert allowed.outcome == "allowed"

    hw = decide_differentiable_curriculum_path(request_hardware_execution=True)
    assert hw.allowed is False
    assert any("qpu" in b.lower() or "hardware" in b.lower() for b in hw.blockers)

    archive = decide_differentiable_curriculum_path(request_full_archive_conversion=True)
    assert archive.allowed is False
    assert any("archive" in b.lower() for b in archive.blockers)


def test_materialise_curriculum_probe_and_paths() -> None:
    """Materialise the core-six manifest and verify repository paths."""
    probe = materialise_curriculum_probe(repo_root=_REPO_ROOT)
    assert probe.notebook_count == 6
    assert probe.hardware_execution_any is False
    assert probe.default_notebook_id == "01_parameter_shift_kuramoto_xy"
    assert probe.missing_path_count == 0
    assert len(probe.notebook_ids) == 6
    curriculum_dir = resolve_curriculum_directory(_REPO_ROOT)
    assert curriculum_dir.is_dir()
    for notebook_id in probe.notebook_ids:
        row = get_curriculum_notebook(notebook_id)
        assert (_REPO_ROOT / row.relative_path).is_file()
    payload = probe.to_dict()
    assert payload["notebook_count"] == 6


def test_public_surfaces_and_registry() -> None:
    """Map the public owner and validate the complete curriculum registry."""
    surfaces = map_differentiable_curriculum_public_surfaces()
    assert surfaces
    assert surfaces[0]["hardware_execution"] is False
    assert surfaces[0]["curriculum_dir"] == "notebooks/differentiable"

    registry = build_differentiable_curriculum_registry()
    assert registry["schema"] == DIFFERENTIABLE_NOTEBOOK_CURRICULUM_SCHEMA
    assert registry["blank_entry_count"] == 0
    assert registry["hardware_execution_policy"] is False
    validated = assert_differentiable_curriculum_integrity(registry)
    assert validated["notebook_count"] == 6
    assert assert_differentiable_curriculum_integrity()["blank_entry_count"] == 0


def test_integrity_rejects_drift_and_hardware() -> None:
    """Reject notebook-set drift and hardware-execution relaxation."""
    registry = build_differentiable_curriculum_registry()
    notebooks = _registry_notebooks(registry)

    broken = dict(registry)
    broken["notebooks"] = notebooks + [
        {
            "notebook_id": "ghost",
            "title": "t",
            "relative_path": "notebooks/differentiable/ghost.ipynb",
            "runtime_class": "cpu_local_fast",
            "summary": "s",
            "required_packages": ["numpy"],
            "hardware_execution": False,
            "order": 99,
            "as_of": "2026-07-24",
            "claim_boundary": DIFFERENTIABLE_NOTEBOOK_CURRICULUM_CLAIM_BOUNDARY,
        }
    ]
    broken["notebook_count"] = len(cast(list[object], broken["notebooks"]))
    with pytest.raises(ValueError, match="drift"):
        assert_differentiable_curriculum_integrity(broken)

    empty = {**registry, "notebooks": [], "notebook_count": 0}
    with pytest.raises(ValueError, match="non-empty notebooks"):
        assert_differentiable_curriculum_integrity(empty)

    hw = dict(registry)
    hw_rows = [dict(row) for row in notebooks]
    hw_rows[0]["hardware_execution"] = True
    hw["notebooks"] = hw_rows
    with pytest.raises(ValueError, match="hardware_execution|invent-green"):
        assert_differentiable_curriculum_integrity(hw)

    policy = dict(registry)
    policy["hardware_execution_policy"] = True
    with pytest.raises(ValueError, match="hardware_execution_policy"):
        assert_differentiable_curriculum_integrity(policy)


def test_integrity_rejects_blank_invalid() -> None:
    """Reject malformed rows, missing defaults, duplicates, and count drift."""
    registry = build_differentiable_curriculum_registry()
    notebooks = _registry_notebooks(registry)

    non_map = dict(registry)
    non_map["notebooks"] = [cast(Any, "nope")]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_differentiable_curriculum_integrity(non_map)

    blank_id = dict(registry)
    rows = [dict(row) for row in notebooks]
    rows[0]["notebook_id"] = "  "
    blank_id["notebooks"] = rows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_differentiable_curriculum_integrity(blank_id)

    bad_rt = dict(registry)
    rrows = [dict(row) for row in notebooks]
    rrows[1]["runtime_class"] = "nope"
    bad_rt["notebooks"] = rrows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_differentiable_curriculum_integrity(bad_rt)

    no_path = dict(registry)
    prows = [dict(row) for row in notebooks]
    prows[0]["relative_path"] = ""
    no_path["notebooks"] = prows
    with pytest.raises(ValueError, match="relative_path"):
        assert_differentiable_curriculum_integrity(no_path)

    no_default = dict(registry)
    renamed = [dict(row) for row in notebooks]
    for row in renamed:
        if row.get("notebook_id") == "01_parameter_shift_kuramoto_xy":
            row["notebook_id"] = "renamed"
    no_default["notebooks"] = renamed
    with pytest.raises(ValueError, match="missing 01_parameter_shift|drift"):
        assert_differentiable_curriculum_integrity(no_default)

    dup = dict(registry)
    drows = [dict(row) for row in notebooks]
    drows.append(dict(drows[0]))
    dup["notebooks"] = drows
    dup["notebook_count"] = len(drows)
    with pytest.raises(ValueError, match="duplicate notebook_id"):
        assert_differentiable_curriculum_integrity(dup)

    blank_count = dict(registry)
    blank_count["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_differentiable_curriculum_integrity(blank_count)

    count_mismatch = dict(registry)
    count_mismatch["notebook_count"] = 0
    with pytest.raises(ValueError, match="notebook_count"):
        assert_differentiable_curriculum_integrity(count_mismatch)


def test_integrity_rejects_stale_contract_metadata() -> None:
    """Reject stale schemas and altered claim boundaries without aliases."""
    registry = build_differentiable_curriculum_registry()
    stale_schema = dict(registry)
    stale_schema["schema"] = "notebook_programme_product.v1"
    with pytest.raises(ValueError, match="schema"):
        assert_differentiable_curriculum_integrity(stale_schema)

    altered_claim = dict(registry)
    altered_claim["claim_boundary"] = "broader claim"
    with pytest.raises(ValueError, match="claim boundary"):
        assert_differentiable_curriculum_integrity(altered_claim)


def test_module_exports() -> None:
    """Keep every documented notebook product entry point public."""
    assert "materialise_curriculum_probe" in curriculum.__all__
    assert "list_curriculum_notebook_ids" in curriculum.__all__
    assert "decide_differentiable_curriculum_path" in curriculum.__all__


def test_row_decision_probe_validation() -> None:
    """Enforce curriculum row, eligibility decision, and probe invariants."""
    base: dict[str, Any] = {
        "notebook_id": "x",
        "title": "t",
        "relative_path": "notebooks/differentiable/x.ipynb",
        "runtime_class": "cpu_local_fast",
        "summary": "s",
        "required_packages": ("numpy",),
        "order": 1,
    }
    assert CurriculumNotebookRow(**base).notebook_id == "x"
    with pytest.raises(ValueError, match="notebook_id"):
        CurriculumNotebookRow(**{**base, "notebook_id": ""})
    with pytest.raises(ValueError, match="title"):
        CurriculumNotebookRow(**{**base, "title": ""})
    with pytest.raises(ValueError, match="relative_path"):
        CurriculumNotebookRow(**{**base, "relative_path": ""})
    with pytest.raises(ValueError, match=".ipynb"):
        CurriculumNotebookRow(**{**base, "relative_path": "nope.md"})
    with pytest.raises(ValueError, match="runtime_class"):
        CurriculumNotebookRow(**{**base, "runtime_class": cast(Any, "nope")})
    with pytest.raises(ValueError, match="summary"):
        CurriculumNotebookRow(**{**base, "summary": ""})
    with pytest.raises(ValueError, match="required_packages"):
        CurriculumNotebookRow(**{**base, "required_packages": ()})
    with pytest.raises(ValueError, match="required_packages entries"):
        CurriculumNotebookRow(**{**base, "required_packages": ("",)})
    with pytest.raises(ValueError, match="hardware_execution"):
        CurriculumNotebookRow(**{**base, "hardware_execution": True})
    with pytest.raises(ValueError, match="order"):
        CurriculumNotebookRow(**{**base, "order": 0})
    with pytest.raises(ValueError, match="as_of"):
        CurriculumNotebookRow(**{**base, "as_of": ""})

    with pytest.raises(ValueError, match="outcome"):
        PathEligibilityDecision(
            outcome=cast(Any, "nope"),
            allowed=False,
            reason="r",
            blockers=("b",),
        )
    with pytest.raises(ValueError, match="reason"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="",
            blockers=("b",),
        )
    with pytest.raises(ValueError, match="outcome=allowed"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=True,
            reason="r",
            blockers=(),
        )
    with pytest.raises(ValueError, match="outcome=refused"):
        PathEligibilityDecision(
            outcome="allowed",
            allowed=False,
            reason="r",
            blockers=("b",),
        )
    with pytest.raises(ValueError, match="cannot list blockers"):
        PathEligibilityDecision(
            outcome="allowed",
            allowed=True,
            reason="r",
            blockers=("x",),
        )
    with pytest.raises(ValueError, match="require blockers"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=(),
        )
    with pytest.raises(ValueError, match="blockers entries"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=("",),
        )
    assert decide_differentiable_curriculum_path().to_dict()["allowed"] is True

    with pytest.raises(ValueError, match="notebook_ids must be non-empty"):
        MaterialisedCurriculumProbe(
            notebook_ids=(),
            notebook_count=0,
            hardware_execution_any=False,
            default_notebook_id="x",
            missing_path_count=0,
        )
    with pytest.raises(ValueError, match="notebook_count must match"):
        MaterialisedCurriculumProbe(
            notebook_ids=("a",),
            notebook_count=2,
            hardware_execution_any=False,
            default_notebook_id="a",
            missing_path_count=0,
        )
    with pytest.raises(ValueError, match="hardware_execution"):
        MaterialisedCurriculumProbe(
            notebook_ids=("a",),
            notebook_count=1,
            hardware_execution_any=True,
            default_notebook_id="a",
            missing_path_count=0,
        )
    with pytest.raises(ValueError, match="default_notebook_id"):
        MaterialisedCurriculumProbe(
            notebook_ids=("a",),
            notebook_count=1,
            hardware_execution_any=False,
            default_notebook_id="",
            missing_path_count=0,
        )
    with pytest.raises(ValueError, match="present in notebook_ids"):
        MaterialisedCurriculumProbe(
            notebook_ids=("a",),
            notebook_count=1,
            hardware_execution_any=False,
            default_notebook_id="b",
            missing_path_count=0,
        )
    with pytest.raises(ValueError, match="missing_path_count"):
        MaterialisedCurriculumProbe(
            notebook_ids=("a",),
            notebook_count=1,
            hardware_execution_any=False,
            default_notebook_id="a",
            missing_path_count=-1,
        )


def test_probe_refused_when_path_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop curriculum materialisation when path policy refuses it."""

    def _refuse(**_kwargs: Any) -> PathEligibilityDecision:
        return PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="forced refuse",
            blockers=("forced",),
        )

    monkeypatch.setattr(curriculum, "decide_differentiable_curriculum_path", _refuse)
    with pytest.raises(ValueError, match="refused"):
        materialise_curriculum_probe(repo_root=_REPO_ROOT)


def test_missing_paths_counted(tmp_path: Path) -> None:
    """Count every absent core-six notebook under an empty repository root."""
    probe = materialise_curriculum_probe(repo_root=tmp_path)
    assert probe.missing_path_count == 6
    assert resolve_curriculum_directory(tmp_path).name == "differentiable"


def test_iter_curriculum_without_runtime_filter() -> None:
    """Unfiltered curriculum iter returns the full catalogue."""
    all_rows = iter_curriculum_notebooks()
    assert len(all_rows) == len(list_curriculum_notebook_ids())
    assert {row.notebook_id for row in all_rows} == set(list_curriculum_notebook_ids())


def test_resolve_curriculum_directory_default_repo_root() -> None:
    """Default repo_root resolves to notebooks/differentiable under package parents."""
    resolved = resolve_curriculum_directory()
    assert resolved.name == "differentiable"
    assert resolved.parent.name == "notebooks"
    # Package lives under src/scpn_quantum_control; parents[2] is the repo root.
    assert resolved.parent.parent == _REPO_ROOT.resolve()


def test_catalogue_map_rejects_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_catalogue_map`` refuses an empty curriculum catalogue."""
    monkeypatch.setattr(curriculum, "_CANONICAL_CURRICULUM", ())
    with pytest.raises(RuntimeError, match="catalogue must be non-empty"):
        curriculum._catalogue_map()


def test_catalogue_map_rejects_blank_notebook_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_catalogue_map`` refuses a blank notebook_id after construction."""
    from dataclasses import replace

    blank = replace(get_curriculum_notebook(list_curriculum_notebook_ids()[0]))
    object.__setattr__(blank, "notebook_id", "  ")
    monkeypatch.setattr(curriculum, "_CANONICAL_CURRICULUM", (blank,))
    with pytest.raises(RuntimeError, match="blank notebook_id"):
        curriculum._catalogue_map()


def test_catalogue_map_rejects_duplicate_notebook_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_catalogue_map`` refuses duplicate notebook identifiers."""
    from dataclasses import replace

    good = replace(get_curriculum_notebook(list_curriculum_notebook_ids()[0]))
    monkeypatch.setattr(
        curriculum,
        "_CANONICAL_CURRICULUM",
        (good, good),
    )
    with pytest.raises(RuntimeError, match="duplicate notebook_id"):
        curriculum._catalogue_map()
