# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 computational-unifier spec tests
"""Tests for Paper 0 EQ0115-EQ0118 validation spec promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_information_thermodynamics_validation_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_information_thermodynamics_validation_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_paper0_info_thermo_specs", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 information-thermodynamics spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(
    ledger_id: str,
    block_index: int,
    *,
    math_ids: list[str] | None = None,
    text: str = "source text",
) -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Paper 0 information thermodynamics test section",
        "math_ids": [] if math_ids is None else math_ids,
        "text": text,
    }


def _complete_records() -> list[dict[str, object]]:
    return [
        _record("P0R05929", 5929),
        _record("P0R05930", 5930),
        _record("P0R05931", 5931),
        _record("P0R05932", 5932, math_ids=["EQ0115"]),
        _record("P0R05933", 5933),
        _record("P0R05934", 5934),
        _record("P0R05935", 5935),
        _record("P0R05936", 5936),
        _record("P0R05937", 5937),
        _record("P0R05938", 5938),
        _record("P0R05939", 5939, math_ids=["EQ0116"]),
        _record("P0R05940", 5940),
        _record("P0R05942", 5942),
        _record("P0R05943", 5943),
        _record("P0R05944", 5944),
        _record("P0R05945", 5945),
        _record("P0R05946", 5946, math_ids=["EQ0117"]),
        _record("P0R05947", 5947),
        _record("P0R05949", 5949),
        _record("P0R05950", 5950),
        _record("P0R05951", 5951, math_ids=["EQ0118"]),
        _record("P0R05952", 5952),
        _record("P0R05953", 5953),
    ]


def test_information_thermodynamics_spec_promotes_eq0117_eq0118() -> None:
    module = _load_module()

    bundle = module.build_information_thermodynamics_validation_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 23
    assert bundle.summary["consumed_source_record_count"] == 23
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 3
    assert bundle.summary["spec_keys"] == [
        "computational.cyclic_operator_boundary",
        "computational.tsvf_abl_boundary",
        "computational.info_thermodynamics",
    ]
    assert bundle.summary["hardware_status"] == "simulator_only_no_provider_submission"


def test_information_thermodynamics_spec_has_controls_and_falsifiers() -> None:
    module = _load_module()

    bundle = module.build_information_thermodynamics_validation_specs(_complete_records())
    by_key = {spec.key: spec for spec in bundle.specs}

    cyclic = by_key["computational.cyclic_operator_boundary"]
    assert cyclic.source_equation_ids == ("EQ0115",)
    assert cyclic.anchor_math_ids == ("EQ0115",)
    assert any("periodicity" in target for target in cyclic.executable_validation_targets)
    assert any("boundary-only" in target for target in cyclic.validation_targets)

    tsvf = by_key["computational.tsvf_abl_boundary"]
    assert tsvf.source_equation_ids == ("EQ0116",)
    assert tsvf.anchor_math_ids == ("EQ0116",)
    assert any("normalisation" in target for target in tsvf.executable_validation_targets)
    assert any("boundary-only" in control for control in tsvf.null_controls)

    spec = by_key["computational.info_thermodynamics"]
    assert spec.key == "computational.info_thermodynamics"
    assert spec.source_equation_ids == ("EQ0117", "EQ0118")
    assert set(spec.anchor_math_ids) == {"EQ0117", "EQ0118"}
    assert "P0R05946" in spec.source_ledger_ids
    assert "P0R05951" in spec.source_ledger_ids
    assert any("GSL" in target for target in spec.validation_targets)
    assert any("Landauer" in control for control in spec.null_controls)
    assert any("mutual information" in target for target in spec.executable_validation_targets)
    assert spec.hardware_status == "simulator_only_no_provider_submission"


def test_information_thermodynamics_builder_rejects_missing_required_anchor() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R05939"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_information_thermodynamics_validation_specs(incomplete)


def test_write_outputs_records_policy_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_information_thermodynamics_validation_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_information_thermodynamics_validation_spec(
        "computational.info_thermodynamics",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert payload["summary"]["hardware_status"] == "simulator_only_no_provider_submission"
    assert loaded["source_equation_ids"] == ["EQ0117", "EQ0118"]
    assert "Paper 0 Information-Thermodynamics Validation Specs" in report
    assert "Provider submission remains out of scope" in report
    assert "computational.cyclic_operator_boundary" in report
