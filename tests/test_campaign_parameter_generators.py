# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — campaign parameter generator tests
"""Tests for campaign parameter provenance gates."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_generator(relative_path: str):
    script_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(
        f"{script_path.parent.name}_generate_params_for_test", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("relative_path", "expected_file"),
    [
        (
            "scripts/primary_campaign_2026/generate_params.py",
            "primary_Knm_12x12.npy",
        ),
        (
            "scripts/hardware_campaign_2026/generate_params.py",
            "tokamak_Knm_12x12.npy",
        ),
        (
            "scripts/sophisticated_campaign_2026/generate_params.py",
            "tokamak_Knm_16x16.npy",
        ),
    ],
)
def test_legacy_campaign_generators_fail_closed_without_synthetic_opt_in(
    relative_path: str,
    expected_file: str,
    tmp_path: Path,
) -> None:
    module = _load_generator(relative_path)

    with pytest.raises(RuntimeError, match="Refusing silent synthetic fallback"):
        module.generate_all_params(tmp_path)

    assert not (tmp_path / expected_file).exists()
    assert not (tmp_path / "PARAMETER_PROVENANCE.json").exists()


@pytest.mark.parametrize(
    ("relative_path", "campaign", "expected_file"),
    [
        (
            "scripts/primary_campaign_2026/generate_params.py",
            "primary_campaign_2026",
            "primary_Knm_12x12.npy",
        ),
        (
            "scripts/hardware_campaign_2026/generate_params.py",
            "hardware_campaign_2026",
            "tokamak_Knm_12x12.npy",
        ),
        (
            "scripts/sophisticated_campaign_2026/generate_params.py",
            "sophisticated_campaign_2026",
            "tokamak_Knm_16x16.npy",
        ),
    ],
)
def test_legacy_campaign_generators_emit_labelled_synthetic_smoke_cache(
    relative_path: str,
    campaign: str,
    expected_file: str,
    tmp_path: Path,
) -> None:
    module = _load_generator(relative_path)

    output_path = module.generate_all_params(
        tmp_path,
        allow_synthetic=True,
        seed=123,
    )

    provenance = json.loads((tmp_path / "PARAMETER_PROVENANCE.json").read_text())
    assert output_path == tmp_path
    assert provenance["campaign"] == campaign
    assert provenance["allow_synthetic"] is True
    assert provenance["seed"] == 123
    assert {entry["source_mode"] for entry in provenance["files"]} == {"synthetic"}
    assert (tmp_path / expected_file).exists()


@pytest.mark.parametrize(
    "relative_path",
    [
        "scripts/primary_campaign_2026/run_primary_campaign.sh",
        "scripts/hardware_campaign_2026/run_campaign.sh",
        "scripts/sophisticated_campaign_2026/run_sophisticated_campaign.sh",
    ],
)
def test_legacy_launchers_do_not_generate_parameters_blindly(relative_path: str) -> None:
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    assert "params/PARAMETER_PROVENANCE.json" in text
    assert "python3 generate_params.py\n" not in text


def test_legacy_campaign_scripts_use_campaign_local_paths() -> None:
    campaign_dirs = [
        REPO_ROOT / "scripts" / "primary_campaign_2026",
        REPO_ROOT / "scripts" / "hardware_campaign_2026",
        REPO_ROOT / "scripts" / "sophisticated_campaign_2026",
    ]

    offenders: list[str] = []
    for campaign_dir in campaign_dirs:
        for script_path in sorted(campaign_dir.glob("test_*.py")):
            text = script_path.read_text(encoding="utf-8")
            if 'np.load("params/' in text or 'np.load(f"params/' in text:
                offenders.append(str(script_path.relative_to(REPO_ROOT)))
            if 'open("results/' in text:
                offenders.append(str(script_path.relative_to(REPO_ROOT)))

    assert offenders == []


def test_legacy_campaign_scripts_keep_license_headers_near_top() -> None:
    campaign_dirs = [
        REPO_ROOT / "scripts" / "primary_campaign_2026",
        REPO_ROOT / "scripts" / "hardware_campaign_2026",
        REPO_ROOT / "scripts" / "sophisticated_campaign_2026",
    ]

    missing_top_headers: list[str] = []
    trailing_headers: list[str] = []
    for campaign_dir in campaign_dirs:
        for script_path in sorted(
            list(campaign_dir.glob("*.py")) + list(campaign_dir.glob("*.sh"))
        ):
            text = script_path.read_text(encoding="utf-8")
            first_lines = "\n".join(text.splitlines()[:12])
            last_lines = "\n".join(text.splitlines()[-12:])

            if "SPDX-License-Identifier" not in first_lines:
                missing_top_headers.append(str(script_path.relative_to(REPO_ROOT)))
            if (
                "SPDX-License-Identifier" in last_lines
                and "SPDX-License-Identifier" not in first_lines
            ):
                trailing_headers.append(str(script_path.relative_to(REPO_ROOT)))

    assert missing_top_headers == []
    assert trailing_headers == []
