# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Native method-comparison CLI entry tests
"""Exercise the public CLI against original files and observe no-overwrite refusals."""

from dataclasses import asdict
from pathlib import Path

import pytest

from scpn_quantum_control.benchmarks.method_comparison_contracts import (
    ComparisonBudget,
    ComparisonMeaning,
    MethodRun,
)
from scpn_quantum_control.studio_workspace.json_transport import read_json, write_json
from tools.export_method_comparison import main


def test_cli_exports_complete_input_without_overwriting_original(tmp_path: Path) -> None:
    """A real source file produces an original native summary through the CLI facade."""
    meaning = ComparisonMeaning(
        "source",
        "metric",
        "1",
        "analytic",
        "float64",
        None,
        "none",
        "fixed",
        "none",
        "none",
        "exact",
    )
    budget = ComparisonBudget(2.0, 3.0, 5.0, 1, 1, 10.0)
    run = MethodRun("original", "reference", "native", 0, meaning, budget, "completed", 3.0)
    text = write_json(
        {
            "schema": "studio.method_comparison_input.v1",
            "runs": [asdict(run)],
            "baseline_method": "reference",
            "missing_data_policy": "require-complete",
            "selected_run_id": None,
        }
    )
    source, output = tmp_path / "input.json", tmp_path / "output.json"
    source.write_text(text)
    args = ["--input", str(source), "--output", str(output)]
    assert main(args) == 0
    original = output.read_bytes()
    assert read_json(original.decode()) is not None
    assert b'"mean":3.0' in original
    assert b'"no_submit":true' in original
    assert main(args) == 1
    assert output.read_bytes() == original
    assert source.read_text() == text


def test_cli_bad_or_missing_input_never_creates_output(tmp_path: Path) -> None:
    """Input/refusal failures leave original files and absent output untouched."""
    source, output = tmp_path / "input.json", tmp_path / "output.json"
    assert main(["--input", str(source), "--output", str(output)]) == 1
    source.write_text('{"best_seed_only":true}')
    assert main(["--input", str(source), "--output", str(output)]) == 1
    assert not output.exists()
    assert source.read_text() == '{"best_seed_only":true}'


def test_original_script_entry_refuses_incomplete_population(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The actual script entry binds CLI arguments and leaves no invalid output."""
    import runpy
    import sys

    source, output = tmp_path / "original.json", tmp_path / "result.json"
    source.write_text('{"best_seed_only":true}')
    script = Path(__file__).resolve().parents[1] / "export_method_comparison.py"
    monkeypatch.setattr(
        sys, "argv", [str(script), "--input", str(source), "--output", str(output)]
    )
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(script), run_name="__main__")
    assert stopped.value.code == 1
    assert not output.exists()


def test_documented_example_retains_all_synthetic_repetitions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The runnable example exports reproducible full populations through the native producer."""
    import runpy

    from examples.method_comparison import main as example

    example()
    public = capsys.readouterr().out
    script = Path(__file__).resolve().parents[2] / "examples/method_comparison.py"
    runpy.run_path(str(script), run_name="__main__")
    assert capsys.readouterr().out == public
    record = read_json(public)
    assert isinstance(record, dict)
    assert b'"population_size":3' in public.encode()
    assert b'"mean":1.0' in public.encode()
    assert b'"mean":2.0' in public.encode()
    assert b'"total_cost_ms":30.0' in public.encode()
