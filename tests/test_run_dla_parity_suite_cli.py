# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — scripts/run_dla_parity_suite.py tests
"""Tests for the DLA-parity validation CLI runner.

Cover:

* Argument parsing — default, ``--verify-integrity``, ``--json``,
  ``--backend``.
* End-to-end run on the real dataset exits 0 and prints the
  summary table.
* ``--json`` emits a parseable JSON document with the expected
  top-level keys.
* Missing data directory exits 3 with a FAIL message on stderr.
* Reproducer breach exits 2 with a FAIL message on stderr.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_dla_parity_suite.py"


def _load_module() -> object:
    spec = importlib.util.spec_from_file_location("run_dla_parity_suite", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cli() -> object:
    return _load_module()


def _has_real_data() -> bool:
    from scpn_quantum_control.dla_parity.dataset import DEFAULT_DATA_DIR, RUN_FILES
    from scpn_quantum_control.dla_parity.reproduce import DEFAULT_PUBLISHED_SUMMARY

    return (
        DEFAULT_DATA_DIR.is_dir()
        and all((DEFAULT_DATA_DIR / f).is_file() for f in RUN_FILES)
        and DEFAULT_PUBLISHED_SUMMARY.is_file()
    )


needs_real_data = pytest.mark.skipif(
    not _has_real_data(),
    reason="DLA-parity dataset or published summary not present",
)


class TestArgParsing:
    def test_default(self, cli: object) -> None:
        ns = cli._parse_args([])  # type: ignore[attr-defined]
        assert ns.data_dir is None
        assert ns.verify_integrity is False
        assert ns.published_summary is None
        assert ns.backend == "auto"
        assert ns.json is False

    def test_verify_integrity_flag(self, cli: object) -> None:
        ns = cli._parse_args(["--verify-integrity"])  # type: ignore[attr-defined]
        assert ns.verify_integrity is True

    def test_json_flag(self, cli: object) -> None:
        ns = cli._parse_args(["--json"])  # type: ignore[attr-defined]
        assert ns.json is True

    def test_backend_choice(self, cli: object) -> None:
        ns = cli._parse_args(["--backend", "numpy"])  # type: ignore[attr-defined]
        assert ns.backend == "numpy"

    def test_data_dir_path(self, cli: object, tmp_path: Path) -> None:
        ns = cli._parse_args(["--data-dir", str(tmp_path)])  # type: ignore[attr-defined]
        assert ns.data_dir == tmp_path


@needs_real_data
class TestMainEndToEnd:
    def test_happy_path_prints_summary(
        self,
        cli: object,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = cli.main([])  # type: ignore[attr-defined]
        captured = capsys.readouterr()
        assert rc == 0
        assert "DLA-parity validation suite" in captured.out
        assert "Fisher" in captured.out

    def test_json_mode_emits_parseable(
        self,
        cli: object,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = cli.main(["--json"])  # type: ignore[attr-defined]
        captured = capsys.readouterr()
        assert rc == 0
        payload = json.loads(captured.out)
        assert payload["n_circuits"] == 342
        assert set(payload["backends"]) == {"ibm_kingston"}
        assert payload["fisher"]["degrees_of_freedom"] == 16
        assert isinstance(payload["depth_summaries"], list)
        assert payload["classical_reference"]["is_zero_within_tolerance"] is True


class TestMainFailurePaths:
    def test_missing_data_dir_exits_3(
        self,
        cli: object,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        missing = tmp_path / "does_not_exist"
        rc = cli.main(["--data-dir", str(missing)])  # type: ignore[attr-defined]
        captured = capsys.readouterr()
        assert rc == 3
        assert "FAIL" in captured.err

    @needs_real_data
    def test_reproducer_breach_exits_2(
        self,
        cli: object,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        # Tamper the published summary so fisher.chi2 mismatches.
        from scpn_quantum_control.dla_parity.reproduce import DEFAULT_PUBLISHED_SUMMARY

        with Path(DEFAULT_PUBLISHED_SUMMARY).open("r", encoding="utf-8") as fh:
            pub = json.load(fh)
        pub["fisher_combined"]["chi2"] = 1.0
        tampered = tmp_path / "phase1_dla_parity_summary.json"
        with tampered.open("w", encoding="utf-8") as fh:
            json.dump(pub, fh)
        rc = cli.main(["--published-summary", str(tampered)])  # type: ignore[attr-defined]
        captured = capsys.readouterr()
        assert rc == 2
        assert "FAIL" in captured.err


def test_script_is_executable_with_main_guard() -> None:
    # Import as __main__ does SystemExit — use runpy-style check on source.
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in src
    assert "raise SystemExit(main())" in src


def test_script_runs_from_command_line_help() -> None:
    # Smoke: the script can be invoked with --help and exits 0.
    import subprocess

    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "DLA-parity" in completed.stdout


def test_unknown_backend_rejected(cli: object) -> None:
    # argparse exits on invalid choice.
    with pytest.raises(SystemExit):
        cli._parse_args(["--backend", "magic"])  # type: ignore[attr-defined]


@needs_real_data
def test_emit_table_covers_formatting(cli: object, capsys: pytest.CaptureFixture[str]) -> None:
    from scpn_quantum_control.dla_parity import run_full_harness

    result = run_full_harness()
    cli._emit_table(result)  # type: ignore[attr-defined]
    captured = capsys.readouterr()
    assert "published claims checked" in captured.out
    assert "classical reference" in captured.out


@needs_real_data
def test_emit_json_payload_matches(cli: object, capsys: pytest.CaptureFixture[str]) -> None:
    from scpn_quantum_control.dla_parity import run_full_harness

    result = run_full_harness()
    cli._emit_json(result)  # type: ignore[attr-defined]
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["n_circuits"] == result.dataset.n_circuits_total


def test_main_respects_baseline_backend_flag(
    cli: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if not _has_real_data():
        pytest.skip("real data not present")
    rc = cli.main(["--backend", "numpy", "--json"])  # type: ignore[attr-defined]
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["classical_reference"]["backend"] == "numpy"
