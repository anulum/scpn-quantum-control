# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — existing provider observation CLI tests
"""Exercise the real CLI entry point on an original locally sampled durable job."""

import asyncio
import json
import runpy
from pathlib import Path
from uuid import uuid4

import pytest
from qiskit import QuantumCircuit

from scpn_quantum_control.hardware.async_runner import AsyncHardwareRunner
from scpn_quantum_control.hardware.provider_job_journal import ProviderJobJournal
from scpn_quantum_control.hardware.runner import HardwareRunner
from tools.export_provider_observations import main


def test_cli_entry_exports_actual_native_job_and_preserves_existing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The original file entry exports once, then refuses overwrite and unknown identities.

    Parameters
    ----------
    tmp_path
        Owned real native Aer results, durable journal and export files.
    monkeypatch
        Set only the actual CLI argument vector; native execution is unchanged.
    capsys
        Capture original CLI diagnostics without replacing transport or result calls.

    """
    runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "results"))
    runner.connect()
    facade = AsyncHardwareRunner(runner)
    journal_root = tmp_path / "journal"
    journal = ProviderJobJournal(journal_root)
    identity = str(uuid4())
    circuit = QuantumCircuit(1)
    circuit.x(0)
    circuit.measure_all()

    async def execute() -> str:
        """Create one genuine local SDK job before invoking the read-only CLI."""
        handle = await facade.submit_one_async(
            [circuit], shots=16, journal=journal, attempt_id=identity
        )
        result = await facade.wait_for_job_async(handle)
        assert result[0].counts == {"1": 16}
        return handle.job_id

    job_id = asyncio.run(execute())
    before = journal.snapshot(identity)
    output = tmp_path / "observation.json"
    args = ["--journal", str(journal_root), "--attempt-id", identity, "--output", str(output)]
    script = Path(__file__).resolve().parents[1] / "tools/export_provider_observations.py"
    monkeypatch.setattr("sys.argv", [str(script), *args])
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(script), run_name="__main__")
    assert exited.value.code == 0
    original = output.read_bytes()
    body = json.loads(original)["body"]
    assert body["provider_job_id"] == job_id
    assert body["events"][-1]["observation"]["results"][0]["counts"] == {"1": 16}
    assert main(args) == 1
    assert "export refused" in capsys.readouterr().err
    assert output.read_bytes() == original
    absent = tmp_path / "absent.json"
    assert (
        main(
            ["--journal", str(journal_root), "--attempt-id", str(uuid4()), "--output", str(absent)]
        )
        == 1
    )
    assert not absent.exists()
    assert journal.snapshot(identity) == before


def test_cli_refuses_a_missing_journal_without_creating_state(tmp_path: Path) -> None:
    """An absent read target cannot become a prepared request or a new database.

    Parameters
    ----------
    tmp_path
        Owned parent of nonexistent input and output paths.

    """
    root = tmp_path / "missing"
    output = tmp_path / "observation.json"
    assert (
        main(["--journal", str(root), "--attempt-id", str(uuid4()), "--output", str(output)]) == 1
    )
    assert not root.exists()
    assert not output.exists()
