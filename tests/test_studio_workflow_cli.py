# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — checkpointed original workflow CLI public tests
"""Exercise the actual module CLI and atomic original checkpoint recovery."""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
import select
import signal
import struct
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import cast

import pytest

from scpn_quantum_control.studio.workflow_cli import run
from scpn_quantum_control.studio.workflow_contracts import MAX_WORKFLOW_BYTES
from scpn_quantum_control.studio.workflow_journal import MAX_JOURNAL_BYTES
from scpn_quantum_control.studio_workspace.canonical import canonical_bytes
from scpn_quantum_control.studio_workspace.json_transport import read_json, write_json


def source_file(directory: Path) -> Path:
    """Write the exact shared compiler graph to the caller-owned test allocation.

    Parameters
    ----------
    directory
        Existing owned pytest allocation.

    Returns
    -------
    Path
        Complete portable original compiler graph file.

    """
    corpus = cast(
        dict[str, object],
        read_json(
            (Path(__file__).parent / "data/studio_workflow/contract_cases.json").read_text()
        ),
    )
    path = directory / "workflow.json"
    path.write_text(write_json(corpus["workflow"]))
    return path


def watch_checkpoint(path: Path, mask: int) -> int:
    """Subscribe to actual Linux checkpoint events without changing the producer.

    Parameters
    ----------
    path
        Actual checkpoint file or its containing directory.
    mask
        Native inotify event mask for the observed filesystem operation.

    Returns
    -------
    int
        Owned nonblocking native event descriptor; the caller closes it.

    """
    libc = ctypes.CDLL(None, use_errno=True)
    libc.inotify_init1.argtypes = [ctypes.c_int]
    libc.inotify_init1.restype = ctypes.c_int
    libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    libc.inotify_add_watch.restype = ctypes.c_int
    descriptor = int(libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC))
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "native checkpoint watcher unavailable")
    if int(libc.inotify_add_watch(descriptor, os.fsencode(path), mask)) < 0:
        os.close(descriptor)
        raise OSError(ctypes.get_errno(), "native checkpoint subscription refused")
    return descriptor


def await_checkpoint_events(descriptor: int, expected: int, name: str = "") -> None:
    """Observe original filesystem events before interfering with the actual CLI.

    Parameters
    ----------
    descriptor
        Owned native inotify descriptor.
    expected
        Number of matching actual events needed at the public checkpoint boundary.
    name
        Optional directory-entry name; a file watch uses its empty native name.

    """
    deadline = time.monotonic() + 20
    observed = 0
    while observed < expected:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([descriptor], [], [], remaining)[0]:
            raise AssertionError("actual checkpoint event was not observed")
        raw = os.read(descriptor, 65536)
        position = 0
        while position < len(raw):
            _, _, _, length = struct.unpack_from("iIII", raw, position)
            entry = os.fsdecode(raw[position + 16 : position + 16 + length].split(b"\0", 1)[0])
            position += 16 + length
            if entry == name:
                observed += 1


def test_actual_module_cli_saves_and_resumes_exact_original_records(tmp_path: Path) -> None:
    """Run the real process CLI twice without duplicating twelve actual compiler records.

    Parameters
    ----------
    tmp_path
        Owned finite temporary allocation from the test harness.

    """
    source = source_file(tmp_path)
    example = Path(__file__).parents[1] / "examples/studio_workflow.py"
    emitted = subprocess.run(
        [sys.executable, str(example)], capture_output=True, text=True, timeout=20, check=False
    )
    assert emitted.returncode == 0, emitted.stderr
    assert canonical_bytes("workflow-example.v1", read_json(emitted.stdout)) == canonical_bytes(
        "workflow-example.v1", read_json(source.read_text())
    )
    source.write_text(emitted.stdout)
    target = tmp_path / "journal.json"
    argv = [
        sys.executable,
        "-m",
        "scpn_quantum_control.studio.workflow_cli",
        str(source),
        "--journal",
        str(target),
    ]
    first = subprocess.run(argv, capture_output=True, text=True, timeout=60, check=False)
    assert first.returncode == 0, first.stderr
    saved = target.read_bytes()
    document = cast(dict[str, object], read_json(first.stdout))
    body = cast(dict[str, object], document["body"])
    assert body["state"] == "complete" and body["evaluations"] == 12
    assert len(cast(list[object], body["entries"])) == 12
    second = subprocess.run(
        [*argv, "--resume"], capture_output=True, text=True, timeout=60, check=False
    )
    assert second.returncode == 0, second.stderr
    assert target.read_bytes() == saved
    assert canonical_bytes("cli-journal.v1", read_json(second.stdout)) == canonical_bytes(
        "cli-journal.v1", document
    )
    assert not list(tmp_path.glob("*.partial"))
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert (
        pyproject["project"]["scripts"]["scpn-studio-workflow"]
        == "scpn_quantum_control.studio.workflow_cli:main"
    )


def test_registered_dispatcher_keeps_actual_compiler_checkpoint_on_resume(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exercise the public dispatcher and atomic saves with actual executive producers.

    Parameters
    ----------
    tmp_path
        Owned allocation for the exact shared graph and its actual journal.
    capsys
        Original caller output capture, including the complete producer journal.

    """
    source = source_file(tmp_path)
    target = tmp_path / "journal.json"
    arguments = [str(source), "--journal", str(target)]
    assert run(arguments) == 0
    first = capsys.readouterr()
    assert first.err == ""
    document = cast(dict[str, object], read_json(first.out))
    body = cast(dict[str, object], document["body"])
    entries = cast(list[dict[str, object]], body["entries"])
    assert body["state"] == "complete" and body["evaluations"] == 12
    assert len(entries) == 12 and all(entry["status"] == "complete" for entry in entries)
    for entry in entries:
        output = cast(dict[str, object], entry["output"])
        result = cast(dict[str, object], output["result"])
        outputs = cast(dict[str, object], result["outputs"])
        assert outputs["execution_status"] == "emitted_not_executed"
    saved = target.read_bytes()
    assert run([*arguments, "--resume"]) == 0
    resumed = capsys.readouterr()
    assert resumed.err == "" and target.read_bytes() == saved
    assert canonical_bytes("cli-journal.v1", read_json(resumed.out)) == canonical_bytes(
        "cli-journal.v1", document
    )
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize(
    "fault", ["malformed", "future", "missing", "same-file", "existing", "resume-missing"]
)
def test_refused_cli_keeps_source_and_prior_checkpoint_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fault: str
) -> None:
    """Refuse invalid source/storage combinations without overwriting original evidence.

    Parameters
    ----------
    tmp_path
        Owned finite temporary allocation.
    capsys
        Actual CLI caller output capture.
    fault
        Concrete malformed input or incompatible checkpoint operation.

    """
    source = source_file(tmp_path)
    target = tmp_path / "journal.json"
    extra: list[str] = []
    if fault == "malformed":
        source.write_text("{broken")
    elif fault == "future":
        document = cast(dict[str, object], read_json(source.read_text()))
        document["schema"] = "experiment_workflow.v2"
        source.write_text(write_json(document))
    elif fault == "missing":
        source = tmp_path / "missing.json"
    elif fault == "same-file":
        target = source
    elif fault == "existing":
        target.write_text("original retained diagnostics")
    else:
        extra.append("--resume")
    before_source = source.read_bytes() if source.exists() else None
    before_target = target.read_bytes() if target.exists() else None
    assert run([str(source), "--journal", str(target), *extra]) == 2
    assert (source.read_bytes() if source.exists() else None) == before_source
    assert (target.read_bytes() if target.exists() else None) == before_target
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "scpn-studio-workflow: source, checkpoint or original runtime refused\n"


def test_invalid_adapter_does_not_silently_use_an_executive_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Keep browser classical workflows unavailable in the original quantum CLI adapter.

    Parameters
    ----------
    tmp_path
        Owned finite temporary allocation.
    capsys
        Actual caller output capture.

    """
    source = source_file(tmp_path)
    document = cast(dict[str, object], read_json(source.read_text()))
    body = cast(dict[str, object], document["body"])
    stages = cast(list[dict[str, object]], body["stages"])
    for stage in stages:
        stage.update(adapter="local-kuramoto", verb="simulate")
    source.write_text(write_json(document))
    target = tmp_path / "journal.json"
    assert run([str(source), "--journal", str(target)]) == 2
    assert not target.exists()
    assert "refused" in capsys.readouterr().err


def test_oversized_source_is_refused_before_checkpoint_allocation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refuse the actual source-file byte boundary before parsing or saving.

    Parameters
    ----------
    tmp_path
        Owned allocation for an oversized input file.
    capsys
        Original dispatcher output capture.

    """
    source = source_file(tmp_path)
    with source.open("r+b") as stream:
        stream.truncate(MAX_WORKFLOW_BYTES + 1)
    before = source.read_bytes()
    target = tmp_path / "journal.json"
    assert run([str(source), "--journal", str(target)]) == 2
    assert source.read_bytes() == before and not target.exists()
    assert not list(tmp_path.glob(".workflow-*"))
    assert "refused" in capsys.readouterr().err


def test_actual_target_lock_refuses_a_second_checkpoint_owner(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Keep the actual filesystem checkpoint lock exclusive to its first owner.

    Parameters
    ----------
    tmp_path
        Owned source and checkpoint allocation.
    capsys
        Original dispatcher output capture.

    """
    source = source_file(tmp_path)
    before = source.read_bytes()
    target = tmp_path / "journal.json"
    lock_name = ".workflow-" + hashlib.sha256(os.fsencode(target.name)).hexdigest() + ".lock"
    with (tmp_path / lock_name).open("a+b") as first_owner:
        fcntl.flock(first_owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert run([str(source), "--journal", str(target)]) == 2
    assert source.read_bytes() == before and not target.exists()
    assert not list(tmp_path.glob("*.partial"))
    assert "refused" in capsys.readouterr().err


def test_source_handler_refusal_returns_partial_with_original_failed_and_blocked_history(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Keep actual compiler-domain failures observable as a partial CLI outcome.

    Parameters
    ----------
    tmp_path
        Owned original graph and resulting partial journal.
    capsys
        Original dispatcher output containing the actual failure history.

    """
    source = source_file(tmp_path)
    document = cast(dict[str, object], read_json(source.read_text()))
    body = cast(dict[str, object], document["body"])
    sweep = cast(dict[str, object], body["sweep"])
    axes = cast(list[dict[str, object]], sweep["axes"])
    axes[0]["values"] = ["invalid original program syntax"]
    source.write_text(write_json(document))
    before = source.read_bytes()
    target = tmp_path / "journal.json"
    assert run([str(source), "--journal", str(target)]) == 1
    emitted = capsys.readouterr()
    assert emitted.err == ""
    journal = cast(dict[str, object], read_json(emitted.out))
    history = cast(dict[str, object], journal["body"])
    entries = cast(list[dict[str, object]], history["entries"])
    assert history["state"] == "partial"
    assert [entry["status"] for entry in entries] == ["failed", "blocked"] * 3
    assert canonical_bytes("cli-partial.v1", read_json(target.read_text())) == canonical_bytes(
        "cli-partial.v1", journal
    )
    assert source.read_bytes() == before and not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("interference", ["cancel", "external-write"])
def test_actual_process_observes_signal_and_external_checkpoint_ownership(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], interference: str
) -> None:
    """Exercise real signals and filesystem races at actual CLI checkpoint boundaries.

    Parameters
    ----------
    tmp_path
        Owned source, actual checkpoint, native event descriptor and process logs.
    capsys
        Original setup dispatcher output capture.
    interference
        Actual SIGINT after reservation or an independent write after resume admission.

    """
    source = source_file(tmp_path)
    target = tmp_path / "journal.json"
    if interference == "external-write":
        assert run([str(source), "--journal", str(target)]) == 0
        capsys.readouterr()
    descriptor = watch_checkpoint(
        target if interference == "external-write" else tmp_path,
        0x00000010 if interference == "external-write" else 0x00000080,
    )
    argv = [
        sys.executable,
        "-m",
        "scpn_quantum_control.studio.workflow_cli",
        str(source),
        "--journal",
        str(target),
    ]
    if interference == "external-write":
        argv.append("--resume")
    process: subprocess.Popen[bytes] | None = None
    try:
        with (
            (tmp_path / "process.out").open("wb") as stdout,
            (tmp_path / "process.err").open("wb") as stderr,
        ):
            process = subprocess.Popen(argv, stdout=stdout, stderr=stderr, start_new_session=True)
            await_checkpoint_events(
                descriptor,
                2 if interference == "external-write" else 1,
                "" if interference == "external-write" else target.name,
            )
            assert process.poll() is None
            if interference == "cancel":
                os.kill(process.pid, signal.SIGINT)
            else:
                target.write_bytes(b"independent caller revision")
            process.wait(timeout=60)
        if interference == "cancel":
            assert process.returncode == 130
            journal = cast(dict[str, object], read_json(target.read_text()))
            assert cast(dict[str, object], journal["body"])["state"] == "cancelled"
            assert canonical_bytes(
                "cli-cancel.v1", read_json((tmp_path / "process.out").read_text())
            ) == canonical_bytes("cli-cancel.v1", journal)
        else:
            assert process.returncode == 2
            assert target.read_bytes() == b"independent caller revision"
        assert not list(tmp_path.glob("*.partial"))
    finally:
        os.close(descriptor)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=15)


def test_resume_keeps_original_checkpoint_when_atomic_wire_exceeds_byte_bound(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refuse a real checkpoint whose retained diagnostics leave no room for its newline.

    Parameters
    ----------
    tmp_path
        Owned original graph, actual producer checkpoint and bounded diagnostic file.
    capsys
        Original dispatcher output; no substituted producer or checkpoint callback.

    """
    source = source_file(tmp_path)
    definition = cast(dict[str, object], read_json(source.read_text()))
    graph = cast(dict[str, object], definition["body"])
    graph["stages"] = [
        stage
        for stage in cast(list[dict[str, object]], graph["stages"])
        if stage["id"] == "source"
    ]
    sweep = cast(dict[str, object], graph["sweep"])
    sweep.update(axes=[], evaluation_budget=1)
    source.write_text(write_json(definition))
    target = tmp_path / "journal.json"
    arguments = [str(source), "--journal", str(target)]
    assert run(arguments) == 0
    capsys.readouterr()
    checkpoint = cast(dict[str, object], read_json(target.read_text()))
    extensions = cast(dict[str, object], checkpoint["extensions"])
    extensions["source_diagnostic"] = ""
    overhead = len(write_json(checkpoint).encode("utf-8"))
    remaining_bytes = MAX_JOURNAL_BYTES - overhead
    extensions["source_diagnostic"] = "😀" * (remaining_bytes // 4) + "x" * (remaining_bytes % 4)
    encoded = write_json(checkpoint).encode("utf-8")
    assert len(encoded) == MAX_JOURNAL_BYTES
    target.write_bytes(encoded)
    original_hash = hashlib.sha256(encoded).hexdigest()
    del encoded, checkpoint, extensions
    assert run([*arguments, "--resume"]) == 2
    assert hashlib.sha256(target.read_bytes()).hexdigest() == original_hash
    refused = capsys.readouterr()
    assert refused.out == ""
    assert refused.err == "scpn-studio-workflow: source, checkpoint or original runtime refused\n"
    assert not list(tmp_path.glob("*.partial"))
