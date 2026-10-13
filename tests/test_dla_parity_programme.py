# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — dla_parity.programme tests
"""Tests of the DLA-parity ordinary programme worker.

Every case goes through the public entry :func:`programme.main` with real
files on disk: the committed raw runs and published summary, or copies of
them changed in one stated way and pinned again. The positive case is
compared with the existing reproducer, not with numbers typed here.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import json
import os
import runpy
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control.dla_parity import programme
from scpn_quantum_control.dla_parity.dataset import (
    DEFAULT_DATA_DIR,
    RUN_FILES,
    load_dla_parity_dataset,
)
from scpn_quantum_control.dla_parity.reproduce import (
    DEFAULT_PUBLISHED_SUMMARY,
    ReproductionTolerance,
    reproduce_statistics,
)

JsonObject = dict[str, Any]
Change = Callable[[list[JsonObject], JsonObject, JsonObject], None]
SCHEMA = json.loads(
    (Path(programme.__file__).parent / "programme_output_v1.schema.json").read_text("utf-8")
)


def _pin(path: Path) -> JsonObject:
    content = path.read_bytes()
    return {
        "path": path.name,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _workspace(tmp_path: Path, change: Change | None = None) -> Path:
    """Copy the committed inputs, apply one change, pin the result and write the request."""
    runs = [json.loads((DEFAULT_DATA_DIR / name).read_text("utf-8")) for name in RUN_FILES]
    summary = json.loads(DEFAULT_PUBLISHED_SUMMARY.read_text("utf-8"))
    request: JsonObject = {
        "schema": programme.REQUEST_SCHEMA,
        "tolerance": dataclasses.asdict(ReproductionTolerance()),
    }
    if change is not None:
        change(runs, summary, request)
    for index, run in enumerate(runs):
        (tmp_path / f"run{index}.json").write_text(json.dumps(run), encoding="utf-8")
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    request.setdefault(
        "raw_run_files", [_pin(tmp_path / f"run{i}.json") for i in range(len(runs))]
    )
    request.setdefault("published_summary", _pin(tmp_path / "summary.json"))
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    return path


def _run(request: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, JsonObject, str]:
    status = programme.main([str(request)])
    text = capsys.readouterr().out
    document: JsonObject = json.loads(text)
    importlib.import_module("jsonschema").Draft202012Validator(SCHEMA).validate(document)
    return status, document, text


def _selected(runs: list[JsonObject]) -> list[JsonObject]:
    return [
        circuit
        for run in runs
        for circuit in run["circuits"]
        if circuit["meta"]["experiment"] == "A_dla_parity_n4" and circuit["meta"]["n_qubits"] == 4
    ]


def test_committed_inputs_complete_with_the_numbers_of_the_existing_reproducer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run the committed inputs and require every scalar and claim of the existing reproducer, one deterministic line."""
    request = _workspace(tmp_path)
    status, document, text = _run(request, capsys)
    reference = reproduce_statistics(load_dla_parity_dataset(verify_integrity=True))

    assert status == 0
    assert document["outcome"] == "completed"
    assert document["depth_summaries"] == [
        dataclasses.asdict(s) for s in reference.depth_summaries
    ]
    assert document["fisher"] == dataclasses.asdict(reference.fisher)
    assert document["result"] == {
        "peak_asymmetry_relative": reference.peak_asymmetry_relative,
        "peak_asymmetry_depth": reference.peak_asymmetry_depth,
        "mean_asymmetry_relative": reference.mean_asymmetry_relative,
        "n_circuits_used": reference.n_circuits_used,
    }
    assert [
        (c["name"], c["published"], c["recomputed"], c["difference"]) for c in document["claims"]
    ] == [tuple(claim) for claim in reference.claims_checked]
    assert all(c["passed"] for c in document["claims"] + document["identity_checks"])
    # Every record is accounted for, and the reasons add up to the whole input.
    records = document["selection"]["records"]
    assert len(records) == sum(document["selection"]["counts"].values())
    assert document["selection"]["counts"]["selected"] == reference.n_circuits_used
    assert [(r["run"], r["circuit"]) for r in records] == sorted(
        (r["run"], r["circuit"]) for r in records
    )
    # The worker repairs nothing: the combined p-value is the existing arithmetic's.
    assert document["fisher"]["combined_p"] == reference.fisher.combined_p
    assert "1.0 - chi2.cdf" in document["comparison_coverage"]["combined_p_arithmetic"]
    # One line, sorted keys, and the same bytes on a second run.
    assert text.endswith("\n") and text.count("\n") == 1
    assert text == programme.serialise(document)
    assert _run(request, capsys)[2] == text


def test_module_entry_writes_the_same_document_and_exit_status(tmp_path: Path) -> None:
    """Run the worker as a module in a fresh interpreter and require a clean exit and an empty error stream."""
    request = _workspace(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-m", "scpn_quantum_control.dla_parity.programme", str(request)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert json.loads(completed.stdout)["outcome"] == "completed"


def test_module_run_as_a_script_exits_with_the_refusal_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Execute the module as a script without a request and require the refusal exit status."""
    monkeypatch.setattr(sys, "argv", ["programme"])
    monkeypatch.delitem(sys.modules, "scpn_quantum_control.dla_parity.programme")
    with pytest.raises(SystemExit) as raised:
        runpy.run_module("scpn_quantum_control.dla_parity.programme", run_name="__main__")
    assert raised.value.code == 2
    assert json.loads(capsys.readouterr().out)["refusal"]["category"] == "malformed_request"


def test_a_numeric_difference_is_a_mismatch_that_lists_every_claim(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Change two published values and require a mismatch that names both and still lists all claims."""

    def change(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
        summary["depth_summaries"][0]["mean_even"] += 1e-3
        summary["depth_summaries"][3]["welch_t"] *= 1.01

    status, document, _ = _run(_workspace(tmp_path, change), capsys)
    failed = [c["name"] for c in document["claims"] if not c["passed"]]
    assert status == 1
    assert document["outcome"] == "mismatch"
    assert failed == ["depth=2.leakage_even", "depth=8.welch_t"]
    assert len(document["claims"]) == 57
    assert {c["rule"] for c in document["claims"]} == {"absolute", "scaled_by_max_abs_or_one"}


def test_a_changed_replicate_count_is_a_mismatch_of_an_identity_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Change two published integers and require the identity checks, not the claims, to fail."""

    def change(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
        summary["depth_summaries"][1]["n_reps_odd"] += 1
        summary["fisher_combined"]["df"] += 2

    status, document, _ = _run(_workspace(tmp_path, change), capsys)
    assert status == 1
    assert all(c["passed"] for c in document["claims"])
    assert [c["name"] for c in document["identity_checks"] if not c["passed"]] == [
        "depth=4.n_reps_odd",
        "fisher.degrees_of_freedom",
    ]


def test_a_selected_experiment_record_of_another_sector_is_accounted_as_excluded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Give one record another sector and require it to be accounted for as excluded."""

    def change(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
        _selected(runs)[0]["meta"]["sector"] = "baseline"

    status, document, _ = _run(_workspace(tmp_path, change), capsys)
    counts = document["selection"]["counts"]
    assert counts["excluded_sector"] == 1
    assert counts["selected"] == 281
    # One replicate fewer changes the statistics of its depth: the run is a mismatch, not a refusal.
    assert status == 1
    assert [c["name"] for c in document["identity_checks"] if not c["passed"]][0].startswith(
        "depth="
    )


def test_an_optional_published_reference_is_compared_when_the_summary_carries_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Add both optional references, one wrong, and require the claim that the reproducer makes."""
    reference = reproduce_statistics(load_dla_parity_dataset())

    def change(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
        summary["mean_asymmetry_relative"] = 999.0
        summary["peak_asymmetry_relative"] = reference.peak_asymmetry_relative

    request = _workspace(tmp_path, change)
    status, document, _ = _run(request, capsys)
    by_name = {claim["name"]: claim for claim in document["claims"]}
    assert status == 1
    assert len(document["claims"]) == 59
    assert by_name["mean_asymmetry_relative"]["passed"] is False
    assert by_name["mean_asymmetry_relative"]["published"] == 999.0
    assert by_name["mean_asymmetry_relative"]["recomputed"] == reference.mean_asymmetry_relative
    assert by_name["peak_asymmetry_relative"]["passed"] is True
    assert document["comparison_coverage"]["emitted_scalars_without_published_reference"] == [
        "result.peak_asymmetry_depth",
        "result.n_circuits_used",
    ]
    # The existing reproducer rejects the same summary: the worker does not weaken it.
    with pytest.raises(AssertionError, match="mean_asymmetry_relative"):
        reproduce_statistics(
            load_dla_parity_dataset(), published_summary=tmp_path / "summary.json"
        )


def test_a_summary_without_the_optional_references_names_them_as_uncompared(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Require the committed summary, which has neither reference, to list both as uncompared."""
    _, document, _ = _run(_workspace(tmp_path), capsys)
    assert document["comparison_coverage"]["emitted_scalars_without_published_reference"] == [
        "result.mean_asymmetry_relative",
        "result.peak_asymmetry_relative",
        "result.peak_asymmetry_depth",
        "result.n_circuits_used",
    ]
    assert len(document["claims"]) == 57


def test_a_number_too_large_for_the_protocol_is_refused_in_a_fresh_interpreter(
    tmp_path: Path,
) -> None:
    """Give a tolerance of ten to the four hundredth and require a refusal document, not a traceback."""
    request = _workspace(tmp_path)
    text = request.read_text("utf-8").replace(
        '"welch_p_rel": 1e-06', '"welch_p_rel": 1' + "0" * 400
    )
    assert "0" * 400 in text
    request.write_text(text, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "scpn_quantum_control.dla_parity.programme", str(request)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert completed.stderr == ""
    assert json.loads(completed.stdout)["refusal"]["category"] == "malformed_request"


@pytest.mark.parametrize(
    ("name", "old", "new", "category"),
    [
        ("request.json", '{"schema":', '{"schema": "other", "schema":', "malformed_request"),
        ("run1.json", '"backend":', '"backend": "other", "backend":', "malformed_input"),
        ("run2.json", '"wall_time_s":', '"wall_time_s": NaN, "ignored":', "malformed_input"),
        ("summary.json", '"fisher_combined":', '"fisher_combined": {}, "fisher_combined":', None),
    ],
)
def test_a_repeated_member_or_a_non_finite_literal_is_refused(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    name: str,
    old: str,
    new: str,
    category: str | None,
) -> None:
    """Edit the text of one input so that a member repeats or is NaN, pin it, and require a refusal."""
    request = _workspace(tmp_path)
    text = (tmp_path / name).read_text("utf-8")
    assert text.count(old) >= 1
    edited = text.replace(old, new, 1).encode("utf-8")
    if name == "request.json":
        request.write_bytes(edited)
    else:
        _replace(request, name, edited)
    status, document, _ = _run(request, capsys)
    assert status == 2
    assert document["refusal"]["category"] == (category or "malformed_published_summary")


def test_a_document_nested_beyond_the_reader_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pin a raw run of one hundred thousand nested lists and require a refusal."""
    request = _workspace(tmp_path)
    _replace(request, "run0.json", b"[" * 100_000 + b"]" * 100_000)
    status, document, _ = _run(request, capsys)
    assert (status, document["refusal"]["category"]) == (2, "malformed_input")


def test_a_request_that_is_too_large_or_not_a_regular_file_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pad a request beyond its limit, then name a directory, and require a refusal each time."""
    request = _workspace(tmp_path)
    request.write_bytes(request.read_bytes() + b" " * programme.MAX_REQUEST_BYTES)
    assert programme.main([str(request)]) == 2
    assert json.loads(capsys.readouterr().out)["refusal"]["category"] == "input_too_large"
    assert programme.main([str(tmp_path)]) == 2
    assert json.loads(capsys.readouterr().out)["refusal"]["category"] == "missing_input"


@pytest.mark.skipif(os.geteuid() == 0, reason="the superuser reads a file whatever its mode")
def test_an_input_that_exists_and_cannot_be_opened_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Remove every permission from a pinned input and require a refusal."""
    request = _workspace(tmp_path)
    (tmp_path / "run3.json").chmod(0)
    status, document, _ = _run(request, capsys)
    assert (status, document["refusal"]["category"]) == (2, "missing_input")


def _fractional_qubits(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
    _selected(runs)[0]["meta"]["n_qubits"] = 4.5


def _fractional_shots(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
    _selected(runs)[0]["meta"]["shots"] += 0.5


def _boolean_depth(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
    _selected(runs)[0]["meta"]["depth"] = True


def _textual_shots(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
    _selected(runs)[0]["meta"]["shots"] = "2000"


def _boolean_count(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
    counts = _selected(runs)[0]["counts"]
    counts[next(iter(counts))] = True


def _fractional_count(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
    counts = _selected(runs)[0]["counts"]
    counts[next(iter(counts))] = 12.0


def _circuit_not_an_object(
    runs: list[JsonObject], summary: JsonObject, request: JsonObject
) -> None:
    runs[0]["circuits"][0] = "circuit"


def _meta_not_an_object(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
    runs[0]["circuits"][0]["meta"] = []


def _unknown_sector(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
    _selected(runs)[0]["meta"]["sector"] = "both"


def _numeric_job_identifier(
    runs: list[JsonObject], summary: JsonObject, request: JsonObject
) -> None:
    runs[1]["job_ids"] = [17]


def _textual_wall_time(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
    runs[1]["wall_time_s"] = "fast"


def _declared_circuit_count_differs(
    runs: list[JsonObject], summary: JsonObject, request: JsonObject
) -> None:
    runs[1]["n_circuits"] += 1


def _optional_reference_not_a_number(
    runs: list[JsonObject], summary: JsonObject, request: JsonObject
) -> None:
    summary["peak_asymmetry_relative"] = "0.17"


def _drop_published_depth(
    runs: list[JsonObject], summary: JsonObject, request: JsonObject
) -> None:
    del summary["depth_summaries"][2]


def _add_published_depth(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
    extra = copy.deepcopy(summary["depth_summaries"][0])
    extra["depth"] = 99
    summary["depth_summaries"].append(extra)


def _repeat_published_depth(
    runs: list[JsonObject], summary: JsonObject, request: JsonObject
) -> None:
    summary["depth_summaries"].append(copy.deepcopy(summary["depth_summaries"][0]))


def _published_row_not_an_object(
    runs: list[JsonObject], summary: JsonObject, request: JsonObject
) -> None:
    summary["depth_summaries"][0] = [1, 2, 3]


def _published_missing_scalar(
    runs: list[JsonObject], summary: JsonObject, request: JsonObject
) -> None:
    del summary["depth_summaries"][0]["welch_p"]


def _published_extra_scalar(
    runs: list[JsonObject], summary: JsonObject, request: JsonObject
) -> None:
    summary["depth_summaries"][0]["unexpected"] = 1.0


def _published_non_finite(
    runs: list[JsonObject], summary: JsonObject, request: JsonObject
) -> None:
    summary["depth_summaries"][0]["sem_even"] = float("nan")


def _published_without_fisher(
    runs: list[JsonObject], summary: JsonObject, request: JsonObject
) -> None:
    del summary["fisher_combined"]


def _published_fisher_malformed(
    runs: list[JsonObject], summary: JsonObject, request: JsonObject
) -> None:
    summary["fisher_combined"]["df"] = 16.5


def _no_shots(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
    circuit = _selected(runs)[0]
    circuit["counts"] = {}
    circuit["meta"]["shots"] = 0


def _shots_differ(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
    _selected(runs)[5]["meta"]["shots"] += 1


def _negative_count(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
    counts = _selected(runs)[0]["counts"]
    counts[next(iter(counts))] = -1


def _short_bitstring(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
    counts = _selected(runs)[0]["counts"]
    counts["01"] = counts.pop(next(iter(counts)))


def _foreign_symbol(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
    _selected(runs)[0]["meta"]["initial"] = "0a01"


def _sector_contradicts_state(
    runs: list[JsonObject], summary: JsonObject, request: JsonObject
) -> None:
    meta = _selected(runs)[0]["meta"]
    meta["sector"] = "odd" if meta["sector"] == "even" else "even"


def _nothing_selected(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
    for circuit in _selected(runs):
        circuit["meta"]["experiment"] = "Z_other"


def _one_replicate(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
    kept = False
    for run in runs:
        remaining = []
        for circuit in run["circuits"]:
            meta = circuit["meta"]
            single = (
                meta["experiment"] == "A_dla_parity_n4"
                and meta["depth"] == 2
                and meta["sector"] == "even"
            )
            if single and kept:
                continue
            kept = kept or single
            remaining.append(circuit)
        run["circuits"] = remaining
        run["n_circuits"] = len(remaining)


def _identical_leakage(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
    # Every circuit of depth 2 measures its own initial state only: zero
    # leakage and zero variance in both sectors, so the Welch test has no p-value.
    for circuit in _selected(runs):
        if circuit["meta"]["depth"] == 2:
            circuit["counts"] = {circuit["meta"]["initial"]: circuit["meta"]["shots"]}


def _odd_sector_without_leakage(
    runs: list[JsonObject], summary: JsonObject, request: JsonObject
) -> None:
    # Zero mean in the odd sector with a varying even sector: the p-value
    # exists and the relative asymmetry divides by zero.
    for circuit in _selected(runs):
        if circuit["meta"]["depth"] == 2 and circuit["meta"]["sector"] == "odd":
            circuit["counts"] = {circuit["meta"]["initial"]: circuit["meta"]["shots"]}


def _raw_missing_key(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
    del runs[1]["backend"]


def _raw_wrong_type(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
    runs[2]["n_circuits"] = None


def _raw_circuits_not_a_list(
    runs: list[JsonObject], summary: JsonObject, request: JsonObject
) -> None:
    runs[0]["circuits"] = "none"


def _request_extra_key(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
    request["entrypoint"] = "os.system"


def _request_other_schema(
    runs: list[JsonObject], summary: JsonObject, request: JsonObject
) -> None:
    request["schema"] = "something.else.v1"


def _request_three_runs(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
    request["raw_run_files"] = [{"path": "run0.json", "bytes": 1, "sha256": "0" * 64}] * 3


def _tolerance_missing_field(
    runs: list[JsonObject], summary: JsonObject, request: JsonObject
) -> None:
    del request["tolerance"]["welch_p_rel"]


def _tolerance_negative(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
    request["tolerance"]["welch_t_rel"] = -1.0


def _pin_without_digest(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
    request["published_summary"] = {"path": "summary.json", "bytes": 10}


def _pin_upper_case_digest(
    runs: list[JsonObject], summary: JsonObject, request: JsonObject
) -> None:
    request["published_summary"] = {"path": "summary.json", "bytes": 10, "sha256": "A" * 64}


def _pin_declared_too_large(
    runs: list[JsonObject], summary: JsonObject, request: JsonObject
) -> None:
    request["published_summary"] = {
        "path": "summary.json",
        "bytes": programme.MAX_INPUT_BYTES + 1,
        "sha256": "0" * 64,
    }


def _pin_absent_file(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
    request["published_summary"] = {"path": "absent.json", "bytes": 10, "sha256": "0" * 64}


def _pin_other_digest(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
    request["published_summary"] = {"path": "summary.json", "bytes": 10, "sha256": "0" * 64}


@pytest.mark.parametrize(
    ("change", "category"),
    [
        (_drop_published_depth, "incomplete_depth_set"),
        (_fractional_qubits, "malformed_input"),
        (_fractional_shots, "malformed_input"),
        (_boolean_depth, "malformed_input"),
        (_textual_shots, "malformed_input"),
        (_boolean_count, "malformed_input"),
        (_fractional_count, "malformed_input"),
        (_circuit_not_an_object, "malformed_input"),
        (_meta_not_an_object, "malformed_input"),
        (_unknown_sector, "malformed_input"),
        (_numeric_job_identifier, "malformed_input"),
        (_textual_wall_time, "malformed_input"),
        (_declared_circuit_count_differs, "malformed_input"),
        (_optional_reference_not_a_number, "malformed_published_summary"),
        (_add_published_depth, "incomplete_depth_set"),
        (_repeat_published_depth, "malformed_published_summary"),
        (_published_row_not_an_object, "malformed_published_summary"),
        (_published_missing_scalar, "malformed_published_summary"),
        (_published_extra_scalar, "malformed_published_summary"),
        (_published_non_finite, "malformed_published_summary"),
        (_published_without_fisher, "malformed_published_summary"),
        (_published_fisher_malformed, "malformed_published_summary"),
        (_no_shots, "empty_shots"),
        (_shots_differ, "inconsistent_counts"),
        (_negative_count, "inconsistent_counts"),
        (_short_bitstring, "malformed_bitstring"),
        (_foreign_symbol, "malformed_bitstring"),
        (_sector_contradicts_state, "inconsistent_sector"),
        (_nothing_selected, "empty_selection"),
        (_one_replicate, "insufficient_replicates"),
        (_identical_leakage, "invalid_p_value"),
        (_odd_sector_without_leakage, "non_finite_output"),
        (_raw_missing_key, "malformed_input"),
        (_raw_wrong_type, "malformed_input"),
        (_raw_circuits_not_a_list, "malformed_input"),
        (_request_extra_key, "malformed_request"),
        (_request_other_schema, "malformed_request"),
        (_request_three_runs, "malformed_request"),
        (_tolerance_missing_field, "malformed_request"),
        (_tolerance_negative, "malformed_request"),
        (_pin_without_digest, "malformed_request"),
        (_pin_upper_case_digest, "malformed_request"),
        (_pin_declared_too_large, "input_too_large"),
        (_pin_absent_file, "missing_input"),
        (_pin_other_digest, "digest_mismatch"),
    ],
)
def test_an_unacceptable_input_is_refused_with_its_category_and_no_scalar(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], change: Change, category: str
) -> None:
    """Apply one unacceptable change and require its refusal category and a document without scalars."""
    status, document, _ = _run(_workspace(tmp_path, change), capsys)
    assert status == 2
    assert document["outcome"] == "refused"
    assert document["refusal"]["category"] == category
    assert category in programme.REFUSAL_CATEGORIES
    assert set(document) == {"schema", "outcome", "refusal"}


def test_content_changed_after_pinning_is_a_digest_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Flip one bit of a pinned input and require a digest refusal."""
    request = _workspace(tmp_path)
    target = tmp_path / "run2.json"
    content = bytearray(target.read_bytes())
    content[100] ^= 0x01
    target.write_bytes(bytes(content))
    status, document, _ = _run(request, capsys)
    assert (status, document["refusal"]["category"]) == (2, "digest_mismatch")


def test_a_file_larger_than_the_limit_is_refused_whatever_its_pin_says(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Replace an input by one byte more than the limit and require a size refusal."""
    request = _workspace(tmp_path)
    (tmp_path / "run0.json").write_bytes(b" " * (programme.MAX_INPUT_BYTES + 1))
    status, document, _ = _run(request, capsys)
    assert (status, document["refusal"]["category"]) == (2, "input_too_large")


def _replace(request: Path, name: str, content: bytes) -> None:
    """Overwrite one input with other bytes and pin those bytes in the request."""
    target = request.parent / name
    target.write_bytes(content)
    document = json.loads(request.read_text("utf-8"))
    if name == "summary.json":
        document["published_summary"] = _pin(target)
    else:
        document["raw_run_files"][int(name[3])] = _pin(target)
    request.write_text(json.dumps(document), encoding="utf-8")


@pytest.mark.parametrize(
    ("name", "content", "category"),
    [
        ("run1.json", b"\xff\xfe", "malformed_input"),
        ("run3.json", b"{not json", "malformed_input"),
        ("run0.json", b"[1, 2]", "malformed_input"),
        ("summary.json", b"{not json", "malformed_published_summary"),
        ("summary.json", b"[]", "malformed_published_summary"),
    ],
)
def test_a_correctly_pinned_input_that_is_not_a_json_object_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], name: str, content: bytes, category: str
) -> None:
    """Pin bytes that are not a JSON object and require the refusal of that input kind."""
    request = _workspace(tmp_path)
    _replace(request, name, content)
    status, document, _ = _run(request, capsys)
    assert (status, document["refusal"]["category"]) == (2, category)


def test_a_request_that_is_not_a_json_object_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Give a request that is a list, then undecodable bytes, and require a request refusal."""
    request = _workspace(tmp_path)
    request.write_bytes(b"[]")
    assert _run(request, capsys)[1]["refusal"]["category"] == "malformed_request"
    request.write_bytes(b"\xff")
    assert _run(request, capsys)[1]["refusal"]["category"] == "malformed_request"


def test_more_circuits_than_the_limit_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Supply one circuit more than the limit and require the refusal before any statistics."""

    def change(runs: list[JsonObject], summary: JsonObject, request: JsonObject) -> None:
        meta = {
            "experiment": "x",
            "n_qubits": 6,
            "depth": 1,
            "sector": "even",
            "initial": "0",
            "rep": 0,
            "shots": 0,
            "t_step": 0.1,
        }
        runs[3]["circuits"] = [{"meta": meta, "counts": {}}] * (programme.MAX_CIRCUITS + 1)
        runs[3]["n_circuits"] = programme.MAX_CIRCUITS + 1

    status, document, _ = _run(_workspace(tmp_path, change), capsys)
    assert (status, document["refusal"]["category"]) == (2, "too_many_circuits")


def test_a_request_that_cannot_be_read_or_is_not_named_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Name an absent request, no request and two requests, and require a refusal each time."""
    assert programme.main([str(tmp_path / "absent.json")]) == 2
    assert json.loads(capsys.readouterr().out)["refusal"]["category"] == "missing_input"
    assert programme.main([]) == 2
    assert json.loads(capsys.readouterr().out)["refusal"]["category"] == "malformed_request"
    assert programme.main(["one", "two"]) == 2
    assert json.loads(capsys.readouterr().out)["refusal"]["category"] == "malformed_request"


def test_a_refusal_carries_its_category_in_its_text() -> None:
    """Require the text of a refusal to start with its category."""
    refusal = programme.ProgrammeRefusal("digest_mismatch", "an input differs from its pin")
    assert str(refusal) == "digest_mismatch: an input differs from its pin"
    assert refusal.category in programme.REFUSAL_CATEGORIES
