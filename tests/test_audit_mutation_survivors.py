# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the mutation survivor ceiling gate
"""Exercise the mutation survivor gate with the pinned mutmut release and real Git repositories."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tools import audit_mutation_survivors as gate

REPOSITORY = Path(__file__).resolve().parents[1]
pytest.importorskip("mutmut", reason="the mutation tool is not installed")

MODULE = "src/calc.py"
RUNNER = "tools/run_calc_tests.sh"
SOURCE = "def area(width, height):\n    return width * height\n"
STRONG_TEST = "from calc import area\n\n\ndef test_area():\n    assert area(2, 3) == 6\n"
WEAK_TEST = "from calc import area\n\n\ndef test_area():\n    assert area(1, 1) == 1\n"
DIGEST = hashlib.sha256(SOURCE.encode()).hexdigest()
_RUNNER_SCRIPT = """#!/bin/sh
# Runs the target's tests. With CALC_SLOW set, a mutated module makes the
# runner hang instead: once (until a marker exists) or always.
if ! cmp -s src/calc.py tests/calc_original.txt; then
  case "${CALC_SLOW:-}" in
    always) exec sleep 60 ;;
    once)
      if [ ! -e .slow-marker ]; then
        : > .slow-marker
        exec sleep 60
      fi
      ;;
  esac
fi
exec "$VENV_PY" -m pytest -x -q -p no:cacheprovider tests/test_calc.py
"""


def _git(repo: Path, *arguments: str) -> str:
    """Run Git in ``repo`` with a fixed identity and return its output."""
    completed = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Gate Test",
            "-c",
            "user.email=gate@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *arguments,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository(
    root: Path,
    test: str = STRONG_TEST,
    *,
    executable: bool = True,
    release: str | None = None,
    **record: Any,
) -> Path:
    """Create a committed repository with one mutation target and its ceiling.

    Parameters
    ----------
    root
        Empty directory that becomes the repository.
    test
        Text of the target's test module.
    executable
        Whether the runner script is committed with its execute permission.
    release
        Release written into the pin file and the ceiling; the installed one when omitted.
    **record
        Fields replaced in the recorded target row.

    Returns
    -------
    Path
        The repository root.

    """
    pinned = gate.installed_release() if release is None else release
    files = {
        MODULE: SOURCE,
        "tests/test_calc.py": test,
        "tests/conftest.py": (
            "import sys\nfrom pathlib import Path\n\n"
            "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
        ),
        RUNNER: _RUNNER_SCRIPT,
        "tests/calc_original.txt": SOURCE,
        str(gate.PIN_FILE): f"mutmut=={pinned} \\\n    --hash=sha256:{'0' * 64}\n",
    }
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (root / RUNNER).chmod(0o755 if executable else 0o644)
    row = {
        "name": "calc",
        "module": MODULE,
        "runner": RUNNER,
        "source_sha256": DIGEST,
        "mutants": 1,
        "survived": 0,
        "timeout": 0,
        **record,
    }
    policy = root / gate.DEFAULT_POLICY
    policy.write_text(
        json.dumps({"schema": gate.SCHEMA, "release": pinned, "targets": [row], "history": []}),
        encoding="utf-8",
    )
    _git(root, "init", "--quiet")
    _git(root, "add", "--all")
    _git(root, "commit", "--quiet", "--message", "fixture")
    return root


def _main(repo: Path, tmp_path: Path, *arguments: str) -> int:
    """Run the gate on ``repo`` with its exported tree under ``tmp_path``."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return gate.main(["--repo", str(repo), "--workspace", str(workspace), *arguments])


def test_killed_mutant_matches_a_ceiling_without_survivors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A test that notices the mutant passes the gate and leaves the working tree untouched."""
    repo = _repository(tmp_path / "repo")

    assert _main(repo, tmp_path) == 0
    assert capsys.readouterr().out.splitlines() == [
        "calc: 1 mutants: 1 killed, 0 survived, 0 timeout, 0 suspicious, 0 skipped, 0 untested",
        "Mutation survivors: 1 targets; 0 problems",
    ]
    assert (repo / MODULE).read_text(encoding="utf-8") == SOURCE
    assert _git(repo, "status", "--porcelain") == ""
    assert list((tmp_path / "workspace").iterdir()) == []


def test_new_survivor_fails_the_gate_and_cannot_be_lowered_in(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A test that misses the mutant fails the gate, and lowering refuses to admit it."""
    repo = _repository(tmp_path / "repo", WEAK_TEST)

    assert _main(repo, tmp_path) == 1
    captured = capsys.readouterr()
    assert captured.err.splitlines() == ["calc: survived mutants grew: 0 -> 1"]
    assert captured.out.endswith("Mutation survivors: 1 targets; 1 problems\n")

    assert _main(repo, tmp_path, "--lower") == 1
    assert "cannot lower the ceiling: calc: survived mutants grew: 0 -> 1" in (
        capsys.readouterr().err
    )


def test_ceiling_above_the_measurement_must_be_lowered(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A recorded survivor that is now killed is reported until ``--lower`` writes it down."""
    repo = _repository(tmp_path / "repo", survived=1)

    assert _main(repo, tmp_path) == 1
    assert capsys.readouterr().err.splitlines() == [
        "calc: ceiling is above the measurement: survived 1 -> 0; lower it with --lower"
    ]
    assert _main(repo, tmp_path, "--lower") == 0
    assert capsys.readouterr().out.endswith("Mutation survivor ceiling written for 1 targets\n")
    assert gate.load_ceiling(repo / gate.DEFAULT_POLICY).targets[0].survived == 0


def test_changed_source_and_changed_mutant_count_need_a_rebaseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stale digest or mutant count fails the gate; a rebaseline records the measurement."""
    stale = _repository(tmp_path / "stale", WEAK_TEST, source_sha256="0" * 64, mutants=7)

    assert _main(stale, tmp_path) == 1
    assert capsys.readouterr().err.splitlines() == [
        "calc: the source of src/calc.py changed; record a new measurement with --rebaseline"
    ]
    assert _main(stale, tmp_path, "--lower") == 1
    assert "cannot lower the ceiling: calc: the source of src/calc.py changed" in (
        capsys.readouterr().err
    )
    assert _main(stale, tmp_path, "--rebaseline") == 0
    capsys.readouterr()
    ceiling = gate.load_ceiling(stale / gate.DEFAULT_POLICY)
    assert ceiling.targets == (gate.Target("calc", MODULE, RUNNER, DIGEST, 1, 1, 0),)
    assert ceiling.history == (
        {
            "release": gate.installed_release(),
            "name": "calc",
            "source_sha256": "0" * 64,
            "mutants": 7,
            "survived": 0,
            "timeout": 0,
        },
    )

    counted = _repository(tmp_path / "counted", mutants=3)
    assert _main(counted, tmp_path) == 1
    assert capsys.readouterr().err.splitlines() == [
        "calc: 1 mutants were generated, 3 are recorded"
    ]


def test_transient_timeout_is_retested_and_a_persistent_one_is_counted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mutant that timed out once is tested again alone; one that always hangs is a timeout.

    The fixture's runner hangs on a mutated module when ``CALC_SLOW`` is set:
    once, as a busy machine would make it, or always, as a mutant that loops
    forever would. The working tree of the repository stays untouched and the
    module in the exported tree is restored after each retest.
    """
    monkeypatch.setenv("CALC_SLOW", "once")
    transient = _repository(tmp_path / "transient")
    assert _main(transient, tmp_path) == 0
    assert capsys.readouterr().out.splitlines()[0] == (
        "calc: 1 mutants: 1 killed, 0 survived, 0 timeout, 0 suspicious, 0 skipped, 0 untested"
    )

    monkeypatch.setenv("CALC_SLOW", "always")
    persistent = _repository(tmp_path / "persistent")
    assert _main(persistent, tmp_path, "--retest-limit", "2") == 1
    captured = capsys.readouterr()
    assert captured.err.splitlines() == ["calc: timeout mutants grew: 0 -> 1"]
    assert captured.out.splitlines()[0] == (
        "calc: 1 mutants: 0 killed, 0 survived, 1 timeout, 0 suspicious, 0 skipped, 0 untested"
    )


def test_runner_without_execute_permission_fails_the_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The defect that kept the weekly job from testing anything is a failure here."""
    repo = _repository(tmp_path / "repo", executable=False)

    assert _main(repo, tmp_path) == 1
    assert capsys.readouterr().err == (
        "mutation survivor gate failed: mutation runner is missing or not executable: "
        "tools/run_calc_tests.sh\n"
    )


def test_tests_that_fail_before_any_mutation_are_a_tool_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A target whose tests do not pass unmutated stops the gate instead of counting mutants."""
    repo = _repository(tmp_path / "repo", STRONG_TEST.replace("== 6", "== 7"))

    assert _main(repo, tmp_path) == 1
    assert capsys.readouterr().err.startswith(
        "mutation survivor gate failed: mutmut could not run calc: "
    )


def test_release_mismatches_and_unknown_targets_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Another pinned release, another recorded release and an unknown target fail the gate."""
    pinned = _repository(tmp_path / "pinned", release="0.0.1")
    assert _main(pinned, tmp_path) == 1
    assert "is not the pinned release 0.0.1" in capsys.readouterr().err

    recorded = _repository(tmp_path / "recorded")
    policy = recorded / gate.DEFAULT_POLICY
    body = json.loads(policy.read_text(encoding="utf-8"))
    body["release"] = "0.0.1"
    policy.write_text(json.dumps(body), encoding="utf-8")
    assert _main(recorded, tmp_path) == 1
    assert "ceiling was measured with mutmut 0.0.1" in capsys.readouterr().err
    assert _main(recorded, tmp_path, "--rebaseline") == 0
    capsys.readouterr()
    rebased = gate.load_ceiling(policy)
    assert rebased.release == gate.installed_release()
    assert [row["release"] for row in rebased.history] == ["0.0.1"]

    assert _main(recorded, tmp_path, "--target", "absent") == 1
    assert "unknown mutation target: absent" in capsys.readouterr().err


def test_unselected_targets_keep_their_record_when_one_target_is_updated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Updating one named target leaves the other recorded target exactly as it was."""
    repo = _repository(tmp_path / "repo", survived=1)
    policy = repo / gate.DEFAULT_POLICY
    body = json.loads(policy.read_text(encoding="utf-8"))
    other = {**body["targets"][0], "name": "other", "survived": 0, "mutants": 9}
    body["targets"].append(other)
    policy.write_text(json.dumps(body), encoding="utf-8")
    _git(repo, "commit", "--quiet", "--all", "--message", "second target")

    assert _main(repo, tmp_path, "--target", "calc", "--target", "calc", "--lower") == 0
    capsys.readouterr()
    targets = gate.load_ceiling(policy).targets
    assert [(target.name, target.survived, target.mutants) for target in targets] == [
        ("calc", 0, 1),
        ("other", 0, 9),
    ]


def test_each_target_is_counted_without_the_mutants_of_an_earlier_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A second target reports its own mutants, not those the tool cached for the first."""
    repo = _repository(tmp_path / "repo")
    source = "def perimeter(width, height):\n    return 2 * (width + height)\n"
    runner = "tools/run_shape_tests.sh"
    files = {
        "src/shape.py": source,
        "tests/test_shape.py": (
            "from shape import perimeter\n\n\n"
            "def test_perimeter():\n    assert perimeter(2, 3) == 10\n"
        ),
        runner: '#!/bin/sh\nexec "$VENV_PY" -m pytest -x -q -p no:cacheprovider tests/test_shape.py\n',
    }
    for name, text in files.items():
        (repo / name).write_text(text, encoding="utf-8")
    (repo / runner).chmod(0o755)
    policy = repo / gate.DEFAULT_POLICY
    body = json.loads(policy.read_text(encoding="utf-8"))
    body["targets"].append(
        {
            "name": "shape",
            "module": "src/shape.py",
            "runner": runner,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "mutants": 3,
            "survived": 0,
            "timeout": 0,
        }
    )
    policy.write_text(json.dumps(body), encoding="utf-8")
    _git(repo, "add", "--all")
    _git(repo, "commit", "--quiet", "--message", "second target")

    report = tmp_path / "report.json"

    assert _main(repo, tmp_path, "--report", str(report)) == 0
    assert capsys.readouterr().out.splitlines() == [
        "calc: 1 mutants: 1 killed, 0 survived, 0 timeout, 0 suspicious, 0 skipped, 0 untested",
        "shape: 3 mutants: 3 killed, 0 survived, 0 timeout, 0 suspicious, 0 skipped, 0 untested",
        "Mutation survivors: 2 targets; 0 problems",
    ]
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["schema"] == gate.REPORT_SCHEMA
    assert written["release"] == gate.installed_release()
    assert list(written["targets"]) == ["calc", "shape"]
    assert written["targets"]["calc"]["killed"] == ["1"]
    assert sorted(written["targets"]["shape"]["killed"]) == ["1", "2", "3"]
    for statuses in written["targets"].values():
        assert set(statuses) == set(gate.STATUSES)
        assert all(statuses[status] == [] for status in gate.STATUSES if status != "killed")


def test_report_names_the_survivor_when_the_gate_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failing gate still writes which mutant survived."""
    repo = _repository(tmp_path / "repo", WEAK_TEST)
    report = tmp_path / "report.json"

    assert _main(repo, tmp_path, "--report", str(report)) == 1
    capsys.readouterr()
    statuses = json.loads(report.read_text(encoding="utf-8"))["targets"]["calc"]
    assert statuses["survived"] == ["1"]
    assert statuses["killed"] == []


def test_measure_refuses_a_missing_module_and_incomplete_runs_are_reported(
    tmp_path: Path,
) -> None:
    """A missing module is refused; untested and skipped mutants are never accepted."""
    target = gate.Target("calc", MODULE, RUNNER, DIGEST, 4, 1, 0)

    with pytest.raises(ValueError, match="mutation target module is missing: src/calc.py"):
        gate.measure_statuses(tmp_path, target, Path(sys.executable))

    counts = {
        "killed": 1,
        "survived": 1,
        "timeout": 0,
        "suspicious": 0,
        "skipped": 1,
        "untested": 1,
    }
    assert gate.compare(target, counts, DIGEST) == ["calc: 2 mutants were not tested"]
    with pytest.raises(ValueError, match="cannot record calc: 2 mutants were not tested"):
        gate.rebaselined(target, counts, DIGEST)
    with pytest.raises(ValueError, match="cannot lower the ceiling: calc: 2 mutants"):
        gate.lowered(target, counts, DIGEST)


def test_listing_or_apply_failure_after_a_completed_run_is_a_tool_failure(tmp_path: Path) -> None:
    """A tool that runs but cannot list or apply its mutants stops the measurement.

    The first stand-in interpreter accepts the run and fails every listing; the
    second lists one timed-out mutant and cannot apply it.
    """
    tree = tmp_path / "tree"
    (tree / "src").mkdir(parents=True)
    (tree / MODULE).write_text(SOURCE, encoding="utf-8")
    runner = tree / RUNNER
    runner.parent.mkdir()
    runner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runner.chmod(0o755)
    stand_in = tmp_path / "stand-in"
    stand_in.write_text('#!/bin/sh\n[ "$3" = run ]\n', encoding="utf-8")
    stand_in.chmod(0o755)
    target = gate.Target("calc", MODULE, RUNNER, DIGEST, 1, 0, 0)

    with pytest.raises(ValueError, match="mutmut could not list the killed mutants of calc"):
        gate.measure_statuses(tree, target, stand_in)

    stand_in.write_text(
        '#!/bin/sh\ncase "$3" in\n  run) exit 0 ;;\n'
        '  result-ids) [ "$4" = timeout ] && echo 7; exit 0 ;;\n  *) exit 1 ;;\nesac\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mutmut could not apply mutant 7 of calc"):
        gate.measure_statuses(tree, target, stand_in)
    assert (tree / MODULE).read_text(encoding="utf-8") == SOURCE


def test_export_needs_an_empty_directory_and_a_repository(tmp_path: Path) -> None:
    """The export refuses a directory with content, a missing one and a tree without Git."""
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "file").write_text("x", encoding="utf-8")
    empty = tmp_path / "empty"
    empty.mkdir()

    for destination in (occupied, tmp_path / "absent"):
        with pytest.raises(ValueError, match="export directory must exist and be empty"):
            gate.export_head(REPOSITORY, destination)
    with pytest.raises(ValueError, match="git archive --format=tar HEAD failed"):
        gate.export_head(tmp_path, empty)


def test_pin_and_installation_failures_are_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing pin file, a file without the pin, an absent distribution and a missing Git fail."""
    with pytest.raises(ValueError, match="cannot read the mutation tool pin"):
        gate.pinned_release(tmp_path)
    (tmp_path / gate.PIN_FILE).write_text("pytest==8.0.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not pin mutmut"):
        gate.pinned_release(tmp_path)
    with pytest.raises(ValueError, match="no-such-distribution is not installed"):
        gate.installed_release("no-such-distribution")

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ValueError, match="cannot run git"):
        gate.export_head(tmp_path, empty)


_VALID: dict[str, Any] = {
    "name": "calc",
    "module": MODULE,
    "runner": RUNNER,
    "source_sha256": DIGEST,
    "mutants": 4,
    "survived": 1,
    "timeout": 1,
}


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ([], "unsupported mutation survivor ceiling schema"),
        ({"schema": "other"}, "unsupported mutation survivor ceiling schema"),
        ({"release": "2.5"}, "must record the mutmut release"),
        ({"targets": {}}, "targets must be a list of objects"),
        ({"targets": [1]}, "targets must be a list of objects"),
        ({"targets": [{**_VALID, "name": "Calc"}]}, "needs a lower-case name"),
        ({"targets": [_VALID, _VALID]}, "recorded twice: calc"),
        ({"targets": [{**_VALID, "source_sha256": "abc"}]}, "needs the digest of its source"),
        ({"targets": [{**_VALID, "module": ""}]}, "needs a non-empty module"),
        ({"targets": [{**_VALID, "runner": "/bin/sh"}]}, "runner must stay inside the repository"),
        ({"targets": [{**_VALID, "module": "../calc.py"}]}, "module must stay inside"),
        ({"targets": [{**_VALID, "mutants": -1}]}, "mutants must be a non-negative integer"),
        ({"targets": [{**_VALID, "survived": True}]}, "survived must be a non-negative integer"),
        ({"targets": [{**_VALID, "survived": 4}]}, "counts exceed its mutants: calc"),
        ({"targets": [{**_VALID, "timeout": 1.5}]}, "timeout must be a non-negative integer"),
        ({"history": {}}, "history must be a list of objects"),
        ({"history": [1]}, "history must be a list of objects"),
    ],
)
def test_invalid_ceilings_are_refused(tmp_path: Path, document: Any, message: str) -> None:
    """Every malformed field of a ceiling document is refused with its own message.

    Parameters
    ----------
    tmp_path
        Directory that receives the document.
    document
        The whole document, or the fields replaced in an otherwise valid one.
    message
        Pattern the refusal must match.

    """
    valid = {"schema": gate.SCHEMA, "release": "2.5.1", "targets": [_VALID], "history": []}
    path = tmp_path / "ceiling.json"
    path.write_text(
        json.dumps({**valid, **document} if isinstance(document, dict) else document),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        gate.load_ceiling(path)


def test_written_ceiling_reads_back_unchanged(tmp_path: Path) -> None:
    """A ceiling survives a write and a read, and the file ends with a newline."""
    ceiling = gate.Ceiling(
        "2.5.1", (gate.Target(**_VALID),), ({"release": "2.5.0", "name": "calc"},)
    )
    path = tmp_path / "ceiling.json"

    gate.write_ceiling(path, ceiling)

    assert gate.load_ceiling(path) == ceiling
    assert path.read_text(encoding="utf-8").endswith("}\n")
