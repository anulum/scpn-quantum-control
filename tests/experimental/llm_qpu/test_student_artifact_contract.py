# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental student artifact contract
"""Exercise offline students against their protocol, split, teacher features and worker."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control.experimental.llm_qpu.contracts import (
    FEATURE_BATCH_SCHEMA,
    STUDENT_ARTIFACT_SCHEMA,
    ArtifactHeader,
    CellKey,
    ExperimentProtocol,
    FeatureBatch,
    FeatureRow,
    SplitManifest,
    StudentArtifact,
    TaskSpec,
    canonical_bytes,
    decode_contract,
    decode_student_artifact,
    f64,
    validate_student_artifact,
)

_ROOT = Path(__file__).resolve().parents[3]
_WORKER = _ROOT / "experimental_workers/llm_qpu/protocol/worker.py"
_BASE = "de259e4837a92ecb09b63c8f4332dbcf3d21021c"
_REVISION = "0883d1e5204ffe4594bb2f5b6e7b6a5d0a915209"
_OBSERVABLES = ("Z0", "Z1", "Z2", "Z3", "Z0Z1", "Z1Z2", "Z2Z3")
_FLOATS = ("learning_rate", "fidelity", "utility")


def _sha(value: object) -> str:
    """Return the SHA-256 of a value's canonical wire bytes."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _header(
    kind: str, content: dict[str, object], parents: tuple[str, ...], origin: str
) -> ArtifactHeader:
    """Return a design-only header bound to the content and parents."""
    return ArtifactHeader(
        kind,
        _sha(content),
        tuple(sorted(parents)),
        _BASE,
        _REVISION,
        "offline_design",
        origin,
        "design_only",
    )


def _design(
    metric: str = "mean_log_loss", *, model: str = "c" * 64
) -> tuple[TaskSpec, SplitManifest, ExperimentProtocol]:
    """Return a frozen task, its split and a two-arm protocol for the metric."""
    task_fields: dict[str, Any] = {
        "task_id": "fixture-label",
        "objective": "Predict held-out generated labels",
        "source_kind": "synthetic_classical",
        "target_origin": "classical_generator",
        "label_schema_digest": "a" * 64,
        "causal_cutoff": 0,
        "primary_metric": metric,
        "group_definition": "One generated scene per group",
    }
    task_content = {
        "schema": "scpn.experimental.llm_qpu.task_spec.v2",
        "object_kind": "task_spec",
        **task_fields,
    }
    task = TaskSpec(
        **task_fields,
        header=_header("task_spec", task_content, ("a" * 64,), "synthetic_classical"),
    )
    split_fields: dict[str, Any] = {
        "task_digest": _sha(task.to_wire()),
        "dataset_digest": "b" * 64,
        "train_groups": ("train-a", "train-b"),
        "dev_groups": ("dev-a",),
        "test_groups": ("test-a",),
        "seed": 7,
        "dedup_rule": "exact_source_digest",
        "test_target_custodian": "separate_locked_evaluator",
        "transform_fit_split": "train_only",
    }
    split_content = {
        "schema": "scpn.experimental.llm_qpu.split_manifest.v2",
        "object_kind": "split_manifest",
        **{k: list(v) if k.endswith("_groups") else v for k, v in split_fields.items()},
    }
    split_parents = (split_fields["task_digest"], split_fields["dataset_digest"])
    split = SplitManifest(
        **split_fields,
        header=_header("split_manifest", split_content, split_parents, "synthetic_classical"),
    )
    fields: dict[str, Any] = {
        "experiment_id": "fixture-protocol",
        "evaluation_id": "heldout-v1",
        "task_digest": _sha(task.to_wire()),
        "split_digest": _sha(split.to_wire()),
        "model_digest": model,
        "compressor_digest": "d" * 64,
        "kernel_plan_digest": "e" * 64,
        "measurement_plan_digest": "f" * 64,
        "analysis_plan_digest": "0" * 64,
        "mode": "contextual_latent_transform",
        "prediction_mode": "teacher_forced_trace",
        "arms": ("classical", "quantum"),
        "classical_arm": "classical",
        "quantum_arm": "quantum",
        "primary_metric": metric,
        "contrast_direction": "quantum_minus_classical"
        if metric == "accuracy"
        else "classical_minus_quantum",
        "minimum_relevant_delta": 0.02,
        "max_dev_fits_per_arm": 8,
        "max_qpu_evaluations": 128,
        "stopping_rule": "fixed_split_no_optional_stopping",
        "failure_policy": "retain_all_ids_report_missing_and_sensitivity",
        "inference_unit": "source_group_paired",
        "analysis_scope": "exploratory",
    }
    content = {
        "schema": "scpn.experimental.llm_qpu.experiment_protocol.v2",
        "object_kind": "experiment_protocol",
        **{
            k: list(v) if k == "arms" else f64(v) if k == "minimum_relevant_delta" else v
            for k, v in fields.items()
        },
    }
    names = (
        "task_digest",
        "split_digest",
        "model_digest",
        "compressor_digest",
        "kernel_plan_digest",
        "measurement_plan_digest",
        "analysis_plan_digest",
    )
    protocol = ExperimentProtocol(
        **fields,
        header=_header(
            "experiment_protocol",
            content,
            tuple(fields[name] for name in names),
            "synthetic_classical",
        ),
    )
    return task, split, protocol


def _teacher(origin: str = "synthetic_classical", digest: str = "9" * 64) -> FeatureBatch:
    """Return a one-row feature batch standing in for the quantum teacher."""
    row = FeatureRow(
        cell_ordinal=0,
        key=CellKey("exp-1", "sample-0", "seq-1", "arm-a", "zz", "rep-0", "epoch-0", "bind-0"),
        measurement_request_digest="8" * 64,
        shots=1,
        values=(1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0),
        covariance=None,
        missing_reason=None,
    )
    content: dict[str, object] = {
        "schema": FEATURE_BATCH_SCHEMA,
        "object_kind": "feature_batch",
        "raw_result_set_digest": digest,
        "estimator": "sampled_mean_v1",
        "basis_id": "Z",
        "observable_ids": list(_OBSERVABLES),
        "rows": [row.to_wire()],
    }
    return FeatureBatch(
        raw_result_set_digest=digest,
        estimator="sampled_mean_v1",
        basis_id="Z",
        observable_ids=_OBSERVABLES,
        rows=(row,),
        header=_header("feature_batch", content, (digest,), origin),
    )


def _student(
    protocol: ExperimentProtocol, teacher: FeatureBatch, **changes: Any
) -> StudentArtifact:
    """Build a student of the protocol distilled from the teacher, hashing its content."""
    hardware = teacher.header.data_origin == "provider_result"
    fields: dict[str, Any] = {
        "student_id": "student-mlp-001",
        "protocol_digest": _sha(protocol.to_wire()),
        "model_digest": protocol.model_digest,
        "compressor_digest": protocol.compressor_digest,
        "teacher_kind": "hardware_features" if hardware else "simulated_features",
        "teacher_digest": _sha(teacher.to_wire()),
        "targets": ("Z0", "Z2", "Z0Z1"),
        "architecture": "mlp-2x64-tanh",
        "weights_digest": "1" * 64,
        "config_digest": "2" * 64,
        "training_groups": ("train-a", "train-b"),
        "optimizer": "adam",
        "learning_rate": 0.001,
        "epochs": 200,
        "seed": 11,
        "device": "cpu",
        "runtime": "numpy",
        "runtime_version": "2.3.0",
        "heldout_split": "dev",
        "fidelity_metric": "feature_mse",
        "fidelity": 0.0125,
        "utility_metric": protocol.primary_metric,
        "utility": 0.61,
        "output_class": "classical_offline",
        **changes,
    }
    content = {
        "schema": STUDENT_ARTIFACT_SCHEMA,
        "object_kind": "student_artifact",
        **{
            name: list(value)
            if type(value) is tuple
            else f64(value)
            if name in _FLOATS and type(value) is float
            else value
            for name, value in fields.items()
        },
    }
    origin = {"hardware_features": "provider_result"}.get(
        fields["teacher_kind"], "synthetic_classical"
    )
    parents = (fields["protocol_digest"], fields["teacher_digest"])
    return StudentArtifact(**fields, header=_header("student_artifact", content, parents, origin))


def _worker(request: dict[str, object], tmp_path: Path) -> subprocess.CompletedProcess[bytes]:
    """Run the standalone protocol worker on one request with poisoned provider tokens."""
    return subprocess.run(
        [sys.executable, "-S", str(_WORKER)],
        input=json.dumps(request).encode(),
        cwd=tmp_path,
        env={"PYTHONPATH": str(tmp_path), "IQM_TOKEN": "poison", "IBM_QUANTUM_TOKEN": "poison"},
        capture_output=True,
        timeout=15,
        check=False,
    )


def _request(
    design: tuple[TaskSpec, SplitManifest, ExperimentProtocol],
    teacher: FeatureBatch,
    student: object,
) -> dict[str, object]:
    """Return one worker student request."""
    task, split, protocol = design
    return {
        "op": "roundtrip_student_artifact",
        "task": task.to_wire(),
        "split": split.to_wire(),
        "protocol": protocol.to_wire(),
        "teacher": teacher.to_wire(),
        "student": student,
    }


def test_students_of_simulated_and_hardware_teachers_validate_and_roundtrip(
    tmp_path: Path,
) -> None:
    """Simulated, hardware and accuracy students validate in both validators."""
    design = _design()
    _, split, protocol = design
    accuracy = _design("accuracy")
    simulated, hardware = _teacher(), _teacher("provider_result")
    cases = (
        (design, simulated, _student(protocol, simulated)),
        (design, hardware, _student(protocol, hardware)),
        (accuracy, simulated, _student(accuracy[2], simulated, utility=1.0, heldout_split="test")),
        (
            design,
            simulated,
            _student(
                protocol,
                simulated,
                targets=_OBSERVABLES,
                fidelity_metric="feature_mae",
                fidelity=0.0,
            ),
        ),
    )
    for used, teacher, student in cases:
        validate_student_artifact(student, used[2], used[1], teacher)
        assert decode_student_artifact(canonical_bytes(student.to_wire())) == student
        response = _worker(_request(used, teacher, student.to_wire()), tmp_path)
        assert response.returncode == 0, response.stdout.decode()
        payload = json.loads(response.stdout)
        assert payload["status"] == "validated_roundtrip_no_compute"
        assert payload["hardware_submission_enabled"] is False
        assert payload["student_sha256"] == _sha(student.to_wire())
    assert cases[0][2].header.data_origin == "synthetic_classical"
    assert cases[1][2].header.data_origin == "provider_result"
    assert cases[0][2].output_class == "classical_offline"


def test_student_refuses_quantum_labels_unbounded_training_and_scores() -> None:
    """A student is classical, trained finitely, and scored within its metric's range."""
    _, _, protocol = _design()
    valid = _student(protocol, _teacher())
    for changes, reason in (
        ({"output_class": "quantum"}, "^a student runs classically; its output is never quantum"),
        ({"output_class": "hardware"}, "^a student runs classically; its output is never quantum"),
        ({"output_class": "gpu"}, "^unknown student output class$"),
        ({"student_id": ""}, "student_id must be bounded nonempty text"),
        ({"weights_digest": "W" * 64}, "^weights_digest must be a lowercase SHA-256 digest$"),
        ({"teacher_kind": "classical_features"}, "^a teacher is hardware or simulated quantum"),
        ({"targets": ()}, "^targets must be a nonempty tuple$"),
        ({"targets": ["Z0"]}, "^targets must be a nonempty tuple$"),
        ({"targets": ("Z0", "Z0")}, "^targets must be unique$"),
        ({"training_groups": ("train-b", "train-a")}, "^training groups must be sorted$"),
        ({"training_groups": ("train-a", "train-a")}, "^training_groups must be unique$"),
        ({"learning_rate": 0.0}, "^learning rate must be positive$"),
        ({"learning_rate": 1}, "^learning rate must be a finite float$"),
        ({"epochs": 0}, "^epochs must be a bounded positive integer$"),
        ({"epochs": 1_000_001}, "^epochs must be a bounded positive integer$"),
        ({"seed": -1}, "^seed must be a non-negative 63-bit integer$"),
        ({"heldout_split": "train"}, "^held-out scores come from the dev or the test split$"),
        ({"fidelity_metric": "feature_r2"}, "^unknown fidelity metric$"),
        ({"fidelity": -0.1}, "^a feature error cannot be negative$"),
        ({"fidelity": float("nan")}, "^fidelity must be a finite float$"),
        ({"utility_metric": "f1"}, "^unknown utility metric$"),
        ({"utility": -0.5}, "^utility must be non-negative, and at most 1 for accuracy$"),
        ({"utility_metric": "accuracy", "utility": 1.5}, "^utility must be non-negative, and at"),
    ):
        with pytest.raises(ValueError, match=reason):
            replace(valid, **changes)
    assert _student(protocol, _teacher(), epochs=1_000_000).epochs == 1_000_000
    with pytest.raises(ValueError, match="^StudentArtifact requires its exact artifact header$"):
        replace(valid, header=replace(valid.header, object_kind="feature_batch"))
    with pytest.raises(ValueError, match="^a student's data origin is its teacher's$"):
        replace(valid, header=replace(valid.header, data_origin="provider_result"))
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(valid, epochs=201)


def test_student_validation_refuses_other_models_teachers_groups_and_relabelling() -> None:
    """The student binds the protocol's model, compressor, split and exact teacher."""
    _, split, protocol = _design()
    _, _, other_protocol = _design(model="7" * 64)
    _, _, accuracy = _design("accuracy")
    teacher = _teacher()
    other_teacher = _teacher(digest="6" * 64)
    for student, used_split, used_teacher, reason in (
        (
            _student(other_protocol, teacher),
            split,
            teacher,
            "^the student names a different frozen",
        ),
        (
            _student(protocol, teacher, model_digest="7" * 64),
            split,
            teacher,
            "^the student names another",
        ),
        (
            _student(protocol, teacher, compressor_digest="7" * 64),
            split,
            teacher,
            "^the student names",
        ),
        (
            _student(protocol, other_teacher),
            split,
            teacher,
            "^the student names a different teacher",
        ),
        (
            _student(protocol, teacher, teacher_kind="hardware_features"),
            split,
            teacher,
            "^teacher kind does not match the teacher's origin; no relabelling$",
        ),
        (
            _student(protocol, teacher, targets=("Z2", "Z0")),
            split,
            teacher,
            "^student targets must be",
        ),
        (
            _student(protocol, teacher, targets=("Z0", "X9")),
            split,
            teacher,
            "^student targets must be",
        ),
        (
            _student(protocol, teacher, training_groups=("dev-a", "train-a")),
            split,
            teacher,
            "^a student trains on train groups only$",
        ),
        (
            _student(protocol, teacher, utility_metric="mse"),
            split,
            teacher,
            "^student utility must use",
        ),
    ):
        with pytest.raises(ValueError, match=reason):
            validate_student_artifact(student, protocol, used_split, used_teacher)
    _, mismatched, _ = _design("accuracy")
    with pytest.raises(ValueError, match="^the protocol binds a different split$"):
        validate_student_artifact(_student(protocol, teacher), protocol, mismatched, teacher)
    not_a_student: Any = "student"
    with pytest.raises(ValueError, match="^student validation needs a StudentArtifact$"):
        validate_student_artifact(not_a_student, accuracy, split, teacher)


def _forge(wire: dict[str, Any], path: tuple[str | int, ...], value: object) -> dict[str, Any]:
    """Return the wire with one nested field replaced."""
    holder: Any = wire
    for key in path[:-1]:
        holder = holder[key]
    holder[path[-1]] = value
    return wire


_WORKER_FORGERIES: tuple[tuple[tuple[str | int, ...], object, str], ...] = (
    (("unexpected",), True, "StudentArtifact fields mismatch"),
    (("schema",), "scpn.experimental.llm_qpu.student_artifact.v2", "unknown StudentArtifact"),
    (("output_class",), "quantum", "its output is never quantum or hardware"),
    (("architecture",), "", "architecture must be bounded normalized text"),
    (("weights_digest",), "W" * 64, "invalid weights_digest"),
    (("teacher_kind",), "classical_features", "a teacher is hardware or simulated quantum"),
    (("targets",), [], "targets must be a nonempty unique list"),
    (("training_groups",), ["train-a", "train-a"], "training_groups must be a nonempty unique"),
    (("training_groups",), ["train-b", "train-a"], "training groups must be sorted"),
    (("learning_rate",), {"$f64": "0x0.0p+0"}, "learning rate must be positive"),
    (("learning_rate",), 0.001, "feature numbers must be f64 tags"),
    (("epochs",), 0, "epochs must be a bounded positive integer"),
    (("seed",), -1, "seed must be a non-negative 63-bit integer"),
    (("heldout_split",), "train", "held-out scores come from the dev or the test split"),
    (("fidelity_metric",), "feature_r2", "unknown fidelity metric"),
    (("fidelity",), {"$f64": "-0x1.0p-4"}, "a feature error cannot be negative"),
    (("utility_metric",), "f1", "unknown utility metric"),
    (("utility",), {"$f64": "-0x1.0p-1"}, "utility must be non-negative"),
    (("header", "data_origin"), "provider_result", "artifact data origin mismatch"),
    (("header", "content_digest"), "f" * 64, "artifact content digest mismatch"),
)


def test_student_wire_is_exact_here_and_in_the_worker(tmp_path: Path) -> None:
    """Edited wires are refused by the decoder and the worker; decode_contract stays closed."""
    design = _design()
    _, split, protocol = design
    teacher = _teacher()
    student = _student(protocol, teacher)
    with pytest.raises(ValueError, match="^unsupported contract schema$"):
        decode_contract(canonical_bytes(student.to_wire()))
    decoder_cases: tuple[tuple[tuple[str | int, ...], object, str], ...] = (
        (("unexpected",), True, "^StudentArtifact fields mismatch$"),
        (("schema",), "scpn.experimental.llm_qpu.student_artifact.v2", "^unknown StudentArtifact"),
        (("object_kind",), "feature_batch", "^unknown StudentArtifact schema$"),
        (("targets",), "Z0", "^targets and training groups must be lists on wire$"),
        (("training_groups",), "train-a", "^targets and training groups must be lists on wire$"),
        (("fidelity",), "0.0125", "^fidelity must be an f64 tag on wire$"),
    )
    for path, value, reason in decoder_cases:
        with pytest.raises(ValueError, match=reason):
            decode_student_artifact(canonical_bytes(_forge(student.to_wire(), path, value)))
    for path, forgery, refusal in _WORKER_FORGERIES:
        response = _worker(
            _request(design, teacher, _forge(student.to_wire(), path, forgery)), tmp_path
        )
        assert response.returncode == 2, path
        assert refusal in json.loads(response.stdout)["reason"], path
    accuracy = _design("accuracy")
    _, _, other_protocol = _design(model="7" * 64)
    consistent: tuple[tuple[Any, FeatureBatch, StudentArtifact, str], ...] = (
        (
            design,
            teacher,
            _student(other_protocol, teacher),
            "the student names a different frozen",
        ),
        (design, teacher, _student(protocol, teacher, model_digest="7" * 64), "another model"),
        (design, _teacher(digest="6" * 64), student, "the student names a different teacher"),
        (
            design,
            teacher,
            _student(protocol, teacher, teacher_kind="hardware_features"),
            "no relabelling",
        ),
        (design, teacher, _student(protocol, teacher, targets=("Z2", "Z0")), "in teacher order"),
        (
            design,
            teacher,
            _student(protocol, teacher, training_groups=("dev-a", "train-a")),
            "a student trains on train groups only",
        ),
        (design, teacher, _student(protocol, teacher, utility_metric="mse"), "primary metric"),
        (
            (accuracy[0], accuracy[1], protocol),
            teacher,
            student,
            "the protocol binds a different split",
        ),
    )
    for used, used_teacher, edited, refusal in consistent:
        response = _worker(_request(used, used_teacher, edited.to_wire()), tmp_path)
        assert response.returncode == 2, refusal
        assert refusal in json.loads(response.stdout)["reason"], refusal
    forged_teacher = _forge(teacher.to_wire(), ("header", "data_origin"), "provider_metadata")
    request = _request(design, teacher, student.to_wire())
    request["teacher"] = forged_teacher
    response = _worker(request, tmp_path)
    assert response.returncode == 2
    assert "a feature batch derives from provider results" in json.loads(response.stdout)["reason"]
    missing = _request(design, teacher, student.to_wire())
    del missing["teacher"]
    response = _worker(missing, tmp_path)
    assert response.returncode == 2
    assert json.loads(response.stdout)["reason"] == "student request fields mismatch"
