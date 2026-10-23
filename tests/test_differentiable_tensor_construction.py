# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native tensor capability refusals
"""Exercise genuine incompatible inputs and caller-owned refusal construction."""

from __future__ import annotations

import shlex
from collections.abc import Callable

import numpy as np
import pytest
import yaml

from scpn_quantum_control.differentiable_tensor_construction import (
    TensorConstructionPrimitives,
    construct_tensorflow_tensor,
    construct_torch_tensor,
    prepare_torch_trainable_tensor,
)
from tools import differentiable_api_quality_gates as quality
from tools import preflight
from tools.ci_workflow_inventory import (
    load_ci_workflow_policy,
    read_ci_workflow_source,
    workflow_path_for_job,
)


def _primitives(
    invalid: Callable[[], type[RuntimeError]] = lambda: RuntimeError,
) -> TensorConstructionPrimitives:
    return TensorConstructionPrimitives(
        attribute=lambda: getattr, predicate=lambda: callable, invalid=invalid
    )


@pytest.mark.parametrize(
    ("construct", "message"),
    [
        (construct_torch_tensor, "PyTorch module does not expose as_tensor or tensor"),
        (construct_tensorflow_tensor, "TensorFlow module does not expose convert_to_tensor"),
    ],
)
def test_incompatible_namespace_has_original_refusal(
    construct: Callable[..., object], message: str
) -> None:
    """An actual Python object has no native tensor factory to execute.

    Parameters
    ----------
    construct
        Public native tensor construction operation.
    message
        Required original capability refusal text.

    """
    with pytest.raises(RuntimeError) as caught:
        construct(object(), np.array([0.2]), primitives=_primitives())
    assert str(caught.value) == message


def test_host_array_cannot_become_a_trainable_native_tensor() -> None:
    """A real NumPy array lacks PyTorch's gradient-recording capability."""
    values = np.array([0.2, -0.4])
    with pytest.raises(RuntimeError, match="PyTorch tensor does not expose requires_grad_"):
        prepare_torch_trainable_tensor(values, primitives=_primitives())
    np.testing.assert_array_equal(values, [0.2, -0.4])


def test_refusal_constructor_is_read_after_capability_lookup() -> None:
    """An actual descriptor may replace the error type before rejection."""
    constructors: list[type[RuntimeError]] = [RuntimeError]

    class CapabilityError(RuntimeError):
        """The constructor selected by descriptor evaluation."""

    class UnavailableConversion:
        """A non-callable conversion attribute with an observable descriptor."""

        @property
        def convert_to_tensor(self) -> None:
            """Select the refusal constructor while resolving the capability."""
            constructors[0] = CapabilityError

    with pytest.raises(CapabilityError) as caught:
        construct_tensorflow_tensor(
            UnavailableConversion(), [0.2], primitives=_primitives(lambda: constructors[0])
        )
    assert str(caught.value) == "TensorFlow module does not expose convert_to_tensor"


def test_preflight_reaches_real_native_tensor_execution_and_exact_coverage() -> None:
    """Keep real SDK execution reachable through ordinary aggregate preflight."""
    actual = dict(preflight.DIFFERENTIABLE_QUALITY_COVERAGE_GATES)
    for name, command in quality.build_native_tensor_coverage_gates(preflight._PY):
        assert actual[name] == command


def test_required_hosted_framework_job_uses_the_exact_native_commands() -> None:
    """The existing CPU-overlay job runs both real SDK tests at exact coverage."""
    workflow = yaml.safe_load(read_ci_workflow_source())
    steps = workflow["jobs"]["differentiable-parity"]["steps"]
    commands = [shlex.split(step["run"]) for step in steps if "run" in step]
    for _name, command in quality.build_native_tensor_coverage_gates("python"):
        assert command in commands
    assert "differentiable-parity" in workflow["jobs"]["ci-gate"]["needs"]


def test_global_policy_combines_exact_commit_native_tensor_coverage() -> None:
    """The base coverage policy waits for and combines the real SDK measurement."""
    policy = load_ci_workflow_policy()
    category = next(item for item in policy["categories"] if item["id"] == "native-integration")
    producer = yaml.safe_load(workflow_path_for_job("differentiable-parity").read_text())
    consumer = yaml.safe_load(workflow_path_for_job("test").read_text())
    assert "framework-parity" in category["caller_needs"]
    upload = next(
        step
        for step in producer["jobs"]["differentiable-parity"]["steps"]
        if step.get("name") == "Upload exact-commit native tensor construction coverage"
    )
    download = next(
        step
        for step in consumer["jobs"]["test"]["steps"]
        if step.get("name")
        == "Download exact-commit native tensor construction coverage (3.12 only)"
    )
    assert upload["with"]["name"] == download["with"]["name"]
    assert "github.sha" in upload["with"]["name"]
    assert upload["with"]["include-hidden-files"] is True
    assert upload["with"]["if-no-files-found"] == "error"
    coverage_step = next(
        step
        for step in consumer["jobs"]["test"]["steps"]
        if step.get("name") == "Run tests with coverage (3.12 only)"
    )["run"]
    combine = 'python -m coverage combine --append --keep "$RUNNER_TEMP/native-tensor-coverage/.coverage.differentiable-tensor-construction"'
    assert combine in coverage_step
    assert coverage_step.index(combine) < coverage_step.index("tools/audit_coverage_policy.py")
