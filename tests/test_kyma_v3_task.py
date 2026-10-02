# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the KYMA v3 symbolic task
"""Tests for the KYMA v3 symbolic ground truth, split and design checks."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.benchmarks.kyma_v3 import task


def test_public_package_exports_preserve_task_identity_and_behavior() -> None:
    """Use deferred public exports to build and inspect the real symbolic dataset."""
    from scpn_quantum_control.benchmarks import kyma_v3

    assert kyma_v3.SymbolicDataset is task.SymbolicDataset
    assert kyma_v3.build_dataset is task.build_dataset
    assert kyma_v3.design_report is task.design_report
    dataset = kyma_v3.build_dataset()
    assert isinstance(dataset, kyma_v3.SymbolicDataset)
    assert dataset.size == 2176
    assert kyma_v3.design_report().test_items == 64
    assert set(kyma_v3.__all__) <= set(dir(kyma_v3))
    name = "undeclared_symbolic_export"
    with pytest.raises(AttributeError, match=name):
        getattr(kyma_v3, name)
    assert name not in vars(kyma_v3)


def test_operations_match_their_definitions() -> None:
    """Check that operations match their definitions."""
    assert task.apply_operation((1, 2, 3), 0) == (2, 3, 1)
    assert task.apply_operation((3, 2, 1), 1) == (1, 2, 1)
    assert task.apply_operation((0, 3, 2), 2) == (0, 0, 2)


def test_unknown_operation_is_refused() -> None:
    """Check that unknown operation is refused."""
    with pytest.raises(ValueError, match="unknown operation"):
        task.apply_operation((0, 0, 0), 3)


def test_configuration_runs_left_to_right() -> None:
    """Check that configuration runs left to right."""
    assert task.run_configuration((0, 1, 0), task.HELD_OUT) == (1, 0, 0)
    assert task.run_configuration((0, 1, 2), task.HELD_OUT) == (3, 2, 0)
    assert task.run_configuration((0, 1, 2), (1, 0)) == (1, 2, 1)


def test_state_and_configuration_inventories() -> None:
    """Check that state and configuration inventories."""
    states = task.all_states()
    assert len(states) == 64
    assert len(set(states)) == 64
    configurations = task.configurations()
    assert configurations[:3] == ((0,), (1,), (2,))
    assert len(configurations) == 12
    assert task.HELD_OUT in configurations


def test_every_configuration_is_a_bijection() -> None:
    """Check that every configuration is a bijection."""
    for configuration in task.configurations():
        finals = {task.run_configuration(state, configuration) for state in task.all_states()}
        assert len(finals) == 64


def test_dataset_split_sizes_and_order() -> None:
    """Check that dataset split sizes and order."""
    dataset = task.build_dataset()
    assert dataset.size == 2112 + 64
    assert int(np.sum(~dataset.is_test)) == 2112
    assert int(np.sum(dataset.is_test)) == 64
    assert not dataset.is_test[:2112].any()
    assert dataset.is_test[2112:].all()


def test_held_out_pair_never_enters_training() -> None:
    """Check that held out pair never enters training."""
    dataset = task.build_dataset()
    train = ~dataset.is_test
    held = (dataset.first_op == task.HELD_OUT[0]) & (dataset.second_op == task.HELD_OUT[1])
    assert not np.any(held & train)
    assert np.all(dataset.query[dataset.is_test] == task.EVALUATED_QUERY)


def test_singles_use_the_no_operation_code() -> None:
    """Check that singles use the no operation code."""
    dataset = task.build_dataset()
    singles = dataset.second_op == task.NO_OPERATION
    assert int(np.sum(singles)) == 3 * 64 * 3


def test_labels_are_the_symbolic_answers() -> None:
    """Check that labels are the symbolic answers."""
    dataset = task.build_dataset()
    for index in (0, 777, 2111, 2112, dataset.size - 1):
        state = tuple(int(v) for v in dataset.states[index])
        second = int(dataset.second_op[index])
        configuration = (int(dataset.first_op[index]),) + (
            () if second == task.NO_OPERATION else (second,)
        )
        final = task.run_configuration((state[0], state[1], state[2]), configuration)
        assert final[int(dataset.query[index])] == dataset.label[index]


def test_only_query_a_is_non_separable() -> None:
    """Check that only query a is non separable."""
    assert task.ambiguous_state_fraction(0) == 1.0
    assert task.ambiguous_state_fraction(1) == 0.0
    assert task.ambiguous_state_fraction(2) == 0.0


def test_held_out_answers_differ_from_every_trained_answer_function() -> None:
    """Check that held out answers differ from every trained answer function."""
    assert task.min_distance_to_trained_answers() == 48


def test_label_counts_are_uniform() -> None:
    """Check that label counts are uniform."""
    assert task.label_counts_are_uniform()


def test_design_report_collects_every_check() -> None:
    """Check that design report collects every check."""
    report = task.design_report()
    assert report == task.DesignReport(
        training_items=2112,
        test_items=64,
        ambiguous_fraction_by_query=(1.0, 0.0, 0.0),
        min_distance_to_trained_answers=48,
        uniform_label_counts=True,
    )


def test_training_marginal_floor_is_measured_on_the_test_items() -> None:
    """Check that training marginal floor is measured on the test items."""
    dataset = task.build_dataset()
    assert task.training_marginal_accuracy(dataset) == 0.25


def test_label_balance_check_detects_an_unbalanced_answer() -> None:
    """Check that label balance check detects an unbalanced answer."""
    uniform = [value % 4 for value in range(64)]
    assert task.answers_are_uniform([uniform])
    assert not task.answers_are_uniform([uniform, [0] * 64])
