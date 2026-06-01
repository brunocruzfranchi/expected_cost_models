import numpy as np
import pytest

from expected_cost import (
    CostMatrix,
    average_cost,
    average_cost_for_bayes_decisions,
    average_cost_for_optimal_decisions,
    bayes_decisions,
    ec,
    generalized_confusion_matrix,
    utils,
)


def test_package_exports_modules_and_core_functions():
    assert ec.average_cost is average_cost
    log_posteriors = utils.llrs_to_logpost(np.array([-1.0, 1.0]), [0.5, 0.5])

    assert log_posteriors.shape == (2, 2)
    np.testing.assert_allclose(np.exp(log_posteriors).sum(axis=1), 1.0)


def test_average_cost_uses_default_zero_one_costs():
    result = average_cost([0, 1], [0, 0])

    assert result == pytest.approx(0.5)


def test_average_cost_accepts_list_priors_with_explicit_costs():
    costs = CostMatrix([[0.0, 2.0], [1.0, 0.0]])

    result = average_cost([0, 1], [1, 0], costs=costs, priors=[0.25, 0.75])

    assert result == pytest.approx(1.25)


def test_generalized_confusion_matrix_validates_lengths():
    with pytest.raises(ValueError, match="same number of samples"):
        generalized_confusion_matrix([0, 1], [0])


def test_generalized_confusion_matrix_normalizes_by_target_class():
    matrix = generalized_confusion_matrix(
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        normalize="true",
    )

    np.testing.assert_allclose(matrix, [[0.5, 0.5], [0.0, 1.0]])


def test_bayes_decisions_uses_default_costs():
    scores = np.log(np.array([[0.9, 0.1], [0.2, 0.8]]))

    decisions, posteriors = bayes_decisions(scores, costs=None)

    np.testing.assert_array_equal(decisions, [0, 1])
    np.testing.assert_allclose(posteriors, [[0.9, 0.1], [0.2, 0.8]])


def test_average_cost_for_bayes_decisions_uses_default_costs():
    targets = np.array([0, 1])
    scores = np.log(np.array([[0.9, 0.1], [0.2, 0.8]]))

    cost, decisions = average_cost_for_bayes_decisions(targets, scores)

    assert cost == pytest.approx(0.0)
    np.testing.assert_array_equal(decisions, [0, 1])


def test_average_cost_for_optimal_decisions_uses_default_costs():
    targets = np.array([0, 0, 1, 1])
    scores = np.log(
        np.array(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.2, 0.8],
                [0.1, 0.9],
            ]
        )
    )

    assert average_cost_for_optimal_decisions(targets, scores) == pytest.approx(0.0)


def test_cost_matrix_from_utilities_normalizes_rows():
    costs = CostMatrix.from_utilities(np.array([[2.0, 0.0], [0.0, 1.0]]))

    np.testing.assert_allclose(costs.get_matrix(), [[0.0, 2.0], [1.0, 0.0]])
