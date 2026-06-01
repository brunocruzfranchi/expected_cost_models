"""Expected cost metrics and utilities."""

from . import ec, utils
from .ec import (
    CostMatrix,
    average_cost,
    average_cost_for_bayes_decisions,
    average_cost_for_optimal_decisions,
    average_cost_from_confusion_matrix,
    bayes_decisions,
    generalized_confusion_matrix,
    get_posteriors_from_scores,
)

__all__ = [
    "CostMatrix",
    "average_cost",
    "average_cost_for_bayes_decisions",
    "average_cost_for_optimal_decisions",
    "average_cost_from_confusion_matrix",
    "bayes_decisions",
    "ec",
    "generalized_confusion_matrix",
    "get_posteriors_from_scores",
    "utils",
]
