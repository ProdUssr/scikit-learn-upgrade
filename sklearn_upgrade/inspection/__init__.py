"""Tools for model inspection."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn_upgrade.inspection._partial_dependence import partial_dependence
from sklearn_upgrade.inspection._permutation_importance import permutation_importance
from sklearn_upgrade.inspection._plot.decision_boundary import DecisionBoundaryDisplay
from sklearn_upgrade.inspection._plot.partial_dependence import PartialDependenceDisplay

__all__ = [
    "DecisionBoundaryDisplay",
    "PartialDependenceDisplay",
    "partial_dependence",
    "permutation_importance",
]
