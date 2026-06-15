"""Ensemble-based methods for classification, regression and anomaly detection."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn_upgrade.ensemble._bagging import BaggingClassifier, BaggingRegressor
from sklearn_upgrade.ensemble._base import BaseEnsemble
from sklearn_upgrade.ensemble._forest import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    RandomTreesEmbedding,
)
from sklearn_upgrade.ensemble._gb import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn_upgrade.ensemble._hist_gradient_boosting.gradient_boosting import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn_upgrade.ensemble._iforest import IsolationForest
from sklearn_upgrade.ensemble._stacking import StackingClassifier, StackingRegressor
from sklearn_upgrade.ensemble._voting import VotingClassifier, VotingRegressor
from sklearn_upgrade.ensemble._weight_boosting import AdaBoostClassifier, AdaBoostRegressor

__all__ = [
    "AdaBoostClassifier",
    "AdaBoostRegressor",
    "BaggingClassifier",
    "BaggingRegressor",
    "BaseEnsemble",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
    "IsolationForest",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "RandomTreesEmbedding",
    "StackingClassifier",
    "StackingRegressor",
    "VotingClassifier",
    "VotingRegressor",
]
