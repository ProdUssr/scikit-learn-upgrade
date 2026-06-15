"""The k-nearest neighbors algorithms."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn_upgrade.neighbors._ball_tree import BallTree
from sklearn_upgrade.neighbors._base import (
    VALID_METRICS,
    VALID_METRICS_SPARSE,
    sort_graph_by_row_values,
)
from sklearn_upgrade.neighbors._classification import (
    KNeighborsClassifier,
    RadiusNeighborsClassifier,
)
from sklearn_upgrade.neighbors._graph import (
    KNeighborsTransformer,
    RadiusNeighborsTransformer,
    kneighbors_graph,
    radius_neighbors_graph,
)
from sklearn_upgrade.neighbors._kd_tree import KDTree
from sklearn_upgrade.neighbors._kde import KernelDensity
from sklearn_upgrade.neighbors._lof import LocalOutlierFactor
from sklearn_upgrade.neighbors._nca import NeighborhoodComponentsAnalysis
from sklearn_upgrade.neighbors._nearest_centroid import NearestCentroid
from sklearn_upgrade.neighbors._regression import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn_upgrade.neighbors._unsupervised import NearestNeighbors

__all__ = [
    "VALID_METRICS",
    "VALID_METRICS_SPARSE",
    "BallTree",
    "KDTree",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "KNeighborsTransformer",
    "KernelDensity",
    "LocalOutlierFactor",
    "NearestCentroid",
    "NearestNeighbors",
    "NeighborhoodComponentsAnalysis",
    "RadiusNeighborsClassifier",
    "RadiusNeighborsRegressor",
    "RadiusNeighborsTransformer",
    "kneighbors_graph",
    "radius_neighbors_graph",
    "sort_graph_by_row_values",
]
