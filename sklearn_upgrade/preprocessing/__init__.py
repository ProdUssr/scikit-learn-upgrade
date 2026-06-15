"""Methods for scaling, centering, normalization, binarization, and more."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn_upgrade.preprocessing._data import (
    Binarizer,
    KernelCenterer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    add_dummy_feature,
    binarize,
    maxabs_scale,
    minmax_scale,
    normalize,
    power_transform,
    quantile_transform,
    robust_scale,
    scale,
)
from sklearn_upgrade.preprocessing._discretization import KBinsDiscretizer
from sklearn_upgrade.preprocessing._encoders import OneHotEncoder, OrdinalEncoder
from sklearn_upgrade.preprocessing._function_transformer import FunctionTransformer
from sklearn_upgrade.preprocessing._label import (
    LabelBinarizer,
    LabelEncoder,
    MultiLabelBinarizer,
    label_binarize,
)
from sklearn_upgrade.preprocessing._polynomial import PolynomialFeatures, SplineTransformer
from sklearn_upgrade.preprocessing._target_encoder import TargetEncoder

__all__ = [
    "Binarizer",
    "FunctionTransformer",
    "KBinsDiscretizer",
    "KernelCenterer",
    "LabelBinarizer",
    "LabelEncoder",
    "MaxAbsScaler",
    "MinMaxScaler",
    "MultiLabelBinarizer",
    "Normalizer",
    "OneHotEncoder",
    "OrdinalEncoder",
    "PolynomialFeatures",
    "PowerTransformer",
    "QuantileTransformer",
    "RobustScaler",
    "SplineTransformer",
    "StandardScaler",
    "TargetEncoder",
    "add_dummy_feature",
    "binarize",
    "label_binarize",
    "maxabs_scale",
    "minmax_scale",
    "normalize",
    "power_transform",
    "quantile_transform",
    "robust_scale",
    "scale",
]
