"""Matrix decomposition algorithms.

These include PCA, NMF, ICA, and more. Most of the algorithms of this module can be
regarded as dimensionality reduction techniques.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn_upgrade.decomposition._dict_learning import (
    DictionaryLearning,
    MiniBatchDictionaryLearning,
    SparseCoder,
    dict_learning,
    dict_learning_online,
    sparse_encode,
)
from sklearn_upgrade.decomposition._factor_analysis import FactorAnalysis
from sklearn_upgrade.decomposition._fastica import FastICA, fastica
from sklearn_upgrade.decomposition._incremental_pca import IncrementalPCA
from sklearn_upgrade.decomposition._kernel_pca import KernelPCA
from sklearn_upgrade.decomposition._lda import LatentDirichletAllocation
from sklearn_upgrade.decomposition._nmf import NMF, MiniBatchNMF, non_negative_factorization
from sklearn_upgrade.decomposition._pca import PCA
from sklearn_upgrade.decomposition._sparse_pca import MiniBatchSparsePCA, SparsePCA
from sklearn_upgrade.decomposition._truncated_svd import TruncatedSVD
from sklearn_upgrade.utils.extmath import randomized_svd

__all__ = [
    "NMF",
    "PCA",
    "DictionaryLearning",
    "FactorAnalysis",
    "FastICA",
    "IncrementalPCA",
    "KernelPCA",
    "LatentDirichletAllocation",
    "MiniBatchDictionaryLearning",
    "MiniBatchNMF",
    "MiniBatchSparsePCA",
    "SparseCoder",
    "SparsePCA",
    "TruncatedSVD",
    "dict_learning",
    "dict_learning_online",
    "fastica",
    "non_negative_factorization",
    "randomized_svd",
    "sparse_encode",
]
