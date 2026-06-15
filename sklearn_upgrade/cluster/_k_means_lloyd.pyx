from cython cimport floating
from cython.parallel import prange, parallel
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memset
from libc.float cimport DBL_MAX, FLT_MAX

from sklearn_upgrade.utils._openmp_helpers cimport omp_lock_t
from sklearn_upgrade.utils._openmp_helpers cimport omp_init_lock
from sklearn_upgrade.utils._openmp_helpers cimport omp_destroy_lock
from sklearn_upgrade.utils._openmp_helpers cimport omp_set_lock
from sklearn_upgrade.utils._openmp_helpers cimport omp_unset_lock
from sklearn_upgrade.utils.extmath import row_norms
from sklearn_upgrade.utils._cython_blas cimport _gemm, _axpy
from sklearn_upgrade.utils._cython_blas cimport RowMajor, Trans, NoTrans
from sklearn_upgrade.cluster._k_means_common import CHUNK_SIZE
from sklearn_upgrade.cluster._k_means_common cimport _relocate_empty_clusters_dense
from sklearn_upgrade.cluster._k_means_common cimport _relocate_empty_clusters_sparse
from sklearn_upgrade.cluster._k_means_common cimport _average_centers, _center_shift


def lloyd_iter_chunked_dense(
        const floating[:, ::1] X,            # IN
        const floating[::1] sample_weight,   # IN
        const floating[:, ::1] centers_old,  # IN
        floating[:, ::1] centers_new,        # OUT
        floating[::1] weight_in_clusters,    # OUT
        int[::1] labels,                     # OUT
        floating[::1] center_shift,          # OUT
        int n_threads,
        bint update_centers=True):

    cdef:
        int n_samples = X.shape[0]
        int n_features = X.shape[1]
        int n_clusters = centers_old.shape[0]

    if n_samples == 0:
        return

    cdef:
        int n_samples_chunk = CHUNK_SIZE if n_samples > CHUNK_SIZE else n_samples
        int n_chunks = n_samples // n_samples_chunk
        int n_samples_rem = n_samples % n_samples_chunk
        int chunk_idx
        int start, end

        int j, k

        floating[::1] centers_squared_norms = row_norms(centers_old, squared=True)

        floating *centers_new_chunk
        floating *weight_in_clusters_chunk

        omp_lock_t lock

    n_chunks += n_samples != n_chunks * n_samples_chunk
    n_threads = min(n_threads, n_chunks)

    if update_centers:
        memset(&centers_new[0, 0], 0, n_clusters * n_features * sizeof(floating))
        memset(&weight_in_clusters[0], 0, n_clusters * sizeof(floating))
        omp_init_lock(&lock)

    with nogil, parallel(num_threads=n_threads):
        centers_new_chunk = <floating*> calloc(n_clusters * n_features, sizeof(floating))
        weight_in_clusters_chunk = <floating*> calloc(n_clusters, sizeof(floating))

        for chunk_idx in prange(n_chunks, schedule='static'):
            start = chunk_idx * n_samples_chunk
            if chunk_idx == n_chunks - 1 and n_samples_rem > 0:
                end = start + n_samples_rem
            else:
                end = start + n_samples_chunk

            _update_chunk_dense_fast(
                X[start: end],
                sample_weight[start: end],
                centers_old,
                centers_squared_norms,
                labels[start: end],
                centers_new_chunk,
                weight_in_clusters_chunk,
                update_centers)

        if update_centers:
            omp_set_lock(&lock)
            for j in range(n_clusters):
                weight_in_clusters[j] += weight_in_clusters_chunk[j]
                for k in range(n_features):
                    centers_new[j, k] += centers_new_chunk[j * n_features + k]
            omp_unset_lock(&lock)

        free(centers_new_chunk)
        free(weight_in_clusters_chunk)

    if update_centers:
        omp_destroy_lock(&lock)
        _relocate_empty_clusters_dense(
            X, sample_weight, centers_old, centers_new, weight_in_clusters, labels
        )
        _average_centers(centers_new, weight_in_clusters)
        _center_shift(centers_old, centers_new, center_shift)


cdef void _update_chunk_dense_fast(
        const floating[:, ::1] X,
        const floating[::1] sample_weight,
        const floating[:, ::1] centers_old,
        const floating[::1] centers_squared_norms,
        int[::1] labels,
        floating *centers_new,
        floating *weight_in_clusters,
        bint update_centers) noexcept nogil:

    cdef:
        int n_samples = labels.shape[0]
        int n_clusters = centers_old.shape[0]
        int n_features = centers_old.shape[1]
        floating sq_dist, min_sq_dist
        int i, j, label
        floating alpha = -2.0, beta = 1.0

        floating *dist_buffer = <floating*> malloc(n_samples * n_clusters * sizeof(floating))

    if not dist_buffer:
        return  

    for i in range(n_samples):
        for j in range(n_clusters):
            dist_buffer[i * n_clusters + j] = centers_squared_norms[j]

    _gemm(RowMajor, NoTrans, Trans, n_samples, n_clusters, n_features,
          alpha, &X[0, 0], n_features, &centers_old[0, 0], n_features,
          beta, dist_buffer, n_clusters)

    for i in range(n_samples):
        min_sq_dist = dist_buffer[i * n_clusters]
        label = 0
        for j in range(1, n_clusters):
            sq_dist = dist_buffer[i * n_clusters + j]
            if sq_dist < min_sq_dist:
                min_sq_dist = sq_dist
                label = j
        labels[i] = label

        if update_centers:
            weight_in_clusters[label] += sample_weight[i]
            _axpy(n_features, sample_weight[i], &X[i, 0], 1,
                  &centers_new[label * n_features], 1)

    free(dist_buffer)


def lloyd_iter_chunked_sparse(
        X,
        const floating[::1] sample_weight,
        const floating[:, ::1] centers_old,
        floating[:, ::1] centers_new,
        floating[::1] weight_in_clusters,
        int[::1] labels,
        floating[::1] center_shift,
        int n_threads,
        bint update_centers=True):
    cdef:
        int n_samples = X.shape[0]
        int n_features = X.shape[1]
        int n_clusters = centers_old.shape[0]

    if n_samples == 0:
        return

    cdef:
        int n_samples_chunk = CHUNK_SIZE if n_samples > CHUNK_SIZE else n_samples
        int n_chunks = n_samples // n_samples_chunk
        int n_samples_rem = n_samples % n_samples_chunk
        int chunk_idx
        int start = 0, end = 0

        int j, k

        floating[::1] X_data = X.data
        int[::1] X_indices = X.indices
        int[::1] X_indptr = X.indptr

        floating[::1] centers_squared_norms = row_norms(centers_old, squared=True)

        floating *centers_new_chunk
        floating *weight_in_clusters_chunk

        omp_lock_t lock

    n_chunks += n_samples != n_chunks * n_samples_chunk
    n_threads = min(n_threads, n_chunks)

    if update_centers:
        memset(&centers_new[0, 0], 0, n_clusters * n_features * sizeof(floating))
        memset(&weight_in_clusters[0], 0, n_clusters * sizeof(floating))
        omp_init_lock(&lock)

    with nogil, parallel(num_threads=n_threads):
        centers_new_chunk = <floating*> calloc(n_clusters * n_features, sizeof(floating))
        weight_in_clusters_chunk = <floating*> calloc(n_clusters, sizeof(floating))

        for chunk_idx in prange(n_chunks, schedule='static'):
            start = chunk_idx * n_samples_chunk
            if chunk_idx == n_chunks - 1 and n_samples_rem > 0:
                end = start + n_samples_rem
            else:
                end = start + n_samples_chunk

            _update_chunk_sparse(
                X_data[X_indptr[start]: X_indptr[end]],
                X_indices[X_indptr[start]: X_indptr[end]],
                X_indptr[start: end+1],
                sample_weight[start: end],
                centers_old,
                centers_squared_norms,
                labels[start: end],
                centers_new_chunk,
                weight_in_clusters_chunk,
                update_centers)

        if update_centers:
            omp_set_lock(&lock)
            for j in range(n_clusters):
                weight_in_clusters[j] += weight_in_clusters_chunk[j]
                for k in range(n_features):
                    centers_new[j, k] += centers_new_chunk[j * n_features + k]
            omp_unset_lock(&lock)

        free(centers_new_chunk)
        free(weight_in_clusters_chunk)

    if update_centers:
        omp_destroy_lock(&lock)
        _relocate_empty_clusters_sparse(
            X_data, X_indices, X_indptr, sample_weight,
            centers_old, centers_new, weight_in_clusters, labels)
        _average_centers(centers_new, weight_in_clusters)
        _center_shift(centers_old, centers_new, center_shift)


cdef void _update_chunk_sparse(
        const floating[::1] X_data,
        const int[::1] X_indices,
        const int[::1] X_indptr,
        const floating[::1] sample_weight,
        const floating[:, ::1] centers_old,
        const floating[::1] centers_squared_norms,
        int[::1] labels,
        floating *centers_new,
        floating *weight_in_clusters,
        bint update_centers) noexcept nogil:
    cdef:
        int n_samples = labels.shape[0]
        int n_clusters = centers_old.shape[0]
        int n_features = centers_old.shape[1]
        floating sq_dist, min_sq_dist
        int i, j, k, label
        floating max_floating = FLT_MAX if floating is float else DBL_MAX
        int s = X_indptr[0]

    for i in range(n_samples):
        min_sq_dist = max_floating
        label = 0
        for j in range(n_clusters):
            sq_dist = 0.0
            for k in range(X_indptr[i] - s, X_indptr[i + 1] - s):
                sq_dist += centers_old[j, X_indices[k]] * X_data[k]
            sq_dist = centers_squared_norms[j] -2 * sq_dist
            if sq_dist < min_sq_dist:
                min_sq_dist = sq_dist
                label = j
        labels[i] = label
        if update_centers:
            weight_in_clusters[label] += sample_weight[i]
            for k in range(X_indptr[i] - s, X_indptr[i + 1] - s):
                centers_new[label * n_features + X_indices[k]] += X_data[k] * sample_weight[i]