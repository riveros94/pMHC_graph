# combinations_filter.pyx

cimport cython
import numpy as np
cimport numpy as np
from libc.time cimport time
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double compute_cv_distance(np.int64_t[:, :] nodes_view, np.int64_t i, np.int64_t j,
                                double[:, :] distance_matrix_view, np.int64_t N) nogil:
    cdef np.int64_t k
    cdef np.int64_t idx_i, idx_j
    cdef double distances_sum = 0.0
    cdef double distances_sq_sum = 0.0
    cdef double distance
    cdef double mean_distance, std_distance, cv_distance
    cdef double temp

    for k in range(N):
        idx_i = nodes_view[i, k]
        idx_j = nodes_view[j, k]

        if idx_i < 0 or idx_i >= distance_matrix_view.shape[0]:
            return -1.0
        if idx_j < 0 or idx_j >= distance_matrix_view.shape[1]:
            return -1.0

        distance = distance_matrix_view[idx_i, idx_j]
        if distance <= 0:
            return -1.0

        distances_sum += distance
        distances_sq_sum += distance * distance

    mean_distance = distances_sum / N
    if mean_distance == 0.0:
        return -1.0  
    temp = (distances_sq_sum / N) - (mean_distance * mean_distance)
    if temp < 0:
        temp = 0.0
    std_distance = sqrt(temp)
    cv_distance = (std_distance / mean_distance) * 100.0

    return cv_distance

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def filtered_combinations(np.ndarray[np.int64_t, ndim=2, mode='c'] nodes not None,
                          np.ndarray[np.float64_t, ndim=2, mode='c'] distance_matrix not None,
                          double max_cv=15):
    cdef np.int64_t len_nodes = nodes.shape[0]
    cdef np.int64_t N = nodes.shape[1]
    cdef np.int64_t i, j, k
    cdef bint all_diff
    cdef double cv_distance
    cdef double start_time, elapsed_time
    cdef double[:, :] distance_matrix_view = distance_matrix
    cdef np.int64_t[:, :] nodes_view = nodes
    cdef list result = []

    start_time = time(NULL)


    for i in range(len_nodes):
        for j in range(i + 1, len_nodes):

            all_diff = True
            for k in range(N):
                if nodes_view[i, k] == nodes_view[j, k]:
                    all_diff = False
                    break

            if all_diff:

                cv_distance = compute_cv_distance(nodes_view, i, j, distance_matrix_view, N)

                if cv_distance >= 0.0 and cv_distance < max_cv:

                    result.append((nodes[i].copy(), nodes[j].copy()))

        if i % 100 == 0:
            elapsed_time = time(NULL) - start_time
            print(f"Progresso: {i}/{len_nodes}, Tempo decorrido: {elapsed_time:.2f} segundos")

    return result
