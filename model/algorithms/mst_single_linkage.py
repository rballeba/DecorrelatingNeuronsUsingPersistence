import numpy as np


def condensed_index(n, i, j):
    """
        Calculate the condensed index of element (i, j) in an n x n condensed
        matrix.
        """
    if i < j:
        return np.rint(n * i - (i * (i + 1) / 2) + (j - i - 1)).astype(np.int32)
    elif i > j:
        return np.rint(n * j - (j * (j + 1) / 2) + (i - j - 1)).astype(np.int32)


def fast_zeroth_persistence_diagram(condensed_distance_matrix: np.ndarray, ordered: bool = True):
    """
    Computes the zeroth persistence diagram of a condensed distance matrix using the single linkage algorithm
    implemented using the SLINK algorithm modified to get the persistence pairs and the death values directly.
    :param ordered: If True, it returns the points of the persistence diagram ordered by increasing death values.
    :param condensed_distance_matrix: the lower triangular part of the distance matrix without including the diagonal.
    :param number_of_observations: the number of observations in the distance matrix
    :return: Persistence diagram of dimension zero for the condensed distance matrix
    """
    number_of_points = int((1 + np.sqrt(1 + 8 * condensed_distance_matrix.shape[0])) / 2)
    persistence_diagram = np.empty((number_of_points - 1, 2))
    persistence_indices = np.empty(number_of_points - 1, dtype=np.int32)
    merged = np.zeros(number_of_points, dtype=bool)
    D = np.full(number_of_points, np.inf)
    D_index = np.empty(number_of_points, dtype=np.int32)
    x, y = 0, 0

    for k in range(number_of_points - 1):
        current_min = np.inf
        merged[x] = True
        for i in range(number_of_points):
            if not merged[i]:
                dist = condensed_distance_matrix[condensed_index(number_of_points, x, i)]
                if D[i] > dist:
                    D[i] = dist
                    D_index[i] = x
                if D[i] < current_min:
                    current_min = D[i]
                    y = i
        persistence_indices[k] = condensed_index(number_of_points, D_index[y], y)
        persistence_diagram[k, 0] = 0
        persistence_diagram[k, 1] = current_min
        x = y
    if ordered:
        order = np.argsort(persistence_diagram[:, 1])
        persistence_diagram = persistence_diagram[order]
        persistence_indices = persistence_indices[order]
    return persistence_diagram, persistence_indices