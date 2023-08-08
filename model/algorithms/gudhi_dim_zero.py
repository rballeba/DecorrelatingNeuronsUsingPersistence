import gudhi
import numpy as np


def generate_rips_complex(distance_matrix):
    rips_complex = gudhi.RipsComplex(distance_matrix=distance_matrix)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
    return simplex_tree


def compute_persistence(rips_complex_simplex_tree, hom_coeff: int = 2):
    persistence_diagrams = rips_complex_simplex_tree.persistence(
        homology_coeff_field=hom_coeff)
    persistence_pairs = rips_complex_simplex_tree.persistence_pairs()
    return persistence_diagrams, persistence_pairs


def compute_indices_persistence_pairs(rips_complex, distance_matrix: np.array, persistence_pairs):
    indices = []
    pers = []
    for _, s2 in persistence_pairs:
        if len(s2) != 0:  # We discard points dying at infinity, specially the max. connected component for H_0 group.
            l2 = np.array(s2)
            i2 = [s2[v] for v in np.unravel_index(np.argmax(distance_matrix[l2, :][:, l2]), [len(s2), len(s2)])]
            pers.append(rips_complex.filtration(s2))
            indices += i2
    # Sort points with distance-to-diagonal
    perm = np.argsort(pers)
    indices = list(np.reshape(indices, [-1, 2])[perm][::-1, :].flatten())

    # Output indices
    indices = indices[:2 * len(persistence_pairs)]
    return indices


def fast_zeroth_persistence_diagram(distance_matrix: np.ndarray, hom_coeff=2):
    rips_complex = generate_rips_complex(distance_matrix)
    persistence_diagram, persistence_pairs = compute_persistence(rips_complex, hom_coeff)
    indices_for_persistence_pairs = compute_indices_persistence_pairs(rips_complex,
                                                                      distance_matrix,
                                                                      persistence_pairs)
    return [point[1][1] for point in persistence_diagram if point[1][1] < np.inf], indices_for_persistence_pairs
