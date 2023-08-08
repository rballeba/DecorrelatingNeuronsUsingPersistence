import gph
import numpy as np


def fast_zeroth_persistence_diagram(distance_matrix: np.ndarray, hom_coeff=2, threads=4):
    result_gph_ripser = gph.ripser_parallel(distance_matrix, maxdim=0, coeff=hom_coeff, metric='precomputed',
                                            n_threads=threads, return_generators=True)
    dgms, generators = result_gph_ripser['dgms'][0], result_gph_ripser['gens'][0]
    deaths = [point[1] for point in dgms if
              point[1] < np.inf]  # Point contains the birth and deaths of the persistence diagrams, but we
    # are only interested in the deaths because we work in dimension zero.
    indices_for_persistence_pairs = [gens[idx] for gens in generators for idx in range(1, 3)]
    return deaths, indices_for_persistence_pairs
