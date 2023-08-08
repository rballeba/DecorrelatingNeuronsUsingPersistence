import numpy as np
import tensorflow as tf

from model.algorithms.ripser_dim_zero import fast_zeroth_persistence_diagram
from model.distances.euclidean import euclidean_distance_tensorflow


# TODO it may be interesting to restrict the code to work only with semidistances and not with pseudo, as
# having d(x,y) = 0 where x!=y can cause problems and also we only allow to have one such distance due to
# differentiability theory
def generate_differentiable_zeroth_persistence_diagrams_deaths(point_cloud: np.array, number_of_points_in_dgm: int,
                                                               distance_function_tf=euclidean_distance_tensorflow):
    number_of_points = point_cloud.shape[0]
    number_of_dimensions = point_cloud.shape[1]
    distance_matrix = distance_function_tf(point_cloud, number_of_points, number_of_dimensions)
    return generate_differentiable_zeroth_persistence_diagrams_deaths_from_distance_matrix(distance_matrix,
                                                                                           number_of_points_in_dgm)


def get_indices_of_birth_death_zeroth_persistence_diagrams(distance_matrix, number_of_points_in_dgm):
    persistence_diagram, persistence_indices = fast_zeroth_persistence_diagram(distance_matrix)
    if len(persistence_diagram) != number_of_points_in_dgm:
        # The persistence diagram has no enough points, so we return a tensor of zeros.
        return [tf.zeros([1], dtype=tf.int64) for _ in range(2 * number_of_points_in_dgm)]
    return persistence_indices[:2*number_of_points_in_dgm]


def generate_differentiable_zeroth_persistence_diagrams_deaths_from_distance_matrix(
        distance_matrix: tf.Tensor,
        number_of_points_in_dgm: int):
    DXX = tf.reshape(distance_matrix, [1, distance_matrix.shape[0], distance_matrix.shape[1]])
    # Turn numpy function into tensorflow function
    RipsTF = lambda DX: tf.numpy_function(get_indices_of_birth_death_zeroth_persistence_diagrams,
                                          [DX, number_of_points_in_dgm],
                                          Tout=tf.int64)

    # Compute vertices associated to positive and negative simplices
    # Don't compute gradient for this operation
    indices_for_persistence = tf.nest.map_structure(tf.stop_gradient,
                                                    tf.map_fn(RipsTF, DXX,
                                                              dtype=[tf.int64 for _ in
                                                                     range(2 * number_of_points_in_dgm)]
                                                              ))
    # If the persistence diagram has no enough points, we return a tensor of zeros.
    if tf.math.reduce_sum(indices_for_persistence) == 0:
        return tf.zeros([number_of_points_in_dgm], dtype=tf.float32)
    # Take using the indices for persistence the deaths of the persistence diagram from the DXX matrix
    deaths_dgm_from_indices = tf.reshape(tf.gather_nd(distance_matrix, tf.reshape(indices_for_persistence,
                                                                                  [number_of_points_in_dgm, 2])),
                                         [number_of_points_in_dgm])
    return deaths_dgm_from_indices
