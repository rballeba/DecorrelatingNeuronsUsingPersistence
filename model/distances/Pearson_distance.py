import tensorflow as tf
import tensorflow_probability as tfp

from model.matrices import get_condensed_matrix


def compute_distance_matrix(point_cloud, number_of_points, number_of_dimensions, noise_to_avoid_nan=0.001):
    # We can have neurons that have constant values for the whole dataset, so we need to add noise to avoid that.
    perturbed_point_cloud = point_cloud + tf.random.uniform(shape=point_cloud.shape, minval=-noise_to_avoid_nan,
                                                            maxval=noise_to_avoid_nan)
    correlations = tfp.stats.correlation(perturbed_point_cloud, sample_axis=1, event_axis=0)
    tf.debugging.assert_all_finite(correlations, "The computed correlation matrix for the neuron space contains not " +
                                   "finite numerical values (perhaps NaN or Inf)")
    # We can have noise in the diagonal values due to the way tensorflow handles the computation of the correlation
    # matrix
    distance_matrix = tf.sqrt((1 + noise_to_avoid_nan) - tf.math.square(correlations, 2))
    tf.debugging.assert_all_finite(distance_matrix, "The computed distance matrix for the neuron space contains not " +
                                   "finite numerical values (perhaps NaN or Inf)")
    return distance_matrix
