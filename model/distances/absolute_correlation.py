import tensorflow as tf
import tensorflow_probability as tfp

from model.matrices import get_condensed_matrix


def compute_distance_matrix(point_cloud, number_of_points, number_of_dimensions, noise_to_avoid_nan=1e-4):
    # We can have neurons that have constant values for the whole dataset, so we need to add noise to avoid that.
    #perturbed_point_cloud = point_cloud + tf.random.uniform(shape=point_cloud.shape, minval=-noise_to_avoid_nan,
    #                                                        maxval=noise_to_avoid_nan)
    perturbed_point_cloud = point_cloud # TODO: We can add noise to avoid NaNs. For now, we don't add noise.
    correlations = tfp.stats.correlation(perturbed_point_cloud, sample_axis=1, event_axis=0)
    # Change the NaN values to 0.0. We assume that if we have a nan value, this means
    # that the neuron has constant values for the whole dataset, and thus is independent from any other neuron
    correlations = tf.where(tf.math.is_nan(correlations), tf.zeros_like(correlations), correlations)
    # Set the diagonal values of the matrix corresponding to the correlation of a neuron with itself to 1.0
    correlations_corrected = tf.linalg.set_diag(correlations, tf.ones(shape=(correlations.shape[0],)))
    tf.debugging.assert_all_finite(correlations_corrected, "The computed correlation matrix for the neuron space contains not " +
                                   "finite numerical values (perhaps NaN or Inf)")
    # We can have noise in the diagonal values due to the way tensorflow handles the computation of the correlation
    # matrix
    distance_matrix = 1.0 - tf.math.abs(correlations_corrected)
    tf.debugging.assert_all_finite(distance_matrix, "The computed distance matrix for the neuron space contains not " +
                                   "finite numerical values (perhaps NaN or Inf)")
    return distance_matrix
