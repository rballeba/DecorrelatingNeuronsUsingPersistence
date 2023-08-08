import tensorflow as tf

from model.neuron_samplings.importance_percentage_neuron_sampling import \
    get_neurons_point_cloud_importance_percentage_sampling
import tensorflow_probability as tfp


def minimize_correlations_regularisation_term(model, dataset, labels, sampling_percentage=0.005):
    activation_x_examples = get_neurons_point_cloud_importance_percentage_sampling(model, dataset,
                                                                                   sampling_percentage=sampling_percentage)
    correlations = tfp.stats.correlation(activation_x_examples, sample_axis=1, event_axis=0)
    # We remove all infinite and NaN values from the correlation matrix
    not_nan_nor_inf_mask = tf.logical_and(tf.math.logical_not(tf.math.is_nan(correlations)),
                                          tf.math.logical_not(tf.math.is_inf(correlations)))
    # We remove all diagonal entries (correlation 1.0 that cannot be minimised) and all zero entries (that are not
    # correlated)
    not_diagonal_nor_zero_mask = tf.logical_and(tf.not_equal(correlations, 0.0),
                                                tf.logical_not(tf.eye(tf.shape(correlations)[0], dtype=tf.bool)))
    complete_mask = tf.logical_and(not_nan_nor_inf_mask, not_diagonal_nor_zero_mask)
    filtered_elements = tf.math.abs(tf.boolean_mask(correlations, complete_mask))
    average_correlations = tf.math.reduce_mean(filtered_elements)
    return average_correlations
