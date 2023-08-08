import tensorflow as tf


def get_condensed_matrix(M):
    indices = tf.where(tf.experimental.numpy.triu(tf.ones((tf.shape(M)[0], tf.shape(M)[0])), k=1) == 1)
    condensed_matrix = tf.gather_nd(M, indices)
    return condensed_matrix


def has_unique_entries(distance_matrix):
    condensed_matrix = get_condensed_matrix(distance_matrix)
    # First we order the condensed matrix vector
    sorted_condensed_matrix = tf.sort(condensed_matrix)
    # Then we check if there are repeated entries by looking if the difference between two consecutive entries is 0
    # Compute the differences between consecutive entries
    differences = sorted_condensed_matrix[1:] - sorted_condensed_matrix[:-1]
    # Check if there are any value in differences very near to zero using  tf.experimental.numpy.isclose
    differences_near_zero = tf.experimental.numpy.isclose(differences, tf.zeros_like(differences))
    return tf.reduce_all(differences_near_zero)
