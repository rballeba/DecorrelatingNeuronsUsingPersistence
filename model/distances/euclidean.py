import numpy as np
import tensorflow as tf

from model.matrices import get_condensed_matrix


def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


def euclidean_distance_tensorflow(point_cloud, number_of_points, number_of_dimensions):
    t1 = tf.reshape(point_cloud, (1, number_of_points, number_of_dimensions))
    t2 = tf.reshape(point_cloud, (number_of_points, 1, number_of_dimensions))
    full_distance_matrix = tf.norm(t1 - t2, ord='euclidean', axis=2, )
    return full_distance_matrix
