import tensorflow as tf


def l2_regularisation_term(model, dataset, labels):
    return tf.reduce_sum([tf.reduce_sum(tf.math.square(w)) for w in model.trainable_weights])
