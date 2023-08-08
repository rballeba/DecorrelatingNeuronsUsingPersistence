import tensorflow as tf


def l1_regularisation_term(model, dataset, labels):
    return tf.reduce_sum([tf.reduce_sum(tf.abs(w)) for w in model.trainable_weights])
