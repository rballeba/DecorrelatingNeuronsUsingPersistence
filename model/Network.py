import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def get_neurons_point_cloud(model, x):
    """
    :param model: neural network in tensorflow keras.
    :param x:  tf.Tensor with shape = (#training_examples, #input_shape)
    :return:
    """
    activation_x_examples = get_neuron_activations_x_examples_matrix(x, model)
    return activation_x_examples


def get_neuron_activations_x_examples_matrix(x, model, num_skipped_layers_from_start=0):
    examples_x_activations = _examples_x_activations_for_input(x, model, num_skipped_layers_from_start)
    activations_x_examples = tf.transpose(examples_x_activations)
    return activations_x_examples


def _examples_x_activations_for_input(x, model, num_skipped_layers_from_start):
    first_layer = True
    skipped_iterations = 0
    for layer in model.layers:
        if skipped_iterations < num_skipped_layers_from_start:
            x = layer(x)
            if not isinstance(layer, tf.keras.layers.Dropout): # We only count non-dropout layers
                skipped_iterations += 1
        else:
            x = layer(x)
            examples_x_neurons = tf.reshape(x, (x.shape[0], -1))
            # We only save layers that are not dropout layers because we are interested in the whole network.
            if not isinstance(layer, tf.keras.layers.Dropout):
                if first_layer:
                    activations_bd = examples_x_neurons
                    first_layer = False
                else:
                    activations_bd = tf.concat((activations_bd, examples_x_neurons), axis=1)
    return activations_bd


def clone_network_with_same_weights(model):
    cloned_model = tf.keras.models.clone_model(model)
    cloned_model.set_weights(model.get_weights())
    return cloned_model


def clone_network_with_new_weights(model):
    return tf.keras.models.clone_model(model)
