import math

from keras.layers import PReLU

from model import Network
import tensorflow as tf


def get_neurons_point_cloud_importance_percentage_sampling(model, x, sampling_percentage=0.005):
    """
    :param model: neural network in tensorflow keras.
    :param x:  tf.Tensor with shape = (#training_examples, #input_shape)
    :param sampling_percentage: Percentage of sampling per layer
    :return:
    """
    activation_x_examples = get_neuron_activations_x_examples_matrix_importance_percentage_sampling(x, model,
                                                                                                    sampling_percentage)
    return activation_x_examples


def get_neuron_activations_x_examples_matrix_importance_percentage_sampling(x, model, sampling_percentage,
                                                                            num_skipped_layers_from_start=0):
    examples_x_activations = _examples_x_activations_for_input_importance_percentage_sampling(x, model,
                                                                                              num_skipped_layers_from_start,
                                                                                              sampling_percentage)
    activations_x_examples = tf.transpose(examples_x_activations)
    return activations_x_examples


def _examples_x_activations_for_input_importance_percentage_sampling(x, model, num_skipped_layers_from_start,
                                                                     sampling_percentage):
    '''
    For each layer with trainable weights we take the sampling_percentage% of neurons that have the highest average activation in
    absolute value
    :param x: dataset
    :param model: neural network in tensorflow keras.
    :param num_skipped_layers_from_start: Number of layers to skip from the start to do the analysis
    :param sampling_percentage The percentage of neurons we take in each layer.
    :return:
    '''
    # We add always the last layer completely
    raw_total_layers = sum([len(layer.trainable_weights) > 0 for layer in model.layers])
    # We remove the last layer (the -1) because we add it completely. The first one is not counted.
    number_of_hidden_layers = raw_total_layers - num_skipped_layers_from_start - 1
    sampled_activations_bd = None  # final shape=(floor(sampling_percentage*total_number_of_neurons), number_of_examples)
    skipped_iterations = 0
    layer_idx = -1
    for layer in model.layers:
        if skipped_iterations < num_skipped_layers_from_start:
            x = layer(x)
            if len(layer.trainable_weights) > 0:  # We only count layers with trainable weights
                skipped_iterations += 1
        else:
            x = layer(x)
            # We avoid PReLU layer because it is almost the same that the previous convolution and we want to save resources
            if len(layer.trainable_weights) > 0 and type(layer) != PReLU:
                layer_idx += 1  # We start with the layer 0 when we have the first layer that is trainable
                examples_x_neurons = tf.reshape(x, (x.shape[0], -1))
                if layer_idx < number_of_hidden_layers:
                    averages_x_neurons = tf.reduce_mean(tf.math.abs(examples_x_neurons), axis=0)
                    averages_x_neurons_args_sorted = tf.argsort(averages_x_neurons, axis=0, direction='DESCENDING')
                    number_of_neurons = averages_x_neurons_args_sorted.shape[0]
                    number_of_samples = math.floor(sampling_percentage * number_of_neurons)
                    selected_examples_x_neurons = tf.gather(examples_x_neurons,
                                                            indices=averages_x_neurons_args_sorted[:number_of_samples],
                                                            axis=1)

                    sampled_activations_bd = selected_examples_x_neurons if sampled_activations_bd is None else \
                        tf.concat((sampled_activations_bd, selected_examples_x_neurons), axis=1)
                else:
                    # We are in the last layer (output layer). We add all the neurons
                    sampled_activations_bd = examples_x_neurons if sampled_activations_bd is None else \
                        tf.concat((sampled_activations_bd, examples_x_neurons), axis=1)
    return sampled_activations_bd
