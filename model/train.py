import os
import time

import numpy as np
import tensorflow as tf

from model.algorithms.ripser_dim_zero import fast_zeroth_persistence_diagram
from model.distances.euclidean import euclidean_distance_tensorflow
from model.filesystem import create_directory_if_it_does_not_exist, save_np_array
from model.neuron_samplings.importance_percentage_neuron_sampling import \
    get_neurons_point_cloud_importance_percentage_sampling
from model.plots.images import build_gif
from model.plots.plot_tool import plot_ppdd


def train(model, train_dataset, validation_dataset, epochs,
          loss_metric_model,
          classical_loss_object_model,
          topological_loss_object,
          topological_loss_weight,
          iterations_to_compute_topological_loss=1,
          homology_distance_fn=euclidean_distance_tensorflow,
          optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.001, momentum=0.9),
          epochs_to_compute_topological_loss=-1,
          epochs_to_save_checkpoints=10,
          experiment_name=None,
          filepath_to_save_experiments='./experiments',
          iterations_to_plot=0,
          plot_x_axis_ppdd_low=None, plot_x_axis_ppdd_high=None,
          plot_y_axis_ppdd_low=None, plot_y_axis_ppdd_high=None,
          verbose=True,
          save_results=True,
          train_batch_size=256,
          validation_batch_size=256,
          sampling_neurons_percentage=0.005,
          early_stopping=True,
          patience=100,
          min_delta=0,
          min_epochs=0,
          accuracy_model=None,
          training_shuffle_buffer_size=1000,
          ):
    """
    The function trains a given model specifying a lot of parameters. It allows the user to use topological regularizers.
    See the parameters to see how they work.

    :param training_shuffle_buffer_size: Size of the buffer to shuffle the training dataset each epoch.
    :param accuracy_model: Model to compute accuracy in validation.
    :param min_epochs: Minimum number of epochs to train the model before starting using early stopping.
    :param update_learning_rate: If True, we update the learning rate every epochs_to_update_learning_rate epochs multiplying
    by multiplier_learning_rate.
    :param epochs_to_save_checkpoints: it saves a checkpoint every epochs_to_save_checkpoints epochs if the parameter
    is higher than zero. Otherwise, it does not save any checkpoint.
    :param epochs_to_compute_topological_loss: It computes the topological loss only the
    first epochs_to_compute_topological_loss epochs. If it is -1, then it computes the topological loss at each epoch
    during the whole training.
    :param iterations_to_compute_topological_loss: Number of iterations to update the topological loss.
    :param multiplier_learning_rate: Number to multiply the learning rate when the learning rate is updated.
    :param epochs_to_update_learning_rate: Number of epochs to wait before updating the learning rate.
    :param initial_learning_rate: Initial learning rate, it is multiplied by multiplier_learning_rate every epochs_to_update_learning_rate
    :param min_delta: The minimum quantity we should improve (decrease) our validation loss to keep training.
    :param patience: Maximum number of consecutive epochs in which we do not improve (decrease) the validation
    loss by min_delta.
    :param early_stopping: True implies that the training finishes when the early stopping criteria happens. The criteria
    is given as follows: each iteration in which we do not improve (decrease) the validation loss by a minimum of min_delta
    we add one to a patience counter. If the counter is bigger than the patience parameter, then we stop the execution.
    If we improve the validation accuracy, then we reset the counter to zero. This is the usual early stopping.
    :param sampling_neurons_percentage: The quantity of neurons we sample from each layer when computing persistence diagrams.
    :param train_batch_size: The batch size for the training dataset.
    :param validation_batch_size: The quantity of elements per batch for computing the validation accuracy. It only
    affects the numerical approximation.
    :param model: Model to train. Keras model.
    :param train_dataset: Training dataset. Type tf.data.Dataset
    :param validation_dataset: Validation dataset. Type tf.data.Dataset
    :param epochs: Number of epochs to train the model.
    batch_cardinality_strategy_fn
    :param loss_metric_model: Loss used to train the model.
    :param classical_loss_object_model:
    :param topological_loss_object:
    :param topological_loss_weight:
    :param homology_distance_fn:
    :param optimizer:
    :param experiment_name:
    :param filepath_to_save_experiments:
    :param iterations_to_plot: 0 if we do not want to plot.
    :param plot_x_axis_ppdd_low:
    :param plot_x_axis_ppdd_high:
    :param verbose:
    :param save_results: If True, it saves the weights of the model at each iteration at the folder
    {experiment_filepath}/weights_training with name {number_of_iteration}.h5. It uses h5py. It also saves in numpy
    format the training and validation loss and accuracy values at each iteration.
    :return:
    """
    # We define the global variables
    # Shuffle the training dataset each epoch
    train_dataset_batched = train_dataset.shuffle(buffer_size=training_shuffle_buffer_size,
                                                  reshuffle_each_iteration=True) \
        .batch(train_batch_size)
    validation_dataset_batched = validation_dataset.batch(validation_batch_size)
    train_original_loss = loss_metric_model(name='train_original_loss')
    train_topological_loss = loss_metric_model(name='train_topological_loss')
    validation_original_loss = loss_metric_model(name='validation_original_loss')
    accuracy_validation = accuracy_model(name='accuracy_validation')
    classical_loss_object = classical_loss_object_model
    topological_loss_object = topological_loss_object
    # We define the topo_gradient globally because we use it each n iterations and we need to store it
    topo_gradient = None
    topological_loss = None
    train_original_losses = []
    train_topological_losses = []
    validation_losses = []
    epoch_time_execution = []
    last_accuracy_validation = None
    epoch = 0
    iteration = 0
    # Variables for controlling the early stopping. We use the loss to stop.
    best_weights = None
    iteration_best_weights = 0
    finished_by_early_stopping = False
    last_good_early_stopping = np.inf
    number_of_epochs_not_improving_loss = 0

    # Internal functions for training
    def get_current_weights():
        # We copy the list of numpy weights before adding to the list of trained weights
        return [np.copy(layer_weights) for layer_weights in model.get_weights()]

    def check_early_stopping(current_loss_validation):
        nonlocal best_weights
        nonlocal iteration_best_weights
        nonlocal last_good_early_stopping
        nonlocal number_of_epochs_not_improving_loss
        nonlocal finished_by_early_stopping
        nonlocal iteration
        nonlocal epoch
        if epoch >= min_epochs:  # We start counting the early stopping when we reach the minimum number of epochs
            # We check if we have improved the previous value of loss in validation at least min delta.
            if last_good_early_stopping > current_loss_validation + min_delta:
                # If so, we update the early stopping variable and we reset the number of iterations in
                # which we do not improve the accuracy in val.
                last_good_early_stopping = current_loss_validation
                number_of_epochs_not_improving_loss = 0
                # We update the best weights obtained
                best_weights = get_current_weights()
                iteration_best_weights = iteration
            else:
                number_of_epochs_not_improving_loss += 1
            if number_of_epochs_not_improving_loss > patience:
                finished_by_early_stopping = True

    def is_compute_topological_loss():
        # We compute the topological loss if the topological loss weight is bigger than zero and if we are in an epoch
        # lower than the epochs to compute the topological loss.
        nonlocal epoch
        nonlocal epochs_to_compute_topological_loss
        nonlocal topological_loss_weight
        return topological_loss_weight > 0 and \
            (epoch < epochs_to_compute_topological_loss or epochs_to_compute_topological_loss == -1)

    def save_numerical_results():
        nonlocal validation_losses
        nonlocal train_original_losses
        nonlocal train_topological_losses
        nonlocal epoch_time_execution
        # Save metrics and model into disc
        validation_losses = np.array(validation_losses)
        train_original_losses = np.array(train_original_losses)
        train_topological_losses = np.array(train_topological_losses)
        epoch_time_execution = np.array(epoch_time_execution)

        save_np_array(validation_losses, generate_filepath_experiment('validation_losses'))
        save_np_array(train_original_losses, generate_filepath_experiment('train_original_losses'))
        save_np_array(train_topological_losses, generate_filepath_experiment('train_topological_losses'))
        save_np_array(epoch_time_execution, generate_filepath_experiment('epoch_time_execution'))

    @tf.function(reduce_retracing=True)
    def classical_train_step(inputs, labels):
        with tf.GradientTape() as tape_classical:
            predictions = model(inputs, training=True)
            classical_loss = classical_loss_object(labels, predictions)
        classical_gradient = tape_classical.gradient(classical_loss, model.trainable_weights)
        return classical_loss, classical_gradient

    def train_step(inputs, labels):
        nonlocal epoch
        nonlocal topo_gradient
        nonlocal topological_loss
        nonlocal epochs_to_compute_topological_loss
        # Compute gradients of the classical loss
        classical_loss, classical_gradient = classical_train_step(inputs, labels)
        # Compute gradients of the topological loss
        if is_compute_topological_loss():
            if iteration % iterations_to_compute_topological_loss == 0:
                with tf.GradientTape() as tape_topo:
                    topological_loss = topological_loss_object(model, inputs, labels)
                topo_gradient = tape_topo.gradient(topological_loss, model.trainable_weights)
        else:
            topological_loss = 0
        # Sum both gradients if topo weight > 0 or use only the classical gradient
        if is_compute_topological_loss():
            # We check if the gradient is None. If so, we set it to zero. If not, we multiply it by its weight.
            # It will be None if the persistence diagram computed in this iteration have less
            # than the number of points specified in the topological loss object.
            # If the gradient cannot be computed for some reason (probably because of the presence of NaN
            # in the correlation matrix), we set it to zero.
            topo_gradient = [(topological_loss_weight * topo_gradient[i]) if topo_gradient[i] is not None else 0.0
                             for i in range(len(topo_gradient))]
            # Put the topo gradient to zero if there are NaN or Inf
            topo_gradient = [tf.cond(tf.math.reduce_all(tf.math.is_finite(topo_gradient[i])), lambda: topo_gradient[i],
                                     lambda: 0.0)
                             for i in range(len(topo_gradient))]

            gradients = [classical_gradient[i] + topo_gradient[i]
                         for i in range(len(classical_gradient))]
        else:
            gradients = classical_gradient
        try:
            assert all(map(lambda gradient_part: tf.cond(tf.math.reduce_all(tf.math.is_finite(gradient_part)),
                                                         lambda: True,
                                                         lambda: False),
                           gradients))  # If all true then there are no tf.inf nor tf.nan
        except AssertionError:
            # Save numerical results
            save_numerical_results()
            # Save a message error in error_training.txt
            with open(f'{experiment_filepath}/error_training.txt', 'w') as f:
                f.write('Error: NaN or Inf in the gradient. Check training losses.')
            # Exit the training
            return None
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        # Update the metrics
        train_original_loss(classical_loss)
        train_topological_loss(topological_loss_weight * topological_loss)
        return classical_loss, topological_loss_weight * topological_loss

    def compute_validation_loss():
        nonlocal last_accuracy_validation
        nonlocal accuracy_validation
        for inputs_validation, labels_validation in validation_dataset_batched:
            validation_step(np.array(inputs_validation), np.array(labels_validation))
        last_accuracy_validation = accuracy_validation.result()

    @tf.function(reduce_retracing=True)
    def validation_step(inputs, labels):
        nonlocal accuracy_validation
        predictions = model(inputs, training=False)
        classical_loss = classical_loss_object(labels, predictions)
        validation_original_loss(classical_loss)
        accuracy_validation(labels, predictions)

    def generate_filepath_experiment(filename):
        return f'{experiment_filepath}/{filename}.npy'

    # Paths to save results
    experiment_filepath = f'{filepath_to_save_experiments}'
    if experiment_name:
        experiment_filepath = f'{experiment_filepath}/{experiment_name}'
    if save_results:
        create_directory_if_it_does_not_exist(experiment_filepath)
    if iterations_to_plot > 0:
        plot_directory = f'{experiment_filepath}/training_plot'
        fixed_measures_plot_directory = f'{plot_directory}/fixed_measures'
        variable_measures_plot_directory = f'{plot_directory}/variable_measures'
        create_directory_if_it_does_not_exist(fixed_measures_plot_directory)
        create_directory_if_it_does_not_exist(variable_measures_plot_directory)

    # Text templates

    template = 'Epoch [{}/{}], Batch {} Train or. loss: {}, Train topo. loss: {}'

    # If we save checkpoints, create its directory
    if save_results and epochs_to_save_checkpoints > 0:
        create_directory_if_it_does_not_exist(f'{experiment_filepath}/checkpoints/')

    # Start of the training function

    while epoch < epochs and (not finished_by_early_stopping):
        if verbose or save_results:
            start_epoch = time.process_time()
        for inputs, labels in train_dataset_batched:
            classical_loss, topological_loss = train_step(inputs, labels)
            # End training
            # Saving data about iterations
            if save_results:
                train_original_losses.append(train_original_loss.result().numpy())
                train_topological_losses.append(train_topological_loss.result().numpy())

            # Printing data about iterations
            if verbose:
                print(template.format(epoch + 1,
                                      epochs,
                                      iteration + 1,
                                      round(float(classical_loss), 3),
                                      round(float(topological_loss), 3)))
            # Plotting the training if necessary

            if iterations_to_plot > 0 and iteration % iterations_to_plot == 0:
                # These computations are done in the function total_persistence but we repeat them for the sake
                # of testing our approach. It is very expensive computationally, so it should be avoided in production
                # or training big models.
                activation_x_examples = get_neurons_point_cloud_importance_percentage_sampling(model, inputs,
                                                                                               sampling_percentage=sampling_neurons_percentage)
                distance_matrix = homology_distance_fn(activation_x_examples, activation_x_examples.shape[0],
                                                       activation_x_examples.shape[1])
                persistence_diagram_deaths, _ = fast_zeroth_persistence_diagram(distance_matrix)
                # Print with fixed frame measures to build gif
                plot_ppdd(persistence_diagram_deaths, fixed_measures_plot_directory, iteration,
                          plot_x_axis_ppdd_low, plot_x_axis_ppdd_high, plot_y_axis_ppdd_low, plot_y_axis_ppdd_high)
                # Print with variable measures to analyze image per image
                plot_ppdd(persistence_diagram_deaths, variable_measures_plot_directory, iteration, None, None)
            # Update the number of iterations
            iteration += 1
        # Check early stopping conditions
        if save_results or early_stopping:
            compute_validation_loss()
            print(
                f'\nEpoch [{epoch + 1}/{epochs}] Validation or. loss: {round(float(validation_original_loss.result()), 3)}.\n'
                f'Validation accuracy: {round(float(last_accuracy_validation), 3)}\n')
            # Clean the validation metrics for the next epoch
            accuracy_validation.reset_states()
        if early_stopping:
            check_early_stopping(validation_original_loss.result())
        if save_results:
            validation_losses.append(validation_original_loss.result().numpy())
        if save_results or early_stopping:
            # Restart the validation for next epoch
            validation_original_loss.reset_states()
        # Update the number of epochs
        epoch += 1
        # Print the time of the epoch
        if verbose or save_results:
            end_epoch = time.process_time()
        if save_results:
            epoch_time_execution.append(end_epoch - start_epoch)
        if verbose:
            print(f'\nEpoch time: {round(end_epoch - start_epoch, 3)} - Topo weight: {topological_loss_weight} - '
                  f'Topo.loss: {type(topological_loss_object)}\n')
        # Print the metric losses
        print(
            f'\nEpoch [{epoch + 1}/{epochs}] Average or. train loss: {round(float(train_original_loss.result()), 3)}.\n'
            f'Average topo. train loss:: {round(float(train_topological_loss.result()), 3)}\n')
        # Reset the metric states
        train_original_loss.reset_states()
        train_topological_loss.reset_states()
        # Save a checkpoint of the model if necessary
        if epochs_to_save_checkpoints > 0 and (epoch % epochs_to_save_checkpoints == 0) and save_results:
            model.save_weights(f'{experiment_filepath}/checkpoints/it_{iteration}/weights')
    # If we are not using early stopping, take the best weights as the last weights
    if not finished_by_early_stopping:
        best_weights = get_current_weights()
        iteration_best_weights = iteration
    # Save metrics and model into disc
    if save_results:
        save_numerical_results()
        # Save the model to disk
        # If we have finished by early stopping then the iteration containing the best weights is
        # current iteration - (patience + 1)
        model.set_weights(best_weights)
        model.save_weights(f'{experiment_filepath}/it_{iteration_best_weights}/weights')
        # We build the gif if we are plotting the training
        if iterations_to_plot > 0:
            build_gif(fixed_measures_plot_directory, frames_per_image=2)
            build_gif(variable_measures_plot_directory, frames_per_image=2)
    return model


def get_last_valid_checkpoint_iteration_number(filepath_to_save_experiments, experiment_name=None):
    """
    It returns the last valid checkpoint iteration number of a given experiment. It returns None if there is no valid checkpoint.

    :param filepath_to_save_experiments: the path to the directory where the experiments are saved.
    :param experiment_name: the name of the experiment.
    :return: the path to the last valid checkpoint.
    """
    experiment_filepath = f'{filepath_to_save_experiments}'
    if experiment_name:
        experiment_filepath = f'{experiment_filepath}/{experiment_name}'
    checkpoints_filepath = f'{experiment_filepath}/checkpoints'
    # Get all the folder names in the checkpoints directory
    checkpoints_iterations = [int(f[3:]) for f in os.listdir(checkpoints_filepath)
                              if os.path.isdir(os.path.join(checkpoints_filepath, f))]
    if len(checkpoints_iterations) == 0:
        return None
    # Select from the experiment filepath the folder with name starting with 'it_'
    last_iteration_training = [int(f[3:]) for f in os.listdir(experiment_filepath)
                               if f[:3] == 'it_'][0]
    # Get the differences between last iteration training and the checkpoints
    differences = [last_iteration_training - checkpoint_iteration for checkpoint_iteration in checkpoints_iterations]
    # Filter the ones that are negative or zero
    differences = [difference for difference in differences if difference > 0]
    # If there is no valid checkpoint, return None
    if len(differences) == 0:
        return None
    # Get the minimum difference
    min_difference = min(differences)
    # Get the iteration of the checkpoint
    checkpoint_iteration = last_iteration_training - min_difference
    return checkpoint_iteration
