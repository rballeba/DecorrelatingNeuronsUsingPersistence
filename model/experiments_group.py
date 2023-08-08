import json
import os

import numpy as np
import time

from model.Google_PGDL.PGDL_task import PGDLTask
from model.Network import clone_network_with_same_weights
from model.train import train, get_last_valid_checkpoint_iteration_number

"""
DO NOT REMOVE IMPORTS: They are needed to execute JSON code.
"""

from functools import partial
import tensorflow as tf

import model.distances.Pearson_distance as PearsonDistance
import model.distances.absolute_correlation as absolute_correlation

from model.topological_losses.std_avg_deaths_importance_percentage_sampling import \
    std_avg_deaths_importance_percentage_sampling
from model.topological_losses.topological_redundancy_importance_percentage_sampling import \
    topological_redundancy_importance_percentage_sampling
from model.other_regularisation_terms.l1_regularisation_term import l1_regularisation_term
from model.other_regularisation_terms.l2_regularisation_term import l2_regularisation_term
from model.other_regularisation_terms.minimize_correlations import minimize_correlations_regularisation_term
"""
Finishing JSON imports
"""


def accuracy_in_test(test_inputs, test_labels, model, accuracy_fn, verbose=True):
    predictions = model(test_inputs, training=False)
    accuracy_fn(test_labels, predictions)
    accuracy = accuracy_fn.result() * 100.0
    accuracy_fn.reset_states()
    if verbose:
        print(f'After training the model has {accuracy}% of accuracy in the test dataset')
    return accuracy


def perform_experiment(experiment_specification, iterations_to_plot=0, verbose=True):
    """
    It performs the experiment specified by the JSON given by parameter, This JSON
    must be loaded using json.load from the json library (import json).
    :param iterations_to_plot: It prints a persistence diagrams every 'iterations_to_plot'.
    :param verbose: Prints information of the experiment execution in real time.
    :param experiment_specification: Experiment specification in JSON format.
    :return: Nothing.
    """
    """
    =======================================================================================
    Experiment data & configuration
    It is a pity that it is required to write manually the information of the functions
    we are using, however I prefer that to use an eval of the strings.
    =======================================================================================
    """
    '--------------------------------------------------------------------------------------'
    tasks_folders = experiment_specification['tasks_folders']
    models_to_analyse = set(experiment_specification['models_to_analyse'])
    '--------------------------------------------------------------------------------------'
    '--------------------------------------------------------------------------------------'
    root_filepath_to_save_experiments = experiment_specification[
        'root_filepath_to_save_experiments']
    '--------------------------------------------------------------------------------------'
    topo_weight = experiment_specification['topo_weight']
    iterations_to_compute_topological_loss = experiment_specification["iterations_to_compute_topological_loss"]
    '--------------------------------------------------------------------------------------'
    batch_size = int(experiment_specification['batch_size'])
    epochs = int(experiment_specification['epochs'])
    '--------------------------------------------------------------------------------------'
    classical_loss_used = experiment_specification['classical_loss']
    loss_object_model = eval(classical_loss_used)
    '--------------------------------------------------------------------------------------'
    topo_loss_used = experiment_specification['topological_loss']
    topological_loss = eval(topo_loss_used)
    '--------------------------------------------------------------------------------------'
    accuracy_model_specs = experiment_specification['accuracy_model']
    accuracy_model = eval(accuracy_model_specs)
    '--------------------------------------------------------------------------------------'
    loss_metric_model_used = experiment_specification['loss_metric_model_used']
    loss_metric_model = eval(loss_metric_model_used)
    '--------------------------------------------------------------------------------------'
    homology_distance_fn_name = experiment_specification['homology_distance_fn']
    homology_distance_fn = eval(homology_distance_fn_name)
    '--------------------------------------------------------------------------------------'
    experiment_patience = experiment_specification['patience']
    '--------------------------------------------------------------------------------------'
    epochs_to_compute_topological_loss = int(experiment_specification['epochs_to_compute_topological_loss'])\
        if 'epochs_to_compute_topological_loss' in experiment_specification else -1
    """
    =======================================================================================
    """
    """
    =======================================================================================
    Experiments functions.
    It contains all the functions used to generate the multiple experiments to avoid
    code duplication
    =======================================================================================
    """

    def training_experiment():
        # We work with a copy of the model
        model_copy = clone_network_with_same_weights(model)
        return train(model_copy, train_dataset, validation_dataset, epochs, loss_metric_model,
                     loss_object_model, topological_loss, topo_weight,
                     train_batch_size=batch_size,
                     iterations_to_compute_topological_loss=iterations_to_compute_topological_loss,
                     patience=experiment_patience,
                     epochs_to_compute_topological_loss=epochs_to_compute_topological_loss,
                     homology_distance_fn=homology_distance_fn,
                     filepath_to_save_experiments=filepath_to_save_experiments, iterations_to_plot=iterations_to_plot,
                     plot_x_axis_ppdd_low=0, plot_x_axis_ppdd_high=5, plot_y_axis_ppdd_low=0, plot_y_axis_ppdd_high=1,
                     verbose=verbose, accuracy_model=accuracy_model)

    def write_experiment_details():
        experiment_details = {
            'experiment_details': {
                'task_name': task.get_task_name(),
                'task_folder': task_folder,
                'model_number': model_number,
                'classical_loss': classical_loss_used,
                'loss_metric_model': loss_metric_model_used,
                'accuracy_model': accuracy_model_specs,
                'weight_topological_regularizer': topo_weight,
                'topological_loss': topo_loss_used,
                'homology_distance_function': homology_distance_fn_name,
                'epochs_trained': epochs,
                'batch_size': batch_size
            }
        }
        with open(f'{filepath_to_save_experiments}/experiment_details.json', 'w', encoding='utf-8') as f:
            json.dump(experiment_details, f, ensure_ascii=False, indent=4)

    # We perform experiments for each model
    for task_folder in tasks_folders:
        task = PGDLTask(task_folder)
        train_dataset, validation_dataset, test_dataset = task.get_datasets()
        for PGDL_model in task.models:
            # We only perform the experiment in the models we want to analyse, set up in the template.json
            if PGDL_model.model_number in models_to_analyse:
                model_number = PGDL_model.model_number
                filepath_to_save_experiments = f'{root_filepath_to_save_experiments}/{task.get_task_name()}/{model_number}'
                # We only perform the experiment if the folder does not exist, being able to generate checkpoints
                if not os.path.exists(filepath_to_save_experiments):
                    model = PGDL_model.get_model()
                    if verbose:
                        print(f'Starting experiment for model: {model_number}')
                        start_time = time.time()
                    training_experiment()
                    if verbose:
                        end_time = time.time()
                        print(f'Finished experiment in {end_time - start_time}s.')
                    # Write (persist) experiment data
                    write_experiment_details()


def experiments_train_from_last_checkpoint(experiment_specification, iterations_to_plot=0, verbose=True):
    """
    This functions continues the training of the experiments without using topology loss. It starts in the latest
    the nearest checkpoint to the iteration in which the experiment finished. We assume that the experiment has been
    previously executed. Theoretically, we save checkpoints every ten epochs, so we should continue training
    from the last checkpoint, at most ten epochs before the algorithm finished. If we have no checkpoint or the only
    checkpoint is the last iteration, we do not do anything.
    :param experiment_specification: Experiment specification in JSON format.
    :param iterations_to_plot: It prints a persistence diagrams every 'iterations_to_plot'.
    :param verbose: Prints information of the experiment execution in real time.
    :return: Nothing.
    """
    """
        =======================================================================================
        Experiment data & configuration
        It is a pity that it is required to write manually the information of the functions
        we are using, however I prefer that to use an eval of the strings.
        =======================================================================================
        """
    '--------------------------------------------------------------------------------------'
    tasks_folders = experiment_specification['tasks_folders']
    models_to_analyse = set(experiment_specification['models_to_analyse'])
    '--------------------------------------------------------------------------------------'
    '--------------------------------------------------------------------------------------'
    root_filepath_to_save_experiments = experiment_specification[
        'root_filepath_to_save_experiments']
    '--------------------------------------------------------------------------------------'
    batch_size = int(experiment_specification['batch_size'])
    epochs = int(experiment_specification['epochs'])
    '--------------------------------------------------------------------------------------'
    classical_loss_used = experiment_specification['classical_loss']
    loss_object_model = eval(classical_loss_used)
    '--------------------------------------------------------------------------------------'
    topo_loss_used = experiment_specification['topological_loss']
    topological_loss = eval(topo_loss_used)
    '--------------------------------------------------------------------------------------'
    accuracy_model = experiment_specification['accuracy_model']
    '--------------------------------------------------------------------------------------'
    loss_metric_model_used = experiment_specification['loss_metric_model_used']
    loss_metric_model = eval(loss_metric_model_used)
    '--------------------------------------------------------------------------------------'
    homology_distance_fn_name = experiment_specification['homology_distance_fn']
    homology_distance_fn = eval(homology_distance_fn_name)
    '--------------------------------------------------------------------------------------'
    experiment_patience = experiment_specification['patience']
    """
    =======================================================================================
    """
    """
    =======================================================================================
    Experiments functions.
    It contains all the functions used to generate the multiple experiments to avoid
    code duplication
    =======================================================================================
    """

    def training_experiment():
        # We work with a copy of the model
        model_copy = clone_network_with_same_weights(model)
        return train(model_copy, train_dataset, validation_dataset, epochs, loss_metric_model,
                     loss_object_model, topological_loss, 0.0,
                     train_batch_size=batch_size,
                     patience=experiment_patience,
                     homology_distance_fn=homology_distance_fn,
                     filepath_to_save_experiments=filepath_to_save_experiments, iterations_to_plot=iterations_to_plot,
                     plot_x_axis_ppdd_low=0, plot_x_axis_ppdd_high=5, plot_y_axis_ppdd_low=0, plot_y_axis_ppdd_high=1,
                     verbose=verbose)

    def write_experiment_details():
        experiment_details = {
            'experiment_details': {
                'task_name': task.get_task_name(),
                'task_folder': task_folder,
                'model_number': model_number,
                'classical_loss': classical_loss_used,
                'loss_metric_model': loss_metric_model_used,
                'accuracy_model': accuracy_model,
                'topological_loss': topo_loss_used,
                'homology_distance_function': homology_distance_fn_name,
                'epochs_trained': epochs,
                'batch_size': batch_size
            }
        }
        with open(f'{filepath_to_save_experiments}/experiment_details.json', 'w', encoding='utf-8') as f:
            json.dump(experiment_details, f, ensure_ascii=False, indent=4)
        # We perform experiments for each model

    for task_folder in tasks_folders:
        task = PGDLTask(task_folder)
        train_dataset, validation_dataset, test_dataset = task.get_datasets()
        for PGDL_model in task.models:
            # We only perform the experiment in the models we want to analyse, set up in the template.json
            if PGDL_model.model_number in models_to_analyse:
                model_number = PGDL_model.model_number
                filepath_existing_experiments = f'{root_filepath_to_save_experiments}/{task.get_task_name()}/{model_number}'
                # We only perform the experiment if the folder exists and if it was successfully executed.
                # The experiment was successfully executed if the train algorithm did not save
                # the file error_training.txt
                successfully_executed = not os.path.exists(f'{filepath_existing_experiments}/error_training.txt')
                if os.path.exists(filepath_existing_experiments) and successfully_executed:
                    filepath_to_save_experiments = f'{root_filepath_to_save_experiments}/continued/' \
                                                   f'{task.get_task_name()}/{model_number}'
                    # We only perform the experiments if previously we have not done it.
                    if not os.path.exists(filepath_to_save_experiments):
                        model = PGDL_model.get_model()
                        checkpoint_number = get_last_valid_checkpoint_iteration_number(filepath_existing_experiments)
                        if checkpoint_number is not None:
                            model.load_weights(f'{filepath_existing_experiments}/checkpoints/it_{checkpoint_number}/weights')
                        if verbose:
                            print(f'Starting experiment for model: {model_number}')
                            start_time = time.time()
                        training_experiment()
                        if verbose:
                            end_time = time.time()
                            print(f'Finished experiment in {end_time - start_time}s.')
                        # Write (persist) experiment data
                        write_experiment_details()
