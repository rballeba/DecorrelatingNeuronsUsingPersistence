# Class used in the notebook analysis_of_results.ipynb
import os
import time
from itertools import product

import tensorflow as tf
from os.path import isfile, join

import numpy as np
import matplotlib.image as mpimg
import mediapy as media
from matplotlib import pyplot as plt
from seaborn import kdeplot
from tabulate import tabulate

from model.Google_PGDL.PGDL_task import PGDLTask
from model.algorithms.ripser_dim_zero import fast_zeroth_persistence_diagram
from model.distances import absolute_correlation
from model.filesystem import load_json, load_np_array, load_dataset
from model.neuron_samplings.importance_percentage_neuron_sampling import \
    get_neurons_point_cloud_importance_percentage_sampling
from model.plots.images import show_image_list


class ExperimentExplorer:
    def __init__(self, experiment_folder, dataset_folder, type='pgdl', load_dataset_and_models=True):
        if type not in ('pgdl', 'tinyimagenet', 'mnist'):
            raise ValueError(f'Invalid type: {type}')
        self.type = type
        self.network_experiments = dict()
        self.experiment_folder = experiment_folder
        self.dataset_folder = dataset_folder
        self.load_dataset_and_models = load_dataset_and_models
        self.get_network_experiments()

    def get_network_experiments(self):
        """
        Loads all the inidividual results for the different networks inside the experiment.
        :return: Nothing.
        """
        subfolder_names = [f.name for f in os.scandir(self.experiment_folder)
                           if f.is_dir()]
        for subfolder in subfolder_names:
            # Check if there is a folder inside subfolder starting with 'it_', if so, add it to the experiments (if not,
            # the experiment has not been trained successfully).
            if any([f.name.startswith('it_') for f in os.scandir(join(self.experiment_folder, subfolder))
                    if f.is_dir()]):
                self.network_experiments[subfolder] = NetworkExperiment(self.experiment_folder, subfolder,
                                                                        self.dataset_folder, self.type,
                                                                        self.load_dataset_and_models)

    def plot_persistence_diagrams_accuracies(self, model_type='trained', dataset='test', figsize_canvas=(40, 20),
                                             cols=10):
        '''
        Plots the persistence diagrams of the networks with the accuracies in the trained network or in the
        initial network.
        :param model_type: 'trained' or 'initial'
        :param dataset: 'train', 'validation', or 'test'
        :param figsize_canvas:
        :param cols:
        :return:
        '''
        if model_type not in ('trained', 'initial'):
            raise ValueError(f'Invalid model_type: {model_type}')
        if dataset not in ('train', 'validation', 'test'):
            raise ValueError(f'Invalid dataset: {dataset}')
        network_experiments_ordered_names = list(self.network_experiments.keys())
        network_experiments_ordered_names.sort()
        ppdd = []
        accuracy_texts = []
        for experiment_name in network_experiments_ordered_names:
            experiment = self.network_experiments[experiment_name]
            index = experiment.best_iteration if model_type == 'trained' else 0  # initial iteration
            accuracy = experiment.get_accuracy(model_type, dataset)
            nearest_persistence_diagram_plot = experiment.get_persistence_diagram_image_at_iteration(index)
            ppdd.append(nearest_persistence_diagram_plot[0])
            accuracy_texts.append(f'Network: {experiment_name} - Acc: {accuracy * 100:.2f}% \nin '
                                  f'iter. {index}')
        title_start = f'{model_type.capitalize()} model {dataset}' if model_type == 'trained' \
            else f'Initial model {dataset}'
        fig, axes = show_image_list(ppdd, accuracy_texts,
                                    general_title=f'{title_start} accuracies for each network and '
                                                  f'their closest persistence diagrams',
                                    num_cols=min(cols, len(self.network_experiments)), figsize=figsize_canvas,
                                    grid=False)
        return fig, axes

    def reproduce_pd_animation_for_network_jupyter(self, name):
        """
        Only valid in Jupyter notebook
        :param name: Name of the network
        :return: Nothing
        """
        self.network_experiments[name].reproduce_training_pd_animation_jupyter()

    def get_average_model_accuracies(self, model_type='trained', dataset='test'):
        if model_type not in ('trained', 'initial'):
            raise ValueError(f'Invalid model_type: {model_type}')
        if dataset not in ('train', 'validation', 'test'):
            raise ValueError(f'Invalid dataset: {dataset}')
        average_accuracy = 0
        network_experiments_ordered_names = list(self.network_experiments.keys())
        network_experiments_ordered_names.sort()
        for experiment_name in network_experiments_ordered_names:
            average_accuracy += self.network_experiments[experiment_name].get_accuracy(model_type, dataset)
        average_accuracy *= (1.0 / len(network_experiments_ordered_names))
        return average_accuracy

    def plot_all_persistence_diagrams_for_selected_models(self, selected_models, x_axis_ppdd_low=None,
                                                          x_axis_ppdd_high=None,
                                                          y_axis_ppdd_low=None, y_axis_ppdd_high=None):
        """
        This method plot the persistence diagrams for all the selected models
        :param selected_models: A list of model names. For PGDL, they are numbers as strings.
        :return:
        """
        for model_name in selected_models:
            self.network_experiments[model_name].plot_persistence_diagrams(x_axis_ppdd_low, x_axis_ppdd_high,
                                                                           y_axis_ppdd_low, y_axis_ppdd_high)


def compare_experiments(*experiments: ExperimentExplorer):
    for idx, experiment in enumerate(experiments):
        average_test_best_accuracy = experiment.get_average_model_accuracies(dataset='test')
        average_train_best_accuracy = experiment.get_average_model_accuracies(dataset='train')
        print(f'Experiment {idx}: Average best test accuracy: {average_test_best_accuracy * 100:.2f} - '
              f'Average best train accuracy: {average_train_best_accuracy * 100:.2f}')
    print(' ')  # Line break
    # We assume that experiments with the same name comes from the same network. In our case we enumerate the networks
    # and the names are the integers identifying the networks, so it holds.

    best_networks_for_experiments_test = {k: [] for k in range(len(experiments))}
    best_networks_for_experiments_training = {k: [] for k in range(len(experiments))}
    total_comparable_networks = 0
    for network_name in experiments[0].network_experiments.keys():
        # If all the experiments contains a network with the same name we can compare the network.
        if all([network_name in experiment.network_experiments.keys() for experiment in experiments]):
            test_accuracies_for_experiments = [experiment.network_experiments[network_name]
                                               .get_accuracy('trained', 'test') for experiment in experiments]
            training_accuracies_for_experiments = [experiment.network_experiments[network_name]
                                                   .get_accuracy['trained', 'train'] for experiment in experiments]
            best_networks_for_experiments_test[np.argmax(test_accuracies_for_experiments)].append(
                network_name)
            best_networks_for_experiments_training[np.argmax(training_accuracies_for_experiments)].append(
                network_name)
            total_comparable_networks += 1
    for experiment_num in range(len(experiments)):
        number_of_best_accuracies_test = len(best_networks_for_experiments_test[experiment_num])
        percentage_of_best_accuracies_test = number_of_best_accuracies_test / total_comparable_networks
        number_of_best_accuracies_training = len(best_networks_for_experiments_training[experiment_num])
        percentage_of_best_accuracies_training = number_of_best_accuracies_training / total_comparable_networks
        print(f'Experiment {experiment_num} obtains the best test accuracies for '
              f'{number_of_best_accuracies_test} networks '
              f'({percentage_of_best_accuracies_test * 100:.2f}%). These are:')
        print(best_networks_for_experiments_test[experiment_num])
        print(f'Experiment {experiment_num} obtains the best train accuracies for '
              f'{number_of_best_accuracies_training} networks '
              f'({percentage_of_best_accuracies_training * 100:.2f}%). These are:')
        print(best_networks_for_experiments_training[experiment_num])


def compare_accuracies_networks(*experiments: ExperimentExplorer, model_type='trained', dataset='test',
                                table_style='github', decimals_round=3):
    """
        It returns a Markdown table with the test accuracies for the best accuracies in the validation dataset.
        To do that, we check, for all the trained neural networks, which is its best validation accuracy,
        we take the index of the iteration in which we obtained this validation accuracy and then extract
        the test accuracy for this iteration.
        :param experiments: List of NetworkExperiment to compare.
        :param model_type: 'trained' or 'initial'.
        :param dataset: Dataset to use to compare the networks. It can be 'test', 'validation' or 'train'.
        :param table_style: tablefmt format for the tabulate function of the tabulate package.
        :param decimals_round: Number of decimals to round the accuracies. By default 3.
        :return:
        """
    if model_type not in ('trained', 'initial'):
        raise ValueError(f'Invalid model_type: {model_type}')
    if dataset not in ('train', 'validation', 'test'):
        raise ValueError(f'Invalid dataset: {dataset}')
    headers = ['Network']
    table_content = []
    table_content_numeric = []
    for experiment_num in range(len(experiments)):
        experiment_regularizer = experiments[experiment_num].experiment_folder.split('/')[-3]
        weight_experiment = experiments[experiment_num].experiment_folder.split('/')[-2].split('_')[-1]
        headers.append(f'{experiment_regularizer} - w:{weight_experiment}: Test. acc.')
    network_experiments_ordered_names = list(map(lambda name: int(name), experiments[0].network_experiments.keys()))
    network_experiments_ordered_names.sort()
    network_experiments_ordered_names = [str(name) for name in network_experiments_ordered_names]
    for network_name in network_experiments_ordered_names:
        # If all the experiments contains a network with the same name we can compare the network.
        if all([network_name in experiment.network_experiments.keys() for experiment in experiments]):
            accuracies_for_experiments = [round(experiment.network_experiments[network_name]
                                                .get_accuracy(model_type, dataset), decimals_round)
                                          for experiment in experiments]
            total_accuracies_numeric = [accuracies_for_experiments[experiment_num]
                                        for experiment_num in range(len(experiments))]
            idx_max_accuracy = np.argmax(total_accuracies_numeric)
            # We put bold on the maximum total accuracy and we convert the rest of the elements to string
            # to be coherent in the type containing the list.
            total_accuracies = [str(acc) for acc in total_accuracies_numeric]
            total_accuracies[idx_max_accuracy] = f'**{total_accuracies[idx_max_accuracy]}**'
            table_content.append([network_name] + total_accuracies)
            table_content_numeric.append(total_accuracies_numeric)
    # Now we add the averages per experiment.
    averages_accuracies = np.round(np.mean(np.vstack(table_content_numeric), axis=0), decimals_round)
    idx_highest_average = np.argmax(averages_accuracies)
    averages_accuracies_text = [str(acc) for acc in list(averages_accuracies)]
    averages_accuracies_text[idx_highest_average] = f'**{averages_accuracies_text[idx_highest_average]}**'
    table_content.append(['Average'] + averages_accuracies_text)
    print(tabulate(table_content, headers=headers, tablefmt=table_style))


def _get_persistence_diagram_in_dataset(dataset, model,
                                        homology_distance_fn=absolute_correlation.compute_distance_matrix,
                                        sampling_neurons_percentage=0.005, number_of_dataset_samples=2000):
    """
    Computes the zeroth persistence diagram for a sample of the dataset given as argument
    :param model: Model for which we compute persistence diagrams
    :param homology_distance_fn: Function to compute the distance matrix used to compute the persistence diagram
    :param sampling_neurons_percentage: Number of neurons taken from the neural network to compute the persistence
    :param dataset:
    :return:
    """
    # These computations are done in the function total_persistence but we repeat them for the sake
    # of testing our approach. It is very expensive computationally, so it should be avoided in production
    # or training big models.
    # Take a random sample of 3000 examples from the dataset
    inputs, labels = list(zip(*dataset.shuffle(10000).take(number_of_dataset_samples).as_numpy_iterator()))
    inputs = tf.stack(inputs)
    # Add one dimension to the inputs
    activation_x_examples = get_neurons_point_cloud_importance_percentage_sampling(model, inputs,
                                                                                   sampling_percentage=sampling_neurons_percentage)
    distance_matrix = homology_distance_fn(activation_x_examples, activation_x_examples.shape[0],
                                           activation_x_examples.shape[1])
    persistence_diagram_deaths, _ = fast_zeroth_persistence_diagram(distance_matrix)
    return persistence_diagram_deaths


class NetworkExperiment:
    def __init__(self, base_folder, name_experiment, dataset_folder, type_experiment, load_dataset_and_models=True):
        self.type_experiment = type_experiment
        self.base_folder = base_folder
        self.name_experiment = name_experiment
        self.dataset_folder = dataset_folder
        self.experiment_folder = f'{self.base_folder}/{self.name_experiment}'
        self.experiment_details = load_json(f'{self.experiment_folder}/experiment_details.json')['experiment_details']
        self.train_original_losses = load_np_array(f'{self.experiment_folder}/train_original_losses.npy')
        self.train_topological_losses = load_np_array(f'{self.experiment_folder}/train_topological_losses.npy')
        self.validation_losses = load_np_array(f'{self.experiment_folder}/validation_losses.npy')
        self.accuracy_model = eval(self.experiment_details['accuracy_model'])('accuracy in explorer')
        self.best_iteration = self.get_best_iteration()
        if load_dataset_and_models:
            self.initial_model = self.get_initial_model()
            self.working_model = tf.keras.models.clone_model(self.initial_model)
            self.recreate_trained_model()
            self.train_dataset, self.validation_dataset, self.test_dataset = self.get_datasets()
        else:
            self.initial_model = None
            self.working_model = None
            self.trained_model = None
            self.train_dataset, self.validation_dataset, self.test_dataset = None, None, None

    def get_datasets(self):
        if self.type_experiment == 'pgdl':
            task = PGDLTask(self.dataset_folder)
            train_dataset, validation_dataset, test_dataset = task.get_datasets()

        elif self.type_experiment == 'tinyimagenet':
            train_dataset = load_dataset(f'{self.dataset_folder}/new_train')
            validation_dataset = load_dataset(f'{self.dataset_folder}/new_validation')
            test_dataset = load_dataset(f'{self.dataset_folder}/new_test')

        elif self.type_experiment == 'mnist':
            train_dataset = load_dataset(f'{self.dataset_folder}/train')
            validation_dataset = load_dataset(f'{self.dataset_folder}/validation')
            test_dataset = load_dataset(f'{self.dataset_folder}/test')
        else:
            raise ValueError(f'Experiment type {self.type_experiment} not supported')
        return train_dataset, validation_dataset, test_dataset

    def recreate_trained_model(self):
        # We recreate the trained model from the initial model using the weights of the best iteration.
        # First, it is needed to set properly the initial model self.initial_model.
        self.trained_model = tf.keras.models.clone_model(self.initial_model)
        # When training in parallel there is a small probability that the model has been trained more than once and
        # saved in the same folder. In that case, we take the last one.
        trained_models = list(filter(lambda folder: folder[:2] == 'it', os.listdir(self.experiment_folder)))
        if len(trained_models) > 1:
            trained_models = sorted(trained_models, key=lambda folder: int(folder[3:]))
        model_folder = trained_models[-1]  # We take the last one, that is the one with the highest number of iterations
        # in case there are more than one and the unique one otherwise.
        self.trained_model.load_weights(f'{self.experiment_folder}/{model_folder}/weights')

    def get_initial_model(self):
        if self.type_experiment == 'pgdl':
            task = PGDLTask(self.dataset_folder)
            pgdl_model = next(filter(lambda model: model.model_number == int(self.name_experiment), task.models))
            return pgdl_model.get_model()
        elif self.type_experiment == 'tinyimagenet':
            return tf.keras.models.load_model(f'{self.dataset_folder}/{self.name_experiment}')
        elif self.type_experiment == 'mnist':
            return tf.keras.models.load_model(f'{self.dataset_folder}/mnist_model_{self.name_experiment}')
        else:
            raise ValueError(f'Experiment type {self.type_experiment} not supported')

    def get_best_iteration(self):
        return next(filter(lambda folder: folder[:2] == 'it', os.listdir(self.experiment_folder)))[3:]

    # Function that given tensorflow dataset and model returns the accuracy of the model in the dataset.
    def get_accuracy_in_dataset(self, dataset, model, batch_size=64, verbose=True):
        if verbose:
            start_compute_accuracy = time.process_time()
            print(f'Computing accuracy')
        # We copy the initial model because keras.save_model() has a bug that does not
        # allow to make predictions with new data. Instead, we copy the initial model
        # and we transfer the weights. This is a workaround that works. It should not be necessary.
        self.accuracy_model.reset_states()  # Reset the accuracy model
        self.working_model.set_weights(model.get_weights())
        batched_dataset = dataset.batch(batch_size)

        @tf.function
        def accuracy_step(inputs, labels):
            test_predictions = self.working_model(inputs, training=False)
            self.accuracy_model(labels, test_predictions)

        for test_inputs, test_labels in batched_dataset:
            accuracy_step(test_inputs, test_labels)
        accuracy = self.accuracy_model.result().numpy()
        self.accuracy_model.reset_states()
        if verbose:
            print(
                f'Accuracy in dataset {dataset} is {accuracy}. Elapsed time {time.process_time() - start_compute_accuracy}')
        return accuracy

    def get_accuracy(self, type: str, dataset: str, recompute: bool = False):
        """
        It returns the accuracy of the model in the initial model or in the best model.
        :param type:'trained' or 'initial'.
        :dataset: 'train', 'validation' or 'test'.
        :return: The accuracy of the selected model evaluated in the test dataset.
        """
        type_str, dataset_str = type, dataset
        if type == 'trained':
            model = self.trained_model
        elif type == 'initial':
            model = self.initial_model
        else:
            raise NotImplemented('Only trained and initial accuracies are implemented in this function')
        if dataset == 'train':
            dataset = self.train_dataset
        elif dataset == 'validation':
            dataset = self.validation_dataset
        elif dataset == 'test':
            dataset = self.test_dataset
        else:
            raise NotImplemented('Only train, validation and test datasets are implemented in this function')
            # Save the accuracy as a float in the disk to avoid computing it again
        accuracy_file = f'{self.experiment_folder}/accuracies_{type_str}_{dataset_str}.npy'
        # If the accuracy has been previously computed, we load it from the disk.
        if os.path.isfile(accuracy_file) and not recompute:
            with open(accuracy_file, 'rb') as acc_f:
                accuracy = float(np.load(acc_f))
        else:
            # Delete the file if it exists
            if os.path.isfile(accuracy_file):
                os.remove(accuracy_file)
            # In case the accuracy has not been computed, we compute it and save it in the disk.
            accuracy = self.get_accuracy_in_dataset(dataset, model)
            with open(accuracy_file, 'wb') as new_acc_f:
                np.save(new_acc_f, accuracy)
        return accuracy

    def get_persistence_diagram_image_at_iteration(self, iteration_number):
        """
        Gets the saved image of the persistence diagram generated at iteration 'iteration_number'. If it does not exist,
        it returns the available plot with the smallest distance to the specified iteration.
        :param iteration_number: The desired iteration number
        :return: numpy array image with the persistence diagram, the iteration of the plotted persistence diagram.
        """
        path_to_plots = f'{self.experiment_folder}/training_plot/variable_measures'
        plot_numbers = [int(f[:-4]) for f in os.listdir(path_to_plots)
                        if (isfile(join(path_to_plots, f)) and f[:-4].isdigit())]
        closest_plot_iteration_idx = np.argmin(
            [abs(int(iteration_number) - plot_number) for plot_number in plot_numbers])
        closest_iteration = plot_numbers[closest_plot_iteration_idx]
        plot = mpimg.imread(f'{self.experiment_folder}/training_plot/variable_measures/{closest_iteration}.png')
        return plot, closest_iteration

    def get_persistence_diagram_images_at_iterations(self, iterations: list):
        """
        Gets the saved images of the persistence diagrams generated at the desired iterations.
        :param iterations: List of integers with iteration numbers
        :return: list of pairs of numpy arrays with the images of the persistence diagrams and the iteration number
        of the plot
        """
        return [self.get_persistence_diagram_image_at_iteration(iteration) for iteration in iterations]

    def reproduce_training_pd_animation_jupyter(self, type='fixed_measures', fps=15):
        """
        Use only in Jupyter notebook
        type: str -> 'fixed_measures' or 'variable_measures'
        :return: Nothing
        """
        animation_filepath = f'{self.experiment_folder}/training_plot/{type}/result.gif'
        animation = media.read_video(animation_filepath)
        media.show_video(animation, fps=fps, codec='gif')

    def plot_train_and_validation_losses(self, max_y=None, figsize=(6.5, 5)):
        """
        Plots the train and validation losses of the experiment given in the attributes train_losses and
        validation_losses. The x axis represents each iteration during the training process and the y axis represents
        the loss value for each of the losses. The train loss is painted in red and the validation loss in blue.
        It uses latex to render the labels. Also, the text are big-enough to be read in a paper.
        :param figsize: Figsize of the plot rendered by matplotlib
        :param max_y: If not None, it sets the maximum value of the y axis to be max_y. If None, the value is set
        automatically by matplotlib.
        :return:
        """
        epoch_size = len(self.train_original_losses) // len(self.validation_losses)
        training_accuracies = self.train_original_losses[::epoch_size]
        assert len(training_accuracies) == len(self.validation_losses)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(training_accuracies, color='red', label='Train loss')
        ax.plot(self.validation_losses, color='blue', label='Validation loss')
        # If fixed_loss_measure=True then we set the y axis maximum value to be max_y and the minimum to be 0
        if max_y is not None:
            ax.set_ylim(0, max_y)
        # Set the title and the labels of the figure
        ax.set_title(f'Train and validation losses for model {self.name_experiment}', fontsize=20)
        ax.set_xlabel('Epoch', fontsize=20)
        ax.set_ylabel('Loss', fontsize=20)
        # Set the ticks fontsize
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.legend(fontsize=20)
        # Add grid
        ax.grid()
        return fig, ax

    def generate_persistence_diagram_for_dataset(self, type: str, dataset: str):
        """
        Computes the zeroth persistence diagram for a sample of the dataset given as argument
        :param dataset:
        :return:
        """
        type_str, dataset_str = type, dataset
        if type == 'trained':
            model = self.trained_model
        elif type == 'initial':
            model = self.initial_model
        else:
            raise NotImplemented('Only trained and initial accuracies are implemented in this function')
        if dataset == 'train':
            dataset = self.train_dataset
        elif dataset == 'validation':
            dataset = self.validation_dataset
        elif dataset == 'test':
            dataset = self.test_dataset
        else:
            raise NotImplemented('Only train, validation and test datasets are implemented in this function')
        path_persistence_diagram = f'{self.experiment_folder}/pd_dim_0_{type_str}_{dataset_str}.npy'
        # If the deaths of the zeroth persistence diagram are not stored into the disk, we compute them
        if not os.path.exists(path_persistence_diagram):
            deaths_persistence_diagram = _get_persistence_diagram_in_dataset(dataset, model)
            with open(path_persistence_diagram, 'wb') as f:
                np.save(f, deaths_persistence_diagram)
        else:
            with open(path_persistence_diagram, 'rb') as f:
                deaths_persistence_diagram = np.load(f)
        return deaths_persistence_diagram

    def plot_persistence_diagrams(self, x_axis_ppdd_low=None, x_axis_ppdd_high=None,
                                  y_axis_ppdd_low=None, y_axis_ppdd_high=None):
        """
        It plots the six types of persistence diagrams depending on type (trained, initial) and dataset (train,
        validation, test) in a 2x3 grid. The first row corresponds to the trained model and the second row to the
        initial model. The first column corresponds to the train dataset, the second to the validation dataset and
        the third to the test dataset. The x axis represents the density of the deaths of the persistence diagram
        and the y axis represents the possible values of the deaths of the persistence diagrams.
        :return:
        """
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.flatten()
        # Set the whole title of the figure
        fig.suptitle(f'Persistence diagrams for model {self.name_experiment}', fontsize=20)
        for i, (type, dataset) in enumerate(product(['trained', 'initial'], ['train', 'validation', 'test'])):
            deaths_persistence_diagram = self.generate_persistence_diagram_for_dataset(type, dataset)
            kdeplot(y=deaths_persistence_diagram, ax=axs[i])
            axs[i].set_title(f'T: {type}, D: {dataset}',
                             fontsize=20)
            # Set the x_label only if it is the second row
            if i >= 3:
                axs[i].set_xlabel('Deaths', fontsize=20)
            else:
                axs[i].set_xlabel('')
            # Set the y_label only if it is the first column
            if i % 3 == 0:
                axs[i].set_ylabel('Density', fontsize=20)
            axs[i].tick_params(axis='x', labelsize=10)
            axs[i].tick_params(axis='y', labelsize=10)
            if (x_axis_ppdd_low is not None) and (x_axis_ppdd_high is not None):
                axs[i].set_ylim(y_axis_ppdd_low, y_axis_ppdd_high)
                axs[i].set_xlim(x_axis_ppdd_low, x_axis_ppdd_high)
        plt.show()
        return fig, axs
