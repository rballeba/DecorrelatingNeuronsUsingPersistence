# Importing libraries
import sys

import numpy as np

from model.critdd import Diagram, stats
from model.experiment_explorer import ExperimentExplorer
from model.filesystem import create_directory_if_it_does_not_exist
from model.statistics import print_p_values_matrix, print_accuracies, \
    generate_friedman_nemenyi_statistics, generate_friedman_dunn_statistics, print_best_iterations

STATISTICS = 'nemenyi'

base_datasets_folder = '/home'
experiments_folder = '/home/completed_experiments'


generate_friedman_p_values_statistics = generate_friedman_nemenyi_statistics

# Regularisation terms to be analysed

regularisers_PGDL = [
    {'folder': 'imp', 'possible_weights': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1]},
    {'folder': 'std_avg', 'possible_weights': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 100.0]},
    {'folder': 'l_one', 'possible_weights': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 100.0]},
    {'folder': 'l_two', 'possible_weights': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 100.0]},
    {'folder': 'min_corr', 'possible_weights': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 100.0]},
]

regularisers_MNIST = [
    {'folder': 'imp', 'possible_weights': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 100.0]},
    {'folder': 'std_avg', 'possible_weights': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 100.0]},
    {'folder': 'l_one', 'possible_weights': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 5.0]},
    {'folder': 'l_two', 'possible_weights': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0]},
    {'folder': 'min_corr', 'possible_weights': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 100.0]},
]

MNIST_networks_analysed = [0, 1, 2]


def get_dataset_folder(type_dataset):
    if type_dataset == 'pgdl':
        dataset_folder = f'{base_datasets_folder}/google_data/public_data/input_data/task1_v4'
    elif type_dataset == 'mnist':
        dataset_folder = f'{base_datasets_folder}/mnist'
    else:
        raise ValueError(f'Dataset {type_dataset} not recognised')
    return dataset_folder


def title_formated(text):
    return f'=========================\n{text}\n========================='


def get_regulariser_folder(reg_name, weight, type_dataset):
    if type_dataset == 'pgdl':
        return f'{experiments_folder}/PGDL/public_data_experiments/{reg_name}/topo_weight_{weight}/task1_v4'
    elif type_dataset == 'mnist':
        return f'{experiments_folder}/mnist/public_data_experiments/{reg_name}/topo_weight_{weight}'
    else:
        raise NotImplementedError('Only PGDL or MNIST datasets')


def get_no_regulariser_folder(type_dataset):
    if type_dataset == 'pgdl':
        return f'{experiments_folder}/PGDL/public_data_experiments/std_avg/topo_weight_0.0/task1_v4'
    elif type_dataset == 'mnist':
        return f'{experiments_folder}/mnist/public_data_experiments/std_avg/topo_weight_0.0'
    else:
        raise NotImplementedError('Only PGDL or MNIST datasets')


def get_ordered_network_names(any_experiment, type_dataset):
    if type_dataset == 'pgdl':
        network_ordered_names = list(
            map(lambda name: int(name), any_experiment.network_experiments.keys()))
        network_ordered_names.sort()
        return [str(name) for name in network_ordered_names]
    elif type_dataset == 'mnist':
        return [str(name) for name in MNIST_networks_analysed]


def get_statistics_for_best_weights(reg_name, possible_parameters, ordered_names, type_dataset, dataset_folder,
                                    statistic='test_accuracy'):
    statistics = []
    for network_name in ordered_names:
        best_validation_acc = -1.0  # It will be always greater than zero so it is always updated in the first iteration.
        best_validation_parameter = None
        for parameter in possible_parameters:
            experiment = ExperimentExplorer(get_regulariser_folder(reg_name, parameter, type_dataset),
                                            dataset_folder,
                                            type=type_dataset, load_dataset_and_models=False)
            validation_acc = experiment.network_experiments[network_name].get_accuracy(type='trained',
                                                                                       dataset='validation')
            if validation_acc > best_validation_acc:
                best_validation_acc = validation_acc
                best_validation_parameter = parameter

        experiment_with_best_validation = ExperimentExplorer(
            get_regulariser_folder(reg_name, best_validation_parameter, type_dataset), dataset_folder,
            type='pgdl', load_dataset_and_models=False)
        if statistic == 'test_accuracy':
            statistics.append(
                experiment_with_best_validation.network_experiments[network_name].get_accuracy(type='trained',
                                                                                               dataset='test'))
        elif statistic == 'number_of_iterations':
            statistics.append(
                int(experiment_with_best_validation.network_experiments[network_name].get_best_iteration()))
        else:
            raise ValueError(f'Unknown statistic {statistic}')
    return statistics


def get_matrix_statistics(regularisers, type_dataset, statistic, dataset_folder):
    best_statistics = []
    headers_stringified = []
    # Experiments without regulariser
    # Add the accuracy for the experiment with no regulariser
    experiment_no_regulariser = ExperimentExplorer(get_no_regulariser_folder(type_dataset), dataset_folder,
                                                   type=type_dataset)
    ordered_network_names = get_ordered_network_names(experiment_no_regulariser, type_dataset)
    if statistic == 'test_accuracy':
        best_statistics.append([experiment_no_regulariser.network_experiments[network_name].get_accuracy(type='trained',
                                                                                                         dataset='test')
                                for network_name in ordered_network_names])
    elif statistic == 'number_of_iterations':
        best_statistics.append(
            [int(experiment_no_regulariser.network_experiments[network_name].get_best_iteration())
             for network_name in ordered_network_names])
    headers_stringified.append('no_reg')
    # Experiments with regulariser and one parameter
    for regulariser in regularisers:
        folder = regulariser['folder']
        possible_weights = regulariser['possible_weights']
        best_statistics.append(get_statistics_for_best_weights(folder, possible_weights, ordered_network_names,
                                                               type_dataset, statistic))
        headers_stringified.append(folder)
    return best_statistics, ordered_network_names, headers_stringified


def get_iterations_selected_regularisers_with_weights(regularisers, type_dataset, dataset_folder):
    best_iterations, ordered_network_names, headers_stringified = get_matrix_statistics(regularisers, type_dataset,
                                                                                        'number_of_iterations',
                                                                                        dataset_folder)
    print("Iterations for best weights:")
    print_best_iterations(best_iterations, ordered_network_names, headers_stringified)


def rank_regulariser_weights(reg_name, possible_parameters, ordered_names, type_dataset, dataset_folder):
    parameter_ranks = {parameter: 0 for parameter in possible_parameters}
    for network_name in ordered_names:
        best_validation_acc = -1.0  # It will be always greater than zero so it is always updated in the first iteration.
        best_validation_parameter = None
        for parameter in possible_parameters:
            experiment = ExperimentExplorer(get_regulariser_folder(reg_name, parameter, type_dataset),
                                            dataset_folder,
                                            type=type_dataset, load_dataset_and_models=False)
            validation_acc = experiment.network_experiments[network_name].get_accuracy(type='trained',
                                                                                       dataset='validation')
            if validation_acc > best_validation_acc:
                best_validation_acc = validation_acc
                best_validation_parameter = parameter
        # Now we add one to the rank of the best performing regulariser for the current network
        parameter_ranks[best_validation_parameter] += 1
    # Now we print, the rank information about each weight of the specific regulariser.
    print(f'Ranking weights for {reg_name}')
    parameter_ranks_sorted = sorted(parameter_ranks.items())
    print(parameter_ranks_sorted)
    print("=====================================")


def critical_differences_diagram_and_stats(regularisers, type_dataset, dataset_folder, title=None, alpha=0.05,
                                           statistical_test='wilcoxon', adjustment=None,
                                           decimals_round_for_accuracies=3):
    """
    Generates a critical differences diagram for the given regularisers and dataset.
    :param decimals_round_for_accuracies:
    :param regularisers:
    :param type_dataset:
    :param title:
    :param alpha:
    :param statistical_test:
    :param adjustment: "holm" or "bonferroni" or "None"
    :return:
    """
    if type_dataset in ('pgdl', 'mnist'):
        accuracies, ordered_network_names, headers_stringified = get_matrix_statistics(regularisers, type_dataset,
                                                                                       'test_accuracy', dataset_folder)
    elif type_dataset == 'combined':
        accuracies_pgdl, ordered_network_names_pgdl, headers_stringified_pgdl = get_matrix_statistics(
            regularisers[0], 'pgdl', 'test_accuracy', dataset_folder[0])
        accuracies_mnist, ordered_network_names_mnist, headers_stringified_mnist = get_matrix_statistics(
            regularisers[1], 'mnist',  'test_accuracy', dataset_folder[1])
        assert headers_stringified_mnist == headers_stringified_pgdl
        accuracies = [accs_mnist + accs_pgd for accs_pgd, accs_mnist in zip(accuracies_pgdl, accuracies_mnist)]
        ordered_network_names = ordered_network_names_mnist + ordered_network_names_pgdl
        headers_stringified = headers_stringified_mnist
    else:
        raise ValueError(f'Unknown type_dataset {type_dataset}')
    # create a CD diagram
    experiments_x_procedures = np.array(accuracies).T
    diagram = Diagram(
        experiments_x_procedures,
        treatment_names=headers_stringified,
        maximize_outcome=True,
        type=statistical_test
    )
    # inspect average ranks and groups of statistically indistinguishable treatments
    print(f'p-value Friedman cridd package: {diagram.r.pvalue}')
    print(f'p-value matrix')
    p_values_matrix = stats.adjust_pairwise_tests(diagram.P, adjustment) if adjustment is not None else diagram.P
    print_p_values_matrix(p_values_matrix, headers_stringified, alpha, 3)
    print(f'Average ranks: {diagram.average_ranks}')
    print(f'Groups of indistinguishable regularisers', diagram.get_groups(alpha=alpha, adjustment=adjustment))
    print(f"")
    print_accuracies(accuracies, ordered_network_names, headers_stringified, decimals_round_for_accuracies)
    title = title if title is not None else type_dataset
    # export the diagram to a file
    if type_dataset == 'pgdl':
        type_dataset_folder = 'PGDL'
    elif type_dataset == 'mnist':
        type_dataset_folder = 'mnist'
    elif type_dataset == 'combined':
        type_dataset_folder = 'combined'
    else:
        raise NotImplementedError("Only 'pgdl' and 'mnist' datasets are supported")
    folder_to_save_file = f'{experiments_folder}/{type_dataset_folder}'
    create_directory_if_it_does_not_exist(folder_to_save_file)
    diagram.to_file(
        f"{folder_to_save_file}/critical_differences_diagram_alpha_{alpha}.tex",
        alpha=alpha,
        adjustment=adjustment,
        reverse_x=True,
        axis_options={"title": title},
    )


def rank_weights_by_validation_accuracies(regularisers, type_dataset, dataset_folder):
    """
    For each regulariser rank the weights by the number of times they are the best ones for a network.
    :return: Nothing. It prints the weights for each regulariser that worked best overall.
    """
    experiment_no_regulariser = ExperimentExplorer(get_no_regulariser_folder(type_dataset), dataset_folder,
                                                   type=type_dataset)
    ordered_network_names = get_ordered_network_names(experiment_no_regulariser, type_dataset)
    for regulariser in regularisers:
        reg_name, possible_weights = regulariser['folder'], regulariser['possible_weights']
        rank_regulariser_weights(reg_name, possible_weights, ordered_network_names, type_dataset, dataset_folder)


# Main function
if __name__ == '__main__':
    if len(sys.argv) < 2:
        "You must introduce as first parameter the dataset to extract statistics from. 'combined', 'mnist' or 'pgdl'"
    type_dataset = sys.argv[1]
    if type_dataset not in {'mnist', 'pgdl', 'combined'}:
        raise NotImplementedError("Only datasets 'mnist', 'pgdl', or 'combined'.")
    if type_dataset == 'pgdl':
        regularisers = regularisers_PGDL
    elif type_dataset == 'mnist':
        regularisers = regularisers_MNIST
    elif type_dataset == 'combined':
        pass
    else:
        raise NotImplementedError("Only datasets 'combined', 'mnist' or 'pgdl'.")
    if type_dataset in {'mnist', 'pgdl'}:
        task_folder = get_dataset_folder(type_dataset)
        print("Computing critical differences diagram")
        critical_differences_diagram_and_stats(regularisers, type_dataset, task_folder, adjustment=None, alpha=0.05,
                                               statistical_test='nemenyi')
    else:
        task_folder_mnist = get_dataset_folder('mnist')
        task_folder_pgdl = get_dataset_folder('pgdl')
        print("Computing critical differences diagram combined")
        critical_differences_diagram_and_stats([regularisers_PGDL, regularisers_MNIST], type_dataset,
                                               [task_folder_pgdl, task_folder_mnist], adjustment=None, alpha=0.05,
                                               statistical_test='nemenyi')

