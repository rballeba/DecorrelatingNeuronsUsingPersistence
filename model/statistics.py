from scipy import stats
import scikit_posthocs as sp
import numpy as np
from tabulate import tabulate


def generate_friedman_nemenyi_statistics(accuracies):
    """
    It produces the p_value for the Friedman test and it computes the Nemenyi post-hoc test, returning its
    p-value matrix.
    :param accuracies: Matrix where each row represents a method and each column represents a dataset. List of
    lists.
    :return: p_value_friedman, p_value_nemenyi_matrix
    """
    _, p_value = stats.friedmanchisquare(*accuracies)
    nemenyi_matrix = sp.posthoc_nemenyi_friedman(np.array(accuracies).T)
    return p_value, nemenyi_matrix


def generate_friedman_dunn_statistics(accuracies):
    _, p_value = stats.friedmanchisquare(*accuracies)
    dunn_matrix = sp.posthoc_dunn(accuracies, p_adjust='bonferroni')
    return p_value, dunn_matrix


def print_p_values_matrix(p_values_matrix, headers, highlight_p_values_below=0.05, decimals_round=3,
                          style='latex_booktabs'):
    p_values_matrix_emphasised = _process_p_values_matrix_to_display(p_values_matrix, highlight_p_values_below,
                                                                     decimals_round)
    print(tabulate(p_values_matrix_emphasised, headers=headers, tablefmt=style,
                   showindex=headers))


def print_accuracies(accuracies, network_names, experiment_names, style='latex_booktabs', decimals_round=3):
    rounded_accuracies = [[round(value, decimals_round) for value in row] for row in accuracies]
    print(tabulate(rounded_accuracies, headers=network_names, tablefmt=style,
                   showindex=experiment_names))


def print_best_iterations(best_iterations, network_names, experiment_names, style='github'):
    print(tabulate(best_iterations, headers=network_names, tablefmt=style,
                   showindex=experiment_names))


def _process_p_values_matrix_to_display(p_values_matrix, highlight_p_values_below=0.05, decimals_round=3):
    for i in range(len(p_values_matrix)):
        p_values_matrix[i, i] = 1
        for j in range(i + 1, len(p_values_matrix)):
            p_values_matrix[i, j] = p_values_matrix[j, i]
    p_values_matrix_emphasised = [
        [str(round(value,
                   decimals_round)) if value >= highlight_p_values_below else f'**{round(value, decimals_round)}**'
         for value in row] for row in p_values_matrix.tolist()]
    return p_values_matrix_emphasised
