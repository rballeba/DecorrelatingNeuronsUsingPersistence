import sys

sys.path.insert(0, '..')

from accuracies.compute_accuracies import compute_general_accuracies

# Imports
MNIST_dataset_folder_public_data = '/home/mnist'
experiments_folder_base = '/home/completed_experiments'

regularisers_MNIST = [
    {'folder': 'imp', 'possible_weights': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 100.0]},
    {'folder': 'std_avg', 'possible_weights': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 100.0]},
    {'folder': 'l_one', 'possible_weights': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 100.0]},
    {'folder': 'l_two', 'possible_weights': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 100.0]},
    {'folder': 'min_corr', 'possible_weights': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 100.0]},
]


def compute_accuracies_MNIST():
    for experiment_specs in regularisers_MNIST:
        folder = experiment_specs['folder']
        possible_weights = experiment_specs['possible_weights']
        for weight in possible_weights:
            print(f"Computing accuracies for {folder} with weight {weight}")
            experiments_folder = f'{experiments_folder_base}/mnist/public_data_experiments/{folder}/topo_weight_{weight}'
            accs = compute_general_accuracies(experiments_folder, MNIST_dataset_folder_public_data,
                                              datasets=('validation', 'test'), model_types=['trained'], type='mnist')
            print(accs)
            print("-----------------------")


# Main function
if __name__ == '__main__':
    compute_accuracies_MNIST()
