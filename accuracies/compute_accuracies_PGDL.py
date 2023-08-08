import sys

sys.path.insert(0, '..')

from accuracies.compute_accuracies import compute_general_accuracies

# Imports

PGDL_dataset_folder_public_data = '/home/google_data/public_data'
experiments_folder_base = '/home/completed_experiments'


PGDL_reference_data = f'{PGDL_dataset_folder_public_data}/reference_data'
PGDL_input_data = f'{PGDL_dataset_folder_public_data}/input_data'
task1_folder = 'task1_v4'

complete_task1_folder = f'{PGDL_input_data}/{task1_folder}'

# Usual and topo regularisers
regularisers_without_beta = [
    {'folder': 'imp', 'possible_weights': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1]},
    {'folder': 'std_avg', 'possible_weights': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 100.0]},
    {'folder': 'l_one', 'possible_weights': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 100.0]},
    {'folder': 'l_two', 'possible_weights': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 100.0]},
    {'folder': 'min_corr', 'possible_weights': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 100.0]},
]


def compute_PGDL_accuracies():
    for experiment_specs in regularisers_without_beta:
        folder = experiment_specs['folder']
        possible_weights = experiment_specs['possible_weights']
        for weight in possible_weights:
            print(f"Computing accuracies for {folder} with weight {weight}")
            experiments_folder = f'{experiments_folder_base}/PGDL/public_data_experiments/{folder}/topo_weight_{weight}/task1_v4'
            accs = compute_general_accuracies(experiments_folder, complete_task1_folder,
                                              datasets=('validation', 'test'), model_types=['trained'], type='pgdl')
            print(accs)
            print("-----------------------")


# Main function
if __name__ == '__main__':
    compute_PGDL_accuracies()
