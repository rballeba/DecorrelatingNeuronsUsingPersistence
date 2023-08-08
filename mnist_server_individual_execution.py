import json
import sys
import os

gpu_number = int(sys.argv[1])
# Make visible only the GPUs specified in the argument

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)
from model.mnist_experiments import perform_experiment


def execute_task(json_template_task, topo_weight_task):
    with open(json_template_task) as json_specs:
        base_experiment_specs = json.load(json_specs)
        base_root_filepath_to_save_experiments = base_experiment_specs['root_filepath_to_save_experiments']
        experiment_specs = base_experiment_specs.copy()

        experiment_specs['topo_weight'] = topo_weight_task
        iterations_to_plot = int(experiment_specs['iterations_to_plot'])
        experiment_specs['root_filepath_to_save_experiments'] = f'{base_root_filepath_to_save_experiments}/topo_weight_{topo_weight_task}'
        perform_experiment(experiment_specification=experiment_specs, iterations_to_plot=iterations_to_plot,
                           verbose=True)


if __name__ == '__main__':
    # First argument: gpu_number
    # Second argument: json_tempalte
    # Third argument: topo_weight
    try:
        json_template = sys.argv[2]
        topo_weight = float(sys.argv[3])
    except Exception:
        print("The arguments must be specified correctly. See the file script for more information")
        sys.exit()
    execute_task(json_template, topo_weight)
