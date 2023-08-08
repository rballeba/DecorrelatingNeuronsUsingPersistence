import json
import os
import sys

from multiprocessing import Pool, Queue

# Queue for taking GPUs
queue = Queue()

experiment_templates_to_execute = [
    # No topological regularisation
    {'json_template': 'experiments/PGDL/specifications/public_data_template_task1_std_avg.json',
     'possible_topo_weights': [0.0]},
    # Topological regularisation
    {'json_template': 'experiments/PGDL/specifications/public_data_template_task1_std_avg.json',
     'possible_topo_weights': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 100]},
    {'json_template': 'experiments/PGDL/specifications/public_data_template_task1_imp.json',
     'possible_topo_weights': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 100]},
    {'json_template': 'experiments/PGDL/specifications/public_data_template_task1_l1_reg.json',
     'possible_topo_weights': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 100]},
    {'json_template': 'experiments/PGDL/specifications/public_data_template_task1_l2_reg.json',
     'possible_topo_weights': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 100]},
    {'json_template': 'experiments/PGDL/specifications/public_data_template_task1_min_corr.json',
     'possible_topo_weights': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 100]},
]


def get_experiment_tasks_from_experiment_templates(template):
    tasks_for_template = []
    json_template = template['json_template']
    possible_topo_weights = template['possible_topo_weights']
    for topo_weight in possible_topo_weights:
        tasks_for_template.append((json_template, topo_weight))
    return tasks_for_template


def get_all_tasks():
    template_tasks = []
    # Experiments with variable topo weights values. To add new multiple experiments of this type add
    # values to the list of dictionaries experiment_templates_to_execute
    for template in experiment_templates_to_execute:
        template_tasks.extend(get_experiment_tasks_from_experiment_templates(template))
    return template_tasks


def worker_execution(task):
    gpu = queue.get()
    try:
        # Execute PGDL_server_individual_execution.py in a new process
        os.system(f'python3 PGDL_server_individual_execution.py {gpu} {task[0]} {task[1]}')
    except Exception as e:
        print(e)
    finally:
        queue.put(gpu)


# Original source
# https://stackoverflow.com/questions/53422761/distributing-jobs-evenly-across-multiple-gpus-with-multiprocessing-pool
def orchestrator_execution(number_of_gpus):
    template_tasks = get_all_tasks()
    # Get the number of GPUs available
    gpus = list(range(number_of_gpus))
    # Create a pool of processes. We use as many processes as there are GPUs
    pool = Pool(len(gpus))
    # Put the GPUs in the queue
    for gpu in gpus:
        queue.put(gpu)
    # Run the processes
    for _ in pool.imap_unordered(worker_execution, template_tasks):
        pass
    pool.close()
    pool.join()


if __name__ == '__main__':
    # Get the first argument from console
    if len(sys.argv) == 2:
        try:
            number_of_gpus = int(sys.argv[1])
            print("The number of GPUs is: ", number_of_gpus)
        except Exception:
            print("The number of GPUs must be specified")
            sys.exit()
    else:
        print("The number of GPUs must be specified")
        sys.exit()  # exit the program
    orchestrator_execution(number_of_gpus)
