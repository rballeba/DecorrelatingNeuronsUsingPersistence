from model.experiment_explorer import ExperimentExplorer


def compute_general_accuracies(experiments_folder, dataset_folder, datasets=('train', 'validation', 'test'),
                               model_types=('trained', 'initial'), type='pgdl'):
    if type not in ('pgdl', 'tinyimagenet', 'mnist'):
        raise ValueError(f'Invalid type: {type}')
    accuracies = {}
    experiment = ExperimentExplorer(experiments_folder, dataset_folder, type=type)
    network_experiments_ordered_names = list(experiment.network_experiments.keys())
    network_experiments_ordered_names.sort()
    network_experiments_ordered_names = [str(name) for name in network_experiments_ordered_names]
    for model_type in model_types:
        accuracies[model_type] = dict()
        for dataset in datasets:
            accuracies_for_dataset_and_type = []
            for network_name in network_experiments_ordered_names:
                # This only computes the accuracy of the trained model and saves it to use it later
                accuracies_for_dataset_and_type.append(experiment.network_experiments[network_name]
                                                       .get_accuracy(type=model_type, dataset=dataset))
            accuracies[model_type][dataset] = accuracies_for_dataset_and_type[:]
    return accuracies
