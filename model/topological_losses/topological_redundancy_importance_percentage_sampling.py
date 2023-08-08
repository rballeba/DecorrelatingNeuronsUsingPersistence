import tensorflow as tf

import model.distances.Pearson_distance as PearsonDistance
from model.differentiable_homology.ppdd_continuation import \
    generate_differentiable_zeroth_persistence_diagrams_deaths_from_distance_matrix
from model.matrices import has_unique_entries

from model.neuron_samplings.importance_percentage_neuron_sampling import \
    get_neurons_point_cloud_importance_percentage_sampling


def topological_redundancy_importance_percentage_sampling(model, dataset, labels,
                                                          number_of_points_in_dgm=None,
                                                          number_of_points_from_ppdd_to_reduce=1,
                                                          sampling_percentage=0.005,
                                                          distance_strategy_fn=PearsonDistance.compute_distance_matrix,
                                                          check_validation_same_distance=False):
    activation_x_examples = get_neurons_point_cloud_importance_percentage_sampling(model, dataset,
                                                                                   sampling_percentage=sampling_percentage)
    if number_of_points_in_dgm is None:
        number_of_points_in_dgm = activation_x_examples.shape[0] - number_of_points_from_ppdd_to_reduce
    distance_matrix = distance_strategy_fn(activation_x_examples, activation_x_examples.shape[0],
                                           activation_x_examples.shape[1])
    if check_validation_same_distance:
        if has_unique_entries(distance_matrix):
            return tf.constant(0.0)
    deaths_dgm_from_indices = generate_differentiable_zeroth_persistence_diagrams_deaths_from_distance_matrix(
        distance_matrix, number_of_points_in_dgm)
    # Minimizing - sum(deaths) is the same as minimizing sum(1 - deaths)
    return - tf.math.reduce_sum(deaths_dgm_from_indices)
