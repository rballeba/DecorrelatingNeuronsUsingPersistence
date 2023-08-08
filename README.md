# Decorrelating neurons using persistence 
This is the GitHub repository for the paper *Decorrelating neurons using persistence*. In this paper, we propose to regularise neural networks by minimising the highest correlations between neurons during training. To achieve this, we minimise the deaths of zero-dimensional persistence diagram computed from neuron activations and the symmetric function $d(x,y)=1-|\text{corr}(x,y)|$. We hypothesise that reducing only the highest correlations between neurons given by persistence diagrams leads to better generalisation of neural networks with respect to training without any regularisation or minimising all the pairwise correlations between neurons.

## Execution of the experiments
To run the experiments proposed in the paper, you need to first download the [public PGDL dataset](http://storage.googleapis.com/gresearch/pgdl/public_data.zip). We assume that you have downloaded the dataset and it is located in the folder ``<google_data>``. This folder must contain the subfolder ``public_data``, which contains the downloaded dataset. Additionally, you need to execute the script ``mnist_dataset_generator.py`` with the console parameter ``<mnist_dataset>``specifying the path where you want to generate the MNIST dataset . Finally, we assume that this code is located in the folder ``<code_folder>`` and that you want to generate the results of the experiments in the folder ``<experiments>``.

To run the experiments, you need to execute the different Python scripts using the Docker image built in the ``Dockerfile`` located in the ``docker`` folder. The following line executes the docker image using the GPUs 0 and 1 according to the script ``nvidia-smi`` and mounting the folders ``<google_data>``, ``<mnist_dataset>``, ``<code_folder>`` and ``<experiments>`` in the container:

```bash
docker run --rm --gpus '"device=0,1,2,3,4,5"' -it -v <google_data>:/home/google_data -v <mnist_dataset>:/home/mnist -v <code_folder>:/home/code  -v <experiments>:/home/completed_experiments <name_of_the_image_built>
```

From now on, we assume all the script are executed from the folder ``<code_folder>`` using the aforementioned Docker image.

### Experiments with the PGDL dataset

To execute the experiments with the PGDL dataset, you first need to generate the training, validation, and test datasets for the CIFAR-10 dataset. To do this, you need to execute the following script:

```bash
python3 split_PGDL_datasets.py /home/google_data
```

Now, to train all the PGDL networks, you need to execute the following script:

```bash
python3 PGDL_server_execution.py <number_of_gpus_used>
```
### Experiments with the MNIST dataset
In this case, we need to generate also the networks that will be trained with the MNIST dataset. To do this, you need to execute the following script:
```bash
python3 mnist_models_generator.py /home/mnist
```
Now, to train all the MNIST networks, you need to execute the following script:
```bash
python3 mnist_server_execution.py <number_of_gpus_used>
```

### Extracting statistics

Once the trainings have been performed, the next step is to compute the validation and test accuracies for the trained and non-trained networks. To do it, you need to execute the following scripts:
```bash
python3 accuracies/compute_accuracies_mnist.py
python3 accuracies/compute_accuracies_PGDL.py
```

The last step is to extract the statistics we computed in the paper. Before executing the script, you need to modify the file ``extract_statistics.py``. In particular, you need to modify the variables ``regularisers_PGDL`` and ``regularisers_PGDL`` and include only the weights for each regulariser for which the training converged. To do this, you need to check the folders where the trainings are saved. The folder structure for each training is:

```
<experiment_name>/public_data_experiments/<regularisation_name>/topo_weight_<associated_weight>/<optional_value_for_PGDL>/<network_name>
```
where ``<experiment_name>`` is either ``PGDL`` or ``mnist`` and ``<optional_value_for_PGDL>`` is only present for the PGDL experiments, with the value ``task1_v4``. The training would have converged if either, there is no folder ``it_<number>`` inside the folder experiment or if the folder experiment does not exist at all. By default, we included the experiments that converged in our experiment executions.  After these modifications, simply execute the modified script:
```bash
python3 extract_statistics.py combined
```
This will print by console all the statistics computed in the paper and will generate the critical difference diagram in the folder
``f"/home/completed_experiments/combined/critical_differences_diagram_alpha_{alpha}.tex"``, where ``{alpha}`` is the significance level of the experiments, that can be adjusted in the script ``extract_statistics.py``.

## Differentiability of persistence diagrams
The different proposed topological regularisers are implemented in the folder ``model/topological_losses``. In particular, $\mathcal T_1$ is the topological regulariser implemented in the file ``topological_redundancy_importance_percentage_sampling.py`` and $\mathcal T_2$ is the topological regulariser implemented in ``std_avg_deaths_importance_percentage_sampling.py``. Both functions use the method ``generate_differentiable_zeroth_persistence_diagrams_deaths_from_distance_matrix`` to generate differentiable persistence diagrams, implemented in the file ``model/differentiable_homology/ppdd_continuation.py``.

## Training procedures

The training procedures for the PGDL and MNIST datasets are implemented in the files ``model/experiments_group.py`` and ``model/mnist_experiments.py``, respectively. Both files are almost identical, but for MNIST, there are some minor changes to perform the training explained in the associated paper. The training specifications are included in the ``json`` files located at the folder ``experiments/<dataset_name>/specifications``, where ``<dataset_name>`` is ``MNIST`` or ``PGDL``.

The main training procedure for both types of experiments is implemented in the file ``model/train.py``. An example of how to use of this procedure can be found in the aforementioned files ``model/experiments_group.py`` and ``model/mnist_experiments.py``. By default, images of density functions of persistence diagram deaths and checkpoints are saved periodically during training in the experiment folder described above, which requires a large amount of memory.