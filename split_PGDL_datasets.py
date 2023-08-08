import math
import sys
import os

from model.Google_PGDL.PGDL_task import PGDLTask
from model.filesystem import persist_dataset


def generate_task_1_dataset(original_dataset_folder, task1_folder='task1_v4', examples_per_label_validation=1000):
    examples_per_label_training = 5000
    task_folder = f'{original_dataset_folder}/{task1_folder}'
    if os.path.exists(task_folder):
        task = PGDLTask(task_folder)
        train_dataset, test_dataset = task.get_original_datasets()
        new_train_dataset = None
        new_validation_dataset = None
        for label in range(10):
            examples_for_label = train_dataset.filter(lambda x, y: y == label)
            examples_for_label = examples_for_label.shuffle(buffer_size=examples_per_label_training)
            examples_for_label_validation = examples_for_label.take(examples_per_label_validation)
            examples_for_label_train = examples_for_label.skip(examples_per_label_validation)
            if new_train_dataset == None:
                new_train_dataset = examples_for_label_train
                new_validation_dataset = examples_for_label_validation
            else:
                new_train_dataset = new_train_dataset.concatenate(examples_for_label_train)
                new_validation_dataset = new_validation_dataset.concatenate(examples_for_label_validation)
        # Shuffle and save the datasets
        new_train_dataset = new_train_dataset.shuffle(10 * (examples_per_label_training - examples_per_label_validation))
        new_validation_dataset = new_validation_dataset.shuffle(10 * examples_per_label_validation)
        persist_dataset(new_train_dataset, f'{original_dataset_folder}/{task1_folder}/dataset_1/new_train')
        persist_dataset(new_validation_dataset, f'{original_dataset_folder}/{task1_folder}/dataset_1/new_validation')
        persist_dataset(test_dataset, f'{original_dataset_folder}/{task1_folder}/dataset_1/new_test')


def main():
    if len(sys.argv) != 2:
        print(
            'The path to the PGDL public phase root folder must be passed as console parameter (folder that contains '
            'the input_data folder)')
        exit(1)
    if not os.path.exists(sys.argv[1]):
        print('The PGDL folder does not exist')
        exit(2)
    original_PGDL_folder = f'{sys.argv[1]}/input_data'
    generate_task_1_dataset(original_PGDL_folder)


if __name__ == "__main__":
    main()
