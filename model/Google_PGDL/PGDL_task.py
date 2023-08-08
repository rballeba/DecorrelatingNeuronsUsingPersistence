import glob
import os
from typing import List, Tuple
import tensorflow as tf

from model.Google_PGDL.PGDL_model import PGDLModel
from model.filesystem import load_dataset


class PGDLTask:
    def __init__(self, directory: str):
        self.directory = directory
        self.models = self._generate_task_models()

    def dataset_folder(self) -> str:
        return f'{self.directory}/dataset_1'

    def get_original_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        train_dataset = self._load_dataset('train')
        test_dataset = self._load_dataset('test')
        return train_dataset, test_dataset

    def get_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        train_dataset = self._load_dataset('new_train')
        validation_dataset = self._load_dataset('new_validation')
        test_dataset = self._load_dataset('new_test')
        return train_dataset, validation_dataset, test_dataset

    def get_PGDL_models(self) -> List[PGDLModel]:
        return self.models

    def get_task_name(self):
        return self.directory.split('/')[-1]

    def _generate_task_models(self) -> List[PGDLModel]:
        model_subfolders = filter(lambda subdir: subdir[:5] == 'model', os.listdir(self.directory))
        return list(map(lambda model_subfolder: PGDLModel(self.directory, model_subfolder), model_subfolders))

    def _load_dataset(self, type: str) -> tf.data.Dataset:
        """
        :param type: 'train',  'test', 'new_train', 'new_validation' or 'new_test' depending in
         which dataset you want to retrieve.
        """
        absolute_dataset_path = os.path.abspath(self.dataset_folder())
        if type in ['train', 'test']:
            path_to_shards = glob.glob(os.path.join(f'{absolute_dataset_path}/{type}', 'shard_*.tfrecord'))
            dataset = tf.data.TFRecordDataset(path_to_shards)
            return dataset.map(_deserialize_example)
        elif type in ['new_train', 'new_validation', 'new_test']:
            return load_dataset(f'{absolute_dataset_path}/{type}')
        else:
            raise ValueError('There does not exist a dataset of the type specified.'
                             ' The available types are train and test')


def _deserialize_example(serialized_example):
    record = tf.io.parse_single_example(
        serialized_example,
        features={
            'inputs': tf.io.FixedLenFeature([], tf.string),
            'output': tf.io.FixedLenFeature([], tf.string)
        })
    inputs = tf.io.parse_tensor(record['inputs'], out_type=tf.float32)
    output = tf.io.parse_tensor(record['output'], out_type=tf.int32)
    return inputs, output
