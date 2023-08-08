import json
import os
import pickle
import shutil

import numpy as np
import tensorflow as tf


def create_directory_if_it_does_not_exist(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def save_np_array(array, filepath):
    with open(filepath, 'wb') as f:
        np.save(f, array)


def load_np_array(filepath):
    with open(filepath, 'rb') as f:
        loaded_array = np.load(f)
    return loaded_array


def remove_file_if_exists(filename):
    if os.path.exists(filename):
        os.remove(filename)
        return True
    return False


def remove_folder_if_exists(dirpath):
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)


def persist_dataset(data, filepath):
    remove_folder_if_exists(filepath)
    data.save(filepath)


def load_dataset(filepath):
    return tf.data.Dataset.load(filepath)


def save_object(object, filepath):
    with open(filepath, 'wb') as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_object(filepath):
    with open(filepath, 'rb') as handle:
        object = pickle.load(handle)
    return object


def load_json(filepath):
    with open(filepath, 'r') as json_file:
        json_object = json.load(json_file)
    return json_object
