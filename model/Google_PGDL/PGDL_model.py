import json
import os

import tensorflow as tf
from keras import Sequential


class PGDLModel:
    def __init__(self, task_directory, model_subfolder: str):
        self.model_directory = model_subfolder
        self.model_directory = f'{task_directory}/{model_subfolder}'
        self.model_number = int(model_subfolder[6:])

    def get_model(self) -> tf.keras.Sequential:
        """
        It gets the model without the PGDL trained weights. Instead, it uses new parameter values by generating
        a new sequential model.
        :return:
        """
        absolute_model_path = os.path.abspath(self.model_directory)
        config_path = f'{absolute_model_path}/config.json'
        model_instance = _create_model_instance(config_path)
        initial_weights_path = os.path.join(absolute_model_path, 'weights_init.hdf5')
        model_instance.load_weights(initial_weights_path)
        return model_instance


def _create_model_instance(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return load_model(config)


def load_model(config):
    model_instance = _model_def_to_keras_sequential(config['model_config'])
    model_instance.build([0] + config['input_shape'])
    return model_instance


def _model_def_to_keras_sequential(model_def):
    """Convert a model json to a Keras Sequential model.
    Args:
        model_def: A list of dictionaries, where each dict describes a layer to add
            to the model.
    Returns:
        A Keras Sequential model with the required architecture.
    """

    def _cast_to_integer_if_possible(dct):
        dct = dict(dct)
        for k, v in dct.items():
            if isinstance(v, float) and v.is_integer():
                dct[k] = int(v)
        return dct

    def parse_layer(layer_def):
        layer_cls = getattr(tf.keras.layers, layer_def['layer_name'])
        # layer_cls = wrap_layer(layer_cls)
        kwargs = dict(layer_def)
        del kwargs['layer_name']
        return _wrap_layer(layer_cls, **_cast_to_integer_if_possible(kwargs))
        # return layer_cls(**_cast_to_integer_if_possible(kwargs))

    return Sequential([parse_layer(l) for l in model_def])


def _wrap_layer(layer_cls, *args, **kwargs):
    """Wraps a layer for computing the jacobian wrt to intermediate layers."""

    class wrapped_layer(layer_cls):
        def __call__(self, x, *args, **kwargs):
            self._last_seen_input = x
            return super(wrapped_layer, self).__call__(x, *args, **kwargs)

    return wrapped_layer(*args, **kwargs)
