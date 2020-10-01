from copy import deepcopy

from .backend import keras
from .custom_grads import CustomReLULayer

__all__ = ['replace_layers', 'replace_relu']


def replace_layers(model: keras.models.Model,
                   layer_mapping: dict,
                   activation_mapping: dict,
                   custom_objects: dict = None,
                   prefix: str = 'mapped_'):
    """Replace all the matched layers in the original model and return a new one.
    The model cannot contain unserializable layers (e.g. Lambda).

    :param model: The original model.
    :param layer_mapping: Configuration mapping rules for layers.
    :param activation_mapping: Configuration mapping rules for activations.
    :param custom_objects: Custom objects for loading the model.
    :param prefix: The prefix added to the names of the new model.
    :return: The mapped new model.
    """
    config = model.get_config()

    def _replace_item(_config, mapping):
        if callable(mapping):
            return mapping(_config)
        return mapping

    def _replace(_config):
        if isinstance(_config, dict):
            if 'class_name' in _config:
                class_name = _config['class_name']
                if class_name in layer_mapping:
                    _config = _replace_item(_config, layer_mapping[class_name])
            if 'activation' in _config:
                act_name = _config['activation']
                if act_name in activation_mapping:
                    _config['activation'] = _replace_item(act_name, activation_mapping[act_name])
            if 'name' in _config:
                _config['name'] = prefix + _config['name']
            for key, val in _config.items():
                _config[key] = _replace(val)
        elif isinstance(_config, list):
            _config = [_replace(item) for item in _config]
        return _config

    new_config = _replace(config)
    new_model = model.__class__.from_config(new_config, custom_objects=custom_objects)
    for layer in model.layers:
        new_model.get_layer(prefix + layer.name).set_weights(layer.get_weights())

    return new_model


def replace_relu(model: keras.models.Model,
                 relu_type: str = 'guided',
                 custom_objects: dict = None,
                 prefix: str = 'mapped_'):
    """Replace ReLUs with custom ReLU function.

    :param model: The original model.
    :param relu_type:
    :param custom_objects: Custom objects for loading the model.
    :param prefix: The prefix added to the names of the new model.
    :return:
    """
    if custom_objects is None:
        custom_objects = {}

    custom_relu = CustomReLULayer(relu_type=relu_type)
    custom_relu_name = custom_relu.relu.__name__
    relu_config = {'class_name': CustomReLULayer.__name__, 'config': custom_relu.get_config()}
    custom_objects[CustomReLULayer.__name__] = CustomReLULayer
    custom_objects[custom_relu_name] = custom_relu.relu

    def _replace_relu_layer(layer_config):
        new_config = deepcopy(relu_config)
        new_config['config']['name'] = layer_config['config']['name']
        return new_config

    layer_mapping = {'ReLU': _replace_relu_layer}
    activation_mapping = {'relu': custom_relu_name}

    return replace_layers(model,
                          layer_mapping=layer_mapping,
                          activation_mapping=activation_mapping,
                          custom_objects=custom_objects,
                          prefix=prefix)
