from typing import Optional, List, Union

import tensorflow as tf
import numpy as np

from .backend import keras

__all__ = ['Categorical', 'get_gradient', 'split_model_by_layer', 'grad_cam']


class Categorical(keras.layers.Layer):
    """Select one target category."""

    def __init__(self, class_index, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.class_index = class_index

    def call(self, inputs):
        return inputs[..., self.class_index]

    def get_config(self):
        config = {
            'class_index': self.class_index,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def get_gradient(model: Union[keras.models.Model, List[keras.models.Model]],
                 inputs: Union[np.ndarray, tf.Tensor, List[Union[np.ndarray, tf.Tensor]]],
                 targets: Optional[Union[tf.Tensor, List[tf.Tensor]]] = None):
    """Get the gradient of input, weights, of intermediate outputs.

    :param model: The keras model.
    :param inputs: The batched input data.
    :param targets: The default is the input tensor.
    :return: The gradients of the targets.
    """
    models = model
    if not isinstance(model, list):
        models = [model]
    if not isinstance(inputs, list):
        inputs = [inputs]
    for i, input_item in enumerate(inputs):
        if isinstance(input_item, np.ndarray):
            inputs[i] = tf.convert_to_tensor(input_item)
    if len(inputs) == 1:
        inputs = inputs[0]
    input_targets = targets
    if targets is None:
        targets = []
    if not isinstance(targets, list):
        targets = [targets]
    model_output = inputs
    with tf.GradientTape() as tape:
        for target in targets:
            tape.watch(target)
        for i, model in enumerate(models):
            if input_targets is None:
                targets.append(model_output)
                tape.watch(model_output)
            model_output = model(model_output)
        gradients = tape.gradient(model_output, targets)
    if len(gradients) == 1:
        gradients = gradients[0]
    return gradients


def split_model_by_layer(model: keras.models.Model,
                         layer_cut: Union[str, keras.layers.Layer]):
    """Split a model into two parts.

    :param model: The keras model.
    :param layer_cut: The layer whose output will be cut. The layer must be a cut point,
                      a.k.a., the output edge is a bridge.
    :return: The two models.
    """
    if isinstance(layer_cut, str):
        layer_cut = model.get_layer(layer_cut)
    head = keras.models.Model(model.inputs, layer_cut.output)
    meet = False
    mappings, depends = {}, set()
    for layer in model.layers:
        if layer_cut is layer:
            meet = True
            mappings[layer.name] = keras.layers.Input(layer.output.shape[1:])
            depends.add(layer.name)
            continue
        if not meet:
            continue
        inputs = layer.input
        if not isinstance(inputs, list):
            inputs = [inputs]
        new_inputs = []
        for input_tensor in inputs:
            name = input_tensor.name.rsplit('/')[0]
            depends.add(name)
            new_inputs.append(mappings[name])
        if not isinstance(layer.input, list):
            new_inputs = new_inputs[0]
        mappings[layer.name] = layer(new_inputs)
    outputs = []
    for layer in model.layers:
        if layer.name in mappings and layer.name not in depends:
            outputs.append(mappings[layer.name])
    tail = keras.models.Model(mappings[layer_cut.name], outputs)
    return head, tail


def grad_cam(model: keras.models.Model,
             layer_cut: Union[str, keras.layers.Layer],
             inputs: Union[np.ndarray, tf.Tensor, List[Union[np.ndarray, tf.Tensor]]],
             target_class: int,
             epsilon: float = 1e-4):
    """Get the class activation map (CAM) based on the gradient.

    @see https://arxiv.org/abs/1610.02391


    :param model: The keras model.
    :param layer_cut: The layer whose output will be cut. The layer must be a cut point,
                      a.k.a., the output edge is a bridge.
    :param inputs: The batched input data.
    :param target_class: The index of the target class.
    :param epsilon: The small number used for the stability of normalization.
    :return:
    """
    # Split model by the last convolutional layer
    head, tail = split_model_by_layer(model, layer_cut)
    last_conv_output = head(inputs)
    gradient_model = keras.models.Sequential()
    gradient_model.add(tail)
    gradient_model.add(Categorical(target_class))
    gradients = get_gradient(gradient_model, last_conv_output)

    # Calculate Grad-CAM
    gradient = gradients.numpy()
    gradient = np.expand_dims(np.mean(gradient, axis=(1, 2)), axis=(1, 2))
    grad_cam = np.mean(last_conv_output.numpy() * gradient, axis=-1)
    grad_cam = grad_cam * (grad_cam > 0).astype(grad_cam.dtype)

    # Normalization
    grad_cam = (grad_cam - np.min(grad_cam)) / (np.max(grad_cam) - np.min(grad_cam) + epsilon)
    return grad_cam
