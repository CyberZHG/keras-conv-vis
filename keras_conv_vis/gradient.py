from typing import Optional, List, Union

import tensorflow as tf
import numpy as np

from .backend import keras

__all__ = ['Categorical', 'get_gradient']


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


def get_gradient(model: keras.models.Model,
                 inputs: Union[np.ndarray, tf.Tensor, List[Union[np.ndarray, tf.Tensor]]],
                 targets: Optional[Union[tf.Tensor, List[tf.Tensor]]] = None):
    if not isinstance(inputs, list):
        inputs = [inputs]
    for i, input_item in enumerate(inputs):
        if isinstance(input_item, np.ndarray):
            inputs[i] = tf.convert_to_tensor(input_item)
    if targets is None:
        target_tensors = inputs
    else:
        target_tensors = []
        if not isinstance(targets, list):
            targets = [targets]
        for target in targets:
            target_tensors.append(target)
    with tf.GradientTape() as tape:
        for target in target_tensors:
            tape.watch(target)
        model_output = model(inputs)
        gradients = tape.gradient(model_output, target_tensors)
    if len(gradients) == 1:
        gradients = gradients[0]
    return gradients
