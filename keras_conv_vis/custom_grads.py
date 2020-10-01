import tensorflow as tf

from .backend import keras

__all__ = [
    'CustomReLU',
    'backward_deconvnet_relu', 'guided_backpropagation_relu',
]


@tf.custom_gradient
def backward_deconvnet_relu(x):
    """The ReLU whose negative gradients will be rectified.

    @see https://arxiv.org/pdf/1312.6034.pdf
    """
    def grad(dy):
        return tf.nn.relu(dy)
    return tf.nn.relu(x), grad


@tf.custom_gradient
def guided_backpropagation_relu(x):
    """The ReLU whose gradients will be rectified by both the inputs and the gradients.

    @see https://arxiv.org/pdf/1412.6806.pdf
    """
    def grad(dy):
        return tf.nn.relu(dy) * tf.cast(x > 0.0, x.dtype)
    return tf.nn.relu(x), grad


class CustomReLU(keras.layers.Layer):

    custom_objects = {
        'deconvnet': backward_deconvnet_relu,
        'guided': guided_backpropagation_relu,
    }

    def __init__(self, relu_type, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        if relu_type not in self.custom_objects:
            raise NotImplementedError(f'Unknown ReLU type {relu_type}. '
                                      f'The choices are {set(self.custom_objects.keys())}.')
        self.relu_type = relu_type
        self.relu = self.custom_objects[relu_type]

    def call(self, inputs):
        return self.relu(inputs)

    def get_config(self):
        config = {
            'relu_type': self.relu_type,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
