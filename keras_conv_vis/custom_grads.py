import tensorflow as tf

__all__ = ['backward_deconvnet_relu', 'guided_backpropagation_relu']


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
