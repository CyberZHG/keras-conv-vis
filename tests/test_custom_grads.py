from unittest import TestCase

import numpy as np
import tensorflow as tf

from keras_conv_vis import backward_deconvnet_relu, guided_backpropagation_relu
from keras_conv_vis.backend import keras
from keras_conv_vis.backend import backend as K


class TestCustomGrads(TestCase):

    def alternating_toy_loss(self, length):
        def loss(y_pred):
            return y_pred * K.constant([1 if i % 2 == 0 else -1 for i in range(length)], dtype=y_pred.dtype)
        return loss

    def test_deconvnet_relu(self):
        input_length = 4

        model = keras.models.Sequential()
        custom_relu_layer = keras.layers.Lambda(lambda x: backward_deconvnet_relu(x))
        model.add(custom_relu_layer)
        model.add(keras.layers.Lambda(lambda x: self.alternating_toy_loss(input_length)(x)))
        model.build((None, input_length,))
        model.summary()
        model.compile(optimizer='sgd', loss='mse')

        sample_input = np.array([[-12.1, 4.4, 0.4, 1.4]])
        with tf.GradientTape() as tape:
            sample_input = tf.convert_to_tensor(sample_input)
            tape.watch(sample_input)
            sample_model_output = model(sample_input)
            self.assertTrue(np.allclose(np.array([[0.0, -4.4, 0.4, -1.4]]), sample_model_output.numpy()))
            sample_gradient = tape.gradient(sample_model_output, sample_input)
            self.assertTrue(np.allclose(np.array([[1.0, 0.0, 1.0, 0.0]]), sample_gradient.numpy()))

    def test_guided_backpropagation_relu(self):
        input_length = 4

        model = keras.models.Sequential()
        custom_relu_layer = keras.layers.Lambda(lambda x: guided_backpropagation_relu(x))
        model.add(custom_relu_layer)
        model.add(keras.layers.Lambda(lambda x: self.alternating_toy_loss(input_length)(x)))
        model.build((None, input_length,))
        model.compile(optimizer='sgd', loss='mse')
        model.summary()

        sample_input = np.array([[-12.1, 4.4, 0.4, 1.4]])
        with tf.GradientTape() as tape:
            sample_input = tf.convert_to_tensor(sample_input)
            tape.watch(sample_input)
            sample_model_output = model(sample_input)
            self.assertTrue(np.allclose(np.array([[0.0, -4.4, 0.4, -1.4]]), sample_model_output.numpy()))
            sample_gradient = tape.gradient(sample_model_output, sample_input)
            self.assertTrue(np.allclose(np.array([[0.0, 0.0, 1.0, 0.0]]), sample_gradient.numpy()))
