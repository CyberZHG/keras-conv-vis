from unittest import TestCase

from keras_conv_vis import replace_relu
from keras_conv_vis.backend import keras


class TestModelReplace(TestCase):

    def test_replace_relu(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'))
        model.add(keras.layers.ReLU())
        model.build((None, 14, 14, 3))
        model.compile(optimizer='sgd', loss='mse')

        new_model = replace_relu(model)
        new_model.summary()
