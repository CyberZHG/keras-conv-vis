from unittest import TestCase

from tensorflow import keras

from keras_conv_vis import replace_layers, replace_relu


class TestModelReplace(TestCase):

    def test_replace_nothing(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'))
        model.add(keras.layers.ReLU())
        model.build((None, 14, 14, 3))
        model.compile(optimizer='sgd', loss='mse')
        replace_layers(model)

    def test_replace_relu(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'))
        model.add(keras.layers.ReLU())
        model.build((None, 14, 14, 3))
        model.compile(optimizer='sgd', loss='mse')
        replace_relu(model)
