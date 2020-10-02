from unittest import TestCase

import numpy as np

from keras_conv_vis import (get_gradient, Categorical, replace_layers,
                            split_model_by_layer, grad_cam)
from keras_conv_vis.backend import keras


class TestGetGradient(TestCase):

    def test_get_gradient(self):
        model = keras.applications.MobileNetV2()
        gradient_model = keras.models.Sequential()
        gradient_model.add(model)
        gradient_model.add(Categorical(7))
        gradient_model.get_config()
        get_gradient(gradient_model, np.random.random((1, 224, 224, 3)))
        get_gradient(gradient_model, np.random.random((1, 224, 224, 3)),
                     targets=model.get_layer('bn_Conv1').trainable_weights[0])

    def test_cut_model(self):
        model = keras.applications.MobileNetV2()
        head, tail = split_model_by_layer(model, 'block_5_add')
        gradient_model = keras.models.Sequential()
        gradient_model.add(tail)
        gradient_model.add(Categorical(7))
        gradients = get_gradient([head, gradient_model], np.random.random((1, 224, 224, 3)))
        self.assertEqual(2, len(gradients))

    def test_grad_cam(self):
        model = keras.applications.MobileNetV2()
        cam = grad_cam(model,
                       layer_cut='Conv_1',
                       inputs=np.random.random((3, 224, 224, 3)),
                       target_class=0)
        self.assertEqual((3, 7, 7), cam.shape)

    def test_grad_cam_pp(self):
        model = keras.applications.MobileNetV2()
        model = replace_layers(model, activation_mapping={'softmax': 'linear'})
        cam = grad_cam(model,
                       layer_cut='Conv_1',
                       inputs=np.random.random((3, 224, 224, 3)),
                       target_class=0,
                       plus=True)
        self.assertEqual((3, 7, 7), cam.shape)
