# Keras Convolution Visualization

[![Travis](https://travis-ci.com/CyberZHG/keras-conv-vis.svg?branch=master)](https://travis-ci.org/CyberZHG/keras-conv-vis)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-conv-vis/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-conv-vis)
![](https://img.shields.io/badge/keras-tensorflow-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras-blue.svg)

\[[中文](https://github.com/CyberZHG/keras-conv-vis/blob/master/README.zh-CN.md)|[English](https://github.com/CyberZHG/keras-conv-vis/blob/master/README.md)\]

## Install

```bash
pip install git+https://github.com/cyberzhg/keras-conv-vis
```

## Guided Backpropagation

See [the paper](https://arxiv.org/pdf/1412.6806.pdf) and [demo](./demo/guided_backpropagation.py).

```python
import keras
import numpy as np
from PIL import Image

from keras_conv_vis import replace_relu, get_gradient, Categorical

model = keras.applications.MobileNetV2()
# Replace all the ReLUs with guided backpropagation
model = replace_relu(model, relu_type='guided')
gradient_model = keras.models.Sequential()
gradient_model.add(model)
# Activate only the target class
gradient_model.add(Categorical(284))  # 284 is the siamese cat in ImageNet
# Get the gradient
gradients = get_gradient(gradient_model, inputs)

# Normalize gradient and convert it to image
gradient = gradients.numpy()[0]
gradient = (gradient - np.min(gradient)) / (np.max(gradient) - np.min(gradient) + 1e-4)
gradient = (gradient * 255.0).astype(np.uint8)
visualization = Image.fromarray(gradient)
```

| Type | Visualization |
|:-:|:-:|
| Input | <img src="https://github.com/CyberZHG/keras-conv-vis/raw/master/samples/cat.jpg" width="224" height="224" /> |
| Gradient | <img src="https://github.com/CyberZHG/keras-conv-vis/raw/master/samples/cat_gradient_relevant.jpg" width="224" height="224" /> |
| Deconvnet without Pooling Switches | <img src="https://github.com/CyberZHG/keras-conv-vis/raw/master/samples/cat_deconvnet_relevant.jpg" width="224" height="224" /> |
| Guided Backpropagation | <img src="https://github.com/CyberZHG/keras-conv-vis/raw/master/samples/cat_guided_relevant.jpg" width="224" height="224" /> |
