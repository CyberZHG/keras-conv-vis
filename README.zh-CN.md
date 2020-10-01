# Keras Convolution Visualization

[![Travis](https://travis-ci.com/CyberZHG/keras-conv-vis.svg?branch=master)](https://travis-ci.org/CyberZHG/keras-conv-vis)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-conv-vis/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-conv-vis)
![](https://img.shields.io/badge/keras-tensorflow-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras-blue.svg)

\[[中文](https://github.com/CyberZHG/keras-conv-vis/blob/master/README.zh-CN.md)|[English](https://github.com/CyberZHG/keras-conv-vis/blob/master/README.md)\]

## 安装

```bash
pip install git+https://github.com/cyberzhg/keras-conv-vis
```

只在启用eager execution的情况下可以使用。

## Guided Backpropagation

参考[论文](https://arxiv.org/pdf/1412.6806.pdf)和[样例](./demo/guided_backpropagation.py)。

```python
import keras
import numpy as np
from PIL import Image

from keras_conv_vis import replace_relu, get_gradient, Categorical

model = keras.applications.MobileNetV2()
# 将模型中所有的ReLU替换为所需的特殊反向传播
model = replace_relu(model, relu_type='guided')
gradient_model = keras.models.Sequential()
gradient_model.add(model)
# 只让特定的类别传递梯度
gradient_model.add(Categorical(284))  # ImageNet第284类是暹罗猫
# 获取输入的梯度
gradients = get_gradient(gradient_model, inputs)

# 将得到梯度归一化和可视化
gradient = gradients.numpy()[0]
gradient = (gradient - np.min(gradient)) / (np.max(gradient) - np.min(gradient) + 1e-4)
gradient = (gradient * 255.0).astype(np.uint8)
visualization = Image.fromarray(gradient)
```

| 类别 | 可视化 |
|:-:|:-:|
| 输入 | <img src="https://github.com/CyberZHG/keras-conv-vis/raw/master/samples/cat.jpg" width="224" height="224" /> |
| 梯度 | <img src="https://github.com/CyberZHG/keras-conv-vis/raw/master/samples/cat_gradient_relevant.jpg" width="224" height="224" /> |
| Deconvnet without Pooling Switches | <img src="https://github.com/CyberZHG/keras-conv-vis/raw/master/samples/cat_deconvnet_relevant.jpg" width="224" height="224" /> |
| Guided Backpropagation | <img src="https://github.com/CyberZHG/keras-conv-vis/raw/master/samples/cat_guided_relevant.jpg" width="224" height="224" /> |


## Grad-CAM


参考[论文](https://arxiv.org/abs/1610.02391)和[样例](https://github.com/CyberZHG/keras-conv-vis/blob/master/demo/grad_cam.py)。

```python
import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from keras_conv_vis import split_model_by_layer, get_gradient, Categorical

model = keras.applications.MobileNetV2()
# 将模型从最后一个卷积处分开，计算出中间结果
head, tail = split_model_by_layer(model, 'Conv_1')
last_conv_output = head(inputs)
# 给最后一个卷积计算梯度
gradient_model = keras.models.Sequential()
gradient_model.add(tail)
gradient_model.add(Categorical(284))  # ImageNet第284类是暹罗猫
gradients = get_gradient(gradient_model, last_conv_output)

# 计算Grad-CAM
gradient = gradients.numpy()[0]
gradient = np.mean(gradient, axis=(0, 1))  # 根据梯度计算每一层输出的权重
grad_cam = np.mean(last_conv_output.numpy()[0] * gradient, axis=-1)  # 对卷积输出进行加权求和
grad_cam = grad_cam * (grad_cam > 0).astype(grad_cam.dtype)  # Grad-CAM输出需要经过ReLU

# 可视化
grad_cam = (grad_cam - np.min(grad_cam)) / (np.max(grad_cam) - np.min(grad_cam) + 1e-4)
heatmap = plt.get_cmap('jet')(grad_cam, bytes=True)
heatmap = Image.fromarray(heatmap[..., :3], mode='RGB')
heatmap = heatmap.resize((original_image.width, original_image.height), resample=Image.BILINEAR)
visualization = Image.blend(original_image, heatmap, alpha=0.5)
```

| Input | Relevant CAM | Irrelevant CAM|
|:-:|:-:|:-:|
| <img src="https://github.com/CyberZHG/keras-conv-vis/raw/master/samples/cat.jpg" width="224" height="224" /> | <img src="https://github.com/CyberZHG/keras-conv-vis/raw/master/samples/cat_grad-cam_relevant.jpg" width="224" height="224" /> | <img src="https://github.com/CyberZHG/keras-conv-vis/raw/master/samples/cat_grad-cam_irrelevant.jpg" width="224" height="224" /> |
