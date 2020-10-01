# Keras Convolution Visualization

[![Travis](https://travis-ci.org/CyberZHG/keras-conv-vis.svg)](https://travis-ci.org/CyberZHG/keras-conv-vis)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-conv-vis/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-conv-vis)
![License](https://img.shields.io/pypi/l/keras-conv-vis.svg)

![](https://img.shields.io/badge/keras-tensorflow-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras-blue.svg)

\[[中文](https://github.com/CyberZHG/keras-conv-vis/blob/master/README.zh-CN.md)|[English](https://github.com/CyberZHG/keras-conv-vis/blob/master/README.md)\]

## 安装

```bash
pip install git+https://github.com/cyberzhg/keras-conv-vis
```

## 使用

See https://arxiv.org/pdf/1412.6806.pdf and [demo](./demo/guided_backpropagation.py).

| Input | Gradient | Deconvnet without Pooling Switches | Guided Backpropagation |
|:-:|:-:|:-:|:-:|
|![](./samples/cat.jpg)|![](./samples/cat_gradient_relevant.jpg)|![](./samples/cat_deconvnet_relevant.jpg)|![](./samples/cat_guided_relevant.jpg)|
