

from keras_conv_vis.backend import keras

model = keras.applications.MobileNetV2()
print(model.get_config())
