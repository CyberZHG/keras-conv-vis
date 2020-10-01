import os

from PIL import Image
import numpy as np

from keras_conv_vis import replace_relu, get_gradient, Categorical
from keras_conv_vis.backend import keras

CLASS_CAT = 284
CLASS_GUITAR = 546

# Load an image
current_path = os.path.dirname(os.path.realpath(__file__))
sample_path = os.path.join(current_path, '..', 'samples')
image_path = os.path.join(sample_path, 'cat.jpg')
image = Image.open(image_path)
image = image.resize((224, 224))
inputs = np.expand_dims(np.array(image).astype(np.float) / 255.0, axis=0)
inputs = inputs * 2.0 - 1.0


def process(relu_type, target_class):
    # Build model and get gradients
    model = keras.applications.MobileNetV2()
    if relu_type is not None:
        model = replace_relu(model, relu_type=relu_type)
    gradient_model = keras.models.Sequential()
    gradient_model.add(model)
    gradient_model.add(Categorical(target_class))
    gradients = get_gradient(gradient_model, inputs)

    # Visualize gradients
    gradient = gradients.numpy()[0]
    gradient = (gradient - np.min(gradient)) / (np.max(gradient) - np.min(gradient) + 1e-4)
    gradient = (gradient * 255.0).astype(np.uint8)
    return Image.fromarray(gradient)


for target_class in [CLASS_CAT, CLASS_GUITAR]:
    for relu_type in [None, 'deconvnet', 'guided']:
        gradient = process(relu_type, target_class)
        cat_name = 'relevant'
        if target_class != CLASS_CAT:
            cat_name = 'irrelevant'
        if relu_type is None:
            relu_type = 'gradient'
        save_name = f'cat_{relu_type}_{cat_name}.jpg'
        gradient.save(os.path.join(sample_path, save_name))
