import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from keras_conv_vis import split_model_by_layer, get_gradient, Categorical
from keras_conv_vis.backend import keras

CLASS_CAT = 284
CLASS_GUITAR = 546

# Load an image
current_path = os.path.dirname(os.path.realpath(__file__))
sample_path = os.path.join(current_path, '..', 'samples')
image_path = os.path.join(sample_path, 'cat.jpg')
original_image = Image.open(image_path)
image = original_image.resize((224, 224))
inputs = np.expand_dims(np.array(image).astype(np.float) / 255.0, axis=0)
inputs = inputs * 2.0 - 1.0


def process(target_class,
            cmap='jet',
            alpha=0.5):
    # Build model and get gradients
    model = keras.applications.MobileNetV2()
    head, tail = split_model_by_layer(model, 'Conv_1')
    last_conv_output = head(inputs)
    gradient_model = keras.models.Sequential()
    gradient_model.add(tail)
    gradient_model.add(Categorical(target_class))
    gradients = get_gradient(gradient_model, last_conv_output)

    # Calculate Grad-CAM
    gradient = gradients.numpy()[0]
    gradient = np.mean(gradient, axis=(0, 1))
    grad_cam = np.mean(last_conv_output.numpy()[0] * gradient, axis=-1)
    grad_cam = grad_cam * (grad_cam > 0).astype(grad_cam.dtype)

    # Visualization
    grad_cam = (grad_cam - np.min(grad_cam)) / (np.max(grad_cam) - np.min(grad_cam) + 1e-4)
    heatmap = plt.get_cmap(cmap)(grad_cam, bytes=True)
    heatmap = Image.fromarray(heatmap[..., :3], mode='RGB')
    heatmap = heatmap.resize((original_image.width, original_image.height), resample=Image.BILINEAR)
    return Image.blend(original_image, heatmap, alpha=alpha)


for target_class in [CLASS_CAT, CLASS_GUITAR]:
    visualization = process(target_class)
    cat_name = 'relevant'
    if target_class != CLASS_CAT:
        cat_name = 'irrelevant'
    save_name = f'cat_grad-cam_{cat_name}.jpg'
    visualization.save(os.path.join(sample_path, save_name))
