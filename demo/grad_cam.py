import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from keras_conv_vis import grad_cam
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
    model = keras.applications.MobileNetV2()
    cam = grad_cam(model=model, layer_cut='Conv_1', inputs=inputs, target_class=target_class)[0]

    # Visualization
    heatmap = plt.get_cmap(cmap)(cam, bytes=True)
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
