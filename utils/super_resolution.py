import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


# Load pre-trained Super-Resolution model from TensorFlow Hub
sr_model = hub.load("https://www.kaggle.com/models/kaggle/esrgan-tf2/TensorFlow2/esrgan-tf2/1")

def super_resolution(image:tf.Tensor, model=sr_model) -> tf.Tensor:
    # scale to [0, 255] if needed
    if image[0, :, :, 0].numpy().max() <= 10:
        image = tf.clip_by_value(image, 0, 1) * 255
    return model(image)