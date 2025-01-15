import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


# Load pre-trained Super-Resolution model from TensorFlow Hub
sr_model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")

def super_resolution(image:tf.Tensor, model) -> tf.Tensor:
    # scale to [0, 255] if needed
    if image[0, :, :, 0].numpy().max() <= 1:
        image = tf.clip_by_value(image, 0, 1) * 255
    return model(image)