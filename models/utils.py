import os, sys
import numpy as np
import tensorflow as tf

project_root = os.path.dirname(os.getcwd())

# Add path to the project root for importing custom modules
if project_root not in sys.path:
    sys.path.append(project_root)

# Color recalibration functions (2 methods)
def match_mean_std(generated_image, content_image):
    """
    Adjust the mean and standard deviation of the generated image to match the content image.

    Args:
        generated_image (np.ndarray): Generated image as a NumPy array.
        content_image (np.ndarray): Content image as a NumPy array.

    Returns:
        np.ndarray: Color-matched image.
    """
    for channel in range(3):  # Assuming RGB
        gen_mean, gen_std = generated_image[..., channel].mean(), generated_image[..., channel].std()
        cont_mean, cont_std = content_image[..., channel].mean(), content_image[..., channel].std()
        generated_image[..., channel] = (
            (generated_image[..., channel] - gen_mean) / (gen_std + 1e-8)
        ) * cont_std + cont_mean
    return np.clip(generated_image, 0, 1)  # Ensure valid pixel range

from skimage.exposure import match_histograms

def match_colors(generated_image, content_image):
    """
    Match the color distribution of the generated image to the content image.

    Args:
        generated_image (np.ndarray): Generated image as a NumPy array.
        content_image (np.ndarray): Content image as a NumPy array.

    Returns:
        np.ndarray: Color-transferred image.
    """
    matched_image = match_histograms(generated_image, content_image, channel_axis=-1)
    return matched_image