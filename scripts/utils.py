import os, sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# Define a function to load and preprocess images
def load_image(image_path, target_size=(128, 128)):
    """
    Load an image, resize it, and preprocess it for neural network input.
    
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Target size for the image, e.g., (128, 128).
    
    Returns:
        tf.Tensor: Preprocessed image tensor.
    """
    # Load the image
    image = Image.open(image_path).convert('RGB')  # Ensure 3 channels (RGB)
    
    # Resize the image
    image = image.resize(target_size, Image.LANCZOS)
    
    # Convert to a NumPy array and normalize pixel values
    image_array = np.array(image) / 255.0  # Scale pixel values to [0, 1]
    
    # Convert to a TensorFlow tensor
    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    
    # Add a batch dimension for processing in neural networks
    image_tensor = tf.expand_dims(image_tensor, axis=0)  # Shape: (1, 128, 128, 3)
    
    return image_tensor

# Display an image
def display_image(image_tensor, title="Image", clip_range=(0, 1)):
    """
    Display an image tensor.
    
    Args:
        image_tensor (tf.Tensor): Image tensor with shape (1, height, width, channels).
        title (str): Title of the plot.
        clip_range: (0, 1) or (0, 255)
    """
    # Remove the batch dimension and clip pixel values
    image = tf.squeeze(image_tensor, axis=0).numpy()  # Shape: (height, width, channels)
    image = np.clip(image, clip_range[0], clip_range[1])
    if clip_range==(0, 255):
        image = image.astype(np.uint8)
    
    # Plot the image
    plt.imshow(image)
    plt.axis('off')
    plt.title(title)
    plt.show()