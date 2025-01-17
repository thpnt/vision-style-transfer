from PIL import Image
from io import BytesIO
import numpy as np
import base64
import tensorflow as tf
import gc
import hashlib

# Function to save uploaded file
def save_uploaded_file(file):
    if file is not None:
        return BytesIO(file.read())
    return None


# Function to download image
def download_image(image):
    # Ensure the input is a PIL Image; no need to reopen it.
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# Helper function : png to tensor
def preprocess_image(image_file, target_size=(256, 256)):
    try:
        image = Image.open(image_file).convert("RGB")
        image = image.resize(target_size, Image.LANCZOS)
        image_array = np.array(image) / 255.0
        image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, axis=0)
        del image, image_array
        gc.collect()
        return image_tensor
    except Exception as e:
        raise Exception(f"Error pre-processing the image: {str(e)}")

# Helper function : tensor to png
def postprocess_image(image_tensor):
    image_array = image_tensor.numpy()
    image_array = np.squeeze(image_array)
    image_array = np.clip(image_array, 0, 1)
    image_array = np.uint8(image_array * 255)
    image = Image.fromarray(image_array)
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

# Helper function : png to base64
def encode_base64(image):
    return base64.b64encode(image).decode('utf-8')

# Helper function : base64 to png
def decode_base64(image):
    return base64.b64decode(image)

import hashlib

# Helper function to compute a file hash
def compute_file_hash(file):
    hasher = hashlib.md5()
    hasher.update(file.read())
    file.seek(0)  # Reset file pointer after reading
    return hasher.hexdigest()