import os, sys
import numpy as np
import tensorflow as tf

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add path to the project root for importing custom modules
if project_root not in sys.path:
    sys.path.append(project_root)
    
from src.convolution import EncodeConvBlock, DecoderConvBlock, ResBlock
from src.neural_optimization import CONTENT_LAYERS, STYLE_LAYERS
    
class TransformerNet(tf.keras.Model):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.model_layers = [
            EncodeConvBlock(filters=32, kernel_size=9, strides=1),
            EncodeConvBlock(filters=64, kernel_size=3, strides=2),
            EncodeConvBlock(filters=128, kernel_size=3, strides=2),
            ResBlock(filters=128, kernel_size=3),
            ResBlock(filters=128, kernel_size=3),
            ResBlock(filters=128, kernel_size=3),
            ResBlock(filters=128, kernel_size=3),
            ResBlock(filters=128, kernel_size=3),
            DecoderConvBlock(filters=64, kernel_size=3, strides=2),
            DecoderConvBlock(filters=32, kernel_size=3, strides=2),
            DecoderConvBlock(filters=3, kernel_size=9, strides=1, use_tanh=True)
        ]

    def build(self, input_shape):
        for layer in self.model_layers:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)
        self.built = True
        self._build_input_shape = input_shape  # Save input shape for serialization
        
    
    def call(self, x):
        for layer in self.model_layers:
            x = layer(x)
        return x
    
    
# Losses computation
# Content loss
def content_loss(content_activation, target_activation):
        return tf.reduce_mean([tf.reduce_mean(tf.square(c - t)) for c, t in zip(content_activation, target_activation)])
    
# Style loss
def gram_matrix(activation):
    # Reshape activation tensor to [height * width, channels]
    shape = tf.shape(activation)
    batch, height, width, channels = shape[0], shape[1], shape[2], shape[3]
    reshaped = tf.reshape(activation, [batch, height * width, channels])
    
    gram = tf.matmul(reshaped, reshaped, transpose_a=True)
    num_elements = tf.cast(height * width * channels, tf.float32)  # Normalize by C * H * W
    
    return gram / num_elements

def style_loss(style_activations, target_activations):
    # Gram matrices
    style_grams = []
    target_grams = []
    for i in range(len(style_activations)):
        style_grams.append([gram_matrix(s) for s in style_activations])
        target_grams.append([gram_matrix(t) for t in target_activations])
    
    # Style loss
    num_layers = len(style_grams)  # Number of style layers
    num_images = len(style_grams[0])  # Number of images

    # Iterate over images in batch
    individual_losses = []
    for j in range(num_images):
        image_loss = 0.0
        for i in range(num_layers):
            diff = style_grams[i][j] - target_grams[i][j]
            layer_loss = tf.reduce_mean(tf.square(diff))  # Mean Frobenius norm
            image_loss += layer_loss
        image_loss /= num_layers  # Average over layers
        individual_losses.append(image_loss)

    # Compute the overall mean style loss
    overall_loss = tf.reduce_mean(individual_losses)

    return overall_loss

# Variation loss
def variation_loss(target_images):
        horizontal_diff = target_images[:, :, 1:, :] - target_images[:, :, :-1, :]
        vertical_diff = target_images[:, 1:, :, :] - target_images[:, :-1, :, :] 
        tv_loss = tf.reduce_sum(tf.square(horizontal_diff)) + tf.reduce_sum(tf.square(vertical_diff))
        return tv_loss / tf.cast(target_images.shape[0], tf.float32)

def total_loss(content_loss, style_loss, tv_loss, weights):
        return weights["content"] * content_loss + weights["style"] * style_loss + weights["tv"] * tv_loss

