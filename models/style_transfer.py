import os, sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import clear_output
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add path to the project root for importing custom modules
if project_root not in sys.path:
    sys.path.append(project_root)


# Model
class StyleTransferModel(tf.keras.Model):
    def __init__(self, target_size=(256, 256), learning_rate=1e-3):
        super().__init__()
        self.vgg = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=target_size + (3,))
        for layer in self.vgg.layers:
            layer.trainable = False
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        self.content_layers = ["block4_conv2"]
        self.style_layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1"]
        self.feature_extractor = self.get_feature_extractor(self.vgg, self.content_layers + self.style_layers)

    def get_feature_extractor(self, pre_trained_model, layer_names):
        outputs = [pre_trained_model.get_layer(name).output for name in layer_names]
        return tf.keras.Model(inputs=pre_trained_model.input, outputs=outputs)

    def get_activations(self, image, feature_extractor):
        preprocessed_image = tf.keras.applications.vgg16.preprocess_input(image * 255) # Useful for VGG
        # Get activations
        activations = feature_extractor(preprocessed_image)
        content_activations = activations[:len(self.content_layers)]
        style_activations = activations[len(self.content_layers):]
        del activations
        return content_activations, style_activations

    @staticmethod
    def content_loss(content_activation, target_activation):
        return tf.reduce_mean([tf.reduce_mean(tf.square(c - t)) for c, t in zip(content_activation, target_activation)])

    @staticmethod
    def gram_matrix(activation):
        batch, height, width, channels = tf.shape(activation)
        reshaped = tf.reshape(activation, [height * width, channels])
        return tf.matmul(reshaped, reshaped, transpose_a=True)

    @staticmethod
    def style_loss(style_activations, target_activations):
        style_grams = [StyleTransferModel.gram_matrix(s) for s in style_activations]
        target_grams = [StyleTransferModel.gram_matrix(t) for t in target_activations]
        return tf.reduce_mean([tf.reduce_mean(tf.square(s - t)) for s, t in zip(style_grams, target_grams)])

    @staticmethod
    def total_loss(content_loss, style_loss, weights):
        return weights["content"] * content_loss + weights["style"] * style_loss
    
    @staticmethod
    def save_image(image_tensor, epoch, method="optimization_method", version="test"):
        image = tf.squeeze(image_tensor, axis=0).numpy()  # Remove batch dimension
        image = np.clip(image, 0, 1)
        image = tf.image.convert_image_dtype(image, tf.uint8)

        # Define the directory and ensure it exists
        save_dir = os.path.join(project_root, f"models/ouputs_monitoring/{method}/{version}")
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

        # Construct file path
        file_path = os.path.join(save_dir, f"output_image_{epoch}.png")

        # Save the image
        tf.keras.utils.save_img(file_path, image)
        
    @staticmethod
    def display_image(image_tensor, title="Image"):
        # Remove the batch dimension and clip pixel values
        image = tf.squeeze(image_tensor, axis=0).numpy()  # Shape: (height, width, channels)
        image = np.clip(image, 0, 1)  # Ensure pixel values are in the range [0, 1]

        # Plot the image
        plt.imshow(image)
        plt.axis('off')
        plt.title(title)
        plt.show()
    
    @tf.function
    def train_step(self, target_image, content_activations, style_activations, weights):
        with tf.GradientTape() as tape:
            target_content, target_style = self.get_activations(target_image, self.feature_extractor)
            c_loss = self.content_loss(content_activations, target_content)
            s_loss = self.style_loss(style_activations, target_style)
            loss = self.total_loss(c_loss, s_loss, weights)
        gradients = tape.gradient(loss, target_image)
        self.optimizer.apply_gradients([(gradients, target_image)])
        return loss

    def fit(self, content_image, style_image, weights, n_epochs=1001, version='test'):
        content_activations, _ = self.get_activations(content_image, self.feature_extractor)
        _, style_activations = self.get_activations(style_image, self.feature_extractor)
        target_image = tf.Variable(content_image)

        for epoch in tqdm(range(n_epochs+1), desc="Image optimization"):
            loss = self.train_step(target_image, content_activations, style_activations, weights, self.optimizer, self.feature_extractor)
            
            if epoch % 250 == 0:
                print(f"Epoch {epoch}: Loss: {loss}")
                clipped_image = tf.clip_by_value(target_image, 0., 1.)
                # Save image
                self.save_image(clipped_image, epoch, method="optimization_method", version=version)
                
                # Display image
                clear_output(wait=True)
                self.display_image(clipped_image, title=f"Epoch {epoch}")