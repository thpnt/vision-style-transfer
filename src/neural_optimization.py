import os, sys, json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add path to the project root for importing custom modules
if project_root not in sys.path:
    sys.path.append(project_root)
    
# Constants 
TARGET_SIZE = (256, 256)
CONTENT_LAYERS = ["block4_conv2"]
STYLE_LAYERS = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1"]
WEIGHTS = {
    "content": 1,
    "style": 1e5,
    "tv": 1e-6
} 

# Feature extractor functions
def get_vgg(input_shape):
    vgg = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=input_shape + (3,))
    for layer in vgg.layers:
        layer.trainable = False
    return vgg

def get_feature_extractor(vgg, layer_names=CONTENT_LAYERS + STYLE_LAYERS):
    outputs = [vgg.get_layer(name).output for name in layer_names]
    return tf.keras.Model(inputs=vgg.input, outputs=outputs)

feature_extractor = get_feature_extractor(get_vgg(TARGET_SIZE), CONTENT_LAYERS + STYLE_LAYERS)

def get_activations(image, feature_extractor=feature_extractor):
    preprocessed_image = tf.keras.applications.vgg16.preprocess_input(image * 255) # Useful for VGG
    # Get activations
    activations = feature_extractor(preprocessed_image)
    content_activations = activations[:len(CONTENT_LAYERS)]
    style_activations = activations[len(CONTENT_LAYERS):]
    del activations
    return content_activations, style_activations


# Model
class StyleTransferModel(tf.keras.Model):
    def __init__(self, get_activations=get_activations, target_size=TARGET_SIZE, learning_rate=1e-3, 
                 content_layers=CONTENT_LAYERS, style_layers=STYLE_LAYERS):
        super().__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.get_activations = get_activations
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.target_size = target_size

    @staticmethod
    def content_loss(content_activation, target_activation):
        return tf.reduce_mean([tf.reduce_mean(tf.square(c - t)) for c, t in zip(content_activation, target_activation)])

    
    @staticmethod
    def gram_matrix(activation):
        # Get the shape of the activation tensor (using tf.shape)
        shape = tf.shape(activation)
        batch, height, width, channels = shape[0], shape[1], shape[2], shape[3]

        # Reshape activation tensor to [height * width, channels]
        reshaped = tf.reshape(activation, [height * width, channels])

        # Compute the Gram matrix by multiplying the reshaped tensor with its transpose
        gram = tf.matmul(reshaped, reshaped, transpose_a=True)
        
        # Normalize the Gram matrix
        num_elements = tf.cast(height * width * channels, tf.float32)  # Normalize by C * H * W
        gram = gram / num_elements

        return gram
    

    @staticmethod
    def style_loss(style_activations, target_activations):
        style_grams = [StyleTransferModel.gram_matrix(s) for s in style_activations]
        target_grams = [StyleTransferModel.gram_matrix(t) for t in target_activations]
        return tf.reduce_mean([tf.reduce_mean(tf.square(s - t)) for s, t in zip(style_grams, target_grams)])
    
    @staticmethod
    def variation_loss(target_image):
        # Ensure computations directly derive from the original target_image variable
        # Calculate differences between adjacent pixels in the horizontal direction
        horizontal_diff = target_image[:, :, 1:, :] - target_image[:, :, :-1, :]

        # Calculate differences between adjacent pixels in the vertical direction
        vertical_diff = target_image[:, 1:, :, :] - target_image[:, :-1, :, :]

        # Compute the total variation loss as the sum of squared differences
        tv_loss = tf.reduce_sum(tf.square(horizontal_diff)) + tf.reduce_sum(tf.square(vertical_diff))

        return tv_loss

    @staticmethod
    def total_loss(content_loss, style_loss, tv_loss, weights):
        return weights["content"] * content_loss + weights["style"] * style_loss + weights["tv"] * tv_loss
    
    @staticmethod
    def save_image(image_tensor, epoch, method="optimization_method", version="test"):
        image = tf.squeeze(image_tensor, axis=0).numpy()  # Remove batch dimension
        image = np.clip(image, 0, 1)
        image = tf.image.convert_image_dtype(image, tf.uint8)

        # Define the directory and ensure it exists
        save_dir = os.path.join(project_root, f"results/style_transfer/{method}/{version}")
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
            target_content, target_style = self.get_activations(target_image)
            c_loss = self.content_loss(content_activations, target_content)
            s_loss = self.style_loss(style_activations, target_style)
            tv_loss = self.variation_loss(target_image)
            loss = self.total_loss(c_loss, s_loss, tv_loss, weights)
        gradients = tape.gradient(loss, target_image)
        self.optimizer.apply_gradients([(gradients, target_image)])
        return loss
    

    def fit(self, content_image, style_image, weights, n_epochs=1001, version='test', save=True, display=True, verbose=20):
        content_activations, _ = self.get_activations(content_image)
        _, style_activations = self.get_activations(style_image)
        target_image = tf.Variable(content_image)

        for epoch in range(n_epochs+1):
            loss = self.train_step(target_image, content_activations, style_activations, weights)
            target_image.assign(tf.clip_by_value(target_image, 0., 1.))
            
            if epoch % verbose == 0:
                # Save image
                if save:
                    self.save_image(target_image, epoch, method="optimization_method", version=version)
                
                # Display image
                if display:
                    clear_output(wait=True)
                    self.display_image(target_image, title=f"Epoch {epoch}")
                
        return target_image
    

# Instantiate model
hyperparams = json.load(open(os.path.join(project_root, "models/hyperparameters.json"), "r"))

def build_style_transfer_model(style="mosaic"):
    return StyleTransferModel(get_activations=get_activations, target_size=TARGET_SIZE, learning_rate=hyperparams[style]['learning_rate'])

model = build_style_transfer_model()

def transform(images, style_image, hyperparams=hyperparams, style='default'):
    output = []  # Use a list to store transformed images
    for i in range(images.shape[0]):
        # Process each image independently
        transformed = model.fit(
            tf.expand_dims(images[i], axis=0),  # Add batch dimension
            style_image,
            hyperparams[style]["weights"],
            n_epochs=hyperparams[style]["n_epochs"],
            save=False,
            display=False
        )
        # Append the transformed image to the output list
        output.append(tf.squeeze(transformed, axis=0))  # Remove batch dimension
    
    # Convert the output list to a tensor
    return tf.stack(output, axis=0)
