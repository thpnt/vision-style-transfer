# Train a Tranformer Network given a dataset of content images and a style image.

import argparse, os, sys, json, gc, psutil
import tensorflow as tf
import numpy as np
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.transformer_net import TransformerNet, content_loss, style_loss, variation_loss, total_loss
from src.neural_optimization import get_activations, transform
from utils.images_utils import load_image

# Load hyperparameters
hyperparams = json.load(open(os.path.join(project_root, "models/hyperparameters.json"), "r"))

# Data path
dataset_path = os.path.join(project_root, "data/train_test/")

# Load style image
style_image = load_image(os.path.join(project_root, "data/style/mosaic.jpg"), target_size=(256, 256))


class TransformerNetTrainer:
    def __init__(self, transformer_net=TransformerNet(), style_image=style_image, dataset_path=dataset_path, batch_size=4, 
                 get_activations=get_activations, train_transform=transform, hyperparams=hyperparams,
                 content_loss=content_loss, style_loss=style_loss, variation_loss=variation_loss, total_loss=total_loss,
                 style="mosaic", carefulness=5, target_size=(256, 256)):
        self.transformer_net = transformer_net
        self.train_transform = train_transform
        self.get_activations = get_activations
        self.dataset = dataset_path
        self.batch_size = batch_size
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.weights = hyperparams[style]["weights"]
        self.carefulness = carefulness
        self.epoch_losses = []  # List to track epoch losses
        self.batch_losses = []  # List to track batch losses
        self.content_loss = content_loss
        self.style_loss = style_loss
        self.variation_loss = variation_loss
        self.total_loss = total_loss
        self.target_size = target_size
        self.style_image = style_image
        self.memory_usage = [] # List to track memory usage
        
        # Build the model explicitly if not already built
        if not self.transformer_net.built:
            self.transformer_net.build(input_shape=(None, 256, 256, 3))

    def log_memory_usage(self):
        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
        self.memory_usage.append(memory)
    
    
    def train_step(self, batch):
        # Compute target images using the style transfer model
        target_images = self.train_transform(batch, self.style_image)
        
        # Compute activations for target
        target_content, target_style = self.get_activations(target_images)

        with tf.GradientTape() as tape:
            # Forward pass
            output = self.transformer_net.call(batch)
            output_content, output_style = self.get_activations(output)

            # Compute losses
            c_loss = self.content_loss(output_content, target_content)
            s_loss = self.style_loss(output_style, target_style)
            tv_loss = self.variation_loss(output)
            loss = self.total_loss(c_loss, s_loss, tv_loss, self.weights)

        # Compute gradients and apply optimization
        gradients = tape.gradient(loss, self.transformer_net.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer_net.trainable_variables))
        
        # Free memory
        del target_images, output, c_loss, s_loss, tv_loss, target_style, target_content, output_style, output_content
        gc.collect()

        return loss

    def train(self, epochs):
        dataset_size = len(os.listdir(self.dataset))
        steps_per_epoch = dataset_size // self.batch_size

        for epoch in tqdm(range(epochs), desc="Epochs", unit="epoch"):
            epoch_loss = 0.0
            
            with tqdm(total=steps_per_epoch, desc="Training", unit="batch") as pbar:
                for i in range(steps_per_epoch):

                    # Load batch
                    batch_list = [f"{self.dataset}/{b:06d}.jpg" for b in range(i * self.batch_size, (i + 1) * self.batch_size)]
                    batch = [load_image(image_path, target_size=self.target_size) for image_path in batch_list]
                    batch = tf.concat(batch, axis=0)

                    # Train
                    loss = self.train_step(batch)

                    # Track batch loss
                    batch_loss = float(loss.numpy())
                    epoch_loss += batch_loss
                    self.batch_losses.append(batch_loss)  # Track batch loss

                    # Free memory
                    del batch, batch_list
                    gc.collect()

                    # Log memory usage
                    self.log_memory_usage()

                    # Intermediate save
                    if (i+1) % self.carefulness == 0:
                        self.transformer_net.save_weights(f"{project_root}/models/transformer_net/batch_{i + 1}.weights.h5")
                        
                        # Save batch losses for the given epoch
                        with open(f"{project_root}/models/epoch_batch_loss.json", "w") as batch_loss_file:
                            json.dump({"epoch": epoch + 1, "batch_losses": self.batch_losses}, batch_loss_file)
                        
                    pbar.set_postfix({"loss": batch_loss})
                    pbar.update(1)

            # Save model weights
            self.transformer_net.save_weights(f"{project_root}/models/transformer_net/epoch_{epoch + 1}.weights.h5")
            
            # save loss and memory usage
            avg_epoch_loss = epoch_loss / steps_per_epoch
            self.epoch_losses.append(avg_epoch_loss)  # Track epoch loss
            
            with open(f"{project_root}/models/loss_memory.json", "w") as loss_file:
                json.dump({"epoch_losses": self.epoch_losses, "batch_losses": self.batch_losses, "memory_usage": self.memory_usage}, loss_file)