# Train a Tranformer Network given a dataset of content images and a style image.

import os, sys, json, gc, psutil, ast
import tensorflow as tf
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(project_root, ".env"))

from src.transformer_net import TransformerNet, content_loss, style_loss, variation_loss, total_loss
from src.neural_optimization import get_activations, transform, build_style_transfer_model
from utils.images_utils import load_image, save_tensor_to_image

# Load hyperparameters
hyperparams = json.load(open(os.path.join(project_root, "models/hyperparameters.json"), "r"))

# Data path
dataset_path = os.path.join(project_root, "data/train_test/")

# Load style image
style_image = load_image(os.path.join(project_root, "data/style/mosaic.jpg"), 
                         target_size=ast.literal_eval(os.getenv("TARGET_SIZE", (256, 256))))





class TransformerNetTrainer:
    def __init__(self, transformer_net=TransformerNet(), style_image=style_image, dataset_path=dataset_path, batch_size=4, 
                 get_activations=get_activations, train_transform=transform, hyperparams=hyperparams,
                 content_loss=content_loss, style_loss=style_loss, variation_loss=variation_loss, total_loss=total_loss,
                 style="mosaic", carefulness=5, target_size=(256, 256), dataset_ratio=1.0):
        self.transformer_net = transformer_net
        self.get_activations = get_activations
        self.dataset = dataset_path
        self.batch_size = batch_size
        self.style = style
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparams[style]["learning_rate"])
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
        self.dataset_ratio = dataset_ratio
        
        # Style transfer model
        self.style_transfer_model = build_style_transfer_model(style=style, target_size=target_size)
        self.train_transform = train_transform
        
        # Build the model explicitly if not already built
        if not self.transformer_net.built:
            self.transformer_net.build(input_shape=(None, 256, 256, 3))

    def log_memory_usage(self):
        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
        self.memory_usage.append(memory)
    
    
    def train_step(self, batch, current_epoch, batch_names):
        
        # if epoch is 0, compute and save the target images
        if current_epoch == 0:
            # Compute target images using the style transfer model
            target_images = self.train_transform(batch, self.style_image, style=self.style, model=self.style_transfer_model)
            for i, target_image in enumerate(target_images):
                input_image_name = os.path.basename(batch_names[i]).split("/")[-1]
                save_tensor_to_image(target_image, 
                                     f"{project_root}/data/coco2017/target_images/{self.style}/{input_image_name}", 
                                     clip_range=(0, 1))
        else: # Load target images
            image_name = [os.path.basename(image) for image in batch_names]
            target_list = [f"{project_root}/data/coco2017/target_images/{self.style}/{image}" for image in image_name]
            target_images = [load_image(image_path, target_size=self.target_size) for image_path in target_list]
            target_images = tf.concat(target_images, axis=0)
        
        
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
        dataset_size = int(len(os.listdir(self.dataset)) * self.dataset_ratio)
        steps_per_epoch = int(dataset_size // self.batch_size)

        for epoch in tqdm(range(epochs), desc="Epochs", unit="epoch"):
            epoch_loss = 0.0
            
            with tqdm(total=steps_per_epoch, desc="Training", unit="batch") as pbar:
                for i in range(steps_per_epoch):

                    # Load batch
                    batch_list = [f"{self.dataset}/{b:06d}.jpg" for b in range(i * self.batch_size, (i + 1) * self.batch_size)]
                    batch = [load_image(image_path, target_size=self.target_size) for image_path in batch_list]
                    batch = tf.concat(batch, axis=0)

                    # Train
                    loss = self.train_step(batch, epoch, batch_list)

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
                        self.transformer_net.save_weights(f"{project_root}/models/transformer_net/{self.style}/batch_{i + 1}.weights.h5")
                        
                        # Save batch losses for the given epoch
                        with open(f"{project_root}/models/transformer_net/{self.style}/epoch_batch_loss.json", "w") as batch_loss_file:
                            json.dump({"epoch": epoch + 1, "batch_losses": self.batch_losses}, batch_loss_file)
                        
                    pbar.set_postfix({"loss": batch_loss})
                    pbar.update(1)

            # Save model weights
            self.transformer_net.save_weights(f"{project_root}/models/transformer_net/{self.style}/epoch_{epoch + 1}.weights.h5")
            
            # save loss and memory usage
            avg_epoch_loss = epoch_loss / steps_per_epoch
            self.epoch_losses.append(avg_epoch_loss)  # Track epoch loss
            
            with open(f"{project_root}/models/transformer_net/{self.style}/loss_memory.json", "w") as loss_file:
                json.dump({"epoch_losses": self.epoch_losses, "batch_losses": self.batch_losses, "memory_usage": self.memory_usage}, loss_file)