# This scripts creates the target images dataset from the original content images.
# The target images are computed from content images with the style transfer method from Gatys et al. (2015).
# This script will create a new folder with the target images dataset.
# The style used is given in the argument --style.
# The target images are saved in the folder specified in the argument --output_folder.
# The script uses the following arguments:
# --content_folder: Folder with the content images.
# --style: Style image.
# --output_folder: Folder to save the target images.
# --image_size: Size of the images.
# --bottom_image_id: id of first image of the batch
# --top_image_id: id of last image of the batch

# Import
import os, sys, argparse, json
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm   
import gc

# Path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
data_path = os.path.join(project_root, 'data')


# Import custom modules
from scripts.utils import load_image
from models.style_transfer import StyleTransferModel, CONTENT_LAYERS, STYLE_LAYERS, get_feature_extractor, get_vgg
hyperparameters = json.load(open(os.path.join(project_root, 'models/hyperparameters.json')))


# Arguments parser
parser = argparse.ArgumentParser(description='Create target images dataset.')
parser.add_argument('--content_folder', default=os.path.join(data_path, "coco2017/raw_images"), 
                    help='Folder with the content images.')
parser.add_argument('--output_folder', default=os.path.join(data_path, "coco2017/target_images"),
                    help="Folder to save the target images.")
parser.add_argument('--style', default="starry_night", help='Name of style image (eg: starry_night).')
parser.add_argument('--image_size', default=hyperparameters['image_size'], type=int, help='Size of the images.')
parser.add_argument('--bottom_image_id', default=0, type=int, help='id of first image of the batch')
parser.add_argument('--top_image_id', default=1000, type=int, help='id of last image of the batch')

# Parse arguments
args = parser.parse_args()
target_size = (args.image_size, args.image_size)

# Test if tensorflow is running on GPU
if tf.config.list_physical_devices('GPU') == []:
    print("No GPU detected. Style transfer will be slow.")
else:
    print(f"GPU detected. {tf.config.list_physical_devices}.")

####################################################################################################
logs = {}

# Create ouptut folder
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)


# Load style image
style_image_path = os.path.join(data_path, "style", args.style + ".jpg")
try:
    style_image = load_image(style_image_path, target_size) # target_size = (image_size, image_size)
except Exception as e:
    logs["style_image"] = f"Error loading style image: {e}"
    sys.exit(1)

 
# List content images
content_images = os.listdir(args.content_folder)


# Instantiate VGG activations extractor
feature_extractor = get_feature_extractor(get_vgg(target_size), CONTENT_LAYERS + STYLE_LAYERS)


# Compute and save target images
logs["content_images"] = {}


# Loop over content images
top_id = args.top_image_id
bottom_id = args.bottom_image_id
total_iterations = top_id - bottom_id

with tqdm(total=total_iterations, desc="Processing", unit="iteration") as pbar:
    for i in range(bottom_id, top_id):
        id = f"{i:05}"
        content_image_path = os.path.join(args.content_folder, content_images[i])
        try:
            content_image = load_image(content_image_path, target_size)
        except Exception as e:
            logs["content_images"][content_images[id]] = {}
            logs["content_images"][content_images[id]]["load_image"] = f"Error loading content image: {e}"
            continue
        
        # Instantiate model
        model = StyleTransferModel(feature_extractor=feature_extractor, target_size=target_size, learning_rate=hyperparameters["learning_rate"])

        # Fit
        try:
            generated_image = model.fit(content_image, style_image, hyperparameters["weights"], 
                                        n_epochs=hyperparameters["n_epochs"], 
                                        version=f"target_images_{i}", save=False, display=False)
            # Save generated image to output folder

            generated_image = np.uint8(generated_image.numpy()[0] * 255)
            generated_image_pil = Image.fromarray(generated_image)
            generated_image_pil.save(os.path.join(args.output_folder, f"transformed_{i}.jpg"))
            del model, generated_image, generated_image_pil, content_image
            tf.keras.backend.clear_session()
            gc.collect()

        except Exception as e:
            if content_images[i] not in logs["content_images"].keys():
                logs["content_images"][content_images[i]] = {}
            logs["content_images"][content_images[i]]["generation"] = f"Error with image {i}: {str(e)}"
            continue
        
        pbar.update(1)
    
    
# Save logs
with open(os.path.join(project_root, "results/target_images/logs.json"), "w") as f:
    json.dump(logs, f)
    
