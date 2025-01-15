### This scripts train the TransformerNet with a TransformerNetTrainer

import argparse, os, sys, json
# Add the project root to the system path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


from src.trainer import TransformerNetTrainer
from src.transformer_net import TransformerNet
from utils.images_utils import load_image




if __name__ == "__main__":
    
    # Argument parser
    parser = argparse.ArgumentParser(description="Train the TransformerNet model") 
    parser.add_argument("--dataset_path", type=str, default=os.path.join(project_root, "data/train_test/"), help="Path to the dataset of content images")
    parser.add_argument("--style_image_path", type=str, default=os.path.join(project_root, "data/style/mosaic.jpg"), help="Path to the style image")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train the model")
    parser.add_argument("--image_size", type=int, default=256, help="Size of the images to be used for training")
    parser.add_argument("--carefulness", type=int, default=5, help="Carefulness parameter for the style loss. Lower values result in saving more often.")
    parser.add_argument("--hyperparameters_path", type=str, default=os.path.join(project_root, "models/hyperparameters.json"), help="Path to the hyperparameters file")
    args = parser.parse_args()
    
    
    # Load hyperparameters
    hyperparams = json.load(open(args.hyperparameters_path, "r"))
    
    # Load style_image
    style_image = load_image(args.style_image_path, target_size=(args.image_size, args.image_size))

    # Instantiate the TransformerNet and StyleTransferModel
    transformer_net = TransformerNet()

    # Initialize the trainer
    trainer = TransformerNetTrainer(transformer_net=transformer_net,
                                    dataset_path=args.dataset_path, 
                                    batch_size=args.batch_size, 
                                    target_size=(args.image_size, args.image_size), 
                                    carefulness=args.carefulness)

    print("Training the model...")
    # Train the model
    trainer.train(epochs=args.epochs)