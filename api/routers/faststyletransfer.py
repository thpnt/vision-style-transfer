import os, sys
import tensorflow as tf
from fastapi import APIRouter, HTTPException, UploadFile, File
from app.components.images import preprocess_image, postprocess_image, encode_base64
from src.transformer_net import TransformerNet

# Path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Model
transformer_net = TransformerNet()
transformer_net.build(input_shape=(None, 256, 256, 3))
transformer_net.load_weights(project_root + '/models/transformer_net/mosaic/epoch_10.weights.h5')

# Router
router = APIRouter()

@router.post("/faststyletransfer")
def style_transfer(content_image: UploadFile = File(...)):
    """
    Endpoint for fast style trasnsfer with pre-trained model (Johnson et al. 2016)
    """
    # Preprocess images
    content_tensor = preprocess_image(content_image.file)

    try:
        output_tensor = transformer_net.call(content_tensor)
        output_image = postprocess_image(output_tensor)
        output_image = encode_base64(output_image)
        
        return {"message": "Style Transfer completed successfully",
            "output_image": output_image
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    