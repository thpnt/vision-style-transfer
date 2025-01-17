import os, sys
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import APIRouter, HTTPException, UploadFile, File
from app.components.images import preprocess_image, postprocess_image, encode_base64

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                
from src.neural_optimization import transform

router = APIRouter()


@router.post("/styletransfer")
def style_transfer(content_image: UploadFile = File(...), style_image: UploadFile = File(...)):
    """
    Endpoint for style transfer with iterative optimization method (Gatys et al. 2015)
    """
    # Preprocess images
    content_tensor = preprocess_image(content_image.file)
    style_tensor = preprocess_image(style_image.file)
    try:
        output_tensor = transform(content_tensor, style_tensor)
        output_image = postprocess_image(output_tensor)
        output_image = encode_base64(output_image)
        
        return {"message": "Style Transfer completed successfully",
            "output_image": output_image
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    