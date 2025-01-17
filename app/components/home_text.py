import sys, os
project_root = os.path.dirname(os.getcwd())
sys.path.append(project_root)

HTEXT = """
# 🎨 Neural Style Transfer Demo

Welcome to the **Neural Style Transfer Demo**! This application showcases two groundbreaking methods in the field of neural style transfer, allowing you to transform images with artistic styles in both real-time and iterative processes.  

Explore the app and learn more about the research that powers these state-of-the-art techniques.  
📂 **[View Full Code on GitHub](https://github.com/thpnt/vision-style-transfer)**

---

## How It Works

### 🎨 Page 1 : Iterative Style Transfer (Gatys et al.)
- **Upload two images**:  
  1. A **content image**, which serves as the base image.  
  2. A **style image**, which defines the artistic appearance to apply.  
- The app uses the **iterative optimization method** proposed by *Gatys et al.* (2015), where the style of the uploaded image is blended with the content image.  
- After computation, a stylized version of the content image is rendered for download.  
- This method optimizes the image iteratively and uses a combination of **content loss** and **style loss** (computed via Gram matrices of feature maps from a pre-trained CNN).

---

### ⚡️ Page 2 : Fast Style Transfer (Johnson et al.)
- **Pre-trained Style Model**:  
  On this page, you can apply a fast style transfer using a pre-trained transformer network.  
  - Currently, a **Mosaic style** model is available, based on *Justin Johnson et al.'s* (2016) work on perceptual losses.  
  - This method provides real-time style transfer with a single forward pass of the network.  

#### Loss Components Used in Training
The Fast Style Transfer model minimizes a combination of three losses during training:
1. **Content Loss**: Ensures the output image preserves the structure and semantics of the input content image.  
2. **Style Loss**: Matches the artistic style of the target style image by aligning Gram matrices of feature maps at multiple layers of a pre-trained network.  
3. **Variation Loss**: A **regularization term** computed on the output (target) image.  
   - **Purpose**: Encourages spatial smoothness by penalizing large variations in pixel intensity between neighboring pixels.  
   - **Implementation**: The total variation loss minimizes artifacts like noise or checkerboard patterns in the stylized image, improving its visual coherence.  
   - **Mathematically**:
   
---
## Behind the Scenes

### Iterative Optimization (Page 1)
The approach follows the seminal work of *Gatys et al. (2015)*:
- **Content Representation**: Extracted from a deep convolutional neural network to preserve the semantic structure of the content image.
- **Style Representation**: Captured by computing the Gram matrix of feature maps to encode the correlations between different feature channels.
- **Optimization**: An iterative process minimizes a weighted combination of content loss and style loss, producing a high-quality stylized image.

### Fast Style Transfer (Page 2)
This method builds on the work of *Justin Johnson et al. (2016)*:
- A **feed-forward transformer network** is trained to minimize perceptual losses:
  - **Content Loss**: Preserves the original image's structure.
  - **Style Loss**: Matches the style of the target using Gram matrices.
  - **Variation Loss**: Improves the smoothness and coherence of the generated image by reducing unwanted artifacts.  
- Once trained, the model can perform style transfer in real-time, making it ideal for applications where speed is critical.

---

## Key Features
- **Two State-of-the-Art Methods**: Explore both iterative optimization and fast style transfer methods.  
- **User-Friendly Interface**: Upload your own content and style images for fully customizable results.  
- **Real-Time Performance**: Enjoy near-instant style transfer with pre-trained models.  

---

## Future Work
- Additional transformer networks for new artistic styles are under training. Stay tuned!  
- Planned improvements to support more advanced style customization and parameter adjustments.  

---

## Try It Yourself
📂 **[Access the Full Code on GitHub](https://github.com/thpnt/vision-style-transfer)**
Explore the repository for implementation details, training scripts, and additional resources.  

---

Thank you for trying the Neural Style Transfer Demo! I hope you enjoy experimenting with the transformative power of neural networks.  
"""