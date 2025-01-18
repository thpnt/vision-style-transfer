import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from components.images import download_image, save_uploaded_file, preprocess_image, decode_base64, compute_file_hash

# Config
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
st.set_page_config(page_title="Style Transfer.", layout="wide")

# Initial state
if "content_image" not in st.session_state:
    st.session_state["content_image"] = None
if "style_image" not in st.session_state:
    st.session_state["style_image"] = None
if "styled_image" not in st.session_state:
    st.session_state["styled_image"] = None
if "content_image_fst" not in st.session_state:
    st.session_state["content_image_fst"] = None
if "style_image_fst" not in st.session_state:
    st.session_state["style_image_fst"] = None
if "styled_image_fst" not in st.session_state:
    st.session_state["styled_image_fst"] = None


# Main page
st.write("## Add your favorites artistic style to your picture.")
col1, col2 = st.columns(2)

# Callback function to reset styled_image
def reset_style_image():
    st.session_state["styled_image"] = None

# Sidebar
st.sidebar.write("## Upload and download :gear:")
style_image = st.sidebar.file_uploader("Upload an artistic style image", type=["png", "jpg", "jpeg"], on_change=reset_style_image)
content_image = st.sidebar.file_uploader("Upload a content image", type=["png", "jpg", "jpeg"], on_change=reset_style_image)
if st.session_state["styled_image"] is not None:
    st.sidebar.download_button(
        "Download your styled image",
        download_image(st.session_state["styled_image"]),
        "my_styled_image.png",
    )

# Save to session state
if style_image is not None:
    st.session_state["style_image"] = save_uploaded_file(style_image)
    

if content_image is not None:
    st.session_state["content_image"] = save_uploaded_file(content_image)



# Retrieve from session state and display
if "style_image" in st.session_state and st.session_state["style_image"] is not None:
    col1.write("Style Image.")
    col1.image(st.session_state["style_image"], use_container_width=True)

if "content_image" in st.session_state and st.session_state["content_image"] is not None:
    col2.write("Content Image.")
    col2.image(st.session_state["content_image"], use_container_width=True)
    
# Check if styled_image exists in session state
if "styled_image" in st.session_state and st.session_state["styled_image"] is not None:
    st.write("## Your Styled Image.")
    st.image(st.session_state["styled_image"], width=256)


# Test API call
API_URL = "http://localhost:8000/api/v1/styletransfer"

if st.button("Apply Style Transfer"):
    if (
        st.session_state["content_image"] is not None
        and st.session_state["style_image"] is not None
    ):
        with st.spinner("Applying style transfer..."):
            try:
                response = requests.post(
                    API_URL,
                    files={
                        "content_image": st.session_state["content_image"],
                        "style_image": st.session_state["style_image"],
                    },
                )
                if response.status_code == 200:

                    styled_image = response.json()["output_image"]
                    styled_image = decode_base64(styled_image)
                    styled_image = Image.open(BytesIO(styled_image))
                    st.session_state["styled_image"] = styled_image

                    st.write("## Your Styled Image.")
                    st.image(styled_image, width=256)

                    # Force rerun to display download button
                    st.rerun()
                else:
                    st.error(f"Error processing image {response.status_code}.")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    else:
        st.info("Please upload both content and style images.")