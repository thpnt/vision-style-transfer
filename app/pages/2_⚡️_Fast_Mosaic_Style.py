import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from components.images import download_image, save_uploaded_file, preprocess_image, decode_base64, compute_file_hash



# Page config
st.set_page_config(page_title="Style Transfer.", layout="wide")
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Callback on file_change
def reset_style_image_fs():
    st.session_state["styled_image_fst"] = None

# Instantiate session_state
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

# Sidebar
st.sidebar.write("## Upload and download :gear:")
content_image_fst = st.sidebar.file_uploader("Upload a content image", type=["png", "jpg", "jpeg"], on_change=reset_style_image_fs)
if st.session_state["styled_image_fst"] is not None:
    st.sidebar.download_button(
        "Download your styled image",
        download_image(st.session_state["styled_image_fst"]),
        "my_fast_styled_image.png",
    )

# Save to session state
if content_image_fst is not None:
    st.session_state["content_image_fst"] = save_uploaded_file(content_image_fst)

# Display images
# Display content image
if "content_image_fst" in st.session_state and st.session_state["content_image_fst"] is not None:
    col1.write("Content Image.")
    col1.image(st.session_state["content_image_fst"], use_container_width=True)

# Check if styled_image exists in session state
if "styled_image_fst" in st.session_state and st.session_state["styled_image_fst"] is not None:
    col2.write("Your Styled Image.")
    col2.image(st.session_state["styled_image_fst"], use_container_width=True)


# API Call
API_URL = "http://localhost:8000/api/v1/faststyletransfer"

if st.button("Apply Fast Style Transfer"):
    if (
        st.session_state["content_image_fst"] is not None
    ):
        with st.spinner("Applying fast style transfer..."):
            try:
                response_fst = requests.post(
                    API_URL,
                    files={
                        "content_image": st.session_state["content_image_fst"],
                    },
                )
                if response_fst.status_code == 200:

                    styled_image_fst = response_fst.json()["output_image"]
                    styled_image_fst = decode_base64(styled_image_fst)
                    styled_image_fst = Image.open(BytesIO(styled_image_fst))
                    st.session_state["styled_image_fst"] = styled_image_fst

                    col2.write("Your Styled Image.")
                    col2.image(styled_image_fst, use_container_width=True)

                    # Force rerun to display download button
                    st.rerun()
                else:
                    st.error(f"Error processing image {response_fst.status_code}.")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    else:
        st.info("Please upload your content image.")
