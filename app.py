import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(page_title="YOLO11 Object Detection", layout="wide")

st.title("🚀 YOLO11 Object Detection App")
st.write("Upload an image to detect objects using the YOLO11n model.")

# Load the model
@st.cache_resource
def load_model():
    return YOLO('yolo11n.pt')

model = load_model()

# Sidebar for configuration
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to PIL Image
    image = Image.open(uploaded_file)
    
    # Create two columns for side-by-side view
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    
    # Run Inference
    with st.spinner('Detecting...'):
        # Convert PIL to numpy for YOLO
        img_array = np.array(image)
        results = model.predict(img_array, conf=conf_threshold)
        
        # Plot results
        res_plotted = results[0].plot() # This returns a BGR numpy array
        
    with col2:
        st.subheader("Detection Result")
        # Streamlit expects RGB, so we show the plotted result
        st.image(res_plotted, channels="BGR", use_container_width=True)
        
    # Show detection details in an expander
    with st.expander("See Detection Details"):
        for result in results:
            boxes = result.boxes
            for box in boxes:
                st.write(f"Detected **{model.names[int(box.cls)]}** with {box.conf[0]:.2f} confidence")

