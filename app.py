import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model("pneumonia_detector_finetuned.h5")
    return model

model = load_model()

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model input shape
    image = image.convert("RGB")  # Ensure image has 3 color channels
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

st.title("PneuScan - Pneumonia Detection from X-ray Images")
st.write("Upload an X-ray image to check for pneumonia.")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(processed_image)[0][0]
    
    # Display the result
    if prediction > 0.5:
        st.error(f"Pneumonia Detected with {prediction * 100:.2f}% confidence")
    else:
        st.success(f"No Pneumonia Detected with {(1 - prediction) * 100:.2f}% confidence")
