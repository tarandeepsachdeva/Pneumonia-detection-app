import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# Load model
model = tf.keras.models.load_model("model.h5")

labels = ["PNEUMONIA", "NORMAL"]
img_resize = 150

def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((img_resize, img_resize))
    img_array = np.array(image)
    img_array = img_array / 255.0  # Normalize
    img_array = img_array.reshape(1, img_resize, img_resize, 1)
    return img_array

# Streamlit UI
st.title("Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image to predict Pneumonia.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        processed = preprocess_image(image)
        prediction = model.predict(processed)
        result = labels[np.argmax(prediction)]
        st.success(f"Prediction: {result}")
