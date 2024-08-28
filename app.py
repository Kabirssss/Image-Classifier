import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Load your trained model
model = tf.keras.models.load_model('imageclassifier.h5')

# Function to preprocess the image
def preprocess_image(image):
    img = np.array(image)
    resize = tf.image.resize(img, (256, 256))
    return resize

# Prediction function
def predict(image):
    image = preprocess_image(image)
    yhat = model.predict(np.expand_dims(image / 255, 0))
    return yhat

# Streamlit app
st.title("Image Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Prediction
    yhat = predict(image)

    if yhat > 0.5:
        st.write(f'Predicted class is **Sad**')
    else:
        st.write(f'Predicted class is **Happy**')
