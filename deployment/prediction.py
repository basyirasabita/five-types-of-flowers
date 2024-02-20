#import library
import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow_hub as hub

from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

#import pickle
import pickle

#load model
def run():
    # Set title
    st.title(':tulip: Flower Type Predictor :tulip:')

    # Set subheader
    st.subheader("This page will allow you to predict an image whether it's a Lily, Tulip, Orchid, Lotus, or Sunflower.")
    st.markdown('---')

    # Image
    st.image('./content/flower_banner1.JPEG', caption='Images taken from Unsplash')
    st.markdown('---')

    st.markdown('## :sunflower: Upload Flower Image :sunflower:')
    file = st.file_uploader("Upload an image", type=["jpg", "png"])

    model = load_model('base_model.keras', custom_objects={'KerasLayer': hub.KerasLayer})
    target_size=(256, 256)

    def import_and_predict(image_data, model):
        image = load_img(image_data, target_size=target_size)
        img_array = img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        # Normalize the image
        img_array = img_array / 255.0

        # Make prediction
        predictions = model.predict(img_array)

        # Get the predicted class
        predicted_class = np.argmax(predictions)

        label = ['Lily', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']
        result = f"Your flower is predicted as **{label[predicted_class]}**."

        return result

    if file is None:
        st.text("Please upload an image file")
    else:
        result = import_and_predict(file, model)
        st.image(file)
        st.markdown(result)
    
    st.markdown('---')
    st.text('Basyira Sabita - 2024')
        
if __name__ == "__main__":
    run()