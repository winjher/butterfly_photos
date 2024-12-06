import os
import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import altair as alt
import plotly.express as px



st.header('Butterfly Classification CNN Model')

butterfly_names = ['Butterfly-Clippers',
 'Butterfly-Common Jay',
 'Butterfly-Common Lime',
 'Butterfly-Common Mime',
 'Butterfly-Common Mormon',
 'Butterfly-Emerald Swallowtail',
 'Butterfly-Golden Birdwing',
 'Butterfly-Gray Glassy Tiger',
 'Butterfly-Great Eggfly',
 'Butterfly-Great Yellow Mormon',
 'Butterfly-Paper Kite',
 'Butterfly-Pink Rose',
 'Butterfly-Plain Tiger',
 'Butterfly-Red Lacewing',
 'Butterfly-Scarlet Mormon',
 'Butterfly-Tailed Jay',
 'Moth-Atlas',
 'Moth-Giant Silk ']


model = load_model('Butterfly_Model.h5')
st.success("Model loaded successfully.")


def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array,0)




    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'The Image belongs to ' + butterfly_names[np.argmax(result)] + ' with a score of '+ str(np.max(result)*100)
    return outcome




uploaded_file = st.file_uploader('Upload an Image')
if uploaded_file is not None:
    with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())
   
    st.image(uploaded_file, width = 200)




    st.markdown(classify_images(uploaded_file))

# import time
# import streamlit as st


# def main():
#     if 'run_button' in st.session_state and st.session_state.run_button == True:
#         st.session_state.running = True
#     else:
#         st.session_state.running = False

#     if st.button('Dashboard', disabled=st.session_state.running, key='run_button'):
#         status = st.progress(0)
#         for t in range(10):
#             time.sleep(.2)
#             status.progress(10*t+10)
#         st.session_state.output = 'Butterfly Output'
#         st.query_params(page='dashboard')  # Set the page query parameter
#         st.experimental_rerun()  # Rerun the app

#     if 'output' in st.session_state:
#         st.write(st.session_state.output)

#     # Check for page query parameter and redirect
#     query_params = st.experimental_get_query_params()
#     if 'page' in query_params and query_params['page'][0] == 'home':
#         st.write("You have been redirected to the Dashboard page!")
#         # Add code for Home page here

# if __name__ == "__main__":
#     main()
