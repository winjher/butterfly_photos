# import os
# import keras
# from keras.models import load_model
# import streamlit as st
# import tensorflow as tf
# import numpy as np




# st.header('Larval Diseases Classification CNN Model')
# larval_disease_names = ['Disease-Larvae_Common_Lime',
#  'Disease-Larvae_Golden_Birdwing',
#  'Healthy-Larvae_Common_Lime',
#  'Healthy-Larvae_Golden_Birdwing',
#  'Tachinid fly']




# model = load_model('Larval_Recog_Model.h5')
# st.success("Model loaded successfully.")




# def classify_images(image_path):
#     input_image = tf.keras.utils.load_img(image_path, target_size=(180,180))
#     input_image_array = tf.keras.utils.img_to_array(input_image)
#     input_image_exp_dim = tf.expand_dims(input_image_array,0)




#     predictions = model.predict(input_image_exp_dim)
#     result = tf.nn.softmax(predictions[0])
#     outcome = 'The Image belongs to ' + larval_disease_names[np.argmax(result)] + ' with a score of '+ str(np.max(result)*100)
#     return outcome




# uploaded_file = st.file_uploader('Upload an Image')
# if uploaded_file is not None:
#     with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
#         f.write(uploaded_file.getbuffer())
   
#     st.image(uploaded_file, width = 200)




#     st.markdown(classify_images(uploaded_file))


# import os
# import keras
# from keras.models import load_model
# import streamlit as st
# import tensorflow as tf
# import numpy as np

# st.header('Larval Diseases Classification CNN Model')
# larval_disease_names = ['Disease-Larvae_Common_Lime',
#  'Disease-Larvae_Golden_Birdwing',
#  'Healthy-Larvae_Common_Lime',
#  'Healthy-Larvae_Golden_Birdwing',
#  'Tachinid fly']

# model = load_model('Larval_Recog_Model.h5')
# st.success("Model loaded successfully.")

# def classify_images(image_path):
#     input_image = tf.keras.utils.load_img(image_path, target_size=(180,180))
#     input_image_array = tf.keras.utils.img_to_array(input_image)
#     input_image_exp_dim = tf.expand_dims(input_image_array, 0)

#     predictions = model.predict(input_image_exp_dim)
#     result = tf.nn.softmax(predictions[0])
#     outcome = 'The Image belongs to ' + larval_disease_names[np.argmax(result)] + ' with a score of ' + str(np.max(result) * 100)
#     return outcome, np.argmax(result)

# uploaded_file = st.file_uploader('Upload an Image')
# if uploaded_file is not None:
#     with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
#         f.write(uploaded_file.getbuffer())
   
#     st.image(uploaded_file, width=200)
#     outcome, label = classify_images(uploaded_file)
#     st.markdown(outcome)

#     if st.button('Show Treatment'):
#         treatment_info = {
#             0: "Treatment for Disease-Larvae_Common_Lime is to provide a net to secure the larvae from injection of fruit flies, gnats and parasites.",
#             1: "Treatment for Disease-Larvae_Golden_Birdwing is to keep away from the ants where they live.",
#             2: "No treatment needed, Healthy-Larvae_Common_Lime.",
#             3: "No treatment needed, Healthy-Larvae_Golden_Birdwing.",
#             4: "Treatment for Tachinid fly removed the larvae who have injected by the tachinid flies."
#         }
#         st.markdown(treatment_info[label])
        
#     # if st.button('Back'):
#     #    st.query_params(page="main")
# html_code = """
# <!DOCTYPE html>
# <html>
# <head>
#     <title>Classification</title>
#     <script type="text/javascript">
#         function showAlert() {
#             alert("This is a Larvae alert!");
#         }
#     </script>
# </head>
# <body>
#     <button onclick="showAlert()">Larvae</button>
# </body>
# </html>
# """

# st.components.v1.html(html_code, height=200)

import os
import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np

st.sidebar.header('Larval Diseases Classification CNN Model')
larval_disease_names = ['Baculovirus',
                        'Gnats Disease',
                        'Healthy-Larvae_Common_Lime',
                        'Healthy-Larvae_Golden_Birdwing',
                        'Tachinid fly']
 
model = load_model('Larval_Recog_Model.h5')
st.sidebar.success("Model loaded successfully.")

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'The Image belongs to ' + larval_disease_names[np.argmax(result)] + ' with a score of ' + str(np.max(result) * 100)
    return outcome, np.argmax(result)

uploaded_file = st.sidebar.file_uploader('Upload an Image')
if uploaded_file is not None:
    with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())
   
    st.image(uploaded_file, width=200)
    outcome, label = classify_images(uploaded_file)
    st.markdown(outcome)

    if st.sidebar.button('Show Treatment'):
        treatment_info = {
            0: "Treatment for Disease-Larvae_Common_Lime is to provide a net to secure the larvae from injection of fruit flies, gnats and parasites.",
            1: "Treatment for Disease-Larvae_Golden_Birdwing is to keep away from the ants where they live.",
            2: "No treatment needed, Healthy-Larvae_Common_Lime.",
            3: "No treatment needed, Healthy-Larvae_Golden_Birdwing.",
            4: "Treatment for Tachinid fly removed the larvae who have injected by the tachinid flies."
        }
        st.markdown(treatment_info[label])

html_code = """
# <!DOCTYPE html>
# <html>
# <head>
#     <title>Larval Disease Prediction</title>
#     <script type="text/javascript">
#         function showAlert() {
#             alert("This is a JavaScript alert!");
#         }
#     </script>
# </head>
# <body>
#     <button onclick="showAlert()">Click Me!</button>
# </body>
# </html>
"""

# st.components.v1.html(html_code, height=200)
