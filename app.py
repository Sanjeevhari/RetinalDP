import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow import argmax
from keras.preprocessing import image
import tensorflow as tf
import numpy as np

st.set_page_config(
    page_title="Retinal Disease Detection",
    page_icon = ":eye:",
    initial_sidebar_state = 'auto'
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

with st.sidebar:
        st.image('bg1.jpg')
        st.title("Retinal Detection")
        st.subheader("Detection of diseases present in the Retinal. This helps an user to easily detect the disease and identify it's cause.")

st.write("""
         # Retinal Disease Detection
         """
         )

model = load_model('Modeleye.h5')

labels= ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

def predict(image_file):
  img = tf.keras.preprocessing.image.load_img(image_file, target_size=(224, 224))
  img_array = tf.keras.preprocessing.image.img_to_array(img)
  img_array = img_array / 255.0
  img_batch = np.expand_dims(img_array, axis=0)
  predictions = model.predict(img_batch)
  predicted_class = argmax(predictions[0])
  return labels[predicted_class]

file = st.file_uploader("", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image_data = file.read()
    st.image(image_data, width=250)
    prediction = predict(file)
    st.write(f"Predicted Disease: {prediction}")

    string = "Detected Disease : " + prediction
    if class_names[np.argmax(predictions)] == 'normal':
        st.balloons()
        st.sidebar.success(string)

    elif class_names[np.argmax(predictions)] == 'cataract':
        st.sidebar.warning(string)
        #st.markdown("## Remedy")
        #st.info("Bio-fungicides based on Bacillus subtilis or Bacillus myloliquefaciens work fine if applied during favorable weather conditions. Hot water treatment of seeds or fruits (48°C for 20 minutes) can kill any fungal residue and prevent further spreading of the disease in the field or during transport.")

    elif class_names[np.argmax(predictions)] == 'diabetic_retinopathy':
        st.sidebar.warning(string)
        #st.markdown("## Remedy")
        #st.info("Prune flowering trees during blooming when wounds heal fastest. Remove wilted or dead limbs well below infected areas. Avoid pruning in early spring and fall when bacteria are most active.If using string trimmers around the base of trees avoid damaging bark with breathable Tree Wrap to prevent infection.")

    elif class_names[np.argmax(predictions)] == 'glaucoma':
        st.sidebar.warning(string)
        #st.markdown("## Remedy")
        #st.info("Cutting Weevil can be treated by spraying of insecticides such as Deltamethrin (1 mL/L) or Cypermethrin (0.5 mL/L) or Carbaryl (4 g/L) during new leaf emergence can effectively prevent the weevil damage.")



