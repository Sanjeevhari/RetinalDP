import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow import argmax
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
from streamlit_image_select import image_select

button_style = """
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 10px;
"""

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

labels= ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

def grad_cam(fname):
    DIM = 224
    img = tf.keras.preprocessing.image.load_img(fname, target_size=(DIM, DIM))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer('conv5_block16_concat')
        iterate = tf.keras.models.Model([model.input], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(x)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.reshape((7, 7))

    img = tf.keras.preprocessing.image.load_img(fname)
    img = tf.keras.preprocessing.image.img_to_array(img)
    alpha=1.1
    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    img1 = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    col2.image(img1)#,use_column_width="always")
    #st.image(img1)


def predict(image_file):
  img = tf.keras.preprocessing.image.load_img(image_file, target_size=(224, 224))
  img_array = tf.keras.preprocessing.image.img_to_array(img)
  #img_array = img_array / 255.0
  img_batch = np.expand_dims(img_array, axis=0)
  predictions = model.predict(img_batch)
  predicted_class = argmax(predictions[0])
  return labels[predicted_class]

upload_img = st.file_uploader("", type=["jpg", "png"])

file = image_select(
    label="Please upload or select an image file",
    images=[
        "Images/10015_left.jpg",
        "Images/1020_left.jpg",
        "Images/1034_left.jpg",
        "Images/112_right.jpg",
    ],
)
class_btn = st.button("Classify", style=button_style)
col1, col2 = st.columns(2)

if class_btn:
    if file is None and upload_img is None:
        st.text("Please upload or select an image file")
    else:
        if upload_img is not None:
            col1.image(upload_img.read())
            file=upload_img
        else:
            col1.image(file)
        prediction = predict(file)
        grad_cam(file)
        #st.write(f"Predicted Disease: {prediction}")

        string = "Detected Disease : " + prediction
        if prediction == 'Normal':
            st.balloons()
            st.sidebar.success(string)

        elif prediction == 'Cataract':
            st.sidebar.warning(string)
            #st.markdown("## Remedy")
            #st.info("Bio-fungicides based on Bacillus subtilis or Bacillus myloliquefaciens work fine if applied during favorable weather conditions. Hot water treatment of seeds or fruits (48°C for 20 minutes) can kill any fungal residue and prevent further spreading of the disease in the field or during transport.")

        elif prediction == 'Diabetic Retinopathy':
            st.sidebar.warning(string)
            #st.markdown("## Remedy")
            #st.info("Prune flowering trees during blooming when wounds heal fastest. Remove wilted or dead limbs well below infected areas. Avoid pruning in early spring and fall when bacteria are most active.If using string trimmers around the base of trees avoid damaging bark with breathable Tree Wrap to prevent infection.")

        elif prediction == 'Glaucoma':
            st.sidebar.warning(string)
            #st.markdown("## Remedy")
            #st.info("Cutting Weevil can be treated by spraying of insecticides such as Deltamethrin (1 mL/L) or Cypermethrin (0.5 mL/L) or Carbaryl (4 g/L) during new leaf emergence can effectively prevent the weevil damage.")



