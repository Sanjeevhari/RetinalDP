import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow import argmax
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import cv2

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

def save_uploaded_file(uploaded_file):
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    return "temp_image.jpg"

def grad_cam(fname):
    DIM = 224
    img = tf.keras.preprocessing.image.load_img(fname, target_size=(DIM, DIM))
    st.image(img)
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
    plt.matshow(heatmap)
    plt.axis('off')  # Turn off axis
    st.pyplot()

    #img = cv2.imread(fname)
    #img = cv2.imdecode(np.fromstring(fname.read(), np.uint8), 1)
    #file_path = save_uploaded_file(fname)
    #image = cv2.imread(file_path)
    #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = tf.keras.preprocessing.image.img_to_array(img)
    #st.image(img)

    INTENSITY = 0.5
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    img1 = heatmap * INTENSITY + img

    #col2.image(img1)
    st.image(img1, clamp=True, channels='RGB')

def predict(image_file):
  img = tf.keras.preprocessing.image.load_img(image_file, target_size=(224, 224))
  img_array = tf.keras.preprocessing.image.img_to_array(img)
  #img_array = img_array / 255.0
  img_batch = np.expand_dims(img_array, axis=0)
  predictions = model.predict(img_batch)
  predicted_class = argmax(predictions[0])
  return labels[predicted_class]

file = st.file_uploader("", type=["jpg", "png"])
col1, col2 = st.columns(2)
if file is None:
    st.text("Please upload an image file")
else:
    #image_data = file.read()
    #col1.image(image)
    #prediction = predict(file)
    grad_cam(file)
    #st.write(f"Predicted Disease: {prediction}")

    string = "Detected Disease : " + prediction
    if prediction == 'normal':
        st.balloons()
        st.sidebar.success(string)

    elif prediction == 'cataract':
        st.sidebar.warning(string)
        #st.markdown("## Remedy")
        #st.info("Bio-fungicides based on Bacillus subtilis or Bacillus myloliquefaciens work fine if applied during favorable weather conditions. Hot water treatment of seeds or fruits (48°C for 20 minutes) can kill any fungal residue and prevent further spreading of the disease in the field or during transport.")

    elif prediction == 'diabetic_retinopathy':
        st.sidebar.warning(string)
        #st.markdown("## Remedy")
        #st.info("Prune flowering trees during blooming when wounds heal fastest. Remove wilted or dead limbs well below infected areas. Avoid pruning in early spring and fall when bacteria are most active.If using string trimmers around the base of trees avoid damaging bark with breathable Tree Wrap to prevent infection.")

    elif prediction == 'glaucoma':
        st.sidebar.warning(string)
        #st.markdown("## Remedy")
        #st.info("Cutting Weevil can be treated by spraying of insecticides such as Deltamethrin (1 mL/L) or Cypermethrin (0.5 mL/L) or Carbaryl (4 g/L) during new leaf emergence can effectively prevent the weevil damage.")



