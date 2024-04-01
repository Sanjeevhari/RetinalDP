import streamlit as st

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
