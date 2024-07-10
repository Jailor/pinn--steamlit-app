import os
import tempfile
import streamlit as st
import tensorflow as tf
import keras
from keras.models import load_model

def show_upload_models():
    st.title("Upload Models")
    st.write("Upload the trained models here:")
    uploaded_model = st.file_uploader("Choose classic model file", type="keras", key="model_uploader")
    uploaded_pinn_model = st.file_uploader("Choose a physics-informed model file", type="keras",
                                           key="pinn_model_uploader")

    if uploaded_model and uploaded_pinn_model:
        try:
            # Create a temporary file for the classic model
            with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as temp_classic:
                temp_classic.write(uploaded_model.getvalue())
                classic_model_path = temp_classic.name

            # Create a temporary file for the PINN model
            with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as temp_pinn:
                temp_pinn.write(uploaded_pinn_model.getvalue())
                pinn_model_path = temp_pinn.name

            # Load the models from the temporary files after closing them
            classic_model = load_model(classic_model_path, compile=False)
            pinn_model = load_model(pinn_model_path, compile=False)

            st.session_state['model'] = classic_model
            st.session_state['pinn_model'] = pinn_model
            st.write("Models uploaded successfully.")

            # Clean up temporary files
            os.remove(classic_model_path)
            os.remove(pinn_model_path)
            st.write("Temporary files removed.")
            st.rerun()

        except Exception as e:
            st.error(f"Error loading models: {e}")