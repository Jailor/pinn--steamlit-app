import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import numpy as np
import pandas as pd
import logging
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
import keras as k
from keras.models import load_model
from threading import Lock
import time
import keras.backend as K
from keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from my_pages.documentation import show_documentation
from my_pages.data_exploration import show_data_exploration_statistics, show_data_exploration_time_series_analysis
from my_pages.interactive_data_filtering import show_interactive_data_filtering
from my_pages.interactive_model_comparison import show_interactive_model_comparison
from my_pages.interactive_model_comparison_with_physics import show_interactive_model_comparison_with_physics
from my_pages.performance_metrics import show_performance_metrics
from common import load_models, load_data, pinn_loss, get_feature_importances, load_simplified_models

def reset_state_and_prompt():
    st.session_state.pop('uploaded_file', None)
    st.session_state.pop('initial_df', None)
    st.session_state['error_message'] = "This dataset is not valid. Please upload a valid CSV file."
    st.rerun()

if 'uploaded_file' not in st.session_state:
    st.title('Choose a dataset to analyze')

    uploaded_file = None
    with st.form("my-form", clear_on_submit=True):
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        submitted = st.form_submit_button("Upload")

    if st.session_state.get('error_message'):
        st.error(st.session_state['error_message'])
        st.session_state.pop('error_message')


    # If a file is uploaded, save it in the session state
    if submitted and uploaded_file is not None:
        try:
            initial_df = load_data(uploaded_file)
            st.session_state['initial_df'] = initial_df
            st.session_state['uploaded_file'] = uploaded_file
            st.rerun()
        except Exception as e:
            print(f"Error loading data: {e}")
            reset_state_and_prompt()


# If a file has been uploaded, use it
if 'uploaded_file' in st.session_state:
    initial_df = None
    standard_model = None
    pinn_model = None
    feature_importance_df = None
    standard_model_simple = None
    pinn_model_simple = None
    try:
        initial_df = st.session_state['initial_df']
        standard_model, pinn_model = load_models()
        standard_model_simple, pinn_model_simple = load_simplified_models()
        feature_importance_df = get_feature_importances(initial_df)
    except Exception as e:
        print(f"Error processing data or loading models: {e}")
        reset_state_and_prompt()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Documentation and Explanation",
                                      "Data Exploration Statistics", "Data Exploration Time Series Analysis",
                                      "Interactive Model Comparison", "Interactive Model Comparison with Physics",
                                      "Performance Metrics", "Interactive Data Filtering"])

    if page == "Documentation and Explanation":
        show_documentation()

    elif page == "Data Exploration Statistics":
        show_data_exploration_statistics(initial_df, feature_importance_df)

    elif page == "Data Exploration Time Series Analysis":
        show_data_exploration_time_series_analysis(initial_df)

    elif page == "Interactive Model Comparison":
        show_interactive_model_comparison(standard_model, pinn_model)
    # TODO: train some models only with the physics-related data
    elif page == "Interactive Model Comparison with Physics":
        show_interactive_model_comparison_with_physics(standard_model_simple, pinn_model_simple)

    elif page == "Performance Metrics":
        show_performance_metrics(initial_df, standard_model, pinn_model)

    elif page == "Interactive Data Filtering":
        show_interactive_data_filtering(initial_df, standard_model, pinn_model)

