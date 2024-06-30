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
import keras
from threading import Lock
import time
import keras.backend as K
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score, mean_absolute_error, \
    mean_absolute_percentage_error
from my_pages.documentation import show_documentation
from my_pages.data_exploration import show_data_exploration_statistics, show_data_exploration_time_series_analysis_generic, show_data_exploration_statistics_generic
from my_pages.interactive_data_filtering import show_interactive_data_filtering
from my_pages.interactive_model_comparison import show_interactive_model_comparison, \
    show_interactive_model_comparison_generic
from my_pages.interactive_model_comparison_with_physics import show_interactive_model_comparison_with_physics, \
    show_interactive_model_comparison_with_physics_generic
from my_pages.performance_metrics import show_performance_metrics
from common import load_models, load_data, load_data_rossland_site1, pinn_loss, get_feature_importances, \
    load_simplified_models, prompt_user_for_columns, process_new_dataset, train_and_evaluate_physics_model, \
    train_and_evaluate_model


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
            initial_df, raw_df = load_data(uploaded_file)
            st.session_state['initial_df'] = initial_df
            st.session_state['raw_df'] = raw_df
            st.session_state['uploaded_file'] = uploaded_file
            st.rerun()
        except Exception as e:
            print(f"Error loading data: {e}")
            reset_state_and_prompt()

# If a file has been uploaded, use it
if 'uploaded_file' in st.session_state:
    initial_df = st.session_state['initial_df']
    raw_df = st.session_state['raw_df']
    rossland_df = load_data_rossland_site1()

    if not raw_df.equals(rossland_df):
        if 'columns_selected' not in st.session_state:
            st.title('New Dataset Detected')
            prompt_user_for_columns(initial_df)
        else:
            columns = st.session_state['columns']

            processed_df = None
            if 'processed_df' not in st.session_state:
                filtered_df = st.session_state['filtered_df']
                processed_df = process_new_dataset(filtered_df, columns['water_flow'], columns['outlet_water_temp'],
                                                   columns['inlet_temp'], columns['timestamp'], columns['heating_load'])
                st.session_state['processed_df'] = processed_df
            else:
                processed_df = st.session_state['processed_df']

            classic_model = None
            pinn_model = None

            # TODO: allow the user to choose some of the hyperparameters
            if 'model' not in st.session_state or 'pinn_model' not in st.session_state:
                st.write("Training classic model...")
                classic_model = train_and_evaluate_model(processed_df, columns['heating_load'])
                st.session_state['model'] = classic_model

                st.write("Training PINN model...")
                pinn_model = train_and_evaluate_physics_model(processed_df, columns['heating_load'])
                st.session_state['pinn_model'] = pinn_model
                st.rerun()
            else:
                classic_model = st.session_state['model']
                pinn_model = st.session_state['pinn_model']
                feature_importance_df = get_feature_importances(processed_df, columns['heating_load'])

                st.sidebar.title("Navigation")
                page = st.sidebar.radio("Go to", ["Documentation and Explanation",
                                                  "Data Exploration Statistics", "Data Exploration Time Series Analysis",
                                                  "Interactive Model Comparison",
                                                  "Performance Metrics", "Interactive Data Filtering"])
                if page == "Documentation and Explanation":
                    show_documentation()

                elif page == "Data Exploration Statistics":
                    show_data_exploration_statistics_generic(processed_df, feature_importance_df)

                elif page == "Data Exploration Time Series Analysis":
                    show_data_exploration_time_series_analysis_generic(processed_df)

                elif page == "Interactive Model Comparison":
                    show_interactive_model_comparison_generic(classic_model, pinn_model, processed_df)

                # todo: train and evaluate the simplified models
                elif page == "Interactive Model Comparison with Physics":
                    st.write("Not implemented yet")
                    # show_interactive_model_comparison_with_physics_generic(classic_model, pinn_model)

                elif page == "Performance Metrics":
                    show_performance_metrics(processed_df, classic_model, pinn_model, columns['heating_load'])

                elif page == "Interactive Data Filtering":
                    show_interactive_data_filtering(processed_df, classic_model, pinn_model, columns['heating_load'])

                else:
                    st.write("Not implemented yet")
    else:
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
            show_data_exploration_time_series_analysis_generic(initial_df)

        elif page == "Interactive Model Comparison":
            show_interactive_model_comparison(standard_model, pinn_model)

        elif page == "Interactive Model Comparison with Physics":
            show_interactive_model_comparison_with_physics(standard_model_simple, pinn_model_simple)

        elif page == "Performance Metrics":
            show_performance_metrics(initial_df, standard_model, pinn_model)

        elif page == "Interactive Data Filtering":
            show_interactive_data_filtering(initial_df, standard_model, pinn_model)

        else:
            st.write("Not implemented yet")
