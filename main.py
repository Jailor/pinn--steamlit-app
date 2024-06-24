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
from my_pages.data_exploration import show_data_exploration
from my_pages.interactive_data_filtering import show_interactive_data_filtering
from my_pages.interactive_model_comparison import show_interactive_model_comparison
from my_pages.interactive_model_comparison_with_physics import show_interactive_model_comparison_with_physics
from my_pages.performance_metrics import show_performance_metrics


# Load models

def pinn_loss(y_true_with_features, y_pred):
    pass


# TODO: see about the custom loss function and further training
# alpha = 0.1
# def pinn_loss(y_true_with_features, y_pred):
#     y_true = y_true_with_features[:, 0:1]
#     outlet_water_temp = y_true_with_features[:, 1:2]
#     inlet_water_temp = y_true_with_features[:, 2:3]
#     inlet_temp = y_true_with_features[:, 3:4]
#     water_flow = y_true_with_features[:, 4:5]
#     watts = y_true_with_features[:, 5:6]
#     heat_trace = y_true_with_features[:, 6:7]
#     specific_heat_capacity = 4174  # J/kg°C, specific heat capacity of water
#     gallons_to_liters = 3.78541  # 1 gallon = 3.78541 liters
#     fahrenheit_to_celsius = lambda f: (f - 32) * 5.0 / 9.0  # Convert °F to °C
#     joules_to_btu = 0.0009478171
#     flow_rate_liters = water_flow * gallons_to_liters
#     inlet_temp_c = fahrenheit_to_celsius(inlet_temp)
#     outlet_temp_c = fahrenheit_to_celsius(outlet_water_temp)
#     heat_output = water_flow * (outlet_water_temp - inlet_temp) * 0.997 * 8.3077
#     return K.mean(K.abs(y_true - y_pred) + alpha * K.abs(heat_output - y_pred), axis=-1)


# TODO: retrain the models
@st.cache_resource
def load_models():
    standard_model = load_model('best_model.h5')
    pinn_model = load_model('pinn_model.h5', custom_objects={'pinn_loss': pinn_loss}, compile=False)

    # TODO: recompiling for further training?
    # standard_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_absolute_error',
    #               metrics=[
    #                   tf.keras.metrics.RootMeanSquaredError(),
    #                   tf.keras.metrics.R2Score(),
    #                   tf.keras.metrics.MeanAbsoluteError()
    #               ])
    # pinn_model.compile(optimizer=Adam(learning_rate=0.001), loss=pinn_loss,
    #                    metrics=[
    #                        tf.keras.metrics.RootMeanSquaredError(),
    #                        tf.keras.metrics.MeanAbsoluteError(),
    #                        tf.keras.metrics.MeanAbsoluteError()])

    return standard_model, pinn_model


@st.cache_resource
def load_data(uploaded_file):
    df = None
    try:
        # Check if the file is not empty
        if uploaded_file.size == 0:
            st.error("The uploaded file is empty. Please upload a valid CSV file.")
            return None

        # Try reading the file with the specified delimiter
        df = pd.read_csv(uploaded_file, sep=';', low_memory=False)
        if df.empty:
            st.error("No data found in the file. Please check the file content.")
            return None
    except pd.errors.EmptyDataError:
        st.error("The uploaded file has no columns to parse. Please check the delimiter and file format.")
        return None
    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
        return None

    # Drop bad data
    df.drop('ID', axis=1, inplace=True)
    df.drop('Uncorrected Water Flow (Gallons)', axis=1, inplace=True)
    df.drop('Uncorrected Hot Water Temp (F)', axis=1, inplace=True)
    df.drop('Uncorrected Cold Water Temp (F)', axis=1, inplace=True)
    df.drop('Site', axis=1, inplace=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)

    df['Watts'] = df['Watts'].astype(str)
    df['Water Heating Load (Btu)'] = df['Water Heating Load (Btu)'].astype(str)

    # Convert non-numeric values to NaN
    df['Watts'] = pd.to_numeric(df['Watts'], errors='coerce')
    df['Water Heating Load (Btu)'] = pd.to_numeric(df['Water Heating Load (Btu)'], errors='coerce')
    df.dropna(inplace=True)

    # Resample the data to hourly intervals
    df = df.resample('h').agg({
        'Hot Water Temp (F)': 'mean',
        'Cold Water Temp (F)': 'mean',
        'Water Flow (Gallons)': 'sum',
        'Inlet Temp (F)': 'mean',
        'Watts': 'mean',
        'Heat Trace (W)': 'mean',
        'Water Heating Load (Btu)': 'sum'
    })
    df.dropna(inplace=True)
    return df


@st.cache_data
def get_feature_importances(initial_df):
    X = initial_df.drop(columns=['Water Heating Load (Btu)'])
    y = initial_df['Water Heating Load (Btu)']
    model = RandomForestRegressor()
    model.fit(X, y)
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    return feature_importance_df



if 'uploaded_file' not in st.session_state:
    st.title('Choose a dataset to analyze')
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    # If a file is uploaded, save it in the session state
    if uploaded_file is not None:
        initial_df = load_data(uploaded_file)
        st.session_state['initial_df'] = initial_df
        st.session_state['uploaded_file'] = uploaded_file
        st.rerun()

# If a file has been uploaded, use it
if 'uploaded_file' in st.session_state:
    initial_df = st.session_state['initial_df']
    standard_model, pinn_model = load_models()
    feature_importance_df = get_feature_importances(initial_df)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Documentation and Explanation", "Data Exploration",
                                      "Interactive Model Comparison", "Interactive Model Comparison with Physics",
                                      "Performance Metrics", "Interactive Data Filtering"])

    if page == "Interactive Model Comparison":
        show_interactive_model_comparison(standard_model, pinn_model)

    # TODO: train some models only with the physics-related data
    elif page == "Interactive Model Comparison with Physics":
        show_interactive_model_comparison_with_physics(standard_model, pinn_model)

    elif page == "Performance Metrics":
        show_performance_metrics(initial_df, standard_model, pinn_model)

    elif page == "Data Exploration":
        show_data_exploration(initial_df, feature_importance_df)

    elif page == "Interactive Data Filtering":
        show_interactive_data_filtering(initial_df, standard_model, pinn_model)

    elif page == "Documentation and Explanation":
        show_documentation()
