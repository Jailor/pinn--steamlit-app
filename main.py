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
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score, mean_absolute_error, \
    mean_absolute_percentage_error
from my_pages.documentation import show_documentation_full, show_documentation_before_process_dataset, \
    show_documentation_before_process_models
from my_pages.data_exploration import show_data_exploration_statistics, \
    show_data_exploration_time_series_analysis_generic, show_data_exploration_statistics_generic
from my_pages.interactive_data_filtering import show_interactive_data_filtering
from my_pages.interactive_model_comparison import show_interactive_model_comparison, \
    show_interactive_model_comparison_generic
from my_pages.interactive_model_comparison_with_physics import show_interactive_model_comparison_with_physics, \
    show_interactive_model_comparison_with_physics_generic
from my_pages.performance_metrics import show_performance_metrics
from common import load_models, load_data, pinn_loss, get_feature_importances, \
    load_simplified_models, prompt_user_for_columns, process_new_dataset, train_and_evaluate_physics_model, \
    train_and_evaluate_model, reset_state_and_prompt, load_existing_datasets, process_existing_dataset, \
    prompt_user_for_partial_columns
import h5py
import io
import tempfile

load_existing_datasets()

if 'uploaded_file' not in st.session_state and 'selected_dataset' not in st.session_state:
    st.title('Upload a dataset to analyze')

    uploaded_file = None
    selected_dataset = None

    with st.form("initial_upload_file_form", clear_on_submit=True):
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="initial_file_uploader")
        submitted = st.form_submit_button("Upload")

    st.header("Or select an existing dataset:")
    selected_dataset = st.selectbox("Select a dataset", list(st.session_state['datasets'].keys()))
    submitted_select = st.button("Select")

    if st.session_state.get('error_message'):
        st.error(st.session_state['error_message'])
        st.session_state.pop('error_message')

    # If a file is uploaded, save it in the session state
    if submitted and uploaded_file is not None:
        try:
            initial_df = load_data(uploaded_file)
            st.session_state['initial_df'] = initial_df
            st.session_state['uploaded_file'] = uploaded_file
            # st.session_state['is_new_dataset'] = True
            st.rerun()
        except Exception as e:
            print(f"Error loading data: {e}")
            reset_state_and_prompt()

    # If an existing dataset is selected, save it in the session state
    if submitted_select and selected_dataset is not None:
        try:
            print(f"Selected dataset: {selected_dataset}")
            initial_df = st.session_state['datasets'][selected_dataset]
            initial_df = process_existing_dataset(initial_df, selected_dataset)
            st.session_state['initial_df'] = initial_df
            st.session_state['selected_dataset'] = selected_dataset
            # st.session_state['is_new_dataset'] = False
            st.rerun()
        except Exception as e:
            print(f"Error loading data: {e}")
            reset_state_and_prompt()


def get_optimizer(optimizer_name, learning_rate):
    if optimizer_name == 'Adam':
        return Adam(learning_rate=learning_rate)
    elif optimizer_name == 'SGD':
        return SGD(learning_rate=learning_rate)
    elif optimizer_name == 'RMSprop':
        return RMSprop(learning_rate=learning_rate)
    else:
        return Adam(learning_rate=learning_rate)


def train_and_evaluate_model(df, target_column, num_layers, neurons_per_layer, learning_rate,
                             optimizer_name, activation_function, epochs, batch_size, validation_split, train_test_size,
                             progress_bar, status_text, log_text, log_file_path):
    # Prepare the data
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_size, random_state=42)

    # Build the model
    model = Sequential()
    model.add(Dense(neurons_per_layer, input_dim=X_train.shape[1], activation=activation_function))
    for _ in range(num_layers - 1):
        model.add(Dense(neurons_per_layer, activation=activation_function))
    model.add(Dense(1, activation='linear'))

    optimizer = get_optimizer(optimizer_name, learning_rate)
    model.compile(optimizer=optimizer, loss='mean_absolute_error',
                  metrics=[
                      keras.metrics.RootMeanSquaredError(),
                      keras.metrics.MeanAbsoluteError()])

    # Callbacks
    custom_callback = StreamlitProgressBar(progress_bar, status_text, log_text, epochs, log_file_path)
    lr_scheduler = ReduceLROnPlateau(factor=0.66, patience=5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    model_checkpoint = ModelCheckpoint('best_classic_trained.keras', monitor='val_loss', save_best_only=True)

    # Train the model
    history = model.fit(X_train, y_train, validation_split=validation_split,
                        epochs=epochs, batch_size=256,
                        callbacks=[early_stopping, model_checkpoint, lr_scheduler, custom_callback])

    # Load the best model
    model = keras.models.load_model('best_classic_trained.keras')

    # Evaluate the model
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    mean_observed = y_test.mean()
    cvrmse = (rmse / mean_observed) * 100
    mae = mean_absolute_error(y_test, y_pred)
    mape = (mae / mean_observed) * 100

    print(f"Root Mean Squared Error: {rmse}")
    print(f"R^2 Score: {r2}")
    print(f"cvRMSE: {cvrmse}%")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Absolute Percentage Error: {mape}")

    return model


def train_and_evaluate_physics_model(df, target_column, num_layers, neurons_per_layer, learning_rate,
                                     optimizer_name, activation_function, epochs, batch_size, validation_split,
                                     train_test_size,
                                     progress_bar, status_text, log_text, log_file_path):
    alpha = 0.089

    def pinn_loss(y_true_with_features, y_pred):
        y_true = y_true_with_features[:, 0:1]
        outlet_water_temp = y_true_with_features[:, 1:2]
        inlet_temp = y_true_with_features[:, 2:3]
        water_flow = y_true_with_features[:, 3:4]

        specific_heat_capacity = 4174  # J/kg°C, specific heat capacity of water
        gallons_to_liters = 3.78541  # 1 gallon = 3.78541 liters
        fahrenheit_to_celsius = lambda f: (f - 32) * 5.0 / 9.0  # Convert °F to °C
        joules_to_btu = 0.0009478171
        flow_rate_liters = water_flow * gallons_to_liters
        inlet_temp_c = fahrenheit_to_celsius(inlet_temp)
        outlet_temp_c = fahrenheit_to_celsius(outlet_water_temp)
        heat_output = water_flow * (outlet_water_temp - inlet_temp) * 0.997 * 8.3077
        return tf.reduce_mean(tf.abs(y_true - y_pred) + alpha * tf.abs(heat_output - y_pred), axis=-1)

    # Prepare the data
    X = df.drop(columns=[target_column])
    y = df[target_column]

    columns = st.session_state['columns']

    outlet_water_temp = X.iloc[:, X.columns.get_loc(columns['outlet_water_temp'])].to_numpy().reshape(-1, 1)
    inlet_temp = X.iloc[:, X.columns.get_loc(columns['inlet_temp'])].to_numpy().reshape(-1, 1)
    water_flow = X.iloc[:, X.columns.get_loc(columns['water_flow'])].to_numpy().reshape(-1, 1)

    y_np = y.to_numpy().reshape(-1, 1)

    y_with_features = np.concatenate(
        [y_np, outlet_water_temp, inlet_temp, water_flow], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y_with_features, test_size=train_test_size, random_state=42)

    model = Sequential()
    model.add(Dense(neurons_per_layer, input_dim=X_train.shape[1], activation=activation_function))
    for _ in range(num_layers - 1):
        model.add(Dense(neurons_per_layer, activation=activation_function))
    model.add(Dense(1, activation='linear'))

    optimizer = get_optimizer(optimizer_name, learning_rate)
    model.compile(optimizer=optimizer, loss=pinn_loss, metrics=[pinn_loss])

    # Callbacks
    custom_callback = StreamlitProgressBar(progress_bar, status_text, log_text, epochs, log_file_path)
    lr_scheduler = ReduceLROnPlateau(factor=0.66, patience=5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    model_checkpoint = ModelCheckpoint('best_physics_trained.keras', monitor='val_loss', save_best_only=True)

    # Train the model
    history = model.fit(X_train, y_train, validation_split=validation_split,
                        epochs=epochs, batch_size=batch_size,
                        callbacks=[early_stopping, model_checkpoint, lr_scheduler, custom_callback])

    model.load_weights('best_physics_trained.keras')

    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test[:, 0], y_pred, squared=False)
    r2 = r2_score(y_test[:, 0], y_pred)
    mean_observed = y_test[:, 0].mean()
    cvrmse = (rmse / mean_observed) * 100
    mae = mean_absolute_error(y_test[:, 0], y_pred)
    mape = (mae * 100) / mean_observed

    print(f"Root Mean Squared Error: {rmse}")
    print(f"R^2 Score: {r2}")
    print(f"cvRMSE: {cvrmse}%")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Absolute Percentage Error: {mape}%")

    return model


class StreamlitProgressBar(Callback):
    def __init__(self, progress_bar, status_text, log_text, total_epochs, log_file_path):
        super().__init__()
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.log_text = log_text
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.logs = []
        self.log_file_path = log_file_path

    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch += 1
        progress = self.current_epoch / self.total_epochs
        self.progress_bar.progress(progress)
        self.status_text.text(f"Training in progress: {int(progress * 100)}% complete")

        # Format and append the current epoch log
        log_message = f"Epoch {epoch + 1}/{self.total_epochs}\n"
        log_message += " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()]) + "\n"
        self.logs.append(log_message)

        # Join all logs and display them in the text area
        full_log_message = "\n".join(self.logs)
        # self.log_text.text_area("Training Log", full_log_message, height=400, max_chars=None)
        self.log_text.text_area("Training Log", log_message, height=400, max_chars=None)

        if self.total_epochs == self.current_epoch:
            with open(self.log_file_path, 'a') as log_file:
                log_file.write("\n".join(self.logs))
                st.write("Training complete. Logs saved to file at:", self.log_file_path)
            self.log_text.text_area("Training Log", full_log_message, height=400, max_chars=None)


# If a file has been uploaded, use it
if 'uploaded_file' in st.session_state and 'initial_df' in st.session_state:
    initial_df = st.session_state['initial_df']

    if 'timestamp_column' not in st.session_state:
        prompt_user_for_partial_columns(initial_df)

    elif 'columns_selected' not in st.session_state:
        filtered_df = st.session_state['filtered_df_initial']
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ["Documentation and Explanation",
                                          "Data Exploration Statistics",
                                          "Data Exploration Time Series Analysis",
                                          "Process Dataset"])
        if 'is_reduced_dataset' in st.session_state and st.session_state['is_reduced_dataset']:
            st.sidebar.warning("The dataset has been automatically reduced to 50k elements to preserve performance.")

        if page == "Documentation and Explanation":
            show_documentation_before_process_dataset()
        elif page == "Data Exploration Statistics":
            show_data_exploration_statistics_generic(filtered_df)
        elif page == "Data Exploration Time Series Analysis":
            show_data_exploration_time_series_analysis_generic(filtered_df)
        elif page == "Process Dataset":
            prompt_user_for_columns(initial_df.copy())
        else:
            st.write("Not implemented yet")
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

        if 'model' not in st.session_state or 'pinn_model' not in st.session_state:
            st.sidebar.title("Navigation")
            page = st.sidebar.radio("Go to", ["Documentation and Explanation",
                                              "Data Exploration Statistics",
                                              "Data Exploration Time Series Analysis",
                                              "Train Models",
                                              "Upload Models"])
            feature_importance_df = get_feature_importances(processed_df, columns['heating_load'])

            if page == "Documentation and Explanation":
                show_documentation_before_process_models()

            elif page == "Data Exploration Statistics":
                show_data_exploration_statistics_generic(processed_df, feature_importance_df)

            elif page == "Data Exploration Time Series Analysis":
                show_data_exploration_time_series_analysis_generic(processed_df)
            elif page == "Train Models":
                # Hyperparameter inputs
                st.header("Hyperparameters")
                st.write("Choose the hyperparameters for training the models:")
                num_layers = st.slider("Number of Layers", 1, 10, 2)
                neurons_per_layer = st.number_input("Neurons per Layer", 5, 1024, 227, help="Between 5 and 1024")
                learning_rate = st.number_input("Learning Rate", 0.00001, 10.00, 0.0009323, format="%.6f",
                                                help="Between 0.00001 and 10.00")
                optimizer_name = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"])
                activation_function = st.selectbox("Activation Function", ["relu", "sigmoid", "tanh"])
                epochs = st.number_input("Epochs", 10, 1000, 100, help="Between 10 and 1000")
                batch_size = st.number_input("Batch Size", 16, 1024, 64, help="Between 16 and 1024")
                validation_split = st.slider("Reserved for Validation", 0.1, 0.4, 0.2)
                train_test_size = st.slider("Reserved for Testing", 0.1, 0.4, 0.2)

                classic_model = None
                pinn_model = None

                if 'model' not in st.session_state and st.button("Train Mdoels"):
                    st.write("Training classic model...")
                    classic_progress_bar = st.progress(0)
                    classic_status_text = st.empty()
                    classic_log_text = st.empty()

                    classic_model = train_and_evaluate_model(processed_df, columns['heating_load'], num_layers,
                                                             neurons_per_layer,
                                                             learning_rate, optimizer_name, activation_function, epochs,
                                                             batch_size, validation_split, train_test_size,
                                                             classic_progress_bar, classic_status_text,
                                                             classic_log_text, "logs/classic_model.log")

                    st.session_state['model'] = classic_model
                    st.write("Finished training classic model.")

                    st.write("Training PINN model...")
                    pinn_progress_bar = st.progress(0)
                    pinn_status_text = st.empty()
                    pinn_log_text = st.empty()
                    pinn_model = train_and_evaluate_physics_model(processed_df, columns['heating_load'], num_layers,
                                                                  neurons_per_layer,
                                                                  learning_rate, optimizer_name, activation_function,
                                                                  epochs, batch_size, validation_split, train_test_size,
                                                                  pinn_progress_bar, pinn_status_text, pinn_log_text,
                                                                  "logs/pinn_model.log")
                    st.session_state['pinn_model'] = pinn_model
                    st.write("Finished training PINN model.")
                    if 'model' in st.session_state and 'pinn_model' in st.session_state:
                        if st.button("Continue"):
                            st.rerun()
            elif page == "Upload Models":
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

            else:
                st.write("Not implemented yet")
        else:
            classic_model = st.session_state['model']
            pinn_model = st.session_state['pinn_model']
            feature_importance_df = get_feature_importances(processed_df, columns['heating_load'])

            st.sidebar.title("Navigation")
            page = st.sidebar.radio("Go to", ["Documentation and Explanation",
                                              "Data Exploration Statistics",
                                              "Data Exploration Time Series Analysis",
                                              "Interactive Model Comparison",
                                              "Performance Metrics", "Interactive Data Filtering"])
            if page == "Documentation and Explanation":
                show_documentation_full()

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

if 'selected_dataset' in st.session_state:
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
        show_documentation_full()

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
