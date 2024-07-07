import os
import tempfile

import pandas as pd
import streamlit as st
import tensorflow as tf
import keras
from keras.models import load_model
from sklearn.ensemble import RandomForestRegressor
import time
import difflib
import keras.backend as K
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from deap import base, creator, tools, algorithms
from deap import tools
import random
import matplotlib.pyplot as plt
import seaborn as sns


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
            with open(self.log_file_path, 'w') as log_file:
                log_file.write("\n".join(self.logs))
                st.write("Training complete. Logs saved to file at:", self.log_file_path)
            self.log_text.text_area("Training Log", full_log_message, height=400, max_chars=None)



@st.cache_resource
def load_models():

    standard_model = load_model('models/classic_full.h5')
    pinn_model = load_model('models/pinn_full.h5', custom_objects={'pinn_loss': pinn_loss}, compile=False)

    return standard_model, pinn_model

@st.cache_resource
def load_simplified_models():
    standard_model = load_model('models/phys/nn_1024.h5')  # classic_simple
    pinn_model = load_model('models/phys/phys_simple.h5', custom_objects={'pinn_loss': pinn_loss},
                            compile=False)  # best_model_phys_only

    return standard_model, pinn_model

def pinn_loss(y_true_with_features, y_pred):
        pass


def reset_state_and_prompt():
    st.session_state.pop('uploaded_file', None)
    st.session_state.pop('initial_df', None)
    st.session_state['error_message'] = "This dataset is not valid. Please upload a valid CSV file."
    st.rerun()


def load_existing_datasets():
    if 'datasets' in st.session_state:
        return
    rossland_original = load_dataset('rossland_original')
    # rossland_renamed_df = load_dataset('rossland_renamed')
    # cleaned_v1_e350 = load_dataset('cleaned_v1_e350')
    st.session_state['rossland_df'] = rossland_original
    st.session_state['datasets'] = {
        'Rossland Site 1 E350': rossland_original,
        # 'Rossland Site 1 renamed E350': rossland_renamed_df,
        # 'Victoria Island 1 E350': cleaned_v1_e350
    }


def load_dataset(dataset_name):
    try:
        df = pd.read_csv(f'datasets/{dataset_name}.csv', low_memory=False)
        return df
    except Exception as e:
        print(f"An error occurred while reading Rossland: {e}")
        return None


@st.cache_resource
def load_data(uploaded_file):
    df = None
    try:
        # Check if the file is not empty
        if uploaded_file.size == 0:
            st.error("The uploaded file is empty. Please upload a valid CSV file.")
            return None

        # Try reading the file with the specified delimiter
        df = pd.read_csv(uploaded_file, low_memory=False)
        if df.empty:
            print("No data found in the file. Please check the file content.")
            return None
    except pd.errors.EmptyDataError:
        print("The uploaded file has no columns to parse. Please check the delimiter and file format.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

    return df


def process_existing_dataset(df, dataset_name):
    if dataset_name == 'Rossland Site 1 E350':
        # Directly process the Rossland site 1 data according to predefined rules
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
    else:
        return df


@st.cache_data
def get_feature_importances(initial_df, heating_load_column='Water Heating Load (Btu)'):
    X = initial_df.drop(columns=[heating_load_column])
    y = initial_df[heating_load_column]
    model = RandomForestRegressor()
    model.fit(X, y)
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    return feature_importance_df


def get_most_similar_column(target, columns):
    matches = difflib.get_close_matches(target, columns, n=1, cutoff=0.0)
    return matches[0] if matches else None


def prompt_user_for_columns(df):
    st.title('New Dataset Detected')
    st.subheader("Dataset Columns")
    st.markdown("""
            Please select the columns to keep and map the necessary columns for the analysis. The kept columns must necessarily match 
             at least the following columns in order to apply physics-informed machine learning for heating load:
              - **water flow** => The amount of water entering the HPWH system.
              - **outlet water temp** => The temperature of the water leaving the HPWH system.
              - **inlet temp** => The temperature of the water entering the HPWH system.
              - **timestamp** => The timestamp of the data. Should be in format 'YYYY-MM-DD HH:MM:SS'.
              - **heating load** => The heating load of the HPWH system.
              """)

    columns_to_keep = st.multiselect("Select columns to keep (Minimum 5)", options=df.columns,
                                     default=df.columns.tolist())

    if len(columns_to_keep) < 5:
        st.error("Please select at least 5 columns to proceed.")
        return

    df = df[columns_to_keep]

    # Find the most similar column names
    default_water_flow_column = get_most_similar_column("water flow", columns_to_keep)
    default_outlet_water_temp_column = get_most_similar_column("outlet water temp", columns_to_keep)
    default_inlet_temp_column = get_most_similar_column("inlet temp", columns_to_keep)
    default_timestamp_column = get_most_similar_column("timestamp", columns_to_keep)
    default_heating_load_column = get_most_similar_column("heating load", columns_to_keep)

    submitted = False
    with st.form("column_mapping"):
        water_flow_column = st.selectbox("Select the column for water flow", options=columns_to_keep,
                                         index=columns_to_keep.index(
                                             default_water_flow_column) if default_water_flow_column else 0)
        outlet_water_temp_column = st.selectbox("Select the column for outlet water temperature",
                                                options=columns_to_keep, index=columns_to_keep.index(
                default_outlet_water_temp_column) if default_outlet_water_temp_column else 0)
        inlet_temp_column = st.selectbox("Select the column for inlet temperature", options=columns_to_keep,
                                         index=columns_to_keep.index(
                                             default_inlet_temp_column) if default_inlet_temp_column else 0)
        timestamp_column = st.selectbox("Select the column for timestamp", options=columns_to_keep,
                                        index=columns_to_keep.index(
                                            default_timestamp_column) if default_timestamp_column else 0)
        heating_load_column = st.selectbox("Select the column for heating load", options=columns_to_keep,
                                           index=columns_to_keep.index(
                                               default_heating_load_column) if default_heating_load_column else 0)
        submitted = st.form_submit_button("Submit")

    if submitted:
        st.session_state['columns_selected'] = True
        st.session_state['columns'] = {
            'water_flow': water_flow_column,
            'outlet_water_temp': outlet_water_temp_column,
            'inlet_temp': inlet_temp_column,
            'timestamp': timestamp_column,
            'heating_load': heating_load_column
        }
        st.session_state['filtered_df'] = df
        st.rerun()


@st.cache_data
def process_new_dataset(df, water_flow_column, outlet_water_temp_column, inlet_temp_column, timestamp_column,
                        heating_load_column):
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    df.set_index(timestamp_column, inplace=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop columns that are entirely NaN
    df.dropna(axis=1, how='all', inplace=True)

    # Create a dictionary for aggregation
    agg_dict = {
        water_flow_column: 'sum',
        outlet_water_temp_column: 'mean',
        inlet_temp_column: 'mean',
        heating_load_column: 'sum'
    }

    # Add other columns to be resampled by their mean
    for col in df.columns:
        if col not in agg_dict:
            agg_dict[col] = 'mean'

    # Resample the data to hourly intervals
    df = df.resample('h').agg(agg_dict)
    df.dropna(inplace=True)
    return df


def prompt_user_for_partial_columns(df):
    st.title('New Dataset Detected')
    st.write("Please select the columns for time stamp and for the heating load.")

    # Find the most similar column names
    columns = df.columns.tolist()

    default_timestamp_column = get_most_similar_column("timestamp", columns)
    default_heating_load_column = get_most_similar_column("heating load", columns)

    submitted = False
    with st.form("column_mapping_initial"):

        timestamp_column = st.selectbox("Select the column for timestamp", options=columns,
                                        index=columns.index(
                                            default_timestamp_column) if default_timestamp_column else 0)
        heating_load_column = st.selectbox("Select the column for heating load", options=columns,
                                           index=columns.index(
                                               default_heating_load_column) if default_heating_load_column else 0)
        submitted = st.form_submit_button("Submit")

    if submitted:
        df_copy = df.copy()

        df_copy[timestamp_column] = pd.to_datetime(df_copy[timestamp_column])
        df_copy.set_index(timestamp_column, inplace=True)

        for col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

        # Drop columns that are entirely NaN
        df_copy.dropna(axis=1, how='all', inplace=True)

        # Reduce df_copy to a maximum of 50000 entries
        if len(df_copy) > 50000:
            df_copy = df_copy.iloc[:50000]
            st.session_state['is_reduced_dataset'] = True
        else:
            st.session_state['is_reduced_dataset'] = False

        st.session_state['filtered_df_initial'] = df_copy
        st.session_state['timestamp_column'] = timestamp_column
        st.session_state['heating_load_column'] = heating_load_column
        st.rerun()


def show_train_models(processed_df, columns):
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

def get_distances(standard_pred, pinn_pred, heat_output_physics):
    d1 = abs(standard_pred[0][0] - heat_output_physics[0])
    d2 = abs(pinn_pred[0][0] - heat_output_physics[0])
    if d2 > d1:
        d1, d2 = d2, d1
        standard_pred[0][0], pinn_pred[0][0] = pinn_pred[0][0], standard_pred[0][0]
    return d1, d2


def plot_population_diversity(population, generation):
    num_layers = [ind[0] for ind in population]
    neurons_per_layer = [ind[1] for ind in population]
    learning_rates_power = [ind[2] for ind in population]

    optimizers = [ind[3] for ind in population]
    activation_functions = [ind[4] for ind in population]

    layer_counts = [num_layers.count(i) for i in range(1, 5)]

    fig, axs = plt.subplots(5, 1, figsize=(10, 20))

    axs[0].bar(range(1, 5), layer_counts, edgecolor='black')
    axs[0].set_title(f'Number of Layers Distribution (Generation {generation})')
    axs[0].set_xlabel('Number of Layers')
    axs[0].set_ylabel('Frequency')
    axs[0].set_xticks(range(1, 5))

    axs[1].hist(neurons_per_layer, bins=range(32, 513, 48), edgecolor='black')
    axs[1].set_title(f'Neurons Per Layer Distribution (Generation {generation})')
    axs[1].set_xlabel('Neurons Per Layer')
    axs[1].set_ylabel('Frequency')

    axs[2].hist(learning_rates_power, bins=12, edgecolor='black')
    axs[2].set_title(f'Learning Rates Distribution (Generation {generation})')
    axs[2].set_xlabel('Learning Rate Power')
    axs[2].set_ylabel('Frequency')

    optimizer_counts = {opt: optimizers.count(opt) for opt in set(optimizers)}
    axs[3].bar(optimizer_counts.keys(), optimizer_counts.values(), color='blue', edgecolor='black')
    axs[3].set_title(f'Distribution of Optimizers (Generation {generation})')
    axs[3].set_xlabel('Optimizer')
    axs[3].set_ylabel('Frequency')

    activation_counts = {act: activation_functions.count(act) for act in set(activation_functions)}
    axs[4].bar(activation_counts.keys(), activation_counts.values(), color='blue', edgecolor='black')
    axs[4].set_title(f'Distribution of Activation Functions (Generation {generation})')
    axs[4].set_xlabel('Activation Function')
    axs[4].set_ylabel('Frequency')

    plt.tight_layout()
    return fig


def plot_fitness_evolution(best_fitness):
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness, label='Best MAE', color='red')
    # plt.plot(avg_fitness, label='Average MAE', color='blue')
    plt.xlabel('Generation')
    plt.ylabel('MAE')
    plt.title('Evolution of MAE over Generations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt.gcf()

class GeneticAlgorithmProgress:
    def __init__(self, progress_bar, status_text, log_text, log_file_path, diversity_plot, fitness_plot):
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.log_text = log_text
        self.log_file_path = log_file_path
        self.diversity_plot = diversity_plot
        self.fitness_plot = fitness_plot
        self.total_generations = 0
        self.current_generation = 0
        self.logs = []
        self.most_recent_diversity_plot = None
        self.most_recent_fitness_plot = None

    def reset(self, total_generations):
        self.total_generations = total_generations
        self.current_generation = 0
        self.logs = []
        self.progress_bar.progress(0)
        self.status_text.text("Genetic Algorithm in progress: 0% complete (Generation 0)")
        self.log_text.text_area("Genetic Algorithm Log", "No results yet", height=400, max_chars=None)
        self.diversity_plot.empty()
        self.fitness_plot.empty()
        self.most_recent_diversity_plot = None
        self.most_recent_fitness_plot = None

    def on_generation_end(self, generation, logs,  population_diversity_plot, best_fitness_plot):
        self.current_generation += 1
        progress = self.current_generation / self.total_generations
        self.progress_bar.progress(progress)
        self.status_text.text(
            f"Genetic Algorithm in progress: {int(progress * 100)}% complete (Generation {generation + 1})")

        # Format and append the current generation log
        log_message = f"Generation {generation + 1}/{self.total_generations}\n"
        formatted_logs = []
        for k, v in logs.items():
            if isinstance(v, (float, int)):
                formatted_logs.append(f"{k}: {v:.4f}")
            else:
                formatted_logs.append(f"{k}: {v}")
        log_message += " - ".join(formatted_logs) + "\n"
        self.logs.append(log_message)

        # Join all logs and display them in the text area
        full_log_message = "\n".join(self.logs)
        self.log_text.text_area("Genetic Algorithm Log", full_log_message, height=400, max_chars=None)

        self.diversity_plot.pyplot(population_diversity_plot)
        self.fitness_plot.pyplot(best_fitness_plot)
        self.most_recent_diversity_plot = population_diversity_plot
        self.most_recent_fitness_plot = best_fitness_plot

        if self.total_generations == self.current_generation:
            self._save_logs()

    def on_user_end(self):
        self.status_text.text("Genetic Algorithm stopped by user at generation " + str(self.current_generation + 1))
        self.progress_bar.progress(1.0)
        full_log_message = "\n".join(self.logs)
        self.log_text.text_area("Genetic Algorithm Log", full_log_message, height=400, max_chars=None)
        if self.most_recent_diversity_plot is not None and self.most_recent_fitness_plot is not None:
            self.diversity_plot.pyplot(self.most_recent_diversity_plot)
            self.fitness_plot.pyplot(self.most_recent_fitness_plot)
        self._save_logs()

    def _save_logs(self):
        with open(self.log_file_path, 'w') as log_file:
            log_file.write("\n".join(self.logs))
        st.write("Logs saved to file at:", self.log_file_path)


def is_valid(individual):
    num_layers, neurons_per_layer, lr, optimizer_name, activation_function = individual
    # Check for valid ranges and types
    if not (1 <= num_layers <= 4):
        return False
    if not (32 <= neurons_per_layer <= 512):
        return False
    if not (-4 <= lr <= -1):
        return False
    if optimizer_name not in ['adam', 'sgd', 'rmsprop']:
        return False
    if activation_function not in ['relu', 'tanh']:
        return False
    return True

def genetic_algorithm(df, NGEN, EPOCHS, POPULATION_SIZE, BATCH_SIZE, target_column = 'Water Heating Load (Btu)'):

    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    optimizers = {'adam': Adam, 'sgd': SGD, 'rmsprop': RMSprop}

    # Define the genetic algorithm functions
    def create_model(num_layers, neurons_per_layer, learning_rate, optimizer_name, activation_function):
        optimizer_class = optimizers[optimizer_name]
        model = Sequential()
        model.add(Dense(neurons_per_layer, input_dim=X_train.shape[1], activation=activation_function))
        for _ in range(num_layers - 1):
            model.add(Dense(neurons_per_layer, activation=activation_function))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=optimizer_class(learning_rate=learning_rate), loss='mean_absolute_error')
        return model

    def evaluate_individual(individual):
        if st.session_state.stopGeneticAlgorithm:
            return (1e7,)
        if not is_valid(individual):
            print(f"1 Failed to train or predict with {individual}")
            return (1e7,),

        num_layers, neurons_per_layer, lr, optimizer_name, activation_function = individual
        model = create_model(num_layers, neurons_per_layer, 10 ** lr, optimizer_name, activation_function)
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        predictions = model.predict(X_test)

        if np.any(np.isnan(predictions)):
            print(f"2 Failed to train or predict with {individual}")
            return (1e7,)

        mae = mean_absolute_error(y_test, predictions)
        return (mae,)

    # Define low and up bounds for numerical attributes
    low = [1, 32, -4]
    up = [4, 512, -1]

    def mutate_individual(individual, num_attrs, cat_attrs, indpb):
        # Mutate numerical attributes
        for i in range(num_attrs):
            if random.random() < indpb:
                if i < 2:
                    individual[i] = int(random.uniform(low[i], up[i]))
                else:
                    individual[i] = random.uniform(low[i], up[i])

        # Mutate categorical attributes
        for i in range(num_attrs, num_attrs + cat_attrs):
            if random.random() < indpb:
                individual[i] = random.choice(['adam', 'sgd', 'rmsprop'] if i == num_attrs else ['relu', 'tanh'])

        return individual,

    # Setup DEAP
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_num_layers", random.randint, 1, 4)
    toolbox.register("attr_neurons_per_layer", random.randint, 32, 512)
    toolbox.register("attr_learning_rate", random.uniform, -4, -1)
    toolbox.register("attr_optimizer", random.choice, ['adam', 'sgd', 'rmsprop'])
    toolbox.register("attr_activation", random.choice, ['relu', 'tanh'])

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_num_layers, toolbox.attr_neurons_per_layer, toolbox.attr_learning_rate,
                      toolbox.attr_optimizer, toolbox.attr_activation),
                     n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize=9)

    # Register the custom mutation function
    toolbox.register("mutate", mutate_individual, num_attrs=3, cat_attrs=2, indpb=0.333)

    # Genetic Algorithm parameters
    population = toolbox.population(n=POPULATION_SIZE)

    all_generations = []
    best_fitness = []
    avg_fitness = []

    for gen in range(NGEN):
        if st.session_state.stopGeneticAlgorithm:
            return
        print("Generation ", gen)
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
        best_ind = tools.selBest(population, 1)[0]
        print(
            f"Best individual this gen: Layers: {best_ind[0]}, Neurons: {best_ind[1]}, Learning Rate: {10 ** best_ind[2]}, Optimizer: {best_ind[3]}, Activation: {best_ind[4]}")
        print(f"Best MAE this gen: {best_ind.fitness.values[0]}")

        # top_individuals = tools.selBest(population, 5)
        # for i, ind in enumerate(top_individuals, 1):
        #     print(f"Top {i} individual: {ind}, MAE: {ind.fitness.values[0]}")
        all_generations.append(list(population))
        best_fitness.append(min([ind.fitness.values[0] for ind in population]))
        # avg_fitness.append(np.mean([ind.fitness.values[0] for ind in population]))
        #   print("Avg MAE this gen: ", avg_fitness[-1])
        # print("Best MAE this gen: ", best_fitness[-1])

        log_data = {
            "Best individual": {
                "Layers": best_ind[0],
                "Neurons": best_ind[1],
                "Learning Rate": 10 ** best_ind[2],
                "Optimizer": best_ind[3],
                "Activation": best_ind[4],
                "MAE": best_ind.fitness.values[0]
            }
        }
        population_diversity_plot = plot_population_diversity(population, gen)
        best_fitness_plot = plot_fitness_evolution(best_fitness)

        st.session_state.progress_callback.on_generation_end(gen, log_data, population_diversity_plot, best_fitness_plot)

def show_genetic_algorithm(df, target_column = 'Water Heating Load (Btu)'):
    st.title("Genetic Algorithm for Model Optimization")
    st.write("Use the buttons below to start or stop the genetic algorithm.")

    st.header("Genetic Algorithm Hyperparameters")
    ngen = st.slider("Number of Generations (NGEN)", min_value=1, max_value=100, value=20)
    population_size = st.slider("Population Size", min_value=1, max_value=100, value=20)
    epochs = st.number_input("Epochs", 5, 1000, 50, help="Between 10 and 1000")
    batch_size = st.number_input("Batch Size", 16, 1024, 128, help="Between 16 and 1024")

    progress_bar = st.progress(0)
    # status_text = st.text("Genetic Algorithm not started yet")
    # log_text = st.text_area("Genetic Algorithm Temporary Log", "No results yet", height=400, max_chars=None)
    status_text = st.empty()
    log_text = st.empty()
    diversity_plot = st.empty()
    fitness_plot = st.empty()
    log_file_path = "logs/genetic_algorithm.log"

    if 'stopGeneticAlgorithm' not in st.session_state:
        st.session_state.stopGeneticAlgorithm = False
        st.session_state.progress_callback = \
            GeneticAlgorithmProgress(progress_bar, status_text, log_text, log_file_path, diversity_plot, fitness_plot)

    start_button = st.button("Start Genetic Algorithm", disabled=False)
    stop_button = st.button("Stop Genetic Algorithm", disabled=False)

    if start_button:
        st.session_state.stopGeneticAlgorithm = False
        progress_callback = st.session_state.progress_callback
        progress_callback.reset(ngen)
        print("GA started")
        genetic_algorithm(df, ngen, epochs, population_size, batch_size, target_column)

    if stop_button:
        st.session_state.stopGeneticAlgorithm = True
        progress_callback = st.session_state.progress_callback
        progress_callback.on_user_end()
        print("GA stopped")