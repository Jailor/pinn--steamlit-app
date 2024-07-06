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
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np


def pinn_loss(y_true_with_features, y_pred):
    pass


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


def train_and_evaluate_model(df, target_column, simple=False):
    # Prepare the data
    X = df.drop(columns=[target_column])
    # if simple:
    #     columns = st.session_state['columns']
    #     X = df[columns['water_flow'])

    y = df[target_column]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the model
    model = Sequential()
    model.add(Dense(256, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer=Adam(), loss='mean_absolute_error',
                  metrics=[
                      keras.metrics.RootMeanSquaredError(),
                      keras.metrics.MeanAbsoluteError()])

    # Callbacks
    lr_scheduler = ReduceLROnPlateau(factor=0.66, patience=5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    model_checkpoint = ModelCheckpoint('best_classic_trained.keras', monitor='val_loss', save_best_only=True)

    # Train the model
    history = model.fit(X_train, y_train, validation_split=0.2,
                        epochs=100, batch_size=256, callbacks=[early_stopping, model_checkpoint, lr_scheduler])

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


def train_and_evaluate_physics_model(df, target_column, simple=False):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y_with_features, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Dense(256, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer=Adam(), loss=pinn_loss, metrics=[pinn_loss])

    # Callbacks
    lr_scheduler = ReduceLROnPlateau(factor=0.66, patience=5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    model_checkpoint = ModelCheckpoint('best_physics_trained.keras', monitor='val_loss', save_best_only=True)

    history = model.fit(X_train, y_train, validation_split=0.2,
                        epochs=100, batch_size=64, callbacks=[early_stopping, model_checkpoint, lr_scheduler])

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
    st.subheader("Dataset Columns")
    st.write("Please select the columns to keep and map the necessary columns for the analysis.")

    columns_to_keep = st.multiselect("Select columns to keep", options=df.columns, default=df.columns.tolist())

    if len(columns_to_keep) == 0:
        st.error("Please select at least one column.")
        return None, None, None, None, None, None

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
