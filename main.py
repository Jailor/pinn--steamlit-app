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
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score, mean_absolute_error, \
    mean_absolute_percentage_error


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
        # Input parameters
        st.title('Heat Pump Power Prediction')
        st.sidebar.header('Input Parameters')


        def user_input_features():
            water_flow = st.sidebar.number_input('Water Flow (Gallons)', value=3.0)
            inlet_temp = st.sidebar.number_input('Inlet Temp (F)',
                                                 value=60.0)  # inlet temperature of the water in the formula
            outlet_temp = st.sidebar.number_input('Outlet Temp (F)',
                                                  value=140.0)  # outlet temperature of the water in the formula
            cold_water_temp = st.sidebar.number_input('Cold Water Temp (F)', value=70.0)
            watts = st.sidebar.number_input('Watts', value=5.0)
            heat_trace = st.sidebar.number_input('Heat Trace (W)', value=50.0)
            data = {'Outlet Temp (F)': outlet_temp,
                    'Cold Water Temp (F)': cold_water_temp,
                    'Water Flow (Gallons)': water_flow,
                    'Inlet Temp (F)': inlet_temp,
                    'Watts': watts,
                    'Heat Trace (W)': heat_trace}
            features = pd.DataFrame(data, index=[0])
            return features


        input_df = user_input_features()

        start_time = time.time()
        standard_pred = standard_model.predict(input_df)
        pinn_pred = pinn_model.predict(input_df)
        end_time = time.time()

        st.subheader('Predictions')
        st.write('Standard Model Prediction:', standard_pred[0][0])
        st.write('PINN Model Prediction:', pinn_pred[0][0])
        st.write(f"Prediction Time: {end_time - start_time:.2f} seconds")

        st.subheader('Comparison Plot')
        comparison_data = pd.DataFrame({
            'Standard Model': [standard_pred[0][0]],
            'PINN Model': [pinn_pred[0][0]]
        })
        fig = go.Figure(data=[go.Bar(name='Standard Model', x=['Standard Model'], y=[standard_pred[0][0]]),
                              go.Bar(name='PINN Model', x=['PINN Model'], y=[pinn_pred[0][0]])]
                        )

        fig.update_layout(barmode='group', xaxis_tickangle=0)
        st.plotly_chart(fig)
    # TODO: train some models only with the physics-related data
    elif page == "Interactive Model Comparison with Physics":
        # Input parameters
        st.title('Heat Pump Power Prediction and Physics')
        st.sidebar.header('Input Parameters')


        def user_input_features():
            water_flow = st.sidebar.number_input('Water Flow (Gallons)', value=50.0)
            inlet_temp = st.sidebar.number_input('Inlet Temp (F)',
                                                 value=60.0)  # inlet temperature of the water in the formula
            outlet_temp = st.sidebar.number_input('Outlet Temp (F)',
                                                  value=140.0)  # outlet temperature of the water in the formula
            data = {'Outlet Temp (F)': outlet_temp,
                    'Cold Water Temp (F)': 53.768377,  # median value
                    'Water Flow (Gallons)': water_flow,
                    'Inlet Temp (F)': inlet_temp,
                    'Watts': 36.24,
                    'Heat Trace (W)': 12.94}
            features = pd.DataFrame(data, index=[0])
            return features


        input_df = user_input_features()

        start_time = time.time()
        standard_pred = standard_model.predict(input_df)
        pinn_pred = pinn_model.predict(input_df)
        end_time = time.time()

        heat_output_physics = input_df['Water Flow (Gallons)'] * (
                input_df['Outlet Temp (F)'] - input_df['Inlet Temp (F)']) * 0.997 * 8.3077

        st.subheader('Predictions')
        st.write('Standard Model Prediction:', standard_pred[0][0])
        st.write('PINN Model Prediction:', pinn_pred[0][0])
        st.write('Physics Prediction:', heat_output_physics[0])
        st.write(f"Prediction Time: {end_time - start_time:.2f} seconds")

        st.subheader('Comparison Plot')
        comparison_data = pd.DataFrame({
            'Standard Model': [standard_pred[0][0]],
            'PINN Model': [pinn_pred[0][0]]
        })
        fig = go.Figure(data=[go.Bar(name='Standard Model', x=['Standard Model'], y=[standard_pred[0][0]]),
                              go.Bar(name='PINN Model', x=['PINN Model'], y=[pinn_pred[0][0]]),
                              go.Bar(name='Physics Formula', x=['Physics Formula'], y=[heat_output_physics[0]])]
                        )

        fig.update_layout(barmode='group', xaxis_tickangle=0)
        st.plotly_chart(fig)

    elif page == "Performance Metrics":
        st.title('Model Performance Metrics')

        X = initial_df.drop(columns=['Water Heating Load (Btu)'])
        y = initial_df['Water Heating Load (Btu)']

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Predict using the standard model
        y_pred_standard = standard_model.predict(X_test)
        y_pred_standard = y_pred_standard.flatten()  # Flatten the predictions

        # Predict using the PINN model
        y_pred_pinn = pinn_model.predict(X_test)
        y_pred_pinn = y_pred_pinn.flatten()  # Flatten the predictions

        # Calculate metrics for the standard model
        rmse_standard = root_mean_squared_error(y_test, y_pred_standard)
        r2_standard = r2_score(y_test, y_pred_standard)
        mean_observed = y_test.mean()
        cvrmse_standard = (rmse_standard / mean_observed) * 100
        mae_standard = mean_absolute_error(y_test, y_pred_standard)
        mape_standard = (mae_standard / mean_observed) * 100

        # Calculate metrics for the PINN model
        rmse_pinn = root_mean_squared_error(y_test, y_pred_pinn)
        r2_pinn = r2_score(y_test, y_pred_pinn)
        cvrmse_pinn = (rmse_pinn / mean_observed) * 100
        mae_pinn = mean_absolute_error(y_test, y_pred_pinn)
        mape_pinn = (mae_pinn / mean_observed) * 100

        metrics_df = pd.DataFrame({
            'Metric': ['RMSE', 'R2 Score', 'MAE', 'MAPE'],
            'Standard Model': [rmse_standard, r2_standard, mae_standard, mape_standard],
            'PINN Model': [rmse_pinn, r2_pinn, mae_pinn, mape_pinn]
        })

        # Display the table
        st.subheader('Model Performance Metrics Comparison')
        st.table(metrics_df)

        metrics = {
            'RMSE': [rmse_standard, rmse_pinn],
            'R2 Score': [r2_standard, r2_pinn],
            'MAE': [mae_standard, mae_pinn],
            'MAPE': [mape_standard, mape_pinn]
        }
        metric_names = ['Standard Model', 'PINN']

        df_metrics = pd.DataFrame(metrics, index=metric_names)

        # Performance Metrics Bar Plot
        fig = px.bar(df_metrics, barmode='group')
        fig.update_layout(title='Model Performance Metrics', xaxis_title='Metrics', yaxis_title='Values')
        st.plotly_chart(fig)

        # Error distributions for both models
        errors_standard = y_test - y_pred_standard
        errors_pinn = y_test - y_pred_pinn

        # Error Distribution Plot for Standard Model
        fig_standard = go.Figure()
        fig_standard.add_trace(go.Histogram(x=errors_standard, name='Standard Model', nbinsx=60))
        fig_standard.update_layout(
            title='Error Distribution for Standard Model',
            xaxis_title='Error',
            yaxis_title='Count',
            yaxis=dict(type='log')  # Set y-axis to logarithmic scale
        )
        st.plotly_chart(fig_standard)

        # Error Distribution Plot for PINN Model
        fig_pinn = go.Figure()
        fig_pinn.add_trace(go.Histogram(x=errors_pinn, name='PINN', opacity=0.75, nbinsx=60))
        fig_pinn.update_layout(
            title='Error Distribution for PINN Model',
            xaxis_title='Error',
            yaxis_title='Count',
            yaxis=dict(type='log')  # Set y-axis to logarithmic scale
        )
        st.plotly_chart(fig_pinn)

    elif page == "Data Exploration":
        st.title('Data Exploration')

        # Add the information section
        st.subheader('Data Description and Insights')
        st.markdown("""
        **Hot Water Temp (F):**
        Hot Water Temp (F) is the temperature of the water after it has been heated by the heat pump. It indicates how hot the water is when it leaves the heat pump or when it's ready to be used or stored.
    
        **Cold Water Temp (F):**
        Cold Water Temp (F) represents the temperature of the outside cold-water in Fahrenheit..
    
        **Inlet Temp (F):**
        Inlet Temp (F) represents the temperature of the water at the inlet in Fahrenheit. It is the temperature of the water before it is heated by the heat pump, useful for calculating the heat output of the system.
    
        **Water Flow (Gallons):**
        The flow of water in gallons. A higher flow rate means more water is being heated, which affects the heating output of the system.
        
        **Watts:**
        Watts  represents power usage of the HPWH in watts. This measurement can be used to understand how much electrical energy the system uses to operate.
    
        **Heat Trace (W):**
        Heat tracing is a method used to maintain or raise the temperature of pipes and vessels using specially designed heating cables or tapes. The purpose of heat tracing is to prevent freezing, maintain fluidity by keeping the temperature above a certain level, or protect sensitive equipment from cold temperatures. The term "Heat Trace" in watts would then refer to the power consumption of the heat tracing system.
    
        **Water Heating Load (Btu):**
        System power of the HVAC unit.
        """)

        st.subheader('Dataset Summary')
        st.write(initial_df.describe())

        st.subheader('Correlation Matrix')
        corr_matrix = initial_df.corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale='hot',
            aspect='auto'
        )
        fig_corr.update_layout(
            xaxis_title='Features',
            yaxis_title='Features',
            title='Correlation Matrix'
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader('Feature Importance')
        # feature importance previously calculated
        fig_importance = px.bar(feature_importance_df, x='Feature', y='Importance')
        fig_importance.update_yaxes(type='log')
        st.plotly_chart(fig_importance)

        # Add histograms of the data
        st.subheader('Data Distributions')

        # Loop through each column in the dataframe and plot histograms
        for col in initial_df.columns:
            fig_hist = px.histogram(initial_df, x=col, nbins=30, title=f'Distribution of {col}')
            fig_hist.update_layout(
                xaxis_title=col,
                yaxis_title='Count',
                title=f'Distribution of {col}'
            )
            st.plotly_chart(fig_hist)

        st.subheader('Time Series Analysis (Weekly Aggregated)')
        initial_df_resampled = initial_df.resample('W').mean()
        for col in ['Hot Water Temp (F)', 'Cold Water Temp (F)', 'Inlet Temp (F)', 'Water Flow (Gallons)', 'Watts',
                    'Heat Trace (W)']:
            fig_ts = px.line(initial_df_resampled, x=initial_df_resampled.index, y=col,
                             title=f'Weekly Time Series of {col}')
            fig_ts.update_layout(
                xaxis_title='Time',
                yaxis_title=col,
                title=f'Weekly Time Series of {col}',
                yaxis=dict(tickformat=".2f")
            )
            st.plotly_chart(fig_ts)

        st.subheader('Time Series Analysis (Interactive)')
        for col in ['Hot Water Temp (F)', 'Cold Water Temp (F)', 'Inlet Temp (F)', 'Water Flow (Gallons)', 'Watts',
                    'Heat Trace (W)']:
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=initial_df.index, y=initial_df[col], mode='lines', name=col))
            fig_ts.update_layout(
                title=f'Time Series of {col}',
                xaxis_title='Time',
                yaxis_title=col,
                xaxis=dict(
                    rangeslider=dict(
                        visible=True
                    ),
                    type="date"
                )
            )
            st.plotly_chart(fig_ts)

        st.subheader('Time Series Analysis (Smoothed with Moving Average)')

        # Add a slider for the moving average window size
        window_size = st.slider('Select Moving Average Window Size', min_value=1, max_value=60, value=21, step=1)

        initial_df_ma = initial_df.copy()
        for col in ['Hot Water Temp (F)', 'Cold Water Temp (F)', 'Inlet Temp (F)', 'Water Flow (Gallons)', 'Watts',
                    'Heat Trace (W)']:
            initial_df_ma[f'{col}_MA'] = initial_df_ma[col].rolling(window=window_size).mean()
            fig_ts = px.line(initial_df_ma, x=initial_df_ma.index, y=f'{col}_MA',
                             title=f'Moving Average Time Series of {col}')
            fig_ts.update_layout(
                xaxis_title='Time',
                yaxis_title=col,
                title=f'Moving Average Time Series of {col} (Window Size: {window_size})',
                yaxis=dict(tickformat=".2f")
            )
            st.plotly_chart(fig_ts)

        # Add scatter matrix plot

        short_column_names = {
            'Hot Water Temp (F)': 'Hot Temp',
            'Cold Water Temp (F)': 'Cold Temp',
            'Inlet Temp (F)': 'Inlet Temp',
            'Water Flow (Gallons)': 'Water Flow',
            'Watts': 'Watts',
            'Heat Trace (W)': 'Heat Trace',
            'Water Heating Load (Btu)': 'Heating Load'
        }
        plot_df_short = initial_df.rename(columns=short_column_names)
        st.subheader('Scatter Matrix Plot')
        fig_scatter_matrix = px.scatter_matrix(plot_df_short)
        fig_scatter_matrix.update_layout(
            title='Scatter Matrix Plot',
            width=1000,
            height=1000,
        )
        st.plotly_chart(fig_scatter_matrix)

        # Add heatmap for missing values
        st.subheader('Heatmap of Missing Values')
        fig_missing = px.imshow(initial_df.isna(), color_continuous_scale='viridis', aspect='auto')
        fig_missing.update_layout(
            xaxis_title='Features',
            yaxis_title='Samples',
            title='Heatmap of Missing Values'
        )
        st.plotly_chart(fig_missing)

    elif page == "Interactive Data Filtering":
        st.title('Interactive Data Filtering')
        st.subheader('Interactive Data Filtering')
        st.sidebar.subheader('Filter Data')

        # Initialize an empty list to store conditions
        conditions = []

        # Create a form for adding conditions
        with st.sidebar.form(key='filter_form'):
            st.write("Add conditions for filtering data:")

            # Get a list of columns
            columns = list(initial_df.columns)

            # Initialize an empty dictionary to store user inputs
            user_conditions = {}

            # Loop through each column to create input fields for conditions
            for col in columns:
                condition = st.text_input(f"Condition for {col} (e.g., > 0.5)", key=col)
                if condition:
                    user_conditions[col] = condition

            # Submit button
            submit_button = st.form_submit_button(label='Apply Filters')

        # Add a button to clear all filters
        if st.sidebar.button('Clear Filters'):
            user_conditions.clear()
            filtered_df = initial_df

        # If the user submits the form, process the conditions
        if submit_button:
            if not user_conditions:
                st.error("No filters applied. Showing the initial dataframe.")
                filtered_df = initial_df
            else:
                query_parts = []
                for col, condition in user_conditions.items():
                    query_parts.append(f"`{col}` {condition}")

                query = " & ".join(query_parts)
                st.write("Filter Query:", query)
                filtered_df = initial_df.query(query)

            st.write(filtered_df)

            # Predict with filtered data
            X_filtered = filtered_df.drop(columns=['Water Heating Load (Btu)'])
            y_filtered = filtered_df['Water Heating Load (Btu)']

            start_time = time.time()
            standard_pred_filtered = standard_model.predict(X_filtered)
            pinn_pred_filtered = pinn_model.predict(X_filtered)
            end_time = time.time()

            st.subheader('Filtered Data Predictions')
            st.write('Standard Model Prediction:', standard_pred_filtered)
            st.write('PINN Model Prediction:', pinn_pred_filtered)
            st.write(f"Prediction Time: {end_time - start_time:.2f} seconds")
        else:
            # If no filters applied, show a message saying so
            st.info("No filters applied.")

    elif page == "Documentation and Explanation":
        st.title('Documentation and Explanation')

        st.subheader('Model Explanation')
        st.markdown("""
        **Standard Machine Learning Model:**
        The standard machine learning model used here is a regression model trained to predict the water heating load in BTU. This model uses historical data and various input features such as water flow, inlet and outlet temperatures, and power consumption to learn patterns and make predictions.
    
        **Physics-Informed Neural Network (PINN):**
        A Physics-Informed Neural Network (PINN) is an advanced type of neural network that incorporates physical laws and principles into the learning process. Unlike standard machine learning models, which rely solely on data, PINNs use governing equations from physics (such as conservation laws) as part of the training process. This helps the model to generalize better, especially in scenarios where data may be sparse or noisy.
    
        **How PINN Incorporates Physical Laws:**
        - The PINN is designed with a custom loss function that includes terms representing the physical laws.
        - For example, the heat equation is used to relate the water flow and temperature differences to the heating load.
        - This ensures that the predictions made by the PINN are not only fitting the data but also adhering to the underlying physical principles.
        """)
