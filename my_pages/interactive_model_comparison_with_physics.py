import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time

from models.model_loading_training import get_distances


def show_interactive_model_comparison_with_physics(standard_model, pinn_model):
    # Input parameters
    st.title('Heat Pump Power Prediction and Physics')
    st.sidebar.header('Input Parameters')

    def user_input_features():
        water_flow = st.sidebar.number_input('Water Flow (Gallons)', value=1.0)
        inlet_temp = st.sidebar.number_input('Inlet Temp (F)',
                                             value=60.0)  # inlet temperature of the water in the formula
        outlet_temp = st.sidebar.number_input('Outlet Temp (F)',
                                              value=140.0)  # outlet temperature of the water in the formula
        data = {'Outlet Temp (F)': outlet_temp,
                'Water Flow (Gallons)': water_flow,
                'Inlet Temp (F)': inlet_temp}
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    start_time = time.time()
    standard_pred = standard_model.predict(input_df)
    pinn_pred = pinn_model.predict(input_df)
    end_time = time.time()

    heat_output_physics = input_df['Water Flow (Gallons)'] * (
            input_df['Outlet Temp (F)'] - input_df['Inlet Temp (F)']) * 0.997 * 8.3077

    comparison_data = pd.DataFrame({
        'Standard Model': [standard_pred],
        'PINN Model': [pinn_pred]
    })

    d1, d2 = get_distances(standard_pred, pinn_pred, heat_output_physics)

    st.subheader('Predictions')
    st.write('Standard Model Prediction:', standard_pred[0][0])
    st.write('Standard model distance:', d1)
    st.write('PINN Model Prediction:', pinn_pred[0][0])
    st.write('Pinn Distance:', d2)
    st.write('Physics Prediction:', heat_output_physics[0])
    st.write(f"Prediction Time: {end_time - start_time:.2f} seconds")

    st.subheader('Comparison Plot')

    fig = go.Figure(data=[go.Bar(name='Standard Model', x=['Standard Model'], y=[standard_pred[0][0]]),
                          go.Bar(name='PINN Model', x=['PINN Model'], y=[pinn_pred[0][0]]),
                          go.Bar(name='Physics Formula', x=['Physics Formula'], y=[heat_output_physics[0]])]
                    )

    fig.update_layout(barmode='group', xaxis_tickangle=0)
    st.plotly_chart(fig)


def show_interactive_model_comparison_with_physics_generic(processed_df, standard_model, pinn_model):
    # Input parameters
    st.title('Heat Pump Power Prediction and Physics')
    st.sidebar.header('Input Parameters')

    columns = st.session_state['columns']

    def user_input_features():
        data = {}
        for col in processed_df.columns:
            if col not in [columns['heating_load'], columns['timestamp']]:
                default_value = processed_df[col].mean()
                if col == columns['water_flow']:
                    default_value = 3.0
                elif col == columns['inlet_temp']:
                    default_value = 60.0
                elif col == columns['outlet_water_temp']:
                    default_value = 140.0
                if col in [columns['water_flow'], columns['inlet_temp'], columns['outlet_water_temp']]:
                    data[col] = st.sidebar.number_input(col, value=default_value)
                else:
                    data[col] = default_value

        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    start_time = time.time()
    standard_pred = standard_model.predict(input_df)
    pinn_pred = pinn_model.predict(input_df)
    end_time = time.time()

    heat_output_physics = input_df[columns['water_flow']] * (
            input_df[columns['outlet_water_temp']] - input_df[columns['inlet_temp']]) * 0.997 * 8.3077

    comparison_data = pd.DataFrame({
        'Standard Model': [standard_pred],
        'PINN Model': [pinn_pred]
    })

    d1, d2 = get_distances(standard_pred, pinn_pred, heat_output_physics)

    st.subheader('Predictions')
    st.write('Standard Model Prediction:', standard_pred[0][0])
    st.write('Standard model distance:', d1)
    st.write('PINN Model Prediction:', pinn_pred[0][0])
    st.write('Pinn Distance:', d2)
    st.write('Physics Prediction:', heat_output_physics[0])
    st.write(f"Prediction Time: {end_time - start_time:.2f} seconds")

    st.subheader('Comparison Plot')

    fig = go.Figure(data=[go.Bar(name='Standard Model', x=['Standard Model'], y=[standard_pred[0][0]]),
                          go.Bar(name='PINN Model', x=['PINN Model'], y=[pinn_pred[0][0]]),
                          go.Bar(name='Physics Formula', x=['Physics Formula'], y=[heat_output_physics[0]])]
                    )

    fig.update_layout(barmode='group', xaxis_tickangle=0)
    st.plotly_chart(fig)


def get_distance(standard_pred, pinn_pred, heat_output_physics):
    d1 = abs(standard_pred[0][0] - heat_output_physics[0])
    d2 = abs(pinn_pred[0][0] - heat_output_physics[0])
    return d1, d2
