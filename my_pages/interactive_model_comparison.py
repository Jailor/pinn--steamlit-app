import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time

def show_interactive_model_comparison(standard_model, pinn_model):
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


def show_interactive_model_comparison_generic(standard_model, pinn_model, processed_df):
    # Input parameters
    st.title('Heat Pump Power Prediction')
    st.sidebar.header('Input Parameters')

    def user_input_features():
        data = {}
        columns = st.session_state['columns']
        for col in processed_df.columns:
            if col not in [columns['heating_load'], columns['timestamp']]:
                default_value = processed_df[col].mean()
                if col == columns['water_flow']:
                    default_value = 3.0
                elif col == columns['inlet_temp']:
                    default_value = 60.0
                elif col == columns['outlet_water_temp']:
                    default_value = 140.0

                data[col] = st.sidebar.number_input(col, value=default_value)
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