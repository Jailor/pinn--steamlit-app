import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time

def show_interactive_model_comparison_with_physics(standard_model, pinn_model):
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