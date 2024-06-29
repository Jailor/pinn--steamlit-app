import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

def show_interactive_data_filtering(initial_df, standard_model, pinn_model, target_column = 'Water Heating Load (Btu)',):
    st.title('Interactive Data Filtering')
    st.subheader('Interactive Data Filtering')
    st.sidebar.subheader('Filter Data')

    # Initialize an empty list to store conditions

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

        if not filtered_df.empty:
            # Predict with filtered data
            X_filtered = filtered_df.drop(columns=[target_column])
            y_filtered = filtered_df[target_column]

            start_time = time.time()
            standard_pred_filtered = standard_model.predict(X_filtered)
            pinn_pred_filtered = pinn_model.predict(X_filtered)
            end_time = time.time()

            #  st.subheader('Filtered Data Predictions')
            # st.write('Standard Model Prediction:', standard_pred_filtered)
            # st.write('PINN Model Prediction:', pinn_pred_filtered)
            st.write(f"Prediction Time: {end_time - start_time:.2f} seconds")

            def calculate_metrics(y_true, y_pred):
                rmse = root_mean_squared_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                mean_observed = y_true.mean()
                cvrmse = (rmse / mean_observed) * 100
                mae = mean_absolute_error(y_true, y_pred)
                mape = (mae / mean_observed) * 100
                return rmse, r2, cvrmse, mae, mape

            metrics_standard = calculate_metrics(y_filtered, standard_pred_filtered)
            metrics_pinn = calculate_metrics(y_filtered, pinn_pred_filtered)

            metrics_df = pd.DataFrame({
                'Metric': ['RMSE', 'R2 Score', 'CVRMSE', 'MAE', 'MAPE'],
                'Standard Model': metrics_standard,
                'PINN Model': metrics_pinn
            })

            st.subheader('Filtered Data Performance Metrics Comparison')
            st.table(metrics_df)

            metrics = {
                'RMSE': [metrics_standard[0], metrics_pinn[0]],
                'R2 Score': [metrics_standard[1], metrics_pinn[1]],
                'MAE': [metrics_standard[3], metrics_pinn[3]],
                'MAPE': [metrics_standard[4], metrics_pinn[4]]
            }
            metric_names = ['Standard Model', 'PINN']

            df_metrics = pd.DataFrame(metrics, index=metric_names)

            # Performance Metrics Bar Plot
            fig = px.bar(df_metrics, barmode='group')
            fig.update_layout(title='Filtered Data Model Performance Metrics', xaxis_title='Metrics', yaxis_title='Values')
            st.plotly_chart(fig)

            # Error distributions for both models
            errors_standard = y_filtered - standard_pred_filtered.flatten()
            errors_pinn = y_filtered - pinn_pred_filtered.flatten()

            # Error Distribution Plot for Standard Model
            fig_standard = go.Figure()
            fig_standard.add_trace(go.Histogram(x=errors_standard, name='Standard Model', nbinsx=60))
            fig_standard.update_layout(
                title='Error Distribution for Standard Model (Filtered Data)',
                xaxis_title='Error',
                yaxis_title='Count',
                yaxis=dict(type='log')  # Set y-axis to logarithmic scale
            )
            st.plotly_chart(fig_standard)

            # Error Distribution Plot for PINN Model
            fig_pinn = go.Figure()
            fig_pinn.add_trace(go.Histogram(x=errors_pinn, name='PINN', opacity=0.75, nbinsx=60))
            fig_pinn.update_layout(
                title='Error Distribution for PINN Model (Filtered Data)',
                xaxis_title='Error',
                yaxis_title='Count',
                yaxis=dict(type='log')  # Set y-axis to logarithmic scale
            )
            st.plotly_chart(fig_pinn)
        else:
            st.warning("No data matches the filter criteria.")

    else:
        # If no filters applied, show a message saying so
        st.info("No filters applied.")
