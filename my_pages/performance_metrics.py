import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error

def show_performance_metrics(initial_df, standard_model, pinn_model):
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