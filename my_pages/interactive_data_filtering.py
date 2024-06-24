import streamlit as st
import pandas as pd
import time

def show_interactive_data_filtering(initial_df, standard_model, pinn_model):
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
