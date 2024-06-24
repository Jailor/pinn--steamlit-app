import streamlit as st


def show_documentation():
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
