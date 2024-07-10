import streamlit as st


def show_documentation_full():
    st.title('Documentation')

    full_application_features()
    model_explanation()

def show_documentation_before_process_dataset():
    st.title('Documentation')

    st.subheader('Using the application')
    st.markdown("""
        You can now explore the dataset in detail and perform various analyses to understand the data better.
        Once you are satisfied with you understanding of the data, you can proceed to the next step of processing the dataset to
        unlock the full features of the application (model training, prediction, etc.).
    - **Documentation**: Provides an overview and instructions on how to use the application.
    - **Data Exploration Statistics**: Displays statistical analysis and summaries of the dataset to help you understand the data better.
    - **Data Exploration Time Series Analysis**: Offers tools for time series analysis of the dataset to identify trends and patterns over time.
    - **Process Dataset**: Allows you to preprocess the dataset, by dropping irrelevant columns and specifying which columns correspond to heating load calculations.
    """)

def show_documentation_before_process_models():
    st.title('Documentation')

    st.subheader('Using the application')
    st.markdown("""
        You can now train machine learning models on the dataset and optimize  their hyperparameters
        using genetic algorithms and simulated annealing.
        
        Current application functionalities:
        - **Documentation**: Provides an overview and instructions on how to use the application.
        - **Data Exploration Statistics**: Displays statistical analysis and summaries of the dataset to help you understand the data better.
        - **Data Exploration Time Series Analysis**: Offers tools for time series analysis of the dataset to identify trends and patterns over time.
        - **Train Models**: Allows you to configure and train machine learning models on the dataset. You can choose hyperparameters such as the number of layers, neurons per layer, learning rate, optimizer, activation function, epochs, batch size, validation split, and train/test split.
        - **Upload Models**: Enables you to upload pre-trained models for prediction and comparison instead of training new models.
        - **Genetic Algorithm**: Provides a genetic algorithm-based optimization tool to find the best hyperparameters for the models.
         - **Simulated Annealing**: Offers a simulated annealing-based optimization tool to find the best alpha parameter for the PINN model.
        """)

def show_documentation_initial():
    st.title('Documentation')

    st.subheader('Using the application')
    st.markdown("""
        This application is designed to help you explore and analyze a dataset related to a heat pump water heater (HPWH) system.
        The application consists of several pages, allowing you to explore and process the dataset, train or upload machine learning models, and optimize hyperparameters
        using genetic algorithms and simulated annealing. **You must upload a time series dataset that contains heating load as a dataset column**.
        After this is done, you can proceed to access the rest of the application's features.
        """)

    full_application_features()
    model_explanation()

def full_application_features():
    st.subheader('Full Application Features')
    st.markdown("""
             This application is designed to help you explore and analyze a dataset related to a heat pump water heater (HPWH) system and train machine learning models 
             to predict the water heating load of the system. 
             The application consists of several pages accessible via the sidebar navigation:
             - **Documentation**: Provides an overview and instructions on how to use the application.
             - **Data Exploration Statistics**: Displays statistical analysis and summaries of the dataset to help you understand the data better.
             - **Data Exploration Time Series Analysis**: Offers tools for time series analysis of the dataset to identify trends and patterns over time.
             - **Interactive Model Comparison**: Allows you to interactively compare the predictions of the standard model and the PINN model.
             - **Interactive Model Comparison with Physics**: Allows you to compare the predictions of the standard model, the PINN model, and the physics-based model (formula).
             - **Performance Metrics**: Shows detailed performance metrics for the models to evaluate their effectiveness, alongside visualizations of the predictions.
             - **Interactive Data Filtering**: Enables interactive filtering of the dataset to focus on specific subsets of data, and see the model predictions for the filtered data.
             - **Genetic Algorithm**: Provides a genetic algorithm-based optimization tool to find the best hyperparameters for the models.
             - **Simulated Annealing**: Offers a simulated annealing-based optimization tool to find the best alpha parameter for the PINN model.
             """)

def model_explanation():
    st.subheader('Model Explanation')
    st.markdown("""
           **Standard Machine Learning Model:**
           The standard machine learning model used here is a regression model trained to predict the water heating load in BTU. This model uses historical data and various input features such as water flow, inlet and outlet temperatures, and power consumption to learn patterns and make predictions.

           **Physics-Informed Neural Network (PINN):**
           A Physics-Informed Neural Network (PINN) is an advanced type of neural network that incorporates physical laws and principles into the learning process. Unlike standard machine learning models, which rely solely on data, PINNs use governing equations from physics (such as conservation laws) as part of the training process. This helps the model to generalize better, especially in scenarios where data may be sparse or noisy.

           **How PINN Incorporates Physical Laws:**
           - The PINN is designed with a custom loss function that includes terms representing the physical laws.
           - In our case the heat equation is used to relate the water flow and temperature differences to the heating load.
           - This ensures that the predictions made by the PINN are not only fitting the data but also adhering to the underlying physical principles.
           """
                )
