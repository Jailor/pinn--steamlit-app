import pandas as pd
import streamlit as st

from models.model_loading_training import train_and_evaluate_model, train_and_evaluate_physics_model
from my_pages.genetic_algorithm import show_genetic_algorithm


def show_train_models(processed_df, columns):
    st.title("Train Models and tune hyperparameters")
    method = st.selectbox("Select Training Method",
                          ["Choose Own Hyperparameters", "Use Optimal Config", "Run Genetic Algorithm"])

    # Variables to store models
    classic_model = None
    pinn_model = None
    if method == "Choose Own Hyperparameters":
        st.header("Choose Hyperparameters")
        st.write("Choose the hyperparameters for training the models. The default values "
                 "are those values found optimal by the genetic algorithm on the Rossland Site 1 E350 dataset.")
        num_layers = st.slider("Number of Layers", 1, 10, 2)
        neurons_per_layer = st.number_input("Neurons per Layer", 5, 1024, 227, help="Between 5 and 1024")
        learning_rate = st.number_input("Learning Rate", 0.00001, 10.00, 0.0009323, format="%.6f",
                                        help="Between 0.00001 and 10.00")
        optimizer_name = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"])
        activation_function = st.selectbox("Activation Function", ["relu", "sigmoid", "tanh"])
        epochs = st.number_input("Epochs", 10, 1000, 100, help="Between 10 and 1000")
        batch_size = st.number_input("Batch Size", 16, 1024, 64, help="Between 16 and 1024")
        validation_split = st.slider("Reserved for Validation", 0.1, 0.4, 0.2)
        train_test_size = st.slider("Reserved for Testing", 0.1, 0.4, 0.2)

        submitted = False

        if st.button("Train Mdoels"):
            st.write("Training classic model...")
            classic_progress_bar = st.progress(0)
            classic_status_text = st.empty()
            classic_log_text = st.empty()

            classic_model = train_and_evaluate_model(processed_df, columns['heating_load'], num_layers,
                                                     neurons_per_layer,
                                                     learning_rate, optimizer_name, activation_function, epochs,
                                                     batch_size, validation_split, train_test_size,
                                                     classic_progress_bar, classic_status_text,
                                                     classic_log_text, "logs/classic_model.log")

            st.write("Finished training classic model.")

            st.write("Training PINN model...")
            pinn_progress_bar = st.progress(0)
            pinn_status_text = st.empty()
            pinn_log_text = st.empty()
            pinn_model = train_and_evaluate_physics_model(processed_df, columns['heating_load'], num_layers,
                                                          neurons_per_layer,
                                                          learning_rate, optimizer_name, activation_function,
                                                          epochs, batch_size, validation_split, train_test_size,
                                                          pinn_progress_bar, pinn_status_text, pinn_log_text,
                                                          "logs/pinn_model.log")
            st.write("Finished training PINN model.")
            st.session_state['model'] = classic_model
            st.session_state['pinn_model'] = pinn_model
            submitted = st.button("Continue")
            if submitted:
                st.rerun()
    elif method == "Use Optimal Config":
        st.header("Use Optimal Configuration")
        st.write("Using the best configuration for training the models as determined by the genetic algorithm using"
                 "20 generations on the Rossland Site 1 E350 dataset.")
        st.write("The configuration is as follows:")
        best_config_df = pd.DataFrame({
            "Hyperparameter": ["Number of Layers", "Neurons per Layer", "Learning Rate", "Optimizer",
                               "Activation Function", "Epochs", "Batch Size", "Validation Split", "Train/Test Split"],
            "Value": [2, 227, 0.0009323, "Adam", "relu", 100, 64, 0.2, 0.2]
        })

        st.table(best_config_df)

        num_layers = 2
        neurons_per_layer = 227
        learning_rate = 0.0009323
        optimizer_name = "Adam"
        activation_function = "relu"
        epochs = 100
        batch_size = 64
        validation_split = 0.2
        train_test_size = 0.2

        if st.button("Train Mdoels"):
            st.write("Training classic model...")
            classic_progress_bar = st.progress(0)
            classic_status_text = st.empty()
            classic_log_text = st.empty()

            classic_model = train_and_evaluate_model(processed_df, columns['heating_load'], num_layers,
                                                     neurons_per_layer,
                                                     learning_rate, optimizer_name, activation_function, epochs,
                                                     batch_size, validation_split, train_test_size,
                                                     classic_progress_bar, classic_status_text,
                                                     classic_log_text, "logs/classic_model.log")

            st.write("Finished training classic model.")

            st.write("Training PINN model...")
            pinn_progress_bar = st.progress(0)
            pinn_status_text = st.empty()
            pinn_log_text = st.empty()
            pinn_model = train_and_evaluate_physics_model(processed_df, columns['heating_load'], num_layers,
                                                          neurons_per_layer,
                                                          learning_rate, optimizer_name, activation_function,
                                                          epochs, batch_size, validation_split, train_test_size,
                                                          pinn_progress_bar, pinn_status_text, pinn_log_text,
                                                          "logs/pinn_model.log")
            st.write("Finished training PINN model.")
            st.session_state['model'] = classic_model
            st.session_state['pinn_model'] = pinn_model
            submitted = st.button("Continue")
            if submitted:
                st.rerun()
    elif method == "Run Genetic Algorithm":
        show_genetic_algorithm(processed_df, columns['heating_load'], True)


