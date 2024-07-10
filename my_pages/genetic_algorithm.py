import pandas as pd
import streamlit as st


from models.model_loading_training import GeneticAlgorithmProgress, genetic_algorithm, train_and_evaluate_model, train_and_evaluate_physics_model


def show_genetic_algorithm(df, target_column='Water Heating Load (Btu)', train_models = False):
    st.write(
        "Choose the hyperparameters of the genetic algorithm, then use buttons below to start or stop the genetic algorithm.")

    st.header("Genetic Algorithm Hyperparameters")
    ngen = st.number_input("Number of Generations (NGEN)", 1, 100, 20, help="Between 1 and 100")
    population_size = st.number_input("Population Size", 1, 100, 20, help="Between 1 and 100")
    epochs = st.number_input("Epochs", 5, 1000, 50, help="Between 5 and 1000")
    batch_size = st.number_input("Batch Size", 16, 1024, 128, help="Between 16 and 1024")

    progress_bar = st.progress(0)
    # status_text = st.text("Genetic Algorithm not started yet")
    # log_text = st.text_area("Genetic Algorithm Temporary Log", "No results yet", height=400, max_chars=None)
    status_text = st.empty()
    log_text = st.empty()
    diversity_plot = st.empty()
    fitness_plot = st.empty()
    log_file_path = "logs/genetic_algorithm.log"

    if 'stopGeneticAlgorithm' not in st.session_state:
        st.session_state.stopGeneticAlgorithm = False
        st.session_state.progress_callback = \
            GeneticAlgorithmProgress(progress_bar, status_text, log_text, log_file_path, diversity_plot, fitness_plot)
        st.session_state.best_individual = None

    start_button = st.button("Start Genetic Algorithm", disabled=False)
    stop_button = st.button("Stop Genetic Algorithm", disabled=False)

    if start_button:
        st.session_state.stopGeneticAlgorithm = False
        progress_callback = st.session_state.progress_callback
        progress_callback.reset(ngen)
        print("GA started")
        genetic_algorithm(df, ngen, epochs, population_size, batch_size, target_column)
        if train_models and progress_callback.ga_completed:  # Check if GA has stopped
            train_best_individual_ga(df, st.session_state['columns'])

    if stop_button:
        st.session_state.stopGeneticAlgorithm = True
        progress_callback = st.session_state.progress_callback
        progress_callback.on_user_end()
        print("GA stopped")
        if train_models and progress_callback.ga_completed:  # Check if GA has completed
            train_best_individual_ga(df, st.session_state['columns'])

def train_best_individual_ga(processed_df, columns):
    if 'best_individual' in st.session_state and st.session_state['best_individual'] is not None:
        st.write("Training classic model...")
        classic_progress_bar = st.progress(0)
        classic_status_text = st.empty()
        classic_log_text = st.empty()
        best_individual = st.session_state['best_individual']


        layers = best_individual["Layers"]
        neurons = best_individual["Neurons"]
        learning_rate = 10 ** best_individual["Learning Rate"]
        optimizer = best_individual["Optimizer"]
        activation = best_individual["Activation"]
        epochs = 100
        batch_size = 64
        validation_split = 0.2
        train_test_size = 0.2

        best_config_df = pd.DataFrame({
            "Hyperparameter": ["Number of Layers", "Neurons per Layer", "Learning Rate", "Optimizer",
                               "Activation Function", "Epochs", "Batch Size", "Validation Split",
                               "Train/Test Split"],
            "Value": [layers, neurons, learning_rate, optimizer, activation, epochs, batch_size, validation_split,
                      train_test_size]
        })

        st.write("The best configuration found by the genetic algorithm is as follows:")
        st.table(best_config_df)

        classic_model = train_and_evaluate_model(processed_df, columns['heating_load'], layers,
                                                 neurons,
                                                 learning_rate, optimizer, activation, epochs,
                                                 batch_size, validation_split, train_test_size,
                                                 classic_progress_bar, classic_status_text,
                                                 classic_log_text, "logs/classic_model.log")

        classic_progress_bar.progress(1.0)

        st.write("Finished training classic model.")

        st.write("Training PINN model...")
        pinn_progress_bar = st.progress(0)
        pinn_status_text = st.empty()
        pinn_log_text = st.empty()
        pinn_model = train_and_evaluate_physics_model(processed_df, columns['heating_load'], layers,
                                                      neurons,
                                                      learning_rate, optimizer, activation,
                                                      epochs, batch_size, validation_split, train_test_size,
                                                      pinn_progress_bar, pinn_status_text, pinn_log_text,
                                                      "logs/pinn_model.log")
        st.write("Finished training PINN model.")
        st.session_state['model'] = classic_model
        st.session_state['pinn_model'] = pinn_model
        pinn_progress_bar.progress(1.0)
        submitted = st.button("Continue")
        if submitted:
            st.rerun()