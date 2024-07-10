import streamlit as st
from models.model_loading_training import simulated_annealing


def show_simulated_annealing(df, heating_load_column='Water Heating Load (Btu)',
                             outlet_water_temp_column='Hot Water Temp (F)',
                             inlet_temp_column='Inlet Temp (F)', water_flow_column='Water Flow (Gallons)'):
    st.title("Simulated Annealing for Hyperparameter Tuning")
    st.write(
        "Choose the hyperparameters of the simulated annealing algorithm and then click the button to start the training.")
    epochs = st.number_input("Epochs", 5, 1000, 10, help="Between 5 and 1000")
    batch_size = st.number_input("Batch Size", 16, 1024, 128, help="Between 16 and 1024")
    initial_alpha = st.number_input("Initial Alpha", 0.1, 10.0, 5.0, help="Between 0.1 and 10.0")
    initial_temp = st.number_input("Initial Temperature", 1.0, 100.0, 50.0, help="Between 1.0 and 100.0")
    cooling_rate = st.number_input("Cooling Rate", 0.1, 1.0, 0.955, help="Between 0.1 and 1.0")
    min_temp = st.number_input("Minimum Temperature", 1.0, 100.0, 5.0, help="Between 1.0 and 100.0")

    progress_bar = st.progress(0)
    status_text = st.empty()
    log_text = st.empty()
    log_file_path = "logs/simulated_annealing_algorithm.log"
    alpha_plot = st.empty()
    temperature_plot = st.empty()

    start_button = st.button("Start Simulated Annealing", disabled=False)

    if start_button:
        print("SA started")
        simulated_annealing(df, epochs=epochs, batch_size=batch_size,
                            heating_load_column=heating_load_column,
                            outlet_water_temp_column=outlet_water_temp_column,
                            inlet_temp_column=inlet_temp_column,
                            water_flow_column=water_flow_column,
                            progress_bar=progress_bar, status_text=status_text, log_text=log_text,
                            log_file_path=log_file_path,
                            alpha_plot=alpha_plot, temperature_plot=temperature_plot,
                            initial_alpha=initial_alpha, initial_temp=initial_temp, cooling_rate=cooling_rate,
                            min_temp=min_temp)
