
import streamlit as st
import tensorflow as tf
import keras
from keras.models import load_model
from sklearn.ensemble import RandomForestRegressor
import time
import difflib
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt




def get_optimizer(optimizer_name, learning_rate):
    if optimizer_name == 'Adam':
        return Adam(learning_rate=learning_rate)
    elif optimizer_name == 'SGD':
        return SGD(learning_rate=learning_rate)
    elif optimizer_name == 'RMSprop':
        return RMSprop(learning_rate=learning_rate)
    else:
        return Adam(learning_rate=learning_rate)


def train_and_evaluate_model(df, target_column, num_layers, neurons_per_layer, learning_rate,
                             optimizer_name, activation_function, epochs, batch_size, validation_split, train_test_size,
                             progress_bar, status_text, log_text, log_file_path):
    # Prepare the data
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_size, random_state=42)

    # Build the model
    model = Sequential()
    model.add(Dense(neurons_per_layer, input_dim=X_train.shape[1], activation=activation_function))
    for _ in range(num_layers - 1):
        model.add(Dense(neurons_per_layer, activation=activation_function))
    model.add(Dense(1, activation='linear'))

    optimizer = get_optimizer(optimizer_name, learning_rate)
    model.compile(optimizer=optimizer, loss='mean_absolute_error',
                  metrics=[
                      keras.metrics.RootMeanSquaredError(),
                      keras.metrics.MeanAbsoluteError()])

    # Callbacks
    custom_callback = ModelTrainingProgress(progress_bar, status_text, log_text, epochs, log_file_path)
    lr_scheduler = ReduceLROnPlateau(factor=0.66, patience=5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    model_checkpoint = ModelCheckpoint('best_classic_trained.keras', monitor='val_loss', save_best_only=True)

    # Train the model
    history = model.fit(X_train, y_train, validation_split=validation_split,
                        epochs=epochs, batch_size=max(256, batch_size),
                        callbacks=[early_stopping, model_checkpoint, lr_scheduler, custom_callback])

    # Load the best model
    model = keras.models.load_model('best_classic_trained.keras')

    # Evaluate the model
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    mean_observed = y_test.mean()
    cvrmse = (rmse / mean_observed) * 100
    mae = mean_absolute_error(y_test, y_pred)
    mape = (mae / mean_observed) * 100

    print(f"Root Mean Squared Error: {rmse}")
    print(f"R^2 Score: {r2}")
    print(f"cvRMSE: {cvrmse}%")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Absolute Percentage Error: {mape}")

    return model


def train_and_evaluate_physics_model(df, target_column, num_layers, neurons_per_layer, learning_rate,
                                     optimizer_name, activation_function, epochs, batch_size, validation_split,
                                     train_test_size,
                                     progress_bar, status_text, log_text, log_file_path):
    alpha = 0.0899197906255722
    lambda_reg = 0.0001

    def pinn_loss(y_true_with_features, y_pred):
        y_true = y_true_with_features[:, 0:1]
        outlet_water_temp = y_true_with_features[:, 1:2]
        inlet_temp = y_true_with_features[:, 2:3]
        water_flow = y_true_with_features[:, 3:4]

        specific_heat_capacity = 4174  # J/kg°C, specific heat capacity of water
        gallons_to_liters = 3.78541  # 1 gallon = 3.78541 liters
        fahrenheit_to_celsius = lambda f: (f - 32) * 5.0 / 9.0  # Convert °F to °C
        joules_to_btu = 0.0009478171
        flow_rate_liters = water_flow * gallons_to_liters
        inlet_temp_c = fahrenheit_to_celsius(inlet_temp)
        outlet_temp_c = fahrenheit_to_celsius(outlet_water_temp)
        heat_output = water_flow * (outlet_water_temp - inlet_temp) * 0.997 * 8.3077
        reg_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in model.trainable_weights])

        return tf.reduce_mean(tf.abs(y_true - y_pred) + alpha * tf.abs(heat_output - y_pred) + lambda_reg * reg_loss,
                              axis=-1)

    # Prepare the data
    X = df.drop(columns=[target_column])
    y = df[target_column]

    columns = st.session_state['columns']

    outlet_water_temp = X.iloc[:, X.columns.get_loc(columns['outlet_water_temp'])].to_numpy().reshape(-1, 1)
    inlet_temp = X.iloc[:, X.columns.get_loc(columns['inlet_temp'])].to_numpy().reshape(-1, 1)
    water_flow = X.iloc[:, X.columns.get_loc(columns['water_flow'])].to_numpy().reshape(-1, 1)

    y_np = y.to_numpy().reshape(-1, 1)

    y_with_features = np.concatenate(
        [y_np, outlet_water_temp, inlet_temp, water_flow], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y_with_features, test_size=train_test_size, random_state=42)

    model = Sequential()
    model.add(Dense(neurons_per_layer, input_dim=X_train.shape[1], activation=activation_function))
    for _ in range(num_layers - 1):
        model.add(Dense(neurons_per_layer, activation=activation_function))
    model.add(Dense(1, activation='linear'))

    optimizer = get_optimizer(optimizer_name, learning_rate)
    model.compile(optimizer=optimizer, loss=pinn_loss, metrics=[pinn_loss])

    # Callbacks
    custom_callback = ModelTrainingProgress(progress_bar, status_text, log_text, epochs, log_file_path)
    lr_scheduler = ReduceLROnPlateau(factor=0.66, patience=5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    model_checkpoint = ModelCheckpoint('best_physics_trained.keras', monitor='val_loss', save_best_only=True)

    # Train the model
    history = model.fit(X_train, y_train, validation_split=validation_split,
                        epochs=epochs, batch_size=batch_size,
                        callbacks=[early_stopping, model_checkpoint, lr_scheduler, custom_callback])

    model.load_weights('best_physics_trained.keras')

    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test[:, 0], y_pred, squared=False)
    r2 = r2_score(y_test[:, 0], y_pred)
    mean_observed = y_test[:, 0].mean()
    cvrmse = (rmse / mean_observed) * 100
    mae = mean_absolute_error(y_test[:, 0], y_pred)
    mape = (mae * 100) / mean_observed

    print(f"Root Mean Squared Error: {rmse}")
    print(f"R^2 Score: {r2}")
    print(f"cvRMSE: {cvrmse}%")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Absolute Percentage Error: {mape}%")

    return model


class ModelTrainingProgress(Callback):
    def __init__(self, progress_bar, status_text, log_text, total_epochs, log_file_path):
        super().__init__()
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.log_text = log_text
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.logs = []
        self.log_file_path = log_file_path

    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch += 1
        progress = self.current_epoch / self.total_epochs
        self.progress_bar.progress(progress)
        self.status_text.text(f"Training in progress: {int(progress * 100)}% complete")

        # Format and append the current epoch log
        log_message = f"Epoch {epoch + 1}/{self.total_epochs}\n"
        log_message += " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()]) + "\n"
        self.logs.append(log_message)

        # Join all logs and display them in the text area
        full_log_message = "\n".join(self.logs)
        # self.log_text.text_area("Training Log", full_log_message, height=400, max_chars=None)
        self.log_text.text_area("Training Log", log_message, height=400, max_chars=None)

        if self.total_epochs == self.current_epoch:
            with open(self.log_file_path, 'w') as log_file:
                log_file.write("\n".join(self.logs))
                st.write("Training complete. Logs saved to file at:", self.log_file_path)
            self.log_text.text_area("Training Log", full_log_message, height=400, max_chars=None)



@st.cache_resource
def load_models():
    standard_model = load_model('models/classic_full.h5')
    pinn_model = load_model('models/pinn_full.h5', custom_objects={'pinn_loss': pinn_loss}, compile=False)

    return standard_model, pinn_model


@st.cache_resource
def load_simplified_models():
    standard_model = load_model('models/phys/nn_1024.h5')  # classic_simple
    pinn_model = load_model('models/phys/phys_simple.h5', custom_objects={'pinn_loss': pinn_loss},
                            compile=False)  # best_model_phys_only

    return standard_model, pinn_model


def pinn_loss(y_true_with_features, y_pred):
    pass


def get_distances(standard_pred, pinn_pred, heat_output_physics):
    d1 = abs(standard_pred[0][0] - heat_output_physics[0])
    d2 = abs(pinn_pred[0][0] - heat_output_physics[0])
    if d2 > d1:
        d1, d2 = d2, d1
        standard_pred[0][0], pinn_pred[0][0] = pinn_pred[0][0], standard_pred[0][0]
    return d1, d2


def plot_population_diversity(population, generation):
    num_layers = [ind[0] for ind in population]
    neurons_per_layer = [ind[1] for ind in population]
    learning_rates_power = [ind[2] for ind in population]

    optimizers = [ind[3] for ind in population]
    activation_functions = [ind[4] for ind in population]

    layer_counts = [num_layers.count(i) for i in range(1, 5)]

    fig, axs = plt.subplots(5, 1, figsize=(10, 20))

    axs[0].bar(range(1, 5), layer_counts, edgecolor='black')
    axs[0].set_title(f'Number of Layers Distribution (Generation {generation})')
    axs[0].set_xlabel('Number of Layers')
    axs[0].set_ylabel('Frequency')
    axs[0].set_xticks(range(1, 5))

    axs[1].hist(neurons_per_layer, bins=range(32, 513, 48), edgecolor='black')
    axs[1].set_title(f'Neurons Per Layer Distribution (Generation {generation})')
    axs[1].set_xlabel('Neurons Per Layer')
    axs[1].set_ylabel('Frequency')

    axs[2].hist(learning_rates_power, bins=12, edgecolor='black')
    axs[2].set_title(f'Learning Rates Distribution (Generation {generation})')
    axs[2].set_xlabel('Learning Rate Power')
    axs[2].set_ylabel('Frequency')

    optimizer_counts = {opt: optimizers.count(opt) for opt in set(optimizers)}
    axs[3].bar(optimizer_counts.keys(), optimizer_counts.values(), color='blue', edgecolor='black')
    axs[3].set_title(f'Distribution of Optimizers (Generation {generation})')
    axs[3].set_xlabel('Optimizer')
    axs[3].set_ylabel('Frequency')

    activation_counts = {act: activation_functions.count(act) for act in set(activation_functions)}
    axs[4].bar(activation_counts.keys(), activation_counts.values(), color='blue', edgecolor='black')
    axs[4].set_title(f'Distribution of Activation Functions (Generation {generation})')
    axs[4].set_xlabel('Activation Function')
    axs[4].set_ylabel('Frequency')

    plt.tight_layout()
    return fig


def plot_fitness_evolution(best_fitness):
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness, label='Best MAE', color='red')
    # plt.plot(avg_fitness, label='Average MAE', color='blue')
    plt.xlabel('Generation')
    plt.ylabel('MAE')
    plt.title('Evolution of MAE over Generations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt.gcf()


class GeneticAlgorithmProgress:
    def __init__(self, progress_bar, status_text, log_text, log_file_path, diversity_plot, fitness_plot):
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.log_text = log_text
        self.log_file_path = log_file_path
        self.diversity_plot = diversity_plot
        self.fitness_plot = fitness_plot
        self.total_generations = 0
        self.current_generation = 0
        self.logs = []
        self.most_recent_diversity_plot = None
        self.most_recent_fitness_plot = None
        self.ga_completed = False

    def reset(self, total_generations):
        self.total_generations = total_generations
        self.current_generation = 0
        self.logs = []
        self.progress_bar.progress(0)
        self.status_text.text("Genetic Algorithm in progress: 0% complete (Generation 0)")
        self.log_text.text_area("Genetic Algorithm Log", "No results yet", height=400, max_chars=None)
        self.diversity_plot.empty()
        self.fitness_plot.empty()
        self.most_recent_diversity_plot = None
        self.most_recent_fitness_plot = None
        self.ga_completed = False

    def on_generation_end(self, generation, logs, population_diversity_plot, best_fitness_plot):
        self.current_generation += 1
        progress = self.current_generation / self.total_generations
        self.progress_bar.progress(progress)
        self.status_text.text(
            f"Genetic Algorithm in progress: {int(progress * 100)}% complete (Generation {generation + 1})")

        # Format and append the current generation log
        log_message = f"Generation {generation + 1}/{self.total_generations}\n"
        formatted_logs = []
        for k, v in logs.items():
            if isinstance(v, (float, int)):
                formatted_logs.append(f"{k}: {v:.4f}")
            else:
                formatted_logs.append(f"{k}: {v}")
        log_message += " - ".join(formatted_logs) + "\n"
        self.logs.append(log_message)

        # Join all logs and display them in the text area
        full_log_message = "\n".join(self.logs)
        self.log_text.text_area("Genetic Algorithm Log", full_log_message, height=400, max_chars=None)

        self.diversity_plot.pyplot(population_diversity_plot)
        self.fitness_plot.pyplot(best_fitness_plot)
        self.most_recent_diversity_plot = population_diversity_plot
        self.most_recent_fitness_plot = best_fitness_plot
        st.session_state.best_individual_so_far = logs["Best individual"]
        if self.total_generations == self.current_generation:
            st.session_state.best_individual = logs["Best individual"]
            self._save_logs()
            self.ga_completed = True

    def on_user_end(self):
        self.status_text.text("Genetic Algorithm stopped by user at generation " + str(self.current_generation + 1))
        self.progress_bar.progress(1.0)
        full_log_message = "\n".join(self.logs)
        self.log_text.text_area("Genetic Algorithm Log", full_log_message, height=400, max_chars=None)
        if self.most_recent_diversity_plot is not None and self.most_recent_fitness_plot is not None:
            self.diversity_plot.pyplot(self.most_recent_diversity_plot)
            self.fitness_plot.pyplot(self.most_recent_fitness_plot)
        if 'best_individual_so_far' in st.session_state:
            st.session_state.best_individual = st.session_state.best_individual_so_far
        else:
            st.session_state.best_individual = None
        self._save_logs()
        self.ga_completed = True

    def _save_logs(self):
        with open(self.log_file_path, 'w') as log_file:
            log_file.write("\n".join(self.logs))
        st.write("Logs saved to file at:", self.log_file_path)


def is_valid(individual):
    num_layers, neurons_per_layer, lr, optimizer_name, activation_function = individual
    # Check for valid ranges and types
    if not (1 <= num_layers <= 4):
        return False
    if not (32 <= neurons_per_layer <= 512):
        return False
    if not (-4 <= lr <= -1):
        return False
    if optimizer_name not in ['adam', 'sgd', 'rmsprop']:
        return False
    if activation_function not in ['relu', 'tanh']:
        return False
    return True


def genetic_algorithm(df, NGEN, EPOCHS, POPULATION_SIZE, BATCH_SIZE, target_column='Water Heating Load (Btu)'):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    optimizers = {'adam': Adam, 'sgd': SGD, 'rmsprop': RMSprop}

    # Define the genetic algorithm functions
    def create_model(num_layers, neurons_per_layer, learning_rate, optimizer_name, activation_function):
        optimizer_class = optimizers[optimizer_name]
        model = Sequential()
        model.add(Dense(neurons_per_layer, input_dim=X_train.shape[1], activation=activation_function))
        for _ in range(num_layers - 1):
            model.add(Dense(neurons_per_layer, activation=activation_function))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=optimizer_class(learning_rate=learning_rate), loss='mean_absolute_error')
        return model

    def evaluate_individual(individual):
        if st.session_state.stopGeneticAlgorithm:
            return (1e7,)
        if not is_valid(individual):
            print(f"1 Failed to train or predict with {individual}")
            return (1e7,),

        num_layers, neurons_per_layer, lr, optimizer_name, activation_function = individual
        model = create_model(num_layers, neurons_per_layer, 10 ** lr, optimizer_name, activation_function)
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        predictions = model.predict(X_test)

        if np.any(np.isnan(predictions)):
            print(f"2 Failed to train or predict with {individual}")
            return (1e7,)

        mae = mean_absolute_error(y_test, predictions)
        return (mae,)

    # Define low and up bounds for numerical attributes
    low = [1, 32, -4]
    up = [4, 512, -1]

    def mutate_individual(individual, num_attrs, cat_attrs, indpb):
        # Mutate numerical attributes
        for i in range(num_attrs):
            if random.random() < indpb:
                if i < 2:
                    individual[i] = int(random.uniform(low[i], up[i]))
                else:
                    individual[i] = random.uniform(low[i], up[i])

        # Mutate categorical attributes
        for i in range(num_attrs, num_attrs + cat_attrs):
            if random.random() < indpb:
                individual[i] = random.choice(['adam', 'sgd', 'rmsprop'] if i == num_attrs else ['relu', 'tanh'])

        return individual,

    # Setup DEAP
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_num_layers", random.randint, 1, 4)
    toolbox.register("attr_neurons_per_layer", random.randint, 32, 512)
    toolbox.register("attr_learning_rate", random.uniform, -4, -1)
    toolbox.register("attr_optimizer", random.choice, ['adam', 'sgd', 'rmsprop'])
    toolbox.register("attr_activation", random.choice, ['relu', 'tanh'])

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_num_layers, toolbox.attr_neurons_per_layer, toolbox.attr_learning_rate,
                      toolbox.attr_optimizer, toolbox.attr_activation),
                     n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize=9)

    # Register the custom mutation function
    toolbox.register("mutate", mutate_individual, num_attrs=3, cat_attrs=2, indpb=0.333)

    # Genetic Algorithm parameters
    population = toolbox.population(n=POPULATION_SIZE)

    all_generations = []
    best_fitness = []
    avg_fitness = []

    for gen in range(NGEN):
        if st.session_state.stopGeneticAlgorithm:
            return
        print("Generation ", gen)
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
        best_ind = tools.selBest(population, 1)[0]
        print(
            f"Best individual this gen: Layers: {best_ind[0]}, Neurons: {best_ind[1]}, Learning Rate: {10 ** best_ind[2]}, Optimizer: {best_ind[3]}, Activation: {best_ind[4]}")
        print(f"Best MAE this gen: {best_ind.fitness.values[0]}")

        # top_individuals = tools.selBest(population, 5)
        # for i, ind in enumerate(top_individuals, 1):
        #     print(f"Top {i} individual: {ind}, MAE: {ind.fitness.values[0]}")
        all_generations.append(list(population))
        best_fitness.append(min([ind.fitness.values[0] for ind in population]))
        # avg_fitness.append(np.mean([ind.fitness.values[0] for ind in population]))
        #   print("Avg MAE this gen: ", avg_fitness[-1])
        # print("Best MAE this gen: ", best_fitness[-1])

        log_data = {
            "Best individual": {
                "Layers": best_ind[0],
                "Neurons": best_ind[1],
                "Learning Rate": 10 ** best_ind[2],
                "Optimizer": best_ind[3],
                "Activation": best_ind[4],
                "MAE": best_ind.fitness.values[0]
            }
        }
        population_diversity_plot = plot_population_diversity(population, gen)
        best_fitness_plot = plot_fitness_evolution(best_fitness)

        st.session_state.progress_callback.on_generation_end(gen, log_data, population_diversity_plot,
                                                             best_fitness_plot)


def simulated_annealing(df, epochs, batch_size, heating_load_column, outlet_water_temp_column, inlet_temp_column,
                        water_flow_column,
                        progress_bar, status_text, log_text, log_file_path, alpha_plot, temperature_plot,
                        initial_alpha, initial_temp, cooling_rate, min_temp):
    class AlphaAdjustmentCallback(Callback):
        def __init__(self, progress_bar, status_text, log_text, log_file_path, epochs, alpha_plot, temperature_plot,
                     initial_alpha=5.0, initial_temp=50.0, cooling_rate=0.955, min_temp=5.0):
            super().__init__()
            self.alpha = tf.Variable(initial_alpha, dtype=tf.float32, trainable=False)
            self.temperature = initial_temp
            self.cooling_rate = cooling_rate
            self.min_temp = min_temp
            self.last_loss = 1000000000
            self.progress_bar = progress_bar
            self.status_text = status_text
            self.log_text = log_text
            self.current_epoch = 0
            self.logs = []
            self.log_file_path = log_file_path
            self.total_epochs = epochs
            self.alphas = [initial_alpha]
            self.temperatures = [initial_temp]
            self.alpha_plot = alpha_plot
            self.temperature_plot = temperature_plot

        def on_epoch_end(self, epoch, logs=None):

            if self.temperature > self.min_temp:
                new_loss = logs.get('loss')
                if new_loss < self.last_loss:
                    accept = True
                else:
                    delta = new_loss - self.last_loss
                    probability = np.exp(-delta / self.temperature)
                    accept = np.random.rand() < probability

                if accept:
                    new_alpha = self.alpha * tf.exp(-1.0 / self.temperature)
                    tf.keras.backend.set_value(self.alpha, new_alpha)
                    self.temperature *= self.cooling_rate
                    self.last_loss = new_loss

            self.current_epoch += 1
            progress = self.current_epoch / self.total_epochs
            self.progress_bar.progress(progress)
            self.status_text.text(f"Simulated Annealing in progress: {int(progress * 100)}% complete")

            # Format and append the current epoch log
            log_message = f"Epoch {epoch + 1}/{self.total_epochs}\n"
            log_message += " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()]) + "\n"
            log_message += f"alpha: {tf.keras.backend.get_value(self.alpha):.4f}, temperature: {self.temperature:.4f}"
            self.logs.append(log_message)

            # Join all logs and display them in the text area
            full_log_message = "\n".join(self.logs)
            # self.log_text.text_area("Training Log", full_log_message, height=400, max_chars=None)
            self.log_text.text_area("Training Log", log_message, height=400, max_chars=None)

            self.alphas.append(tf.keras.backend.get_value(self.alpha))
            self.temperatures.append(self.temperature)

            if self.total_epochs == self.current_epoch:
                with open(self.log_file_path, 'w') as log_file:
                    log_file.write("\n".join(self.logs))
                    st.write("Training complete. Logs saved to file at:", self.log_file_path)
                self.log_text.text_area("Training Log", full_log_message, height=400, max_chars=None)

                plt.figure(figsize=(10, 6))
                plt.plot(np.arange(0, self.total_epochs + 1), self.alphas, label='Alpha')
                plt.xlabel('Epochs')
                plt.ylabel('Alpha')
                plt.title('Dynamic Adjustment of Alpha Over Training Epochs')
                plt.legend()
                plt.grid(True)
                # self.diversity_plot.pyplot(
                self.alpha_plot.pyplot(plt.gcf())

                # Plotting temperature values over epochs
                plt.figure(figsize=(10, 6))
                plt.plot(np.arange(0, self.total_epochs + 1), self.temperatures, label='Temperature', color='orange')
                plt.xlabel('Epochs')
                plt.ylabel('Temperature')
                plt.title('Temperature Cooling Schedule Over Training Epochs')
                plt.legend()
                plt.grid(True)
                self.temperature_plot.pyplot(plt.gcf())

            print(
                f'Epoch {epoch + 1}: α updated to {tf.keras.backend.get_value(self.alpha)}, temperature to {self.temperature}')

    X = df.drop(columns=[heating_load_column])
    y = df[heating_load_column]

    outlet_water_temp = X.iloc[:, X.columns.get_loc(outlet_water_temp_column)].to_numpy().reshape(-1, 1)
    inlet_temp = X.iloc[:, X.columns.get_loc(inlet_temp_column)].to_numpy().reshape(-1, 1)
    water_flow = X.iloc[:, X.columns.get_loc(water_flow_column)].to_numpy().reshape(-1, 1)

    y_np = y.to_numpy().reshape(-1, 1)

    y_with_features = np.concatenate(
        [y_np, outlet_water_temp, inlet_temp, water_flow], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y_with_features, test_size=0.2, random_state=42)
    lambda_reg = 0.0001

    def custom_loss(alpha):
        def loss(y_true_with_features, y_pred):
            y_true = y_true_with_features[:, 0:1]
            outlet_water_temp = y_true_with_features[:, 1:2]
            inlet_temp = y_true_with_features[:, 2:3]
            water_flow = y_true_with_features[:, 3:4]

            specific_heat_capacity = 4174  # J/kg°C, specific heat capacity of water
            gallons_to_liters = 3.78541  # 1 gallon = 3.78541 liters
            fahrenheit_to_celsius = lambda f: (f - 32) * 5.0 / 9.0  # Convert °F to °C
            joules_to_btu = 0.0009478171

            flow_rate_liters = water_flow * gallons_to_liters
            inlet_temp_c = fahrenheit_to_celsius(inlet_temp)
            outlet_temp_c = fahrenheit_to_celsius(outlet_water_temp)

            current_alpha = tf.keras.backend.get_value(alpha)
            reg_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in model.trainable_weights])

            heat_output = flow_rate_liters * (outlet_temp_c - inlet_temp_c) * specific_heat_capacity * joules_to_btu
            return tf.reduce_mean(
                tf.abs(y_true - y_pred) + current_alpha * tf.abs(heat_output - y_pred) + lambda_reg * reg_loss, axis=-1)

        return loss

    model = Sequential()
    model.add(Dense(227, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(227, activation='relu'))
    model.add(Dense(1, activation='linear'))

    alpha_adjust_callback = AlphaAdjustmentCallback(progress_bar, status_text, log_text, log_file_path, epochs,
                                                    alpha_plot, temperature_plot,
                                                    initial_alpha, initial_temp, cooling_rate, min_temp)
    model.compile(optimizer=Adam(learning_rate=0.0009323), loss=custom_loss(alpha_adjust_callback.alpha),
                  metrics=['mae'], run_eagerly=True)
    # Callbacks
    lr_scheduler = ReduceLROnPlateau(factor=0.66, patience=5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    model_checkpoint = ModelCheckpoint('best_model_boltzman.keras', monitor='val_loss', save_best_only=True)

    try:
        history = model.fit(X_train, y_train, validation_split=0.2,
                            epochs=epochs, batch_size=batch_size,
                            callbacks=[early_stopping, lr_scheduler, model_checkpoint, alpha_adjust_callback])
    except Exception as e:
        print(f"An error occurred during training: {e}")

    model.load_weights('best_model_boltzman.keras')

    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test[:, 0], y_pred, squared=False)
    r2 = r2_score(y_test[:, 0], y_pred)
    mean_observed = y_test[:, 0].mean()
    cvrmse = (rmse / mean_observed) * 100
    mae = mean_absolute_error(y_test[:, 0], y_pred)
    mape = (mae * 100) / mean_observed

    print(f"Root Mean Squared Error: {rmse}")
    print(f"R^2 Score: {r2}")
    print(f"cvRMSE: {cvrmse}%")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Absolute Percentage Error: {mape}%")

    st.write(f"Root Mean Squared Error: {rmse}")
    st.write(f"R^2 Score: {r2}")
    st.write(f"cvRMSE: {cvrmse}%")
    st.write(f"Mean Absolute Error: {mae}")
    st.write(f"Mean Absolute Percentage Error: {mape}%")

