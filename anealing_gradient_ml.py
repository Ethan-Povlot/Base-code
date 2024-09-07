import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

#define datasets as a lisy of lists, [[X1, Y1], [X2, Y2], ...]

def reshape_data(X, Y, look_back):
    X_list, Y_list = [], []
    for i in range(len(X) - look_back):
        X_list.append(X.iloc[i:(i + look_back)].values)
        Y_list.append(Y.iloc[i + look_back].values)
    return np.array(X_list), np.array(Y_list)

def normalize_data(X, Y):
    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()

    X_normalized = X_scaler.fit_transform(X)
    Y_normalized = Y_scaler.fit_transform(Y.values.reshape(-1, 1))

    return X_normalized, Y_normalized, X_scaler, Y_scaler

def build_model(lstm_units_list=None, gru_units_list=None, dense_node_list=None, 
                activation='relu', optimizer='adam', dropout_rate=0.2, input_shape=None):
    
    model = Sequential()
    
    if lstm_units_list is not None:
        for i, units in enumerate(lstm_units_list):
            if i == 0:
                model.add(LSTM(units, input_shape=input_shape, return_sequences=(i < len(lstm_units_list) - 1)))
            else:
                model.add(LSTM(units, return_sequences=(i < len(lstm_units_list) - 1)))
            model.add(Dropout(dropout_rate))
    
    if gru_units_list is not None:
        for i, units in enumerate(gru_units_list):
            if lstm_units_list is None and i == 0:
                model.add(GRU(units, input_shape=input_shape, return_sequences=(i < len(gru_units_list) - 1)))
            else:
                model.add(GRU(units, return_sequences=(i < len(gru_units_list) - 1)))
            model.add(Dropout(dropout_rate))
    
    if dense_node_list is not None:
        for nodes in dense_node_list:
            model.add(Dense(nodes, activation=activation))
            model.add(Dropout(dropout_rate))
    
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer=optimizer, loss='mse')
    return model

def evaluate_model(params, X_train, Y_train, X_val, Y_val):
    model = build_model(**params, input_shape=(X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, Y_train, batch_size=params['batch_size'], epochs=params['epochs'], verbose=0)
    loss = model.evaluate(X_val, Y_val, verbose=0)
    return loss

def simulated_annealing_search(initial_params, X_train, Y_train, X_val, Y_val, max_iterations=10, initial_temperature=1.0, cooling_rate=0.9, tolerance=1e-3):
    current_params = initial_params.copy()
    best_params = initial_params.copy()
    best_score = evaluate_model(best_params, X_train, Y_train, X_val, Y_val)
    
    temperature = initial_temperature
    
    for iteration in range(max_iterations):
        improved = False
        
        param_steps = {
            'units': [int(temperature * 10), int(-temperature * 10)],
            'dense_nodes': [int(temperature * 5), int(-temperature * 5)],
            'dropout_rate': [temperature * 0.05, -temperature * 0.05],
            'num_lstm_layers': [int(temperature * 1), int(-temperature * 1)],
            'num_gru_layers': [int(temperature * 1), int(-temperature * 1)],
            'num_dense_layers': [int(temperature * 1), int(-temperature * 1)],
            'look_back': [int(temperature * 1), int(-temperature * 1)],
            'batch_size': [int(temperature * 16), int(-temperature * 16)],
            'epochs': [int(temperature * 10), int(-temperature * 10)]
        }
        
        activation_options = ['relu', 'tanh', 'sigmoid']
        optimizer_options = ['adam', 'rmsprop', 'sgd']
        
        for activation in activation_options:
            if activation != current_params['activation']:
                new_params = current_params.copy()
                new_params['activation'] = activation
                score = evaluate_model(new_params, X_train, Y_train, X_val, Y_val)
                if score < best_score - tolerance:
                    best_score = score
                    best_params = new_params
                    improved = True
                    print(f"Iteration {iteration + 1}: Changed activation to {activation} with score {best_score}")
        
        for optimizer in optimizer_options:
            if optimizer != current_params['optimizer']:
                new_params = current_params.copy()
                new_params['optimizer'] = optimizer
                score = evaluate_model(new_params, X_train, Y_train, X_val, Y_val)
                if score < best_score - tolerance:
                    best_score = score
                    best_params = new_params
                    improved = True
                    print(f"Iteration {iteration + 1}: Changed optimizer to {optimizer} with score {best_score}")
        
        for key in param_steps:
            current_value = current_params[key]
            for step in param_steps[key]:
                new_value = current_value + step
                if new_value > 0: 
                    new_params = current_params.copy()
                    new_params[key] = new_value
                    score = evaluate_model(new_params, X_train, Y_train, X_val, Y_val)
                    if score < best_score - tolerance: #yay
                        best_score = score
                        best_params = new_params
                        improved = True
                        print(f"Iteration {iteration + 1}: Improved {key} to {new_value} with score {best_score}")
        
        if not improved:
            print(f"No further improvements after {iteration + 1} iterations.")
            break
        
        temperature *= cooling_rate
    
    return best_params, best_score

# Initial hyperparameters
initial_best_params = {
    'lstm_units_list': [50],
    'gru_units_list': [],
    'dense_node_list': [10],
    'activation': 'relu',
    'optimizer': 'adam',
    'dropout_rate': 0.2,
    'look_back': 1,
    'batch_size': 32,
    'epochs': 10
}

X_combined = []
Y_combined = []
scalers = []

for X, Y in datasets:
    X_normalized, Y_normalized, X_scaler, Y_scaler = normalize_data(X, Y)
    X_combined.append(X_normalized)
    Y_combined.append(Y_normalized)
    scalers.append((X_scaler, Y_scaler))

X_combined = np.vstack(X_combined)
Y_combined = np.vstack(Y_combined)

look_back = initial_best_params['look_back']
X_reshaped, Y_reshaped = reshape_data(pd.DataFrame(X_combined), pd.DataFrame(Y_combined), look_back)

X_train, X_val, Y_train, Y_val = train_test_split(X_reshaped, Y_reshaped, test_size=0.2, random_state=42)

num_instantiations = 15
best_global_params = None
best_global_score = float('inf')

for i in range(num_instantiations):
    print(f"Random instantiation {i+1}")
    
    random_params = {
        'lstm_units_list': [np.random.randint(20, 100) for _ in range(np.random.randint(1, 3))],
        'gru_units_list': [np.random.randint(20, 100) for _ in range(np.random.randint(0, 2))],
        'dense_node_list': [np.random.randint(5, 50) for _ in range(np.random.randint(1, 3))],
        'activation': np.random.choice(['relu', 'tanh', 'sigmoid']),
        'optimizer': np.random.choice(['adam', 'rmsprop', 'sgd']),
        'dropout_rate': np.random.uniform(0.1, 0.5),
        'look_back': np.random.randint(1, 5),
        'batch_size': np.random.choice([16, 32, 64]),
        'epochs': np.random.choice([10, 20, 30])
    }
    
    best_params, best_score = simulated_annealing_search(random_params, X_train, Y_train, X_val, Y_val)
    
    if best_score < best_global_score:
        best_global_score = best_score
        best_global_params = best_params
        print(f"New best global parameters found: {best_global_params} with score {best_global_score}")

print(f"Best global parameters after all instantiations: {best_global_params}")
print(f"Best validation score from combined dataset: {best_global_score}")

base_model = build_model(**best_global_params, input_shape=(X_train.shape[1], X_train.shape[2]))
base_model.fit(X_train, Y_train, batch_size=best_global_params['batch_size'], epochs=best_global_params['epochs'], verbose=1)

for i, (X, Y) in enumerate(datasets):
    print(f"Fine-tuning on dataset {i+1}")
    
    # Normalize the data
    X_normalized, Y_normalized = scalers[i][0].transform(X), scalers[i][1].transform(Y.reshape(-1, 1))
    
    X_reshaped, Y_reshaped = reshape_data(pd.DataFrame(X_normalized), pd.DataFrame(Y_normalized), best_global_params['look_back'])
    
    # Split the data into training and val
    X_train, X_val, Y_train, Y_val = train_test_split(X_reshaped, Y_reshaped, test_size=0.2, random_state=42)
    
    fine_tuned_model = tf.keras.models.clone_model(base_model)
    fine_tuned_model.set_weights(base_model.get_weights())  
    fine_tuned_model.compile(optimizer=best_global_params['optimizer'], loss='mse') 
    fine_tuned_model.fit(X_train, Y_train, batch_size=best_global_params['batch_size'], epochs=2, verbose=1)
    
    model_name = f'fine_tuned_model_dataset_{i+1}.keras'
    fine_tuned_model.save(model_name)
    print(f"Saved fine-tuned model for dataset {i+1} as {model_name}')







def simulated_annealing_search(initial_params, X_train, Y_train, X_val, Y_val, max_iterations=10, initial_temperature=1.0, cooling_rate=0.9, tolerance=1e-3):
    current_params = initial_params.copy()
    best_params = initial_params.copy()
    best_score = evaluate_model(best_params, X_train, Y_train, X_val, Y_val)
    
    temperature = initial_temperature
    
    for iteration in range(max_iterations):
        improved = False
        
        # Scale the step sizes based on the current temperature
        param_steps = {
            'lstm_units_list': [int(temperature * 10), int(-temperature * 10)],
            'gru_units_list': [int(temperature * 10), int(-temperature * 10)],  
            'dense_node_list': [int(temperature * 5), int(-temperature * 5)],
            'dropout_rate': [temperature * 0.05, -temperature * 0.05],
            'look_back': [int(temperature * 1), int(-temperature * 1)],
            'batch_size': [int(temperature * 16), int(-temperature * 16)],
            'epochs': [int(temperature * 10), int(-temperature * 10)]
        }
        
        activation_options = ['relu', 'tanh', 'sigmoid']
        optimizer_options = ['adam', 'rmsprop', 'sgd']
        
        for activation in activation_options:
            if activation != current_params['activation']:
                new_params = current_params.copy()
                new_params['activation'] = activation
                score = evaluate_model(new_params, X_train, Y_train, X_val, Y_val)
                if score < best_score - tolerance:
                    best_score = score
                    best_params = new_params
                    improved = True
                    print(f"Iteration {iteration + 1}: Changed activation to {activation} with score {best_score}")
        
        for optimizer in optimizer_options:
            if optimizer != current_params['optimizer']:
                new_params = current_params.copy()
                new_params['optimizer'] = optimizer
                score = evaluate_model(new_params, X_train, Y_train, X_val, Y_val)
                if score < best_score - tolerance:
                    best_score = score
                    best_params = new_params
                    improved = True
                    print(f"Iteration {iteration + 1}: Changed optimizer to {optimizer} with score {best_score}")
        
        for key in param_steps:
            current_value = current_params[key]
            if isinstance(current_value, list):  # Handle lists like lstm_units_list, gru_units_list, etc.
                for i in range(len(current_value)):
                    for step in param_steps[key]:
                        new_value = current_value[i] + step
                        if new_value > 0:  # Ensure the parameters remain positive and valid
                            new_params = current_params.copy()
                            new_params[key][i] = new_value
                            score = evaluate_model(new_params, X_train, Y_train, X_val, Y_val)
                            if score < best_score - tolerance:  # Improvement found
                                best_score = score
                                best_params = new_params
                                improved = True
                                print(f"Iteration {iteration + 1}: Improved {key} to {new_value} with score {best_score}")
            else:  # Handle scalar parameters like dropout_rate, batch_size, etc.
                for step in param_steps[key]:
                    new_value = current_value + step
                    if new_value > 0:  # Ensure the parameters remain positive and valid
                        new_params = current_params.copy()
                        new_params[key] = new_value
                        score = evaluate_model(new_params, X_train, Y_train, X_val, Y_val)
                        if score < best_score - tolerance:  # Improvement found
                            best_score = score
                            best_params = new_params
                            improved = True
                            print(f"Iteration {iteration + 1}: Improved {key} to {new_value} with score {best_score}")
        
        if not improved: 
            print(f"No further improvements after {iteration + 1} iterations.")
            break
        
        temperature *= cooling_rate
    
    return best_params, best_score
