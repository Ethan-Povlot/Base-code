import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
#this is probably worse than the annealing code, I am no longer updating this code, this is older just adding it here now


def reshape_data(X, Y, look_back):
    X_list, Y_list = [], []
    for i in range(len(X) - look_back):
        X_list.append(X.iloc[i:(i + look_back)].values)
        Y_list.append(Y.iloc[i + look_back].values)
    return np.array(X_list), np.array(Y_list)

def build_model(units=50, dense_nodes=10, activation='relu', optimizer='adam', 
                dropout_rate=0.2, num_lstm_layers=1, num_gru_layers=0, 
                num_dense_layers=1, look_back=1):
    
    model = Sequential()
    for i in range(num_lstm_layers):
        if i == 0:
            model.add(LSTM(units, input_shape=(look_back, X.shape[1]), return_sequences=True))
        else:
            model.add(LSTM(units, return_sequences=(i < num_lstm_layers - 1)))
        model.add(Dropout(dropout_rate))
    
    for i in range(num_gru_layers):
        if num_lstm_layers == 0 and i == 0: 
            model.add(GRU(units, input_shape=(look_back, X.shape[1]), return_sequences=True))
        else:
            model.add(GRU(units, return_sequences=(i < num_gru_layers - 1)))
        model.add(Dropout(dropout_rate))
    
    for _ in range(num_dense_layers):
        model.add(Dense(dense_nodes, activation=activation))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer=optimizer, loss='mse')
    return model

X_reshaped, Y_reshaped = reshape_data(X, Y, look_back=1)

model = KerasRegressor(
    model=build_model, 
    units=50, 
    dense_nodes=10, 
    activation='relu', 
    optimizer='adam',
    dropout_rate=0.2,
    num_lstm_layers=1,
    num_gru_layers=0,
    num_dense_layers=1,
    look_back=1,
    batch_size=None, 
    epochs=10, 
    verbose=0
)

param_grid = {
    'units': [50, 100],  
    'dense_nodes': [10, 20], 
    'activation': ['relu', 'tanh'],
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.2, 0.3],
    'num_lstm_layers': [1, 2],
    'num_gru_layers': [0, 1],
    'num_dense_layers': [1, 2],
    'look_back': [1, 3, 5],
    'batch_size': [16, 32, 64],
    'epochs': [10, 20]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_result = grid.fit(X_reshaped, Y_reshaped)

print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

best_model = grid_result.best_estimator_.model_
best_model.save('best_model.keras')
