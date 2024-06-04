import pandas as pd
import numpy as np
import csv
import tensorflow as tf
from keras import layers, models, regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import optuna

# Loading data from CSV file
def readData():
    inputs = []     # Inputs (positions x, y, z and orientations alpha, beta, gamma)
    outputs = []    # Outputs (joint angles q1 to q7)

    with open('panda_random_data.csv', mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:  
            inputs.append([float(coord) for coord in row[:6]])   # Add inputs (positions and orientations)
            outputs.append([float(angle) for angle in row[6:]])  # Add outputs (joint angles)
    return np.array(inputs), np.array(outputs)

# Loading and preparing data
inputs, outputs = readData()

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

# Normalize the features and labels
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train_scaled, y_train_scaled, test_size=0.2, random_state=42)

# Data augmentation
def augment_data(X, y, noise_level=0.01):
    noise_X = X + noise_level * np.random.randn(*X.shape)
    noise_y = y + noise_level * np.random.randn(*y.shape)
    return np.vstack([X, noise_X]), np.vstack([y, noise_y])

# Augment the training data
X_train_augmented, y_train_augmented = augment_data(X_train, y_train)

# Objective function for hyperparameter optimization
def objective(trial):
    model = models.Sequential()
    model.add(layers.Dense(trial.suggest_int('units_l1', 128, 512), activation='relu', input_shape=(6,)))
    model.add(layers.Dropout(trial.suggest_float('dropout_l1', 0.2, 0.5)))
    model.add(layers.Dense(trial.suggest_int('units_l2', 128, 512), activation='relu'))
    model.add(layers.Dropout(trial.suggest_float('dropout_l2', 0.2, 0.5)))
    model.add(layers.Dense(trial.suggest_int('units_l3', 64, 256), activation='relu'))
    model.add(layers.Dense(7))  # Output layer for 7 joint angles

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=trial.suggest_loguniform('lr', 1e-5, 1e-3)),
                  loss='mean_squared_error')
    
    # Train the model
    history = model.fit(X_train_augmented, y_train_augmented, epochs=100, batch_size=trial.suggest_int('batch_size', 32, 128),validation_split=0.2, verbose=0)
    
    # Evaluate the model on the test set
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    return test_loss

# Run the optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Get the best trial
best_trial = study.best_trial
print(f'Best trial: {best_trial.params}')

best_params = best_trial.params

model = models.Sequential()
model.add(layers.Dense(best_params['units_l1'], activation='relu', input_shape=(6,)))
model.add(layers.Dropout(best_params['dropout_l1']))
model.add(layers.Dense(best_params['units_l2'], activation='relu'))
model.add(layers.Dropout(best_params['dropout_l2']))
model.add(layers.Dense(best_params['units_l3'], activation='relu'))
model.add(layers.Dense(7))  # Output layer for 7 joint angles

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['lr']), loss='mean_squared_error')

history = model.fit(X_train_augmented, y_train_augmented, epochs=200, batch_size=best_params['batch_size'], validation_split=0.2)

test_loss = model.evaluate(X_test, y_test)
print(f'Test loss: {test_loss}')

# Example end-effector position for inference
new_position = np.array([[0.5, 0.2, 0.3, 0, 0, 0]])

# Normalize the new position
new_position_scaled = scaler_X.transform(new_position)

# Predict joint angles
predicted_joint_angles_scaled = model.predict(new_position_scaled)

# Inverse transform to get the original scale
predicted_joint_angles = scaler_y.inverse_transform(predicted_joint_angles_scaled)

print(f'Predicted joint angles: {predicted_joint_angles}')