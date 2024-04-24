import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Forward Kinematics function
def forward_kinematics(theta1, theta2):
    # Lengths of the robot links
    l1 = 1.0
    l2 = 1.0

    # Calculate end-effector position (x, y)
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    return x, y

# Inverse Kinematics function
def inverse_kinematics(x, y):
    # Lengths of the robot links
    l1 = 1.0
    l2 = 1.0

    # Calculate joint angles (θ1, θ2) using inverse kinematics
    # This is just a placeholder example, replace it with your actual inverse kinematics solution
    theta2 = np.arccos((x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2))
    theta1 = np.arctan2(y, x) - np.arctan2((l2 * np.sin(theta2)), (l1 + l2 * np.cos(theta2)))
    
    return theta1, theta2

# Generate sample data for training using inverse kinematics
def generate_data(num_samples):
    X = np.random.uniform(low=-1, high=1, size=(num_samples, 2))  # End-effector positions (x, y)
    y = np.zeros((num_samples, 2))  # Joint angles (θ1, θ2)
    for i in range(num_samples):
        x, y_ = X[i]
        theta1, theta2 = inverse_kinematics(x, y_)
        y[i] = [theta1, theta2]
    return X, y

# Define neural network architecture
model = Sequential()
model.add(Dense(128, input_shape=(2,), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(2))  # Output layer, no activation for regression

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Generate training data
X_train, y_train = generate_data(100000)

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Test the model with some sample input
sample_input = np.array([[1.0, 1.0]])  # Example end-effector position (x, y)
predicted_angles = model.predict(sample_input)
print("Predicted joint angles:", predicted_angles)
print(forward_kinematics(predicted_angles[0][0],predicted_angles[0][1]))