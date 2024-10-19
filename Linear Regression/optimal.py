import numpy as np
import pandas as pd

# Load the training and test data
train_data = pd.read_csv('concrete/train.csv')
test_data = pd.read_csv('concrete/test.csv')

# Assuming the last column is the target and the rest are features
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Function to calculate the optimal weight vector using the analytical formula
def calculate_optimal_weights(X, y):
    X_bias = np.hstack([np.ones((X.shape[0], 1)), X])  # Adding bias term
    # w* = (X^T X)^-1 X^T y
    optimal_weights = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
    return optimal_weights

# Calculate the optimal weights using the closed-form solution
optimal_weights = calculate_optimal_weights(X_train, y_train)

# Calculate the test MSE for the analytical solution
X_test_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
test_predictions_optimal = np.dot(X_test_bias, optimal_weights)
test_mse_optimal = np.mean((test_predictions_optimal - y_test) ** 2)

# Print the optimal weight vector and the MSE for the test set
print("Optimal weight vector:", optimal_weights)
print("Test MSE using optimal weights:", test_mse_optimal)
