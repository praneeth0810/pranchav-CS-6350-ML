import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Load the training and test data
train_data = pd.read_csv('concrete/train.csv')
test_data = pd.read_csv('concrete/test.csv')

# Extract features and target
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

def stochastic_gradient_descent(X, y, initial_lr=0.01, epochs=50, tolerance=1e-6):
    np.random.seed(0)  # for reproducibility
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # Adding bias term
    weights = np.zeros(X.shape[1])  # Initialize weights to zero
    learning_rate = initial_lr
    costs = []
    previous_cost = float('inf')
    
    # Performing epochs
    for epoch in range(epochs):
        for i in range(len(y)):
            # Randomly pick one data point
            index = np.random.randint(0, len(y))
            xi = X[index]
            yi = y[index]

            # Compute prediction
            prediction = np.dot(xi, weights)
            error = prediction - yi
            gradient = xi * error  # Stochastic gradient

            # Update weights
            weights -= learning_rate * gradient

            # Calculate cost (MSE) for convergence check
            predictions = np.dot(X, weights)
            cost = np.mean((predictions - y) ** 2)
            costs.append(cost)

            # Check for convergence
            if abs(previous_cost - cost) < tolerance:
                return weights, costs, learning_rate
            previous_cost = cost

        # Reduce learning rate
        learning_rate /= 1.02

    return weights, costs, learning_rate

# Run SGD
final_weights, cost_history, final_lr = stochastic_gradient_descent(X_train, y_train)

# Plotting the cost function changes over updates
plt.figure(figsize=(10, 6))
plt.plot(cost_history, marker='o', linestyle='-', markersize=4)
plt.title('Cost Function Values vs. Number of Updates')
plt.xlabel('Number of Updates')
plt.ylabel('Cost Function Value')
plt.grid(True)
plt.show()

# Calculate Mean Squared Error (MSE) on test data
X_test_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
test_predictions = np.dot(X_test_bias, final_weights)
test_mse = np.mean((test_predictions - y_test) ** 2)

# Print the results
print("Final learning rate:", final_lr)
print("Final weight vector:", final_weights)
print("Cost function value on test data:", test_mse)

'''
Final learning rate: 0.005975792848316459
Final weight vector: [ 0.03044219 -0.04116582 -0.17005131 -0.223324  0.51840129 -0.016597480.25025085 -0.07470251]
Cost function value on test data: 0.9175234682486892
'''