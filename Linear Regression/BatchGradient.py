import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('concrete/train.csv')
test_data = pd.read_csv('concrete/test.csv')

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

def batch_gradient_descent(X, y, initial_lr=1.0, tolerance=1e-6, max_epochs=1000):
    X = np.hstack([np.ones((X.shape[0], 1)), X]) 
    weights = np.zeros(X.shape[1])
    learning_rate = initial_lr
    costs = []


    for epoch in range(max_epochs):
        predictions = np.dot(X, weights)
        errors = predictions - y
        gradient = np.dot(X.T, errors) / len(y)
        new_weights = weights - learning_rate * gradient

        norm_diff = np.linalg.norm(new_weights - weights)


        cost = np.mean(errors ** 2)
        costs.append(cost)

        if norm_diff < tolerance:
            break

        weights = new_weights

        if learning_rate > 0.001 and norm_diff > tolerance:
            learning_rate *= 0.5

    return weights, costs, learning_rate

final_weights, cost_history, final_lr = batch_gradient_descent(X_train, y_train)

plt.figure(figsize=(10, 6))
plt.plot(cost_history, marker='o', linestyle='-', markersize=4, color='blue')
plt.title('Cost Function vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost Function Value')
plt.grid(True)
plt.show()

X_test_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
test_predictions = np.dot(X_test_bias, final_weights)
test_cost = np.mean((test_predictions - y_test) ** 2)

print("Final learning rate:", final_lr)
print("Cost of the test data:", test_cost)
print("Final weight vector:", final_weights)


'''

Final learning rate: 0.0009765625
Cost of the test data: 0.8367320770004505
Final weight vector: [-0.03058548 -0.03616832 -0.20504727 -0.1964894   0.42849298 -0.037361670.15104357 -0.02493451]

'''
