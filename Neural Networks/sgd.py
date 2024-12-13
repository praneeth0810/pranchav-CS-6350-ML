import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self, width):
        """Initialize the neural network."""
        self.width = width
        self.params = {}

    def initialize_weights(self, input_size, initialize_random=True):
        """Initialize the network weights and biases."""
        if initialize_random:
            self.params = {
                "W1": np.random.normal(size=(input_size, self.width)),
                "b1": np.random.normal(size=(self.width,)),
                "W2": np.random.normal(size=(self.width, self.width)),
                "b2": np.random.normal(size=(self.width,)),
                "W3": np.random.normal(size=(self.width, 1)),
                "b3": np.random.normal(size=(1,))
            }
        else:
            self.params = {
                "W1": np.zeros((input_size, self.width)),
                "b1": np.zeros((self.width,)),
                "W2": np.zeros((self.width, self.width)),
                "b2": np.zeros((self.width,)),
                "W3": np.zeros((self.width, 1)),
                "b3": np.zeros((1,))
            }
    
    def sigmoid(self, x):
        """Sigmoid activation function with overflow prevention."""
        x = np.clip(x, -500, 500)  # Avoid overflow in exp()
        return 1 / (1 + np.exp(-x))


    def forward_pass(self, X):
        """Forward pass through the neural network."""
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        S1 = np.dot(X, W1) + b1
        Z1 = self.sigmoid(S1)
        S2 = np.dot(Z1, W2) + b2
        Z2 = self.sigmoid(S2)
        output = np.dot(Z2, W3) + b3

        return output, (S1, Z1, S2, Z2)

    def backward_propagation(self, X, y, output, forward_cache):
        """Backward propagation to compute gradients."""
        S1, Z1, S2, Z2 = forward_cache
        W3 = self.params["W3"]

        # Gradients for output layer
        d_output = (output - y).reshape((1, 1))
        dW3 = np.dot(Z2.T, d_output)
        db3 = np.sum(d_output, axis=0)
        dZ2 = np.dot(d_output, W3.T)

        # Gradients for second layer
        dS2 = self.sigmoid(S2) * (1 - self.sigmoid(S2)) * dZ2
        dW2 = np.dot(Z1.T, dS2)
        db2 = np.sum(dS2, axis=0)
        dZ1 = np.dot(dS2, self.params["W2"].T)

        # Gradients for first layer
        dS1 = self.sigmoid(S1) * (1 - self.sigmoid(S1)) * dZ1
        dW1 = np.dot(X.T, dS1)
        db1 = np.sum(dS1, axis=0)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}

    def train(self, X, y, epochs, threshold, learning_rate, initialize_random=True, lr_schedule=None):
        """Train the neural network using gradient descent."""
        num_samples, input_size = X.shape
        self.initialize_weights(input_size, initialize_random)

        for epoch in range(epochs):
            if lr_schedule is not None:
                learning_rate = lr_schedule[epoch]

            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for i in indices:
                x = X[i].reshape(1, -1)
                output, forward_cache = self.forward_pass(x)
                gradients = self.backward_propagation(x, y[i], output, forward_cache)

                # Update weights and biases
                for key in self.params:
                    self.params[key] -= learning_rate * gradients["d" + key]

            # Compute error for early stopping
            outputs, _ = self.forward_pass(X)
            error = np.mean(0.5 * ((outputs.flatten() - y) ** 2))
            if epoch > 0 and abs(previous_error - error) < threshold:
                break
            previous_error = error

    def predict(self, X):
        """Predict output for given input data."""
        outputs, _ = self.forward_pass(X)
        return np.sign(outputs.flatten())

def prediction_helper(y):
    """Convert labels from {0, 1} to {-1, 1}."""
    y_copy = y.copy()
    y_copy[y_copy == 0] = -1
    return y_copy

def generate_learning_rate_schedule(base_rate, decay, T):
    """Generate a decaying learning rate schedule."""
    return base_rate / (1 + decay * np.arange(T))

if __name__ == "__main__":
    # Load datasets
    train_data = pd.read_csv('bank-note/train.csv', header=None)
    test_data = pd.read_csv('bank-note/test.csv', header=None)

    X_train = train_data.iloc[:, :4].values
    y_train = prediction_helper(train_data.iloc[:, 4].values)
    X_test = test_data.iloc[:, :4].values
    y_test = prediction_helper(test_data.iloc[:, 4].values)

    # Training with random initialization
    print("Training with Random Initialization:")
    widths = [5, 10, 25, 50, 100]
    for width in widths:
        learning_rate_schedule = generate_learning_rate_schedule(0.1, 0.1 / 0.01, 100)
        nn = NeuralNetwork(width)
        nn.train(X_train, y_train, epochs=100, threshold=1e-9, learning_rate=0.1, lr_schedule=learning_rate_schedule)

        train_error = np.mean(nn.predict(X_train) != y_train)
        test_error = np.mean(nn.predict(X_test) != y_test)

        print(f"Width: {width}, Train Error: {train_error:.4f}, Test Error: {test_error:.4f}")

    # Training with zero initialization
    print("\nTraining with Zero Initialization:")
    for width in widths:
        learning_rate_schedule = generate_learning_rate_schedule(0.1, 0.1 / 0.1, 100)
        nn = NeuralNetwork(width)
        nn.train(X_train, y_train, epochs=100, threshold=1e-9, learning_rate=0.1, initialize_random=False, lr_schedule=learning_rate_schedule)

        train_error = np.mean(nn.predict(X_train) != y_train)
        test_error = np.mean(nn.predict(X_test) != y_test)

        print(f"Width: {width}, Train Error: {train_error:.4f}, Test Error: {test_error:.4f}")
