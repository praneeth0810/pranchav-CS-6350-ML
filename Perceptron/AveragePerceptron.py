import pandas as pd

train_data = pd.read_csv("bank-note/train.csv", header=None)
test_data = pd.read_csv("bank-note/test.csv", header=None)

train_data[4] = train_data[4].apply(lambda x: -1 if x == 0 else 1)
test_data[4] = test_data[4].apply(lambda x: -1 if x == 0 else 1)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

def average_perceptron_train(X, y, learning_rate=0.01, epochs=10):
    weights = [0.0] * X.shape[1]
    bias = 0.0
    cumulative_weights = [0.0] * X.shape[1]
    cumulative_bias = 0.0
    total_updates = 0 

    for epoch in range(epochs):
        for xi, target in zip(X, y):
            activation = sum(weight * feature for weight, feature in zip(weights, xi)) + bias
            prediction = 1 if activation >= 0 else -1
            
            if prediction != target:
                update = learning_rate * target
                weights = [w + update * x for w, x in zip(weights, xi)]
                bias += update

            cumulative_weights = [cw + w for cw, w in zip(cumulative_weights, weights)]
            cumulative_bias += bias
            total_updates += 1
    
    average_weights = [cw / total_updates for cw in cumulative_weights]
    average_bias = cumulative_bias / total_updates
    return average_weights, average_bias

def perceptron_predict_average(X, weights, bias):
    predictions = []
    for xi in X:
        activation = sum(weight * feature for weight, feature in zip(weights, xi)) + bias
        prediction = 1 if activation >= 0 else -1
        predictions.append(prediction)
    return predictions

average_weights, average_bias = average_perceptron_train(X_train, y_train)

y_pred_average = perceptron_predict_average(X_test, average_weights, average_bias)

average_test_error_average = 1 - (sum(1 for actual, predicted in zip(y_test, y_pred_average) if actual == predicted) / len(y_test))

print("Learned Average Weight Vector:", average_weights)
print("Average Test Error on Test Dataset:", average_test_error_average)
