import pandas as pd

train_data = pd.read_csv("bank-note/train.csv", header=None)
test_data = pd.read_csv("bank-note/test.csv", header=None)

train_data[4] = train_data[4].apply(lambda x: -1 if x == 0 else 1)
test_data[4] = test_data[4].apply(lambda x: -1 if x == 0 else 1)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

def standard_perceptron(X, y, learning_rate=0.01, epochs=10):
    weights = [0.0] * X.shape[1]
    bias = 0.0

    for epoch in range(epochs):
        for xi, target in zip(X, y):

            activation = sum(weight * feature for weight, feature in zip(weights, xi)) + bias
            prediction = 1 if activation >= 0 else -1
            
            if prediction != target:
                update = learning_rate * target
                weights = [w + update * x for w, x in zip(weights, xi)]
                bias += update
    return weights, bias

def perceptron_predict(X, weights, bias):
    predictions = []
    for xi in X:
        activation = sum(weight * feature for weight, feature in zip(weights, xi)) + bias
        prediction = 1 if activation >= 0 else -1
        predictions.append(prediction)
    return predictions

weights_fixed_epochs, bias_fixed_epochs = standard_perceptron(X_train, y_train)

y_pred_fixed_epochs = perceptron_predict(X_test, weights_fixed_epochs, bias_fixed_epochs)

average_prediction_error = 1 - (sum(1 for actual, predicted in zip(y_test, y_pred_fixed_epochs) if actual == predicted) / len(y_test))

print("Learned Weight Vector:", weights_fixed_epochs)
print("Average Prediction Error on Test Dataset:", average_prediction_error)
