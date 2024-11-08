import pandas as pd

train_data = pd.read_csv("bank-note/train.csv", header=None)
test_data = pd.read_csv("bank-note/test.csv", header=None)

train_data[4] = train_data[4].apply(lambda x: -1 if x == 0 else 1)
test_data[4] = test_data[4].apply(lambda x: -1 if x == 0 else 1)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

def voted_perceptron_train(X, y, learning_rate=0.01, epochs=10):
    weights = [0.0] * X.shape[1]
    bias = 0.0
    count = 1 
    weight_vectors = [] 
    counts = []
    for epoch in range(epochs):
        for xi, target in zip(X, y):
            activation = sum(weight * feature for weight, feature in zip(weights, xi)) + bias
            prediction = 1 if activation >= 0 else -1
            
            if prediction != target:
                weight_vectors.append((weights[:], bias))
                counts.append(count)
                update = learning_rate * target
                weights = [w + update * x for w, x in zip(weights, xi)]
                bias += update

                count = 1
            else:
                count += 1
    weight_vectors.append((weights, bias))
    counts.append(count)
    return weight_vectors, counts

def voted_perceptron_predict(X, weight_vectors, counts):
    predictions = []
    for xi in X:
        vote_sum = 0
        for (weights, bias), count in zip(weight_vectors, counts):

            activation = sum(weight * feature for weight, feature in zip(weights, xi)) + bias
            prediction = 1 if activation >= 0 else -1

            vote_sum += count * prediction
        final_prediction = 1 if vote_sum >= 0 else -1
        predictions.append(final_prediction)
    return predictions

voted_weight_vectors, voted_counts = voted_perceptron_train(X_train, y_train)
y_pred_voted = voted_perceptron_predict(X_test, voted_weight_vectors, voted_counts)
average_test_error_voted = 1 - (sum(1 for actual, predicted in zip(y_test, y_pred_voted) if actual == predicted) / len(y_test))

print("The Weight Vectors:")
for weights, bias in voted_weight_vectors:
    print(f"Weights: {weights}")

print("\nCounts for Each Vector:")
print(voted_counts)
print("\nAverage Test Error on Test Dataset:", average_test_error_voted)
