import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


train_df = pd.read_csv('bank-note/train.csv') 
test_df = pd.read_csv('bank-note/test.csv')

# Extract features and labels
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss_and_gradient(X, y, w, v):
    m = X.shape[0]
    z = X.dot(w)
    predictions = sigmoid(z)
    
    
    log_loss = -(1 / m) * np.sum(y * np.log(predictions + 1e-9) + (1 - y) * np.log(1 - predictions + 1e-9))
    regularization = (1 / (2 * v)) * np.sum(w ** 2)
    total_loss = log_loss + regularization

    gradient = (1 / m) * X.T.dot(predictions - y) + (1 / v) * w
    return total_loss, gradient

def logistic_regression_sgd(X, y, v, gamma_0=0.1, d=100, T=100):
    n_features = X.shape[1]
    w = np.zeros(n_features)  
    t = 0 
    
    for epoch in range(T):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]
        
        for i in range(X.shape[0]):
            t += 1
            gamma_t = gamma_0 / (1 + (gamma_0 / d) * t) 
            _, gradient = compute_loss_and_gradient(X[i:i+1], y[i:i+1], w, v)
            w -= gamma_t * gradient  
    return w

variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
results = []

for v in variances:
    w = logistic_regression_sgd(X_train, y_train, v)
    

    train_preds = sigmoid(X_train.dot(w)) >= 0.5
    test_preds = sigmoid(X_test.dot(w)) >= 0.5
    
    train_error = 1 - accuracy_score(y_train, train_preds)
    test_error = 1 - accuracy_score(y_test, test_preds)
    results.append((v, train_error, test_error))

results_df = pd.DataFrame(results, columns=["Variance", "Train Error", "Test Error"])
print(results_df)
