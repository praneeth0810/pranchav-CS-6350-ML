import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


train_df = pd.read_csv('bank-note/train.csv')  
test_df = pd.read_csv('bank-note/test.csv')    

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

def sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def compute_loss_and_gradient_mle(X, y, w):
    m = X.shape[0]
    z = X.dot(w)
    predictions = sigmoid(z)
    

    log_loss = -(1 / m) * np.sum(y * np.log(predictions + 1e-9) + (1 - y) * np.log(1 - predictions + 1e-9))

    gradient = (1 / m) * X.T.dot(predictions - y)
    return log_loss, gradient

def logistic_regression_mle_sgd(X, y, gamma_0=0.1, d=100, T=100):
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
            _, gradient = compute_loss_and_gradient_mle(X[i:i+1], y[i:i+1], w)
            w -= gamma_t * gradient
    return w

gamma_0, d = 0.1, 100  
results = []

w_mle = logistic_regression_mle_sgd(X_train, y_train, gamma_0=gamma_0, d=d)

train_preds = sigmoid(X_train.dot(w_mle)) >= 0.5
test_preds = sigmoid(X_test.dot(w_mle)) >= 0.5

train_error = 1 - accuracy_score(y_train, train_preds)
test_error = 1 - accuracy_score(y_test, test_preds)

print(f"Train Error: {train_error:.4f}")
print(f"Test Error: {test_error:.4f}")
