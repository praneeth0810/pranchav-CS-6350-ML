import numpy as np
import pandas as pd


def svm_sgd_schedule_a(X, y, C, T, gamma_0=0.01, a=0.01):
    n, d = X.shape
    w = np.zeros(d) 
    b = 0  
    train_errors = []

    for epoch in range(T):
        # Shuffle the data
        indices = np.random.permutation(n)
        X, y = X[indices], y[indices]

        for i in range(n):
            
            t = epoch * n + i  
            gamma_t = gamma_0 / (1 + gamma_0 * a * t)

            
            if y[i] * (np.dot(w, X[i]) + b) < 1:
                w = (1 - gamma_t) * w + gamma_t * C * y[i] * X[i]
                b += gamma_t * C * y[i]
            else:
                w = (1 - gamma_t) * w

        
        predictions = np.sign(np.dot(X, w) + b)
        train_error = np.mean(predictions != y)
        train_errors.append(train_error)

    return w, b, train_errors


train_data = pd.read_csv("bank-note/train.csv", header=None)
test_data = pd.read_csv("bank-note/test.csv", header=None)


train_data[4] = train_data[4].replace({0: -1, 1: 1})
test_data[4] = test_data[4].replace({0: -1, 1: 1})


X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values


C_values = [100/873, 500/873, 700/873]
T = 100  
gamma_0 = 0.01  
a = 0.01 


results = []

for C in C_values:
    w, b, train_errors = svm_sgd_schedule_a(X_train, y_train, C, T, gamma_0=gamma_0, a=a)
    predictions = np.sign(np.dot(X_test, w) + b)
    test_error = np.mean(predictions != y_test)
    results.append({
        "C": f"{C:.6f}",
        "Train Error": round(train_errors[-1], 6),
        "Test Error": round(test_error, 6)
    })


results_df = pd.DataFrame(results)

print(results_df)
