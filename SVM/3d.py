import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

def gaussian_kernel(x1, x2, gamma):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / gamma)


def kernel_perceptron(X_train, y_train, gamma, max_epochs=100):
    n_samples = X_train.shape[0]
    alphas = np.zeros(n_samples)  
    
 
    for _ in range(max_epochs):
        for i in range(n_samples):
            
            decision = np.sum([
                alphas[j] * y_train[j] * gaussian_kernel(X_train[j], X_train[i], gamma)
                for j in range(n_samples)
            ])
            if y_train[i] * decision <= 0:
                alphas[i] += 1
    return alphas

def predict_kernel_perceptron(X_train, X_test, alphas, y_train, gamma):
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    y_pred = []
    for i in range(n_test):
        decision = np.sum([
            alphas[j] * y_train[j] * gaussian_kernel(X_train[j], X_test[i], gamma)
            for j in range(n_train)
        ])
        y_pred.append(np.sign(decision))
    return np.array(y_pred)

def main():

    train_data = pd.read_csv("bank-note/train.csv", header=None)
    test_data = pd.read_csv("bank-note/test.csv", header=None)
    columns = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Label']
    train_data.columns = columns
    test_data.columns = columns

    X_train = train_data.iloc[:, :-1].values
    y_train = train_data['Label'].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data['Label'].values

    y_train = np.where(y_train == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    gamma_values = [0.1, 0.5, 1, 5, 100]
    results = []

    for gamma in gamma_values:
        alphas = kernel_perceptron(X_train, y_train, gamma)

        y_train_pred = predict_kernel_perceptron(X_train, X_train, alphas, y_train, gamma)
        y_test_pred = predict_kernel_perceptron(X_train, X_test, alphas, y_train, gamma)

        train_error = 1 - accuracy_score(y_train, y_train_pred)
        test_error = 1 - accuracy_score(y_test, y_test_pred)

        results.append((gamma, train_error, test_error))
        print(f"Gamma = {gamma}, Train Error = {train_error:.4f}, Test Error = {test_error:.4f}")

    results_df = pd.DataFrame(results, columns=["Gamma", "Train Error", "Test Error"])
    print("\nResults:")
    print(results_df)

if __name__ == "__main__":
    main()
