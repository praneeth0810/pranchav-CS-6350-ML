import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

def gaussian_kernel(x1, x2, gamma):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / gamma)


def compute_kernel_matrix(X, gamma):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = gaussian_kernel(X[i], X[j], gamma)
    return K

def dual_svm_gaussian(X, y, C, gamma):
    n_samples = X.shape[0]
    K = compute_kernel_matrix(X, gamma)
    

    def objective(alpha):
        return 0.5 * np.dot(alpha, np.dot(K * np.outer(y, y), alpha)) - np.sum(alpha)
    

    def eq_constraint(alpha):
        return np.dot(alpha, y)
    

    bounds = [(0, C) for _ in range(n_samples)]
    alpha0 = np.zeros(n_samples)  # Initial guess
    constraints = {'type': 'eq', 'fun': eq_constraint}
    
    solution = minimize(objective, alpha0, bounds=bounds, constraints=constraints, method='SLSQP')
    alphas = solution.x

    support_vector_indices = np.where((alphas > 1e-5) & (alphas < C))[0]
    b = np.mean([y[i] - np.sum(alphas * y * K[i]) for i in support_vector_indices])
    return alphas, b

def predict_gaussian(X_train, X_test, alphas, y_train, gamma, b):
    n_samples = X_train.shape[0]
    y_pred = []
    for x_test in X_test:
        decision = 0
        for i in range(n_samples):
            decision += alphas[i] * y_train[i] * gaussian_kernel(X_train[i], x_test, gamma)
        decision += b
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
    C_values = [100 / 873, 500 / 873, 700 / 873]


    results = []
    for gamma in gamma_values:
        for C in C_values:

            alphas, b = dual_svm_gaussian(X_train, y_train, C, gamma)

            y_train_pred = predict_gaussian(X_train, X_train, alphas, y_train, gamma, b)
            y_test_pred = predict_gaussian(X_train, X_test, alphas, y_train, gamma, b)
            

            train_error = 1 - accuracy_score(y_train, y_train_pred)
            test_error = 1 - accuracy_score(y_test, y_test_pred)

            results.append((gamma, C, train_error, test_error))
            

            print(f"Gamma: {gamma}, C: {C:.4f}")
            print(f"Train Error: {train_error:.4f}, Test Error: {test_error:.4f}\n")
    
    results_df = pd.DataFrame(results, columns=["Gamma", "C", "Train Error", "Test Error"])
    best_result = results_df.loc[results_df["Test Error"].idxmin()]
    print("Best Combination:")
    print(best_result)

if __name__ == "__main__":
    main()


