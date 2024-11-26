import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


def compute_kernel_matrix(X):
    return np.dot(X, X.T)


def dual_svm(X, y, C):
    n_samples = X.shape[0]
    K = compute_kernel_matrix(X)
    

    def objective(alpha):
        return 0.5 * np.dot(alpha, np.dot(K * np.outer(y, y), alpha)) - np.sum(alpha)
    

    def eq_constraint(alpha):
        return np.dot(alpha, y)
    

    bounds = [(0, C) for _ in range(n_samples)]
    alpha0 = np.zeros(n_samples)
    

    constraints = {'type': 'eq', 'fun': eq_constraint}
    solution = minimize(objective, alpha0, bounds=bounds, constraints=constraints, method='SLSQP')
    alphas = solution.x
    
    w = np.sum((alphas * y)[:, None] * X, axis=0)
    
    support_vector_indices = np.where((alphas > 1e-5) & (alphas < C))[0]
    b = np.mean(y[support_vector_indices] - np.dot(X[support_vector_indices], w))
    
    return w, b, alphas


def primal_svm(X, y, C, learning_rate=0.01, epochs=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    
    for _ in range(epochs):
        for i in range(n_samples):
            if y[i] * (np.dot(w, X[i]) + b) < 1:
                # Update rule for misclassified points
                w = (1 - learning_rate) * w + learning_rate * C * y[i] * X[i]
                b += learning_rate * C * y[i]
            else:
                # Update rule for correctly classified points
                w = (1 - learning_rate) * w
    return w, b


def predict(X, w, b):
    return np.sign(np.dot(X, w) + b)


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
    

    C_values = [100 / 873, 500 / 873, 700 / 873]
    results_dual = {}
    results_primal = {}
    
    for C in C_values:

        w_dual, b_dual, alphas = dual_svm(X_train, y_train, C)
        results_dual[C] = {'w': w_dual, 'b': b_dual, 'alphas': alphas}
        

        w_primal, b_primal = primal_svm(X_train, y_train, C)
        results_primal[C] = {'w': w_primal, 'b': b_primal}
        
 
        y_train_pred_dual = predict(X_train, w_dual, b_dual)
        y_test_pred_dual = predict(X_test, w_dual, b_dual)
        
        y_train_pred_primal = predict(X_train, w_primal, b_primal)
        y_test_pred_primal = predict(X_test, w_primal, b_primal)
        

        dual_train_accuracy = accuracy_score(y_train, y_train_pred_dual)
        dual_test_accuracy = accuracy_score(y_test, y_test_pred_dual)
        
        primal_train_accuracy = accuracy_score(y_train, y_train_pred_primal)
        primal_test_accuracy = accuracy_score(y_test, y_test_pred_primal)
        

        dual_train_error = 1 - dual_train_accuracy
        dual_test_error = 1 - dual_test_accuracy
        
        primal_train_error = 1 - primal_train_accuracy
        primal_test_error = 1 - primal_test_accuracy
        

        print(f"For C={C:.4f}:")
        print(f"Dual SVM -> Train Error: {dual_train_error:.4f}, Test Error: {dual_test_error:.4f}")
        print(f"Primal SVM -> Train Error: {primal_train_error:.4f}, Test Error: {primal_test_error:.4f}")
        print(f"Dual SVM -> Weight vector (w): {w_dual}, Bias (b): {b_dual}")
        print(f"Primal SVM -> Weight vector (w): {w_primal}, Bias (b): {b_primal}")
        print(f"Number of support vectors (Dual SVM): {np.sum(alphas > 1e-5)}\n")
    
    return results_dual, results_primal

if __name__ == "__main__":
    results_dual, results_primal = main()

