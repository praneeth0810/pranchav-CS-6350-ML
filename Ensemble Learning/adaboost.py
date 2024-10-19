import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def median_thresholding(df, attribute):
    threshold = df[attribute].median()
    df[attribute] = (df[attribute] >= threshold).astype(int)


bank_columns = [
    'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
    'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y'
]


df_train = pd.read_csv("bank-1/train.csv", header=None, names=bank_columns)
df_test = pd.read_csv("bank-1/test.csv", header=None, names=bank_columns)

numeric_attributes = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

for numeric_attr in numeric_attributes:
    median_thresholding(df_train, numeric_attr)
    median_thresholding(df_test, numeric_attr)


X_train = df_train.drop(columns=['y']).values
y_train = df_train['y'].apply(lambda x: 1 if x == 'yes' else -1).values

X_test = df_test.drop(columns=['y']).values
y_test = df_test['y'].apply(lambda x: 1 if x == 'yes' else -1).values


class DecisionTreeStump:
    def __init__(self, attribute, threshold, label_lesser, label_greater):
        self.attribute = attribute
        self.threshold = threshold
        self.label_lesser = label_lesser
        self.label_greater = label_greater

    def predict(self, X):
        return np.where(X[:, self.attribute] <= self.threshold, self.label_lesser, self.label_greater)

    @staticmethod
    def weighted_information_gain(X, y, attribute, threshold, weights):
        indices_less_threshold = X[:, attribute] <= threshold
        indices_greater_threshold = X[:, attribute] > threshold

        weighted_less = np.sum(weights[indices_less_threshold])
        weighted_greater = np.sum(weights[indices_greater_threshold])

        if weighted_less == 0 or weighted_greater == 0:
            return 0

        entropy_less = -np.sum(weights[indices_less_threshold] * np.log2(weights[indices_less_threshold] / weighted_less))
        entropy_greater = -np.sum(weights[indices_greater_threshold] * np.log2(weights[indices_greater_threshold] / weighted_greater))

        total_entropy = (weighted_less / len(y)) * entropy_less + (weighted_greater / len(y)) * entropy_greater
        return total_entropy

    @staticmethod
    def split_max_IG(X, y, weights):
        num_features = X.shape[1]
        selected_threshold = None
        selected_attribute = None
        min_entropy = float('inf')

        for feature in range(num_features):
            unique_values = np.unique(X[:, feature])
            epsilon = 1e-9
            thresholds = (unique_values[:-1] + unique_values[1:] + epsilon) / 2

            for threshold in thresholds:
                entropy = DecisionTreeStump.weighted_information_gain(X, y, feature, threshold, weights)
                if entropy < min_entropy:
                    min_entropy = entropy
                    selected_threshold = threshold
                    selected_attribute = feature

        return selected_attribute, selected_threshold


class AdaBoost:
    def __init__(self, iterations, learning_rate=0.5):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.alpha = []
        self.stumps = []
        self.train_errors = []
        self.test_errors = []
        self.stump_errors = []

    def fit(self, X, y):
        X = self.numeric_encoding(X)
        y = y.astype(float)
        weights = np.full(len(y), (1 / len(y))) 

        for t in range(self.iterations):
            stump = self.train_stump(X, y, weights)
            self.stumps.append(stump)
            predictions = stump.predict(X)
            error = np.sum(weights * (predictions != y))

            alpha = self.learning_rate * np.log((1 - error) / max(error, 1e-10)) if error != 0 else 1.0
            self.alpha.append(alpha)

            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

            train_error = 1 - accuracy_score(y_train, self.predict(X_train))
            test_error = 1 - accuracy_score(y_test, self.predict(X_test))
            self.train_errors.append(train_error)
            self.test_errors.append(test_error)
            self.stump_errors.append(error / len(y))

            if error == 0 or train_error == 0:
                print(f"Early stopping at iteration {t} due to zero error.")
                break

    @staticmethod
    def numeric_encoding(X):
        for i in range(X.shape[1]):
            column = X[:, i]
            if not np.issubdtype(column.dtype, np.number):
                column = pd.to_numeric(column, errors='coerce')
                column[np.isnan(column)] = 0
            X[:, i] = column
        return X

    def train_stump(self, X, y, weights):
        best_error = float('inf')
        selected_stump = None

        for _ in range(2):  
            attribute, threshold = DecisionTreeStump.split_max_IG(X, y, weights)
            for label_lesser in [-1, 1]:
                predictions = np.where(X[:, attribute] <= threshold, label_lesser, -label_lesser)
                error = np.sum(weights * (predictions != y))
                if best_error >= error:
                    best_error = error
                    selected_stump = DecisionTreeStump(attribute, threshold, label_lesser, -label_lesser)
        return selected_stump

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for alpha, stump in zip(self.alpha, self.stumps):
            predictions += alpha * stump.predict(X)
        return np.sign(predictions)

def plot_errors(iterations, train_errors, test_errors, stump_errors):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_errors)), train_errors, label='Train Error')
    plt.plot(range(len(test_errors)), test_errors, label='Test Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Train and Test Errors vs. Iteration')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(len(stump_errors)), stump_errors, label='Decision Stump Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Decision Stump Errors vs. Iteration')
    plt.legend()

    plt.tight_layout()
    plt.show()

iterations = 500
learning_rate = 0.5
boosting = AdaBoost(iterations, learning_rate)

boosting.fit(X_train, y_train)

plot_errors(iterations, boosting.train_errors, boosting.test_errors, boosting.stump_errors)

train_error = 1 - accuracy_score(y_train, boosting.predict(X_train))
test_error = 1 - accuracy_score(y_test, boosting.predict(X_test))

print("Training Error = ", train_error)
print("Testing Error = ", test_error)