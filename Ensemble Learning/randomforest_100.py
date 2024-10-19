import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

bank_columns = [
    "age", "job", "marital", "education", "default", "balance", "housing", "loan", 
    "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"
]
training_dataset = "bank-1/train.csv"
test_dataset = "bank-1/test.csv"

def calculate_entropy(df, attribute):
    probabilities = df[attribute].value_counts(normalize=True)
    return -(probabilities * np.log2(probabilities + np.finfo(float).eps)).sum()

def select_attribute(df, attributes, num_features):
    if not isinstance(attributes, list):
        attributes = list(attributes)

    if num_features < len(attributes):
        attributes = random.sample(attributes, num_features)
    
    entropy_values = {attr: calculate_entropy(df, attr) for attr in attributes}
    return min(entropy_values, key=entropy_values.get)


def build_tree(df, attributes=None, num_features=None):
    if attributes is None:
        attributes = df.columns[:-1]
    
    if attributes.empty:
        return df[df.columns[-1]].mode()[0]

    if df[df.columns[-1]].nunique() == 1:
        return df.iloc[0, -1]
    
    selected_attribute = select_attribute(df, attributes, num_features)
    node = {selected_attribute: {}}
    
    for value, subset in df.groupby(selected_attribute):
        if subset.empty:
            node[selected_attribute][value] = df[df.columns[-1]].mode()[0]
        else:
            remaining_attributes = attributes.drop(selected_attribute)
            node[selected_attribute][value] = build_tree(subset, remaining_attributes, num_features)
    return node


def predict_tree(instance, tree):
    if not isinstance(tree, dict):
        return tree
    attribute = next(iter(tree))
    if instance[attribute] in tree[attribute]:
        return predict_tree(instance, tree[attribute][instance[attribute]])
    return np.nan

def evaluate_trees(df, forest):
    correct_predictions = sum(
        max(set([predict_tree(row, tree) for tree in forest]), key=[predict_tree(row, tree) for tree in forest].count) == row.iloc[-1]
        for _, row in df.iterrows()
    )
    return correct_predictions / len(df)

def random_forest(df_train, df_test, num_features, num_trees, sample_size):
    trees = [build_tree(df_train.sample(n=sample_size), num_features=num_features) for _ in range(num_trees)]
    train_accuracy = evaluate_trees(df_train, trees)
    test_accuracy = evaluate_trees(df_test, trees)
    return 1 - train_accuracy, 1 - test_accuracy

def median_threshold(df, attributes):
    for attr in attributes:
        median = df[attr].median()
        df[attr] = (df[attr] >= median).astype(int)

def bias_variance_decomposition(errors, true_values):
    bias = np.mean((np.mean(errors, axis=0) - true_values) ** 2)
    variance = np.mean(np.var(errors, axis=0))
    return bias, variance

if __name__ == "__main__":
    df_train = pd.read_csv(training_dataset)
    df_train.columns = bank_columns
    df_test = pd.read_csv(test_dataset)
    df_test.columns = bank_columns          

    df_train['y'] = df_train['y'].map({'yes': 1, 'no': 0})
    df_test['y'] = df_test['y'].map({'yes': 1, 'no': 0})

    median_threshold(df_train, ["age", "balance", "day", "duration", "campaign", "pdays", "previous"])
    median_threshold(df_test, ["age", "balance", "day", "duration", "campaign", "pdays", "previous"])

    num_features = [2, 4, 6]
    num_iterations = 10
    num_trees = 20
    sample_size = 1000

    for features in num_features:
        train_errors = []
        test_errors = []
        for _ in range(num_iterations):
            train_error, test_error = random_forest(df_train, df_test, features, num_trees, sample_size)
            train_errors.append(train_error)
            test_errors.append(test_error)

        test_errors_array = np.array(test_errors)
        true_values = df_test['y'].values
        bias, variance = bias_variance_decomposition(test_errors_array, true_values)
        print(f"Features: {features} - Bias: {bias:.4f}, Variance: {variance:.4f}")

    final_train_error, final_test_error = random_forest(df_train, df_test, 6, 100, 1000)
    print(f"Final Training Error: {final_train_error:.4f}")
    print(f"Final Testing Error: {final_test_error:.4f}")