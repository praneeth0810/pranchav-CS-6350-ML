import pandas as pd
import numpy as np
from random import sample
import matplotlib.pyplot as plt

bank_columns = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact",
                "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"]
training_dataset = "bank-1/train.csv"
test_dataset = "bank-1/test.csv"

def compute_entropy(distribution):
    distribution += 1e-9
    return -np.sum(distribution * np.log2(distribution))

def get_class_distribution(df, column):
    return df[column].value_counts(normalize=True).values

def calculate_attribute_entropy(df, attribute):
    total_length = len(df)
    total_entropy = 0.0
    for value, subgroup in df.groupby(attribute):
        class_distribution = get_class_distribution(subgroup, df.columns[-1])
        total_entropy += (len(subgroup) / total_length) * compute_entropy(class_distribution)
    return total_entropy

def select_optimal_attribute(df, potential_attributes, max_features=None):
    if max_features and len(potential_attributes) > max_features:
        potential_attributes = sample(potential_attributes, max_features)
    entropy_dict = {attr: calculate_attribute_entropy(df, attr) for attr in potential_attributes}
    return min(entropy_dict, key=entropy_dict.get)

def build_decision_tree(df, attributes=None, feature_limit=None):
    if attributes is None:
        attributes = df.columns[:-1].tolist()
    if df[df.columns[-1]].nunique() == 1:
        return df.iloc[0, -1]
    if not attributes:
        return df[df.columns[-1]].mode()[0]

    best_attribute = select_optimal_attribute(df, attributes, feature_limit)
    decision_tree = {best_attribute: {}}
    remaining_attributes = [attr for attr in attributes if attr != best_attribute]

    for attribute_value, subset in df.groupby(best_attribute):
        decision_tree[best_attribute][attribute_value] = build_decision_tree(subset, remaining_attributes, feature_limit)

    return decision_tree

def make_prediction(data_point, decision_tree):
    if not isinstance(decision_tree, dict):
        return decision_tree
    root_attribute = next(iter(decision_tree))
    attribute_value = data_point[root_attribute]
    next_node = decision_tree[root_attribute].get(attribute_value, "NotLeaf")
    return make_prediction(data_point, next_node)

def calculate_accuracy(dataframe, forest):
    correct = 0
    for _, row in dataframe.iterrows():
        predictions = [make_prediction(row, tree) for tree in forest]
        if max(set(predictions), key=predictions.count) == row.iloc[-1]:
            correct += 1
    return correct / len(dataframe)


def RandomForest(df_train, df_test, num_features, num_trees, sample_size):
    train_errors = []
    test_errors = []
    forest = []
    sample_size = min(sample_size, len(df_train))
    for i in range(num_trees):
        sample_df = df_train.sample(n=sample_size, replace=True)
        tree = build_decision_tree(sample_df, feature_limit=num_features)
        forest.append(tree)
        train_accuracy = calculate_accuracy(df_train, forest)
        test_accuracy = calculate_accuracy(df_test, forest)
        train_errors.append(1 - train_accuracy)
        test_errors.append(1 - test_accuracy)

    return train_accuracy, test_accuracy

if __name__ == "__main__":

    df_train = pd.read_csv(training_dataset)
    df_train.columns = bank_columns
    df_test = pd.read_csv(test_dataset)
    df_test.columns = bank_columns
    
    if df_train.empty or df_test.empty:
        raise ValueError("DataFrames are empty or not loaded properly. Check file paths and data.")

    numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    for feature in numeric_features:
        median = df_train[feature].median()
        df_train[feature] = (df_train[feature] >= median).astype(int)
        df_test[feature] = (df_test[feature] >= median).astype(int)

    num_features = 6
    num_trees = 50
    sample_size = 1000
    train_accuracy, test_accuracy = RandomForest(df_train, df_test, num_features, num_trees, sample_size)
    print(f"Train Error: {1-train_accuracy:.2f}")
    print(f"Test Error: {1-test_accuracy:.2f}")
