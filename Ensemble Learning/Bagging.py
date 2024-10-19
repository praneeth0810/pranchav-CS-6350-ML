import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

BANK_COLUMNS = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", 
                "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"]
TRAINING_DATASET_PATH = "bank-1/train.csv"
TESTING_DATASET_PATH = "bank-1/test.csv"

def calculate_entropy(target):
    probabilities = target.value_counts(normalize=True)
    return -np.sum(probabilities * np.log2(probabilities))

def choose_best_attribute(data, attributes):
    entropies = {}
    for attribute in attributes:
        grouped_data = data.groupby(attribute)
        weighted_entropy = sum((group.shape[0] / data.shape[0]) * calculate_entropy(group.iloc[:, -1]) for _, group in grouped_data)
        entropies[attribute] = weighted_entropy
    return min(entropies, key=entropies.get)

def build_decision_tree(data, attributes):
    if len(data.iloc[:, -1].unique()) == 1:
        return data.iloc[:, -1].iloc[0]
    if not attributes:
        return data.iloc[:, -1].mode()[0]
    
    best_attribute = choose_best_attribute(data, attributes)
    tree = {best_attribute: {}}
    for attribute_value, subset in data.groupby(best_attribute):
        subtree = build_decision_tree(subset, [attr for attr in attributes if attr != best_attribute])
        tree[best_attribute][attribute_value] = subtree
    return tree

def predict_instance(instance, tree):
    if not isinstance(tree, dict):
        return tree
    attribute = next(iter(tree))
    attribute_value = instance.get(attribute, None)
    if attribute_value not in tree[attribute]:
        return None
    return predict_instance(instance, tree[attribute][attribute_value])

def evaluate_accuracy(data, tree):
    correct_predictions = sum(1 for _, row in data.iterrows() if predict_instance(row.to_dict(), tree) == row.iloc[-1])
    return correct_predictions / len(data)


def bagging_ensemble(data, num_trees, sample_size):
    samples = [data.sample(n=sample_size, replace=True) for _ in range(num_trees)]
    with ThreadPoolExecutor() as executor:
        trees = list(executor.map(lambda sample: build_decision_tree(sample, list(sample.columns[:-1])), samples))
    return trees

def median_threshold(data, attribute):
    threshold = data[attribute].median()
    data[attribute] = (data[attribute] >= threshold).astype(int)

def plot_errors(train_errors, test_errors):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_errors) + 1), train_errors, label='Train Errors', color='blue')
    plt.plot(range(1, len(test_errors) + 1), test_errors, label='Test Errors', color='red')
    plt.title('Train and Test Errors vs. Number of Trees')
    plt.xlabel('Number of Trees')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    df_train = pd.read_csv(TRAINING_DATASET_PATH)
    df_train.columns = BANK_COLUMNS
    df_test = pd.read_csv(TESTING_DATASET_PATH)
    df_test.columns = BANK_COLUMNS
    
    numeric_attributes = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    for numeric_attr in numeric_attributes:
        median_threshold(df_train, numeric_attr)
        median_threshold(df_test, numeric_attr)
    
    trees = bagging_ensemble(df_train, 500, 1000)
    train_errors = [1 - evaluate_accuracy(df_train, tree) for tree in trees]
    test_errors = [1 - evaluate_accuracy(df_test, tree) for tree in trees]

    plot_errors(train_errors, test_errors)

if __name__ == "__main__":
    main()
