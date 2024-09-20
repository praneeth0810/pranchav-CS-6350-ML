import pandas as pd
import numpy as np

# Function to compute entropy
def entropy(data):
    target = data.iloc[:, -1]  # Assuming that last column contains the class labels
    value_counts = target.value_counts(normalize=True)
    return -sum(value_counts * np.log2(value_counts + 1e-9))

# Function to compute majority error
def majority_error(data):
    target = data.iloc[:, -1]
    value_counts = target.value_counts(normalize=True)
    return 1 - value_counts.max()

# Function to compute Gini index
def gini_index(data):
    target = data.iloc[:, -1]
    value_counts = target.value_counts(normalize=True)
    return 1 - sum(value_counts ** 2)

# Function to calculate information gain for a split
def information_gain(data, attr, method):
    initial_impurity = method(data)
    values = data[attr].unique()
    weighted_impurity = 0

    for value in values:
        subset = data[data[attr] == value]
        prob = len(subset) / len(data)
        weighted_impurity += prob * method(subset)

    return initial_impurity - weighted_impurity

# Function to find the best attribute for splitting
def best_split(data, attributes, method):
    gains = {attr: information_gain(data, attr, method) for attr in attributes}
    return max(gains, key=gains.get)

# Function to create a decision tree
def build_tree(data, attributes, method, depth, max_depth):
    target = data.iloc[:, -1]
    
    # Base cases: all labels are the same or maximum depth reached
    if len(target.unique()) == 1:
        return target.values[0]
    if depth >= max_depth or len(attributes) == 0:
        return target.mode()[0]  # Return majority class

    # Choose the best attribute to split
    best_attr = best_split(data, attributes, method)
    tree = {best_attr: {}}
    remaining_attrs = [attr for attr in attributes if attr != best_attr]

    # Recursive case: create branches
    for value in data[best_attr].unique():
        subset = data[data[best_attr] == value]
        if subset.empty:
            tree[best_attr][value] = target.mode()[0]
        else:
            tree[best_attr][value] = build_tree(subset, remaining_attrs, method, depth + 1, max_depth)

    return tree

# Function to classify a single instance
def classify(tree, instance):
    if not isinstance(tree, dict):
        return tree
    attribute = next(iter(tree))
    value = instance[attribute]
    return classify(tree[attribute][value], instance) if value in tree[attribute] else None

# Function to calculate error rate
def error_rate(data, tree):
    predictions = data.apply(lambda row: classify(tree, row), axis=1)
    return (predictions != data.iloc[:, -1]).mean()

# Function to run decision tree training and evaluation
def evaluate_decision_tree(train_data, test_data,max_depth):
    results = []
    methods = {
        'entropy': entropy,
        'majority_error': majority_error,
        'gini_index': gini_index
    }

    for method_name, method in methods.items():
        for depth in range(1, max_depth + 1):
            tree = build_tree(train_data, train_data.columns[:-1], method, 0, depth)
            train_error = error_rate(train_data, tree)
            test_error = error_rate(test_data, tree)
            results.append({
                'Method': method_name,
                'Depth': depth,
                'Train Error': train_error,
                'Test Error': test_error
            })

    return pd.DataFrame(results)

if __name__ == "__main__":
    # Load training and test datasets
    train_df = pd.read_csv('car/train.csv',header = None )
    test_df = pd.read_csv('car/test.csv',header = None)
    
    max_depth = int(input("Give your input for maximum depth of the tree: "))

    # Run evaluation and print results
    results = evaluate_decision_tree(train_df, test_df,max_depth)
    print(results)
