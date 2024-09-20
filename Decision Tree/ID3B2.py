import pandas as pd
import numpy as np

# Function to compute entropy
def entropy(data):
    target = data.iloc[:, -1]  # Last column contains the class labels
    value_counts = target.value_counts(normalize=True)
    return -sum(value_counts * np.log2(value_counts + 1e-9))  # Avoid log(0) by adding small value

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

# Function to replace "unknown" with the majority value in categorical columns
def replace_unknowns_with_majority(train_data, test_data):
    categorical_columns = train_data.select_dtypes(include='object').columns

    for col in categorical_columns:
        # Find the most frequent value in the training set for each categorical column
        majority_value = train_data[col].mode()[0]
        
        # Replace "unknown" with the majority value in both train and test datasets
        train_data[col] = train_data[col].replace('unknown', majority_value)
        test_data[col] = test_data[col].replace('unknown', majority_value)

    return train_data, test_data

# Function to convert numerical attributes to binary (based on median)
def binarize_numerical_attributes(train_data, test_data, numerical_columns):
    for column in numerical_columns:
        median = train_data[column].median()
        # Convert to binary for both train and test
        train_data[column] = np.where(train_data[column] > median, 1, 0)
        test_data[column] = np.where(test_data[column] > median, 1, 0)
    return train_data, test_data

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
def evaluate_decision_tree(train_data, test_data, max_depth, numerical_columns):
    #Note
    print("The code is running...it may take upto 3 min")
    # Handle "unknown" values by replacing them with the majority value
    train_data, test_data = replace_unknowns_with_majority(train_data, test_data)

    # Binarize numerical attributes in both train and test datasets
    train_data, test_data = binarize_numerical_attributes(train_data, test_data, numerical_columns)
    
    train_results = pd.DataFrame(index=range(1, max_depth + 1), columns=['Entropy', 'Majority Error', 'Gini Index'])
    test_results = pd.DataFrame(index=range(1, max_depth + 1), columns=['Entropy', 'Majority Error', 'Gini Index'])
    
    methods = {
        'Entropy': entropy,
        'Majority Error': majority_error,
        'Gini Index': gini_index
    }

    for method_name, method in methods.items():
        for depth in range(1, max_depth + 1):
            tree = build_tree(train_data, train_data.columns[:-1], method, 0, depth)
            train_error = error_rate(train_data, tree)
            test_error = error_rate(test_data, tree)
            train_results.at[depth, method_name] = train_error
            test_results.at[depth, method_name] = test_error

    return train_results, test_results

# Example usage
if __name__ == "__main__":
    # Load training and test datasets
    train_df = pd.read_csv('bank/train.csv', header=None)
    test_df = pd.read_csv('bank/test.csv', header=None)
    
    # List of numerical columns based on the attribute description
    numerical_columns = [0, 5, 9, 11, 12, 13, 14]  # Indices of numerical columns in the dataset
    
    # Set maximum depth to 16
    max_depth = int(input("Give your input for maximum depth of the tree: "))
    
    # Run evaluation
    train_results, test_results = evaluate_decision_tree(train_df, test_df, max_depth, numerical_columns)
    
    print("Training data prediction Error Results:")
    print(train_results)
    print("\nTest data prediction Error Results:")
    print(test_results)
