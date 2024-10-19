import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

bank_columns = ["age", "job", "marital", "education", "default", "balance", "housing", "loan",
                "contact", "day", "month", "duration", "campaign", "pdays", "previous",
                "poutcome", "y"]


def calc_entropy(df):
    attribute = df.keys()[-1]  
    prob = df[attribute].value_counts(normalize=True) 
    entropy = -np.sum(prob * np.log2(prob))
    return entropy

def max_entropy_attribute(df, attributes):
    avg_entropy = float('inf')
    selected_attribute = None
    np.random.shuffle(attributes) 
    for attribute in attributes:
        values = df[attribute].unique()
        entropy = 0.0
        for value in values:
            subset = df[df[attribute] == value]
            if len(subset) > 0:
                entropy += len(subset) / len(df) * calc_entropy(subset)
        if entropy < avg_entropy:
            avg_entropy = entropy
            selected_attribute = attribute
    return selected_attribute

def DecisionTreeClassifier(df, attributes=None, maxDepth=17):
    if attributes is None:
        attributes = df.keys()[:-1].tolist()

    if len(df[df.keys()[-1]].unique()) == 1:
        return df[df.keys()[-1]].iloc[0] 
    if len(attributes) == 0:
        return df[df.keys()[-1]].value_counts().idxmax() 

    selected_attribute = max_entropy_attribute(df, attributes)
    tree = {selected_attribute: {}}
    attributes.remove(selected_attribute) 

    for value in df[selected_attribute].unique():
        subset = df[df[selected_attribute] == value]
        if len(subset) == 0:
            tree[selected_attribute][value] = df[df.keys()[-1]].value_counts().idxmax()
        else:
            maxDepth -= 1
            if maxDepth == 0:
                tree[selected_attribute][value] = df[df.keys()[-1]].mode()[0]
            else:
                tree[selected_attribute][value] = DecisionTreeClassifier(subset, attributes.copy(), maxDepth)

    return tree


def predict(instance, tree):
    for node, branches in tree.items():
        value = instance[node]
        if value not in branches:
            return "NotLeaf"
        branch = branches[value]
        if isinstance(branch, dict):
            return predict(instance, branch)
        else:
            return branch

def evaluate(df, trees):
    correct_predictions = 0
    for i in range(len(df)):
        instance = df.iloc[i, :-1] 
        predictions = [predict(instance, tree) for tree in trees]
        final_prediction = max(set(predictions), key=predictions.count)
        if df.iloc[i, -1] == final_prediction:
            correct_predictions += 1
    accuracy = correct_predictions / len(df)
    return accuracy

def train_single_tree(train_sample):
    return DecisionTreeClassifier(train_sample)

def bagging_decision_trees(train_df, test_df, num_trees, sample_size, max_workers=4):
    each_tree_train_errors = []
    each_tree_test_errors = []
    trees = []
    single_tree_predictions = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for _ in range(num_trees):
            train_sample = train_df.sample(n=sample_size, replace=False)
            futures.append(executor.submit(train_single_tree, train_sample))

        for future in futures:
            tree = future.result()
            trees.append(tree)

            train_accuracy = evaluate(train_df.sample(n=sample_size, replace=False), [tree])
            test_accuracy = evaluate(test_df, [tree])
            each_tree_train_errors.append(1 - train_accuracy)
            each_tree_test_errors.append(1 - test_accuracy)

            if len(trees) == 1:
                single_tree_predictions = [predict(instance, tree) for _, instance in test_df.iterrows()]

    return each_tree_train_errors, each_tree_test_errors, single_tree_predictions


def median_thresholding(df, attribute):
    threshold = df[attribute].median()
    df[attribute] = (df[attribute] >= threshold).astype(int)

if __name__ == "__main__":
    df_train = pd.read_csv("bank-1/train.csv")
    df_train.columns = bank_columns
    df_test = pd.read_csv("bank-1/test.csv")
    df_test.columns = bank_columns

    numeric_attributes = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    for numeric_attr in numeric_attributes:
        median_thresholding(df_train, numeric_attr)
        median_thresholding(df_test, numeric_attr)

    train_errors = []
    test_errors = []
    single_tree_test_errors = []
    iterations = 100
    numberOfTrees = 500

    for iteration in range(iterations):
        bag_train_error, bag_test_error, single_tree_predictions = bagging_decision_trees(df_train, df_test, numberOfTrees, 1000, max_workers=8)
        train_errors.append(bag_train_error)
        test_errors.append(bag_test_error)
        single_tree_test_errors.append(single_tree_predictions)

    value = np.array(df_test.iloc[:, -1].tolist())
    value[value == 'yes'] = 1
    value[value == 'no'] = -1
    value = value.astype(int)

    single_tree_predictions_numeric = np.array([
        [1 if pred == 'yes' else -1 for pred in single_tree_pred_list]
        for single_tree_pred_list in single_tree_test_errors
    ])

    bias_single = np.mean(np.square(single_tree_predictions_numeric.mean(axis=0) - value))
    variance_single = np.var(single_tree_predictions_numeric.mean(axis=0))
    print(f"Single Tree - Bias: {bias_single}, Variance: {variance_single}, Total Error: {bias_single + variance_single}")

    mean_bagged = np.mean(np.mean(test_errors, axis=0))
    bias_bagged = np.mean(np.square(mean_bagged - value))
    variance_bagged = np.var(np.mean(test_errors, axis=0))
    print(f"Bagged Trees - Bias: {bias_bagged}, Variance: {variance_bagged}, Total Error: {bias_bagged + variance_bagged}")

    avg_train_errors = np.mean(train_errors, axis=0)
    avg_test_errors = np.mean(test_errors, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, numberOfTrees+1), avg_train_errors, label="Train Error", color='blue')
    plt.plot(range(1, numberOfTrees+1), avg_test_errors, label="Test Error", color='orange')
    plt.xlabel("Number of Trees")
    plt.ylabel("Error")
    plt.title("Train and Test Error vs. Number of Trees")
    plt.legend()
    plt.show()
