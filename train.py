
from sklearn import datasets  # helps us load sample datasets
from sklearn.model_selection import train_test_split  # helps us split data into train/test
import numpy as np  # helps us work with numbers
from DecisionTree import DecisionTree  # our custom decision tree

# Load the breast cancer dataset (real medical data!)
data = datasets.load_breast_cancer()
X, y = data.data, data.target  # X = features (measurements), y = answers (cancer or not)

# Split our data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234  # random_state makes results repeatable
)

# Create our decision tree with max depth of 10
clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)  # teach the tree using training data
predictions = clf.predict(X_test)  # make predictions on test data

def accuracy(y_test, y_pred):
    """Calculates what percentage of predictions were correct."""
    return np.sum(y_test == y_pred) / len(y_test)  # count correct predictions / total predictions

# Calculate and print how accurate our tree is
acc = accuracy(y_test, predictions)
print(f"Accuracy: {acc:.2%}")  # print as percentage with 2 decimal places