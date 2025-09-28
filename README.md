# Decision Tree Implementation

A custom implementation of a Decision Tree classifier from scratch using Python and NumPy. This project demonstrates the core concepts of decision tree algorithms including entropy calculation, information gain, and tree construction.

## Features

- **Custom Decision Tree Class**: Complete implementation with configurable parameters
- **Entropy-based Splitting**: Uses information gain to determine optimal splits
- **Flexible Parameters**: Configurable minimum samples per split, maximum depth, and number of features
- **Breast Cancer Dataset**: Example implementation using sklearn's breast cancer dataset
- **Performance Evaluation**: Built-in accuracy calculation

## Files

- `DecisionTree.py` - Main decision tree implementation
- `train.py` - Training script with example usage
- `README.md` - This documentation file

## Algorithm Details

### Core Components

1. **Node Class**: Represents tree nodes with feature, threshold, and child nodes
2. **DecisionTree Class**: Main classifier with fit and predict methods
3. **Entropy Calculation**: Measures impurity in data splits
4. **Information Gain**: Determines the best feature and threshold for splitting

### Key Methods

- `fit(X, y)`: Trains the decision tree on training data
- `predict(X)`: Makes predictions on test data
- `_grow_tree()`: Recursively builds the decision tree
- `_best_split()`: Finds optimal feature and threshold for splitting
- `_information_gain()`: Calculates information gain for splits
- `_entropy()`: Computes entropy for impurity measurement

## Usage

### Basic Example

```python
from DecisionTree import DecisionTree
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset
data = datasets.load_breast_cancer()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# Create and train model
clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)

# Calculate accuracy
def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, predictions)
print(f"Accuracy: {acc}")
```

### Parameters

- `min_sample_split` (default: 2): Minimum number of samples required to split a node
- `max_depth` (default: 10): Maximum depth of the tree
- `n_features` (default: None): Number of features to consider for splitting (None = all features)

## Running the Code

1. Ensure you have the required dependencies:
   ```bash
   pip install numpy scikit-learn
   ```

2. Run the training script:
   ```bash
   python train.py
   ```

## Algorithm Explanation

### Decision Tree Construction

1. **Root Node**: Start with the entire dataset
2. **Feature Selection**: For each node, evaluate all features and thresholds
3. **Best Split**: Choose the split that maximizes information gain
4. **Recursive Splitting**: Continue splitting until stopping criteria are met
5. **Leaf Nodes**: Assign the most common class label

### Stopping Criteria

- Maximum depth reached
- All samples belong to the same class
- Number of samples is less than minimum required for splitting

### Information Gain Formula

```
Information Gain = Entropy(parent) - Weighted Average of Entropy(children)
```

Where entropy is calculated as:
```
Entropy = -Σ(p(x) * log(p(x)))
```

## Dependencies

- Python 3.6+
- NumPy
- scikit-learn (for dataset loading and train/test split)

## Educational Purpose

This implementation is designed for educational purposes to understand:
- How decision trees work internally
- Entropy and information gain concepts
- Tree construction algorithms
- Machine learning model evaluation

## License

This project is open source and available under the MIT License.
